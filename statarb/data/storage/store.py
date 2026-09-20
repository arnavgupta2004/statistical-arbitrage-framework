"""Local store: one Parquet file per ticker and table, plus a DuckDB catalog.

Layout::

    <root>/prices/<ticker>.parquet    date, ticker, open, high, low, close, volume   (vendor basis)
    <root>/actions/<ticker>.parquet   date, ticker, kind, value
    <root>/universe/                  point-in-time universe snapshots
    <root>/catalog.duckdb             manifest (one row per ticker) + fetch_log (append-only)

Parquet holds the data (portable, columnar, diff-able by hash); DuckDB holds provenance and lets
notebooks query everything with SQL (``store.sql("select ... from prices")``).  The catalog is
derivable from the Parquet files, never the other way round.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from urllib.parse import quote

import duckdb
import pandas as pd

from statarb.data.sources.base import ACTION_COLUMNS, PRICE_COLUMNS, empty_actions

_MANIFEST_COLS = [
    "ticker",
    "source",
    "status",
    "first_date",
    "last_date",
    "n_rows",
    "content_hash",
    "last_attempt_at",
    "last_success_at",
    "last_full_refresh_at",
    "last_empty_at",
    "message",
]


def frame_hash(prices: pd.DataFrame, actions: pd.DataFrame) -> str:
    """Content hash of the values (not the file bytes, which vary across pyarrow versions)."""
    h = hashlib.sha256()
    for df in (prices[PRICE_COLUMNS], actions[ACTION_COLUMNS] if len(actions) else actions):
        if len(df):
            h.update(pd.util.hash_pandas_object(df, index=True).to_numpy().tobytes())
        h.update(b"|")
    return h.hexdigest()[:16]


def _ts(x):
    return None if x is None or pd.isna(x) else pd.Timestamp(x).to_pydatetime()


class ParquetStore:
    def __init__(self, root: str | Path):
        self.root = Path(root)
        for sub in ("prices", "actions", "universe", "reports"):
            (self.root / sub).mkdir(parents=True, exist_ok=True)
        self._con = duckdb.connect(str(self.root / "catalog.duckdb"))
        self._con.execute(
            """CREATE TABLE IF NOT EXISTS manifest (
                ticker VARCHAR PRIMARY KEY, source VARCHAR, status VARCHAR,
                first_date TIMESTAMP, last_date TIMESTAMP, n_rows BIGINT, content_hash VARCHAR,
                last_attempt_at TIMESTAMP, last_success_at TIMESTAMP,
                last_full_refresh_at TIMESTAMP, last_empty_at TIMESTAMP, message VARCHAR)"""
        )
        self._con.execute(
            """CREATE TABLE IF NOT EXISTS ingest_issues (
                ts TIMESTAMP, ticker VARCHAR, code VARCHAR, severity VARCHAR, count BIGINT,
                first_date TIMESTAMP, last_date TIMESTAMP, detail VARCHAR)"""
        )
        self._con.execute(
            """CREATE TABLE IF NOT EXISTS fetch_log (
                ts TIMESTAMP, ticker VARCHAR, mode VARCHAR, status VARCHAR, rows_written BIGINT,
                message VARCHAR)"""
        )

    # ---- lifecycle ---------------------------------------------------------------------
    def close(self) -> None:
        self._con.close()

    def __enter__(self) -> ParquetStore:
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    # ---- paths -------------------------------------------------------------------------
    def _path(self, table: str, ticker: str) -> Path:
        return self.root / table / f"{quote(ticker, safe='-._')}.parquet"

    # ---- data --------------------------------------------------------------------------
    def write_history(self, ticker: str, prices: pd.DataFrame, actions: pd.DataFrame) -> str:
        """Atomically replace a ticker's stored history; returns the content hash."""
        p = prices[PRICE_COLUMNS].copy()
        p.insert(0, "ticker", ticker)
        self._atomic_parquet(p.reset_index(), self._path("prices", ticker))
        a = actions[ACTION_COLUMNS].copy() if len(actions) else empty_actions()
        a.insert(1, "ticker", ticker)
        self._atomic_parquet(a.reset_index(drop=True), self._path("actions", ticker))
        return frame_hash(prices, actions)

    @staticmethod
    def _atomic_parquet(df: pd.DataFrame, path: Path) -> None:
        tmp = path.with_suffix(".parquet.tmp")
        df.to_parquet(tmp, index=False)
        os.replace(tmp, path)

    def read_prices(self, ticker: str) -> pd.DataFrame | None:
        path = self._path("prices", ticker)
        if not path.exists():
            return None
        df = pd.read_parquet(path)
        df["date"] = df["date"].astype("datetime64[ns]")
        return df.drop(columns="ticker").set_index("date").sort_index()[PRICE_COLUMNS]

    def read_actions(self, ticker: str) -> pd.DataFrame:
        path = self._path("actions", ticker)
        if not path.exists():
            return empty_actions()
        df = pd.read_parquet(path)
        if df.empty:
            return empty_actions()
        df["date"] = df["date"].astype("datetime64[ns]")
        return df[ACTION_COLUMNS].reset_index(drop=True)

    def has_history(self, ticker: str) -> bool:
        return self._path("prices", ticker).exists()

    # ---- manifest ----------------------------------------------------------------------
    def get_entry(self, ticker: str) -> dict | None:
        df = self._con.execute("SELECT * FROM manifest WHERE ticker = ?", [ticker]).df()
        if df.empty:
            return None
        row = df.iloc[0].to_dict()
        for k, v in row.items():
            if pd.isna(v):
                row[k] = None
            elif k.endswith(("_date", "_at")):
                row[k] = pd.Timestamp(v)
        return row

    def upsert_entry(self, ticker: str, **fields) -> None:
        entry = self.get_entry(ticker) or {c: None for c in _MANIFEST_COLS}
        entry.update(fields)
        entry["ticker"] = ticker
        vals = [_ts(entry[c]) if c.endswith(("_date", "_at")) else entry[c] for c in _MANIFEST_COLS]
        self._con.execute(
            f"INSERT OR REPLACE INTO manifest VALUES ({', '.join('?' * len(_MANIFEST_COLS))})", vals
        )

    def manifest(self) -> pd.DataFrame:
        df = self._con.execute("SELECT * FROM manifest ORDER BY ticker").df()
        for c in df.columns:
            if c.endswith(("_date", "_at")):
                df[c] = pd.to_datetime(df[c])
        return df

    def log(
        self, ts: pd.Timestamp, ticker: str, mode: str, status: str, rows: int, message: str
    ) -> None:
        self._con.execute(
            "INSERT INTO fetch_log VALUES (?, ?, ?, ?, ?, ?)",
            [_ts(ts), ticker, mode, status, int(rows), message],
        )

    def log_issues(self, ts: pd.Timestamp, issues) -> None:
        """Persist ingest-time findings: repairs and flags are results, not console noise."""
        for i in issues:
            self._con.execute(
                "INSERT INTO ingest_issues VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    _ts(ts),
                    i.ticker,
                    i.code,
                    str(i.severity.value),
                    int(i.count),
                    _ts(i.first_date),
                    _ts(i.last_date),
                    i.detail,
                ],
            )

    def ingest_issues(self) -> pd.DataFrame:
        return self._con.execute("SELECT * FROM ingest_issues ORDER BY ts, ticker, code").df()

    def fetch_log(self) -> pd.DataFrame:
        return self._con.execute("SELECT * FROM fetch_log ORDER BY ts").df()

    # ---- ad-hoc SQL --------------------------------------------------------------------
    def sql(self, query: str) -> pd.DataFrame:
        """Run SQL with ``prices`` and ``actions`` views over all stored Parquet files."""
        for table in ("prices", "actions"):
            if any((self.root / table).glob("*.parquet")):
                glob = str(self.root / table / "*.parquet").replace("'", "''")
                self._con.execute(
                    f"CREATE OR REPLACE VIEW {table} AS SELECT * FROM read_parquet('{glob}')"
                )
        return self._con.execute(query).df()
