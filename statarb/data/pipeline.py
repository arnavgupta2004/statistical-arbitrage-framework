"""Facade tying the data layer together:

    Universe -> Download -> Validation -> Corporate actions -> Alignment -> Local storage

``DataPipeline.panel`` is the single point through which research code should read prices.  It
enforces the point-in-time contract: a panel ends at ``end``, is expressed on the split basis a
vendor would have used on ``as_of`` (default: ``end``), and contains no dividend, split or bar
dated after it.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from collections.abc import Iterable
from pathlib import Path

import pandas as pd

from statarb.config import DataConfig, load_config
from statarb.data.cleaning.alignment import AlignedPanel, build_panel
from statarb.data.cleaning.calendar import TradingCalendar
from statarb.data.cleaning.corporate_actions import as_traded_close, rebase_to_as_of
from statarb.data.cleaning.issues import Issue, issues_to_frame
from statarb.data.cleaning.validation import audit_history
from statarb.data.download.refresh import Refresher, RefreshReport
from statarb.data.sources.base import DataSource
from statarb.data.storage.store import ParquetStore
from statarb.data.universe.builders import Universe, build_universe
from statarb.data.universe.identity import check_identity


def make_source(cfg: DataConfig) -> DataSource:
    if cfg.source == "yahoo":
        from statarb.data.sources.yahoo import YahooSource

        return YahooSource()
    if cfg.source == "csv":
        from statarb.data.sources.csv_source import CsvDirectorySource

        return CsvDirectorySource(**cfg.source_options)
    raise ValueError(
        f"source {cfg.source!r} cannot be built from config alone "
        "(synthetic sources are passed in code)"
    )


class DataPipeline:
    def __init__(
        self,
        cfg: DataConfig,
        source: DataSource | None = None,
        store: ParquetStore | None = None,
        calendar: TradingCalendar | None = None,
    ):
        self.cfg = cfg
        self.store = store or ParquetStore(cfg.store_path)
        self.source = source or make_source(cfg)
        self.calendar = calendar or TradingCalendar(cfg.calendar)
        self._universe: Universe | None = None

    @classmethod
    def from_config_path(cls, path: str | Path) -> DataPipeline:
        return cls(load_config(path))

    def close(self) -> None:
        self.store.close()

    # ---- universe ----------------------------------------------------------------------
    def universe(self, rebuild: bool = False) -> Universe:
        if self._universe is None or rebuild:
            self._universe = build_universe(self.cfg.universe, self.store.root)
        return self._universe

    # ---- download ----------------------------------------------------------------------
    def refresh(
        self,
        tickers: Iterable[str] | None = None,
        *,
        force_full: bool = False,
        end: pd.Timestamp | None = None,
    ) -> RefreshReport:
        """Refresh ``tickers`` (default: every ticker ever in the universe since ``start``)."""
        if tickers is None:
            tickers = self.universe().tickers_ever(self.cfg.start, pd.Timestamp.now())
        return Refresher(self.store, self.source, self.cfg, self.calendar).refresh(
            tickers, force_full=force_full, end=end
        )

    # ---- read path ---------------------------------------------------------------------
    def panel(
        self,
        tickers: Iterable[str],
        start,
        end,
        as_of=None,
    ) -> AlignedPanel:
        """Aligned panel of ``tickers`` over sessions in [start, end], as known on ``as_of``.

        ``as_of`` defaults to ``end``: by default a panel is exactly what could have been observed
        on its last session, so a training window built with ``end = train_end`` cannot see later
        splits or dividends.  Pass a later ``as_of`` only for descriptive (non-causal) work.
        """
        start, end = pd.Timestamp(start), pd.Timestamp(end)
        as_of = pd.Timestamp(as_of) if as_of is not None else end
        sessions = self.calendar.sessions(start, end)
        histories, missing = {}, {}
        for t in sorted(set(tickers)):
            prices = self.store.read_prices(t)
            if prices is None:
                missing[t] = "not in store"
                continue
            p, a = rebase_to_as_of(prices, self.store.read_actions(t), as_of)
            histories[t] = (p.loc[start:end], a)
        panel = build_panel(histories, sessions, as_of=as_of)
        panel.dropped.update(missing)
        return panel

    def raw_close(self, ticker: str, start, end) -> pd.Series:
        """As-traded close: the price that actually printed, reconstructed by undoing every split.

        Not point-in-time safe for *research inputs* (it encodes all splits through today by
        construction) -- use it for what genuinely needs the printed price: tick-size and
        minimum-price screens, share counts, and bps-of-price cost conversions.
        """
        prices = self.store.read_prices(ticker)
        if prices is None:
            raise KeyError(f"{ticker} not in store")
        raw = as_traded_close(prices["close"], self.store.read_actions(ticker))
        return raw.loc[pd.Timestamp(start) : pd.Timestamp(end)]

    def identity_report(self) -> pd.DataFrame:
        """Per membership interval: is the stored price series plausibly that company's?"""
        today = pd.Timestamp.now().normalize()
        sessions = self.calendar.sessions(self.cfg.start, today)
        return check_identity(
            self.universe(),
            self.store.read_prices,
            sessions,
            self.cfg.start,
            today,
            self.cfg.universe.min_volume_share,
        )

    def members_mask(self, panel: AlignedPanel, verified: bool = False) -> pd.DataFrame:
        """Point-in-time membership as a boolean (session x ticker) matrix aligned to ``panel``."""
        m = self.universe().membership.matrix(panel.close.index)
        mask = m.reindex(columns=panel.close.columns, fill_value=False)
        if verified:  # drop (ticker, interval) cells whose prices are probably another company's
            bad = self.identity_report()
            bad = bad[bad["verdict"].isin(["suspect", "no_prices"])]
            for t, s, e in bad[["ticker", "start", "end"]].itertuples(index=False):
                if t in mask.columns:
                    keep = pd.Series(False, index=mask.index)
                    keep |= (mask.index >= s) if pd.notna(s) else True
                    keep &= (mask.index < e) if pd.notna(e) else True
                    mask.loc[keep, t] = False
        return mask

    # ---- audit -------------------------------------------------------------------------
    def audit(self, tickers: Iterable[str] | None = None) -> list[Issue]:
        now = pd.Timestamp.now(tz="UTC")
        last_final = self.calendar.last_complete_session(
            now, self.cfg.refresh.finalize_buffer_hours
        )
        manifest = self.store.manifest()
        tickers = (
            list(tickers)
            if tickers is not None
            else list(manifest.loc[manifest["status"] == "ok", "ticker"])
        )
        issues: list[Issue] = []
        sessions = self.calendar.sessions(self.cfg.start, last_final + pd.Timedelta(days=10))
        for t in tickers:
            prices = self.store.read_prices(t)
            if prices is None:
                continue
            issues += audit_history(
                t, prices, self.store.read_actions(t), sessions, self.cfg.validation, last_final
            )
        return issues

    def audit_frame(self, tickers: Iterable[str] | None = None) -> pd.DataFrame:
        return issues_to_frame(self.audit(tickers))

    # ---- reproducibility ---------------------------------------------------------------
    def fingerprint(self, tickers: Iterable[str] | None = None) -> str:
        """Hash of the stored *values* (per-ticker content hashes) for ``tickers``."""
        m = self.store.manifest()
        m = m[m["status"] == "ok"]
        if tickers is not None:
            m = m[m["ticker"].isin(set(tickers))]
        h = hashlib.sha256()
        for t, c in zip(m["ticker"], m["content_hash"], strict=True):
            h.update(f"{t}:{c};".encode())
        return h.hexdigest()[:16]

    def dataset_manifest(self, tickers: Iterable[str] | None = None) -> dict:
        """Everything needed to state exactly which data a result was computed on."""
        import numpy
        import pyarrow

        m = self.store.manifest()
        m = m[m["status"] == "ok"]
        if tickers is not None:
            m = m[m["ticker"].isin(set(tickers))]
        uni = self.universe()
        return {
            "created_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "fingerprint": self.fingerprint(tickers),
            "source": self.source.name,
            "n_tickers": len(m),
            "first_date": str(m["first_date"].min().date()) if len(m) else None,
            "last_date": str(m["last_date"].max().date()) if len(m) else None,
            "config": self.cfg.model_dump(mode="json"),
            "universe": {"provenance": uni.provenance, "caveats": uni.caveats},
            "versions": {
                "pandas": pd.__version__,
                "numpy": numpy.__version__,
                "pyarrow": pyarrow.__version__,
            },
            "git_commit": _git_commit(),
        }

    def write_dataset_manifest(
        self, path: str | Path | None = None, tickers: Iterable[str] | None = None
    ) -> Path:
        path = Path(path) if path else self.store.root / "reports" / "dataset_manifest.json"
        path.write_text(json.dumps(self.dataset_manifest(tickers), indent=2, default=str))
        return path


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=5
        )
        return out.stdout.strip() or None
    except Exception:
        return None
