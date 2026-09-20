"""Command line: ``python -m statarb.data <command> [--config configs/data_default.yaml]``.

universe [--fetch]        build the universe (optionally download a fresh Wikipedia snapshot)
refresh [--limit N] [--full] [--tickers A B ...]
audit                     validate every stored ticker; writes reports/validation_issues.csv
survivorship              coverage / naive-overlap / size-check tables for the universe
snapshot                  write reports/dataset_manifest.json (fingerprint + provenance)
status                    manifest summary
"""

from __future__ import annotations

import argparse
import logging
import sys

import pandas as pd

from statarb.config import load_config
from statarb.data.cleaning.issues import issues_to_frame, summarize
from statarb.data.pipeline import DataPipeline
from statarb.data.universe import sp500
from statarb.data.universe.survivorship import survivorship_report


def _sample_dates(pipe: DataPipeline, freq: str = "YS") -> pd.DatetimeIndex:
    """Yearly Jan-1 samples from ``start`` plus the last finalised session."""
    start = pd.Timestamp(pipe.cfg.start).normalize()
    last = pipe.calendar.last_complete_session(
        pd.Timestamp.now(tz="UTC"), pipe.cfg.refresh.finalize_buffer_hours
    )
    return pd.date_range(start, last, freq=freq).append(pd.DatetimeIndex([last]))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="statarb.data")
    ap.add_argument(
        "command", choices=["universe", "refresh", "audit", "survivorship", "snapshot", "status"]
    )
    ap.add_argument("--config", default="configs/data_default.yaml")
    ap.add_argument(
        "--fetch", action="store_true", help="universe: download a fresh Wikipedia snapshot"
    )
    ap.add_argument(
        "--limit", type=int, default=None, help="refresh: only the first N tickers (sorted)"
    )
    ap.add_argument("--full", action="store_true", help="refresh: force full re-download")
    ap.add_argument("--tickers", nargs="*", default=None, help="refresh: explicit tickers")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s: %(message)s")
    pd.set_option("display.width", 200, "display.max_columns", 30, "display.max_rows", 200)

    cfg = load_config(args.config)
    pipe = DataPipeline(cfg)
    try:
        if args.command == "universe":
            if args.fetch:
                label = sp500.fetch_snapshot(pipe.store.root / "universe")
                print(f"stored S&P 500 snapshot {label}")
            uni = pipe.universe(rebuild=True)
            print("provenance:", uni.provenance)
            for c in uni.caveats:
                print("CAVEAT:", c)
            print(
                f"members today: {len(uni.members_on(pd.Timestamp.now()))}; "
                f"tickers ever since {cfg.start}: {len(uni.tickers_ever(cfg.start, pd.Timestamp.now()))}"
            )

        elif args.command == "refresh":
            tickers = args.tickers or pipe.universe().tickers_ever(cfg.start, pd.Timestamp.now())
            tickers = sorted(tickers)[: args.limit] if args.limit else sorted(tickers)
            print(f"refreshing {len(tickers)} tickers from {pipe.source.name} ...", flush=True)
            report = pipe.refresh(tickers, force_full=args.full)
            print("through", report.end.date(), "->", report.counts())
            errs = report.to_frame().query("action == 'error'")
            if len(errs):
                print(errs.to_string(index=False))
            print(summarize(report.issues).to_string(index=False))

        elif args.command == "audit":
            issues = pipe.audit()
            df = issues_to_frame(issues)
            out = pipe.store.root / "reports" / "validation_issues.csv"
            df.to_csv(out, index=False)
            print(summarize(issues).to_string(index=False))
            print(f"{len(df)} findings written to {out}")
            ing = pipe.store.ingest_issues()
            ing.to_csv(pipe.store.root / "reports" / "ingest_issues.csv", index=False)
            print(
                f"{len(ing)} ingest-time findings (repairs/flags at download) in ingest_issues.csv"
            )

        elif args.command == "survivorship":
            uni = pipe.universe()
            ident = pipe.identity_report()
            ident.to_csv(pipe.store.root / "reports" / "identity_check.csv", index=False)
            tbl, summary = survivorship_report(
                uni,
                pipe.store.manifest(),
                _sample_dates(pipe),
                _sample_dates(pipe)[-1],
                cfg.universe.expected_size,
                cfg.universe.size_tolerance,
                identity=ident,
            )
            print(tbl.round(3).to_string())
            print(summary)
            tbl.to_csv(pipe.store.root / "reports" / "survivorship.csv")
            for c in uni.caveats:
                print("CAVEAT:", c)

        elif args.command == "snapshot":
            print("wrote", pipe.write_dataset_manifest())
            print("fingerprint:", pipe.fingerprint())

        elif args.command == "status":
            m = pipe.store.manifest()
            print(m["status"].value_counts().to_string())
            ok = m[m["status"] == "ok"]
            if len(ok):
                print(
                    "first_date:",
                    ok["first_date"].min().date(),
                    " last_date range:",
                    ok["last_date"].min().date(),
                    "->",
                    ok["last_date"].max().date(),
                    " rows:",
                    int(ok["n_rows"].sum()),
                )
            print("fingerprint:", pipe.fingerprint())
    finally:
        pipe.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
