#!/usr/bin/env python3
"""Scheduled ERCOT dashboard ingestion utility."""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd

# Ensure project root is on path for package imports
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from python.data import ERCOTDataClient
from python.storage import DashboardStorage


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Ingest ERCOT dashboard data on a fixed cadence.")
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Polling interval in seconds (default: 300 = 5 minutes)",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        default=None,
        help="Maximum number of iterations before exiting (default: run forever)",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "raw" / "ercot_dashboard",
        help="Directory for raw snapshot Parquet files",
    )
    parser.add_argument(
        "--history",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "ercot_dashboard_history.parquet",
        help="Path to consolidated history Parquet file",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging verbosity",
    )
    return parser.parse_args(argv)


def ingest_once(client: ERCOTDataClient, storage: DashboardStorage) -> None:
    df = client._fetch_ercot_dashboard_data()

    if df.empty:
        logging.warning("No dashboard data returned; skipping snapshot")
        return

    snapshot_path = storage.write_snapshot(df)
    history_path = storage.update_history(df)

    logging.info(
        "Snapshot stored at %s; history updated at %s with %d rows",
        snapshot_path,
        history_path,
        len(df),
    )


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(asctime)s %(levelname)s %(message)s")

    client = ERCOTDataClient()
    storage = DashboardStorage(raw_dir=args.raw_dir, history_path=args.history)

    iteration = 0
    logging.info("Starting ERCOT dashboard ingestion with %s-second interval", args.interval)

    try:
        while True:
            iteration += 1
            logging.info("Ingestion iteration %d", iteration)

            try:
                ingest_once(client, storage)
            except Exception as exc:  # pylint: disable=broad-except
                logging.exception("Ingestion iteration %d failed: %s", iteration, exc)

            if args.max_iterations is not None and iteration >= args.max_iterations:
                logging.info("Reached max iterations (%d); exiting", args.max_iterations)
                break

            time.sleep(args.interval)
    except KeyboardInterrupt:
        logging.info("Interrupted; shutting down ingestion loop")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
