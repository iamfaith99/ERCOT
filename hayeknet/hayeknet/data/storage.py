"""Persistent storage helpers for ERCOT data ingestion."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import pandas as pd


@dataclass
class DashboardStorage:
    """Utility for writing ERCOT dashboard snapshots and consolidated history."""

    raw_dir: Path
    history_path: Path
    index_columns: Iterable[str] = field(
        default_factory=lambda: ("timestamp", "data_source", "is_forecast")
    )

    def __post_init__(self) -> None:
        self.raw_dir = self.raw_dir.expanduser().resolve()
        self.history_path = self.history_path.expanduser().resolve()
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        if self.history_path.parent != self.raw_dir:
            self.history_path.parent.mkdir(parents=True, exist_ok=True)

    def write_snapshot(self, df: pd.DataFrame) -> Path:
        """Persist a single snapshot as a Parquet file in the raw directory."""
        if df.empty:
            raise ValueError("Cannot write empty dataframe snapshot")

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        path = self.raw_dir / f"ercot_dashboard_{timestamp}.parquet"
        df.to_parquet(path, index=False)
        return path

    def update_history(self, df: pd.DataFrame) -> Path:
        """Append data to the consolidated history Parquet file with de-duplication."""
        if df.empty:
            raise ValueError("Cannot update history with empty dataframe")

        history_df = pd.DataFrame()
        if self.history_path.exists():
            history_df = pd.read_parquet(self.history_path)

        combined = pd.concat([history_df, df], ignore_index=True)
        if self.index_columns:
            dedupe_subset = [col for col in self.index_columns if col in combined.columns]
            if dedupe_subset:
                combined = combined.drop_duplicates(subset=dedupe_subset, keep="last")
            else:
                combined = combined.drop_duplicates(keep="last")
        else:
            combined = combined.drop_duplicates(keep="last")

        combined = combined.sort_values("timestamp").reset_index(drop=True)
        combined.to_parquet(self.history_path, index=False)
        return self.history_path
