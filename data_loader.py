
import os
import pandas as pd
from typing import Tuple, Optional
from datetime import datetime
from config import (
    ID_COLUMN, TEXT_COLUMN, TIME_COLUMN, TIME_IS_EPOCH,
    OUTPUT_DIR, DEDUPLICATE_BY_ID
)

class DataLoader:
    def __init__(self, path: str):
        self.path = path

    def load(self) -> pd.DataFrame:
        if not os.path.exists(self.path):
            raise FileNotFoundError(f"Input file not found: {self.path}")
        try:
            if self.path.lower().endswith(".csv"):
                df = pd.read_csv(self.path)
            elif self.path.lower().endswith(".json"):
                df = pd.read_json(self.path, lines=self._is_json_lines(self.path))
            else:
                raise ValueError("Unsupported file format. Use CSV or JSON.")
        except Exception as e:
            raise RuntimeError(f"Failed reading data: {e}")

        # Basic validation
        missing = [c for c in [TEXT_COLUMN, TIME_COLUMN] if c not in df.columns]
        if missing:
            raise KeyError(f"Missing expected columns: {missing}. Columns present: {list(df.columns)}")

        # Deduplicate by ID if available
        if DEDUPLICATE_BY_ID and ID_COLUMN in df.columns:
            before = len(df)
            df = df.drop_duplicates(subset=[ID_COLUMN])
            after = len(df)
            if before != after:
                print(f"[DataLoader] Deduplicated by {ID_COLUMN}: {before} -> {after}")

        # Parse time
        df = self._parse_time(df)
        return df

    def _parse_time(self, df: pd.DataFrame) -> pd.DataFrame:
        if TIME_IS_EPOCH:
            try:
                df["_timestamp"] = pd.to_datetime(df[TIME_COLUMN], unit="s", errors="coerce")
            except Exception:
                # Fallback: try milliseconds
                df["_timestamp"] = pd.to_datetime(df[TIME_COLUMN], unit="ms", errors="coerce")
        else:
            df["_timestamp"] = pd.to_datetime(df[TIME_COLUMN], errors="coerce")
        nulls = df["_timestamp"].isna().sum()
        if nulls > 0:
            print(f"[DataLoader] Warning: {nulls} rows had unparseable timestamps and will be dropped.")
            df = df.dropna(subset=["_timestamp"]).copy()
        return df

    @staticmethod
    def _is_json_lines(path: str) -> bool:
        # naive heuristic
        return True if path.lower().endswith(".jsonl") else False
