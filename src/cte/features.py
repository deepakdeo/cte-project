#!/usr/bin/env python3
"""
CTE — features.py (Phase 2: Feature Engineering MVP)

Reads:    data/interim/clean.parquet   (output from src/cte/data.py)
Writes:   data/interim/features.parquet
          notebooks/reports/features_preview.csv  (small preview for quick inspection)

Features included
-----------------
Time-of-day (cyclical):
  - wakeup_time_minutes, dinner_time_minutes, bed_time_minutes → sin/cos transforms

Lags (t−1..t−3) for:
  - productivity_pct, sleep_duration_h, deep_sleep_pct, rem_sleep_pct

Rolling 7-day stats (mean, std) for the same signals.

Interaction aggregates:
  - interaction_total_score = partner_score + family_score + friends_score
  - any_no_interaction = 1 if any of {partner|family|friends}_no_interaction == 1

Temporal:
  - day_of_week one-hots (Mon..Sun)
  - day_idx (0,1,2,...) since first date (useful as a trend regressor)

Notes
-----
- All operations are deterministic. We sort by date and reset index.
- We do not forward-fill; lags/rollings will be NaN at the top (expected).
- Keep text fields out of the feature matrix; they’ll be used later in NLP phase.

CLI
---
poetry run python src/cte/features.py \
  --in data/interim/clean.parquet \
  --out data/interim/features.parquet
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Iterable, List, Tuple

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

# ------------------- helpers -------------------

def _ensure_datetime(df: pd.DataFrame, col: str = "date") -> pd.DataFrame:
    if col in df.columns:
        df[col] = pd.to_datetime(df[col], errors="coerce")
        df = df.sort_values(col).reset_index(drop=True)
    else:
        raise ValueError("Expected a 'date' column in the cleaned table.")
    return df

def _cyclical_minutes(df: pd.DataFrame, col: str, period: int = 1440) -> pd.DataFrame:
    """Add sin/cos transforms for a minutes-after-midnight column."""
    if col not in df: 
        return df
    x = (df[col].astype(float) / period) * 2 * np.pi
    df[f"{col}_sin"] = np.sin(x)
    df[f"{col}_cos"] = np.cos(x)
    return df

def _add_lags(df: pd.DataFrame, cols: Iterable[str], lags: Iterable[int]) -> pd.DataFrame:
    for c in cols:
        if c in df:
            for L in lags:
                df[f"{c}_lag{L}"] = df[c].shift(L)
    return df

def _add_rollings(df: pd.DataFrame, cols: Iterable[str], window: int = 7) -> pd.DataFrame:
    for c in cols:
        if c in df:
            r = df[c].rolling(window=window, min_periods=2)
            df[f"{c}_roll{window}_mean"] = r.mean()
            df[f"{c}_roll{window}_std"]  = r.std()
    return df

def _day_of_week_onehots(df: pd.DataFrame) -> pd.DataFrame:
    dow = df["date"].dt.dayofweek  # Monday=0
    dummies = pd.get_dummies(dow, prefix="dow", dtype="Int64")
    dummies.columns = ["dow_mon","dow_tue","dow_wed","dow_thu","dow_fri","dow_sat","dow_sun"][:dummies.shape[1]]
    df = pd.concat([df, dummies], axis=1)
    return df

def _interaction_aggregates(df: pd.DataFrame) -> pd.DataFrame:
    # total score
    parts = [c for c in ["partner_score","family_score","friends_score"] if c in df]
    if parts:
        df["interaction_total_score"] = df[parts].sum(axis=1, skipna=True)
    # any no-interaction flag
    flags = [c for c in ["partner_no_interaction","family_no_interaction","friends_no_interaction"] if c in df]
    if flags:
        df["any_no_interaction"] = (df[flags].fillna(0).sum(axis=1) > 0).astype("Int64")
    return df

def _select_feature_columns(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Choose a clean feature set (exclude free text, raw time strings, original interaction labels, etc.)
    We keep engineered numerics + meaningful binary indicators.
    """
    # Start with everything numeric
    numeric_cols = df.select_dtypes(include=[np.number, "Int64"]).columns.tolist()

    # Drop raw minutes columns (we keep sin/cos instead), keep day_idx as trend
    drop_exact = ["wakeup_time_minutes","dinner_time_minutes","bed_time_minutes"]
    numeric_cols = [c for c in numeric_cols if c not in drop_exact]

    # Ensure we carry target candidates for modeling later (here just keep them; model step will choose target)
    # For EDA/training we keep productivity_pct as a column (not as a feature if you prefer strictness).
    keep_extra = []
    for c in ["productivity_pct"]:
        if c in df.columns and c not in numeric_cols:
            keep_extra.append(c)

    cols = numeric_cols + keep_extra

    # Always include date for reference (not a model feature)
    final = ["date"] + cols
    return df.loc[:, final], cols  # return df and list of feature columns (excluding date)

# ------------------- main pipeline -------------------

def build_features(clean_path: Path, out_path: Path, preview_path: Path | None = None) -> pd.DataFrame:
    LOG.info("Reading cleaned table: %s", clean_path)
    df = pd.read_parquet(clean_path)

    df = _ensure_datetime(df, "date")

    # Cyclical encodings from minutes-after-midnight
    for c in ["wakeup_time_minutes","dinner_time_minutes","bed_time_minutes"]:
        if c in df:
            df = _cyclical_minutes(df, c)

    # Lags and rollings
    base_cols = ["productivity_pct","sleep_duration_h","deep_sleep_pct","rem_sleep_pct"]
    df = _add_lags(df, base_cols, lags=[1,2,3])
    df = _add_rollings(df, base_cols, window=7)

    # Interactions
    df = _interaction_aggregates(df)

    # Temporal helpers
    df["day_idx"] = (df["date"] - df["date"].min()).dt.days.astype("Int64")
    df = _day_of_week_onehots(df)

    # Select/export
    feat_df, feature_cols = _select_feature_columns(df)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    feat_df.to_parquet(out_path, index=False)
    LOG.info("Wrote features to %s (rows=%d, cols=%d)", out_path, len(feat_df), feat_df.shape[1])

    # Small CSV preview for quick eyeballing in the notebook/repo
    if preview_path is not None:
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        # show first 40 rows of a compact subset (date + a handful)
        preview_cols = ["date","productivity_pct","sleep_duration_h","interaction_total_score","any_no_interaction",
                        "wakeup_time_minutes_sin","wakeup_time_minutes_cos","day_idx"]
        preview_cols = [c for c in preview_cols if c in feat_df.columns]
        feat_df.loc[:, preview_cols].head(40).to_csv(preview_path, index=False)
        LOG.info("Wrote preview CSV to %s", preview_path)

    LOG.info("Feature columns (excluding 'date'): %d", len(feature_cols))
    return feat_df

# ------------------- CLI -------------------

def _argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CTE Feature Engineering (Phase 2)")
    p.add_argument("--in",  dest="in_path",  type=Path, required=True,
                   help="Path to cleaned parquet (e.g., data/interim/clean.parquet)")
    p.add_argument("--out", dest="out_path", type=Path, default=Path("data/interim/features.parquet"),
                   help="Where to write features parquet (default: data/interim/features.parquet)")
    p.add_argument("--preview", dest="preview_path", type=Path, default=Path("notebooks/reports/features_preview.csv"),
                   help="Optional small CSV preview (default: notebooks/reports/features_preview.csv)")
    return p

if __name__ == "__main__":
    args = _argparser().parse_args()
    build_features(args.in_path, args.out_path, args.preview_path)
