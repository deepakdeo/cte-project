#!/usr/bin/env python3
"""
CTE — data.py
--------------
Load the raw self-tracking CSV and produce a clean, typed Parquet table.

Highlights (interview-ready):
- Canonical header schema (explicit rename map for your CSV).
- Deterministic parsing for dates/times (no "guessing" formats).
- Yes/No coercion supports many variants (yes/no/y/n/true/false/t/f/1/0).
- Social interactions = Option B:
    * score:      negative=-1, neutral=0, positive=+1,  'na' (no interaction) -> 0
    * no_interaction flag: 1 if raw == 'na', else 0
    * genuine missing (blank/unknown) remains NaN in score (distinct from 'na')
- Decode "when_most_productive" codebook (1,2,3,12,13,23,123,5)
  and add one-hot flags (morning/afternoon/evening/none).
- Robust numeric coercion for %, durations (e.g., '7h38m'), and free-text numbers.
- Clip percentages to [0, 100].

CLI
---
poetry run python src/cte/data.py \
  --in  data/raw/cte-project_data.csv \
  --out data/interim/clean.parquet
"""

from __future__ import annotations

import argparse
import logging
import re
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd

# ---------------- Logging ----------------
LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

# --------- Explicit header mapping (from the provided CSV) ----------
# (Fixes newlines/spacing/typos → canonical snake_case names.)
RENAME_MAP: Dict[str, str] = {
    "Date": "date",
    "Reflection": "reflection",
    "primary mood": "primary_mood",
    "secondary mood": "secondary_mood",
    "productivity\npercentage": "productivity_pct",
    "when most \nproductive": "when_most_productive",
    "studied \nat home": "studied_at_home",
    "studied \nat school": "studied_at_school",
    "breakfast qualtity": "breakfast_quality",  # fix typo
    "lunch quality": "lunch_quality",
    "dinner quality": "dinner_quality",
    "water \ndrank": "water_drank_l",           # liters
    "workout \ndid": "workout_did",
    "meditation": "meditation",
    "morning \nshower": "morning_shower",
    "played \nsports": "played_sports",
    "interaction \nw/ partner": "interaction_partner",
    "interaction \nw/ family": "interaction_family",
    "interaction \nw/ friends": "interaction_friends",
    "sickness": "sickness",
    "sleep \nduration": "sleep_duration_h",
    "nap \ntoday": "nap_today",
    "deep sleep \npercentage": "deep_sleep_pct",
    "REM sleep \npercentage": "rem_sleep_pct",
    "dinner time": "dinner_time",
    "bed time": "bed_time",
    "wakeup time": "wakeup_time",
}

# Columns by semantic type
BOOLEAN_COLS = [
    "studied_at_home",
    "studied_at_school",
    "workout_did",
    "meditation",
    "morning_shower",
    "played_sports",
    "sickness",
    "nap_today",
]

INTERACTION_COLS = [
    "interaction_partner",
    "interaction_family",
    "interaction_friends",
]

TIME_COLS = ["dinner_time", "bed_time", "wakeup_time"]

# Numeric columns that are **not** percentages and don't need special duration parsing
NUMERIC_COLS = [
    "water_drank_l",
    "breakfast_quality",
    "lunch_quality",
    "dinner_quality",
]

# Duration-like columns (e.g., '7h38m', '7:38', '7.5')
DURATION_COLS = ["sleep_duration_h"]

MOOD_COLS = ["primary_mood", "secondary_mood"]

# text columns (kept as-is, just trimmed)
TEXT_COLS = ["reflection"]

# percent-like columns
PERCENT_COLS = ["productivity_pct", "deep_sleep_pct", "rem_sleep_pct"]

# ---- "when most productive" codebook (decoder) ----
PRODUCTIVE_CODE_MAP = {
    1: "morning",
    2: "afternoon",
    3: "evening",
    12: "morning_afternoon",
    13: "morning_evening",
    23: "afternoon_evening",
    123: "morning_afternoon_evening",
    5: "not_productive",
}

# ---- robust yes/no tokens ----
_BOOL_TRUE = {"yes", "y", "true", "t", "1", "home", "school"}  # 'home'/'school' appear in study flags
_BOOL_FALSE = {"no", "n", "false", "f", "0"}


# ---------------- Deterministic parsers ----------------
def _parse_date_with_default_year(s: pd.Series) -> pd.Series:
    """
    Handle rows like 'Jan 27, 2025' and 'Jan 28' (missing year).
    1) Find a default year from any row that already has a year, else use current year.
    2) Append that year to rows missing it.
    3) Parse using fixed '%b %d, %Y'.
    """
    year_pat = re.compile(r",\s*(\d{4})$")
    default_year: Optional[int] = None
    for v in s.dropna().astype(str):
        m = year_pat.search(v.strip())
        if m:
            default_year = int(m.group(1))
            break
    if default_year is None:
        default_year = pd.Timestamp.today().year

    def add_year(x: str) -> str:
        x = x.strip()
        return x if year_pat.search(x) else f"{x}, {default_year}"

    with_year = s.astype("string").map(lambda x: add_year(x) if x is not pd.NA else x)
    parsed = pd.to_datetime(with_year, format="%b %d, %Y", errors="coerce")
    return parsed


def _parse_time_to_minutes(s: pd.Series) -> pd.Series:
    """
    Parse '10:15 PM'/'6:33 AM' → integer minutes after midnight.
    Uses fixed format '%I:%M %p' to avoid inference.
    """
    def to_minutes(x) -> Optional[int]:
        if x is None or x is pd.NA:
            return None
        xs = str(x).strip()
        if not xs:
            return None
        try:
            t = pd.to_datetime(xs, format="%I:%M %p")
            return int(t.hour) * 60 + int(t.minute)
        except Exception:
            return None

    return s.astype("string").map(to_minutes)


# ---------------- Coercion helpers ----------------
def _norm_token(x) -> Optional[str]:
    if pd.isna(x):
        return None
    s = str(x).strip().lower()
    return s if s else None


def _coerce_yes_no(s: pd.Series) -> pd.Series:
    def to01(x):
        tok = _norm_token(x)
        if tok is None:
            return np.nan
        if tok in _BOOL_TRUE:
            return 1
        if tok in _BOOL_FALSE:
            return 0
        return np.nan
    return s.map(to01).astype("Int64")


_pct_pat = re.compile(r"^\s*([+-]?\d+(?:\.\d+)?)\s*%?\s*$")

def _coerce_percent_or_number(s: pd.Series) -> pd.Series:
    def to_float(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, (int, float)) and not isinstance(x, bool):
            return float(x)
        m = _pct_pat.match(str(x))
        return float(m.group(1)) if m else np.nan
    return s.map(to_float)


def _coerce_float(s: pd.Series) -> pd.Series:
    """
    Convert a Series to float robustly:
    - Allows plain numbers ('7.5', '100', '1,500')
    - Extracts numeric portion from strings with units ('2 L', '500ml')
    - Returns NaN when nothing numeric is present
    """
    def to_float(x):
        if pd.isna(x):
            return np.nan
        text = str(x).strip()
        if text == "":
            return np.nan
        # Try fast path
        try:
            return float(text.replace(",", ""))
        except Exception:
            pass
        # Extract first number like -12.34
        m = re.search(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
        return float(m.group(0)) if m else np.nan

    return s.map(to_float)


def _parse_duration_to_hours(val) -> float:
    """
    Parse durations like:
      - '7h38m', '7 h 38 m'
      - '7:38'
      - '7.5'
      - '7h', '45m'
    Returns hours as float, or NaN if unknown.
    """
    if pd.isna(val):
        return np.nan
    s = str(val).strip().lower()
    if s in {"", "na", "none"}:
        return np.nan

    # 1) 7h38m / 7h / 38m
    m = re.fullmatch(r"(?:(\d+)\s*h)?\s*(?:(\d+)\s*m)?", s)
    if m:
        h = int(m.group(1)) if m.group(1) else 0
        mins = int(m.group(2)) if m.group(2) else 0
        if h == 0 and mins == 0:
            return np.nan
        return h + mins / 60.0

    # 2) 7:38
    m = re.fullmatch(r"(\d{1,2}):(\d{1,2})", s)
    if m:
        h = int(m.group(1))
        mins = int(m.group(2))
        return h + mins / 60.0

    # 3) Plain float like 7.5
    try:
        return float(s)
    except Exception:
        return np.nan


# -------- Social interactions (Option B: score 0 + flag) --------
def _map_interaction(series: pd.Series, base: str) -> pd.DataFrame:
    """
    Returns two columns:
      {base}_score:          negative=-1, neutral=0, positive=+1, 'na'→0, NaN stays NaN
      {base}_no_interaction: 1 if raw=='na' (case-insensitive), else 0
    """
    def score_val(x):
        tok = _norm_token(x)
        if tok is None:        # genuine missing/unknown
            return np.nan
        if tok == "negative":
            return -1.0
        if tok == "neutral":
            return 0.0
        if tok == "positive":
            return 1.0
        if tok == "na":        # no interaction that day
            return 0.0
        return np.nan

    def na_flag(x):
        tok = _norm_token(x)
        return 1 if tok == "na" else 0

    return pd.DataFrame({
        f"{base}_score": series.map(score_val),
        f"{base}_no_interaction": series.map(na_flag).astype("Int64"),
    })


# ---------------- Cleaning pipeline ----------------
def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    # warn for missing expected raw headers (helps catch typos in source)
    missing = [k for k in RENAME_MAP.keys() if k not in df.columns]
    if missing:
        LOG.warning("Some expected raw columns are missing: %s", missing)
    return df.rename(columns=RENAME_MAP)


def clean_types(df: pd.DataFrame) -> pd.DataFrame:
    # date
    if "date" in df.columns:
        df["date"] = _parse_date_with_default_year(df["date"])

    # time → minutes (keep original strings too)
    for col in TIME_COLS:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip()
            df[f"{col}_minutes"] = _parse_time_to_minutes(df[col])

    # durations → hours as float
    for col in DURATION_COLS:
        if col in df.columns:
            df[col] = df[col].map(_parse_duration_to_hours)

    # percentages (0–100)
    for col in PERCENT_COLS:
        if col in df.columns:
            df[col] = _coerce_percent_or_number(df[col]).clip(0, 100)

    # simple numerics
    for col in NUMERIC_COLS:
        if col in df.columns:
            df[col] = _coerce_float(df[col])

    # booleans
    for col in BOOLEAN_COLS:
        if col in df.columns:
            df[col] = _coerce_yes_no(df[col])

    # interactions (Option B)
    if "interaction_partner" in df.columns:
        df = pd.concat([df, _map_interaction(df["interaction_partner"], "partner")], axis=1)
    if "interaction_family" in df.columns:
        df = pd.concat([df, _map_interaction(df["interaction_family"], "family")], axis=1)
    if "interaction_friends" in df.columns:
        df = pd.concat([df, _map_interaction(df["interaction_friends"], "friends")], axis=1)

    # moods (normalized lower-case strings)
    for col in MOOD_COLS:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype("string")
                .str.strip()
                .str.lower()
                .replace({"": pd.NA, "nan": pd.NA})
            )

    # reflection text (trim; keep as string)
    for col in TEXT_COLS:
        if col in df.columns:
            df[col] = df[col].astype("string").str.strip().replace({"": pd.NA})

    # decode when_most_productive + one-hot convenience flags
    if "when_most_productive" in df.columns:
        codes = pd.to_numeric(df["when_most_productive"], errors="coerce")
        decoded = codes.map(PRODUCTIVE_CODE_MAP)
        df["when_most_productive_decoded"] = decoded.astype("category")

        # one-hots for morning/afternoon/evening/not_productive
        df["prod_morning"] = decoded.str.contains("morning", na=False).astype("Int64")
        df["prod_afternoon"] = decoded.str.contains("afternoon", na=False).astype("Int64")
        df["prod_evening"] = decoded.str.contains("evening", na=False).astype("Int64")
        df["prod_none"] = (decoded == "not_productive").astype("Int64")

    # preferred column order (date + key numerics first)
    prefer_first = [
        "date",
        "wakeup_time_minutes", "dinner_time_minutes", "bed_time_minutes",
        "sleep_duration_h", "productivity_pct",
        "deep_sleep_pct", "rem_sleep_pct",
        "water_drank_l",
        "studied_at_home", "studied_at_school",
        "workout_did", "meditation", "morning_shower", "played_sports",
        "sickness", "nap_today",
        "partner_score", "partner_no_interaction",
        "family_score", "family_no_interaction",
        "friends_score", "friends_no_interaction",
        "when_most_productive_decoded", "prod_morning", "prod_afternoon", "prod_evening", "prod_none",
        "primary_mood", "secondary_mood",
        "reflection",
    ]
    cols = [c for c in prefer_first if c in df.columns] + [c for c in df.columns if c not in prefer_first]
    return df.loc[:, cols]


def clean_csv(in_path: Path, out_path: Path) -> pd.DataFrame:
    LOG.info("Loading raw CSV: %s", in_path)
    df = pd.read_csv(in_path)
    LOG.info("Loaded shape: %s", df.shape)
    df = standardize_columns(df)
    df = clean_types(df)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    LOG.info("Wrote cleaned data to %s (rows=%d, cols=%d)", out_path, len(df), df.shape[1])
    return df


# ---------------- CLI ----------------
def _argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Clean CTE raw CSV into typed Parquet.")
    p.add_argument("--in", dest="in_path", required=True, type=Path, help="Path to raw CSV")
    p.add_argument("--out", dest="out_path", required=True, type=Path, help="Path to write Parquet")
    return p


if __name__ == "__main__":
    args = _argparser().parse_args()
    clean_csv(args.in_path, args.out_path)
