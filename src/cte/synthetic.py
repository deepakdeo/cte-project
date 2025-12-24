#!/usr/bin/env python3
"""
CTE — synthetic.py

Generate realistic synthetic behavioral tracking data for demos and testing.

This module creates synthetic datasets that mimic real self-tracking patterns:
- Temporal correlations (sleep affects next-day productivity)
- Weekly cycles (weekday vs weekend patterns)
- Realistic distributions for all metrics
- Coherent narratives in reflection text

CLI
---
poetry run python src/cte/synthetic.py --days 90 --out data/sample/synthetic_90d.csv
"""

from __future__ import annotations

import argparse
import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

# Seed phrases for generating reflections
REFLECTION_TEMPLATES = {
    "high_productivity": [
        "Great day! Managed to complete all my tasks and felt really focused.",
        "Productive morning session, got through most of my priority items.",
        "Felt in the zone today. Deep work session was particularly effective.",
        "High energy day. Accomplished more than expected.",
        "Solid focus throughout the day. Made good progress on the project.",
        "Everything clicked today. Efficient workflow and clear thinking.",
    ],
    "medium_productivity": [
        "Decent day overall. Some distractions but managed to stay on track.",
        "Mixed bag today. Morning was good, afternoon was slower.",
        "Made progress but not as much as I hoped. Some interruptions.",
        "Steady day. Nothing exceptional but got the basics done.",
        "Okay productivity. Had to context-switch more than I'd like.",
        "Average day. Some tasks took longer than expected.",
    ],
    "low_productivity": [
        "Struggled to focus today. Felt scattered and unfocused.",
        "Tired and unmotivated. Didn't accomplish much.",
        "Hard to concentrate. Too many distractions and interruptions.",
        "Low energy day. Just couldn't get into the flow.",
        "Felt overwhelmed and couldn't prioritize well.",
        "Brain fog most of the day. Need to rest more.",
    ],
    "sick": [
        "Not feeling well today. Took it easy and rested.",
        "Fighting off something. Minimal work, focused on recovery.",
        "Under the weather. Did only essential tasks.",
    ],
}

MOODS_PRIMARY = ["happy", "productive", "neutral", "tired", "energetic", "stressed", "calm", "focused"]
MOODS_SECONDARY = ["energetic", "neutral", "tired", "relaxed", "anxious", "motivated", "content"]

MEAL_CATEGORIES = ["carb_heavy", "protein_heavy", "fat_heavy", "balanced", "na"]
INTERACTION_VALUES = ["positive", "neutral", "negative", "na"]

# Productivity codes
PRODUCTIVE_CODES = [1, 2, 3, 12, 13, 23, 123, 5]  # morning, afternoon, evening, combos, not_productive


def _time_str(hour: int, minute: int) -> str:
    """Convert hour (0-23) and minute to '8:30 PM' format."""
    dt = datetime(2000, 1, 1, hour, minute)
    return dt.strftime("%-I:%M %p")


def _duration_str(hours: float) -> str:
    """Convert decimal hours to '7h38m' format."""
    h = int(hours)
    m = int((hours - h) * 60)
    return f"{h}h{m}m"


def _generate_day(
    date: datetime,
    prev_sleep: Optional[float],
    prev_productivity: Optional[float],
    is_weekend: bool,
    rng: np.random.Generator,
) -> dict:
    """Generate one day of synthetic tracking data with realistic correlations."""

    # Base probabilities influenced by day type
    weekend_factor = 0.8 if is_weekend else 1.0

    # Sleep duration: 5-10 hours, normally distributed around 7.5
    sleep_hours = np.clip(rng.normal(7.5, 1.0), 5.0, 10.0)

    # Sleep quality affects productivity
    sleep_quality_factor = (sleep_hours - 5) / 5  # 0 to 1 scale

    # Previous day carryover effect
    carryover = 0
    if prev_sleep is not None:
        carryover += (prev_sleep - 7) * 3  # Good sleep helps
    if prev_productivity is not None:
        carryover += (prev_productivity - 50) * 0.1  # Momentum effect

    # Sickness (rare, ~5% chance, but clustered)
    is_sick = rng.random() < 0.05

    # Base productivity
    if is_sick:
        productivity = np.clip(rng.normal(15, 10), 0, 30)
    else:
        base_prod = 55 + carryover + sleep_quality_factor * 20
        productivity = np.clip(rng.normal(base_prod, 15) * weekend_factor, 0, 100)

    # Determine productivity level for reflection selection
    if productivity >= 70:
        prod_level = "high_productivity"
    elif productivity >= 40:
        prod_level = "medium_productivity"
    else:
        prod_level = "low_productivity"

    if is_sick:
        prod_level = "sick"

    # Select mood based on productivity
    if productivity >= 70:
        primary_mood = rng.choice(["happy", "productive", "energetic", "focused"])
        secondary_mood = rng.choice(["energetic", "motivated", "content"])
    elif productivity >= 40:
        primary_mood = rng.choice(["neutral", "productive", "calm", "focused"])
        secondary_mood = rng.choice(["neutral", "relaxed", "tired"])
    else:
        primary_mood = rng.choice(["tired", "stressed", "neutral"])
        secondary_mood = rng.choice(["tired", "anxious", "neutral"])

    if is_sick:
        primary_mood = "tired"
        secondary_mood = "neutral"

    # When most productive (correlates with actual productivity pattern)
    if productivity >= 60:
        when_prod = rng.choice([1, 12, 123])  # morning or combo
    elif productivity >= 30:
        when_prod = rng.choice([2, 3, 23])  # afternoon/evening
    else:
        when_prod = 5  # not productive

    # Boolean behaviors (more likely on good days)
    good_day_prob = productivity / 100

    studied_home = int(rng.random() < 0.4 * weekend_factor)
    studied_school = int(rng.random() < 0.3 * (1 if not is_weekend else 0.1))
    workout = int(rng.random() < (0.4 + good_day_prob * 0.2))
    meditation = int(rng.random() < (0.3 + good_day_prob * 0.2))
    morning_shower = int(rng.random() < 0.85)
    played_sports = int(rng.random() < 0.15 * (1.5 if is_weekend else 1))
    nap = int(rng.random() < (0.3 if productivity < 40 else 0.15))

    # Meal quality (1-5 scale, but we'll use categories for the cleaning pipeline)
    def pick_meal():
        if rng.random() < 0.1:  # 10% skip meal
            return "na"
        weights = [0.2, 0.25, 0.15, 0.4]  # carb, protein, fat, balanced
        return rng.choice(MEAL_CATEGORIES[:4], p=weights)

    breakfast_q = pick_meal()
    lunch_q = pick_meal()
    dinner_q = pick_meal()

    # Water intake (1-4 liters)
    water = np.clip(rng.normal(2.2, 0.6), 0.5, 4.0)

    # Social interactions
    def pick_interaction(base_positive_prob: float):
        if rng.random() < 0.2:  # No interaction
            return "na"
        p = base_positive_prob + good_day_prob * 0.2
        weights = [p, 0.5, 1 - p - 0.5]
        weights = [max(0, w) for w in weights]
        total = sum(weights)
        weights = [w / total for w in weights]
        return rng.choice(["positive", "neutral", "negative"], p=weights)

    partner_int = pick_interaction(0.5)
    family_int = pick_interaction(0.6)
    friends_int = pick_interaction(0.4)

    # Sleep percentages
    deep_pct = np.clip(rng.normal(20, 5), 10, 35)
    rem_pct = np.clip(rng.normal(22, 4), 12, 30)

    # Times
    # Wakeup: 5-9 AM
    wakeup_hour = int(np.clip(rng.normal(6.5 if not is_weekend else 8, 1), 5, 10))
    wakeup_min = rng.integers(0, 60)

    # Bed time: 9 PM - 12 AM
    bed_hour = int(np.clip(rng.normal(22.5 if not is_weekend else 23.5, 1), 21, 24))
    if bed_hour >= 24:
        bed_hour = 23
    bed_min = rng.integers(0, 60)

    # Dinner: 6-9 PM
    dinner_hour = int(np.clip(rng.normal(19.5, 1), 18, 21))
    dinner_min = rng.integers(0, 60)

    # Reflection text
    reflection = rng.choice(REFLECTION_TEMPLATES[prod_level])

    # Format date
    date_str = date.strftime("%b %d, %Y")

    return {
        "Date": date_str,
        "Reflection": reflection,
        "primary mood": primary_mood,
        "secondary mood": secondary_mood,
        "productivity\npercentage": int(round(productivity)),
        "when most \nproductive": when_prod,
        "studied \nat home": "yes" if studied_home else "no",
        "studied \nat school": "yes" if studied_school else "no",
        "breakfast qualtity": breakfast_q,  # Note: typo matches original schema
        "lunch quality": lunch_q,
        "dinner quality": dinner_q,
        "water \ndrank": round(water, 1),
        "workout \ndid": "yes" if workout else "no",
        "meditation": "yes" if meditation else "no",
        "morning \nshower": "yes" if morning_shower else "no",
        "played \nsports": "yes" if played_sports else "no",
        "interaction \nw/ partner": partner_int,
        "interaction \nw/ family": family_int,
        "interaction \nw/ friends": friends_int,
        "sickness": "yes" if is_sick else "no",
        "sleep \nduration": _duration_str(sleep_hours),
        "nap \ntoday": "yes" if nap else "no",
        "deep sleep \npercentage": int(round(deep_pct)),
        "REM sleep \npercentage": int(round(rem_pct)),
        "dinner time": _time_str(dinner_hour, dinner_min),
        "bed time": _time_str(bed_hour, bed_min),
        "wakeup time": _time_str(wakeup_hour, wakeup_min),
    }


def generate_synthetic_dataset(
    n_days: int = 90,
    start_date: Optional[datetime] = None,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a synthetic behavioral tracking dataset.

    Parameters
    ----------
    n_days : int
        Number of days to generate (default 90)
    start_date : datetime, optional
        Start date for the dataset. Defaults to 90 days ago.
    seed : int
        Random seed for reproducibility

    Returns
    -------
    pd.DataFrame
        DataFrame with synthetic tracking data matching the CTE schema
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)  # For random.choice in reflection selection

    if start_date is None:
        start_date = datetime.now() - timedelta(days=n_days)

    rows: List[dict] = []
    prev_sleep: Optional[float] = None
    prev_productivity: Optional[float] = None

    for day_offset in range(n_days):
        current_date = start_date + timedelta(days=day_offset)
        is_weekend = current_date.weekday() >= 5

        row = _generate_day(
            date=current_date,
            prev_sleep=prev_sleep,
            prev_productivity=prev_productivity,
            is_weekend=is_weekend,
            rng=rng,
        )
        rows.append(row)

        # Parse for carryover effects
        sleep_match = row["sleep \nduration"]
        h, m = 7, 0
        if "h" in sleep_match:
            parts = sleep_match.replace("m", "").split("h")
            h = int(parts[0])
            m = int(parts[1]) if parts[1] else 0
        prev_sleep = h + m / 60
        prev_productivity = float(row["productivity\npercentage"])

    return pd.DataFrame(rows)


def _argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Generate synthetic CTE tracking data.")
    p.add_argument("--days", type=int, default=90, help="Number of days to generate")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--out", type=Path, required=True, help="Output CSV path")
    p.add_argument(
        "--start-date",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD). Defaults to N days ago.",
    )
    return p


if __name__ == "__main__":
    args = _argparser().parse_args()

    start = None
    if args.start_date:
        start = datetime.strptime(args.start_date, "%Y-%m-%d")

    df = generate_synthetic_dataset(
        n_days=args.days,
        start_date=start,
        seed=args.seed,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Generated {len(df)} days of synthetic data → {args.out}")
