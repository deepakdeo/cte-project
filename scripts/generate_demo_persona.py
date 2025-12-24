#!/usr/bin/env python3
"""
Generate a demo persona from synthetic data for the CTE demo experience.

This script:
1. Uses the synthetic data generator to create behavioral data
2. Runs it through the cleaning pipeline
3. Runs feature engineering
4. Calculates trait scores based on the patterns
5. Saves a complete demo persona

Usage:
    poetry run python scripts/generate_demo_persona.py
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from cte.synthetic import generate_synthetic_dataset
from cte.data import clean_csv


def calculate_trait_scores(df: pd.DataFrame) -> dict:
    """
    Calculate trait scores from behavioral data.

    Returns a per_trait dict with scores, confidence, and evidence.
    """
    n = len(df)

    # Helper to normalize and calculate confidence based on variance
    def score_with_confidence(values, higher_is_better=True):
        if len(values) == 0:
            return 0.5, 0.3
        mean = np.nanmean(values)
        std = np.nanstd(values)
        # Normalize to 0-1 if not already
        if np.nanmax(values) > 1:
            mean = mean / 100  # Assume percentage
        # Confidence based on sample size and consistency
        confidence = min(0.95, 0.4 + 0.5 * (n / 90) + 0.1 * (1 - min(std / 30, 1) if std else 0))
        return round(mean, 3), round(confidence, 3)

    traits = {}

    # Focus: based on productivity and deep work patterns
    if "productivity_pct" in df.columns:
        focus_score, focus_conf = score_with_confidence(df["productivity_pct"].dropna())
        traits["focus"] = {
            "score": focus_score,
            "confidence": focus_conf,
            "evidence": [{"ts": datetime.now().isoformat(), "desc": f"Avg productivity: {focus_score*100:.0f}%", "link": None}]
        }

    # Planning: based on study patterns and routine consistency
    plan_indicators = []
    if "studied_at_home" in df.columns:
        plan_indicators.append(df["studied_at_home"].mean())
    if "prod_morning" in df.columns:
        plan_indicators.append(df["prod_morning"].mean())
    if plan_indicators:
        plan_score = np.mean(plan_indicators)
        traits["planning"] = {
            "score": round(0.4 + 0.5 * plan_score, 3),
            "confidence": round(0.5 + 0.3 * (n / 90), 3),
            "evidence": []
        }

    # Communication: based on social interaction patterns
    comm_scores = []
    for col in ["partner_score", "family_score", "friends_score"]:
        if col in df.columns:
            # Score is -1, 0, 1, normalize to 0-1
            normalized = (df[col].dropna() + 1) / 2
            if len(normalized) > 0:
                comm_scores.append(normalized.mean())
    if comm_scores:
        traits["communication"] = {
            "score": round(np.mean(comm_scores), 3),
            "confidence": round(0.5 + 0.3 * (n / 90), 3),
            "evidence": []
        }

    # Teamwork: similar to communication but weighted by interaction frequency
    no_int_cols = ["partner_no_interaction", "family_no_interaction", "friends_no_interaction"]
    interaction_freq = 1.0
    for col in no_int_cols:
        if col in df.columns:
            interaction_freq -= df[col].mean() * 0.2
    teamwork_base = traits.get("communication", {}).get("score", 0.5)
    traits["teamwork"] = {
        "score": round(teamwork_base * interaction_freq, 3),
        "confidence": round(0.5 + 0.3 * (n / 90), 3),
        "evidence": []
    }

    # Adaptability: variance in productivity (handling different situations)
    if "productivity_pct" in df.columns:
        prod_std = df["productivity_pct"].std()
        # Moderate variance is good (too low = rigid, too high = inconsistent)
        adapt_score = 1 - abs(prod_std - 20) / 40  # Optimal std around 20
        traits["adaptability"] = {
            "score": round(max(0.3, min(0.9, adapt_score)), 3),
            "confidence": round(0.5 + 0.3 * (n / 90), 3),
            "evidence": []
        }

    # Learning mindset: workout, meditation, consistent improvement
    learn_indicators = []
    if "workout_did" in df.columns:
        learn_indicators.append(df["workout_did"].mean())
    if "meditation" in df.columns:
        learn_indicators.append(df["meditation"].mean())
    if learn_indicators:
        traits["learning_mindset"] = {
            "score": round(0.4 + 0.5 * np.mean(learn_indicators), 3),
            "confidence": round(0.5 + 0.3 * (n / 90), 3),
            "evidence": []
        }

    # Impact: productivity combined with consistency
    if "productivity_pct" in df.columns:
        avg_prod = df["productivity_pct"].mean() / 100
        consistency = 1 - (df["productivity_pct"].std() / 50)
        traits["impact"] = {
            "score": round(0.4 * avg_prod + 0.6 * max(0.3, consistency), 3),
            "confidence": round(0.5 + 0.3 * (n / 90), 3),
            "evidence": []
        }

    # Reliability: consistency in routines
    routine_indicators = []
    if "morning_shower" in df.columns:
        routine_indicators.append(df["morning_shower"].mean())
    if "wakeup_time_minutes" in df.columns:
        # Low variance in wakeup time = reliable
        wakeup_std = df["wakeup_time_minutes"].std()
        routine_indicators.append(1 - min(wakeup_std / 120, 1))  # Normalize by 2 hours
    if routine_indicators:
        traits["reliability"] = {
            "score": round(np.mean(routine_indicators), 3),
            "confidence": round(0.5 + 0.3 * (n / 90), 3),
            "evidence": []
        }

    # Resilience: recovery from low days, no sickness impact
    resilience_score = 0.6
    if "sickness" in df.columns:
        sick_rate = df["sickness"].mean()
        resilience_score = 0.7 - 0.3 * sick_rate
    if "productivity_pct" in df.columns:
        # Check recovery: days after low productivity
        prod = df["productivity_pct"].values
        recoveries = 0
        recovery_count = 0
        for i in range(1, len(prod)):
            if prod[i-1] < 40 and prod[i] > 50:
                recoveries += 1
                recovery_count += 1
            elif prod[i-1] < 40:
                recovery_count += 1
        if recovery_count > 0:
            resilience_score = 0.4 + 0.5 * (recoveries / recovery_count)
    traits["resilience"] = {
        "score": round(max(0.4, min(0.9, resilience_score)), 3),
        "confidence": round(0.5 + 0.3 * (n / 90), 3),
        "evidence": []
    }

    # Independence: working at home, self-study
    indep_indicators = []
    if "studied_at_home" in df.columns:
        indep_indicators.append(df["studied_at_home"].mean())
    if traits.get("focus"):
        indep_indicators.append(traits["focus"]["score"])
    if indep_indicators:
        traits["independence"] = {
            "score": round(np.mean(indep_indicators), 3),
            "confidence": round(0.5 + 0.3 * (n / 90), 3),
            "evidence": []
        }

    return traits


def main():
    print("Generating demo persona from synthetic data...")

    # Generate 90 days of synthetic data
    print("1. Generating 90 days of synthetic behavioral data...")
    df_raw = generate_synthetic_dataset(n_days=90, seed=42)

    # Save raw synthetic data
    raw_path = Path("data/sample/synthetic_90d.csv")
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    df_raw.to_csv(raw_path, index=False)
    print(f"   Saved raw data to {raw_path}")

    # Clean the data
    print("2. Running data cleaning pipeline...")
    clean_path = Path("data/sample/synthetic_90d_clean.parquet")
    df_clean = clean_csv(raw_path, clean_path)
    print(f"   Saved cleaned data to {clean_path}")

    # Calculate trait scores
    print("3. Calculating trait scores from behavioral patterns...")
    traits = calculate_trait_scores(df_clean)

    # Create persona
    persona = {
        "name": "Demo User",
        "created": datetime.now().isoformat(),
        "data_source": "synthetic_90d",
        "per_trait": traits
    }

    # Save persona
    persona_path = Path("data/sample/demo_persona.json")
    persona_path.write_text(json.dumps(persona, indent=2))
    print(f"   Saved demo persona to {persona_path}")

    # Print summary
    print("\n=== Demo Persona Summary ===")
    print(f"Traits: {len(traits)}")
    avg_score = np.mean([t["score"] for t in traits.values()])
    avg_conf = np.mean([t["confidence"] for t in traits.values()])
    print(f"Average Score: {avg_score:.1%}")
    print(f"Average Confidence: {avg_conf:.1%}")
    print("\nTrait Breakdown:")
    for name, info in sorted(traits.items(), key=lambda x: x[1]["score"], reverse=True):
        print(f"  {name:20s}: {info['score']:.1%} (conf: {info['confidence']:.1%})")

    print("\nDemo persona ready! Use in the app with:")
    print("  Persona path: data/sample/demo_persona.json")


if __name__ == "__main__":
    main()
