# CTE — Project Story (Recruiter One-Pager)

## Problem
I wanted to quantify how daily habits (sleep, routines, social context, reflections) affect real productivity and career readiness. Most tools stop at dashboards; I wanted a full data science pipeline that produces interpretable insights and a job‑fit verdict.

## Approach
- Collected 27 daily signals over ~72 days (sleep, mood, habits, reflections).
- Built a deterministic cleaning pipeline (no fuzzy parsing).
- Engineered temporal features (lags, rolling stats, circadian encodings).
- Trained baseline models with time‑aware validation.
- Added NLP sentiment + trait extraction and a persona scoring layer.
- Built a Streamlit dashboard + CLI to evaluate job fit from a JD.

## Results
- Cleaned, typed Parquet table; reproducible pipelines.
- Baseline models with CV metrics and diagnostics.
- Trait profile + job‑fit scoring that’s explainable and extensible.

## Why It Matters
This shows end‑to‑end DS/ML skills: data engineering, modeling, evaluation, NLP, interpretability, and productization.

## Limitations
- Small sample size (personal data).
- No external ground truth labels for traits.
- Job‑fit scoring is heuristic and can be calibrated further.

## Next Steps
- Expand to multi‑user dataset and reduce cold‑start with baseline priors.
- Add advanced models (XGBoost/LightGBM) + SHAP interpretability.
- Provide a “day‑1” onboarding and optional data imports.
