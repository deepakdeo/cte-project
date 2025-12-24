# Character Traits Evaluator (CTE)

![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Poetry](https://img.shields.io/badge/poetry-managed-blue)
![Streamlit](https://img.shields.io/badge/streamlit-app-red)
![License](https://img.shields.io/badge/license-MIT-green)

![CTE Overview](assets/cte_overview.png)

A project that analyzes 27 wearable metrics and daily reflections, runs EDA/NLP/ML, and uses an LLM to profile traits and generate data-driven recommendations.


CTE is a machine-learning pipeline that turns mixed personal-tracking data (sleep, mood, productivity, habits, daily reflections) into clean, model-ready features and interpretable behavioral insights.

The system analyzes behavioral patterns to estimate personality traits and evaluates personality, job compatibility against specific role requirements. All processing runs locally for privacy; raw personal data is excluded from version control.

## Overview

I wanted to know how sleep, routines, and social context actually show up in my day-to-day work productivity and performance. So I tracked 27 different parameters - sleep, mood, productivity, habits, and daily reflections - for 72 days.

**CTE** takes those raw logs (timestamps, durations, booleans, text), cleans them with deterministic rules (fixed formats, no guessing), and produces a tidy, typed Parquet table that’s ready for feature engineering and modeling.

> **Status:** Phase 3 (feature engineering and baseline modeling complete, moving toward advanced modeling) ✅

## Impact (What This Demonstrates)

- End-to-end DS workflow: data cleaning, feature engineering, modeling, NLP, evaluation.
- Reproducibility: typed Parquet outputs, deterministic parsing, time-aware validation.
- Product thinking: interactive dashboard + CLI and a job-fit scoring layer.


## Current Features

- **Robust Data Cleaning**
  - Header normalization and schema standardization
  - Deterministic date parsing with default-year fallback
  - Time parsing to **minutes-after-midnight** features (`*_time_minutes`)
  - Duration parsing (e.g., `7h38m`, `7:38`) to decimal hours
  - Flexible boolean coercion (`yes/no/y/n/true/false/t/f/1/0`)
  - Social interaction encoding with **no-interaction flags**
  - Percentage/number coercion with validation and clipping
  - “When most productive” codebook decoding + one-hot indicators

- **Privacy-First**
  - All processing runs locally
  - Optional LLM calls; local-only path supported
  - Raw personal data excluded from version control

- **Production-Ready Engineering**
  - Type-safe transformations with explicit parsing rules
  - CLI interface with argument parsing
  - Reproducible environment via Poetry + Python 3.11

## Recent Progress

With **03_Baselines.ipynb**, the project now includes a predictive layer that connects behavioral features to daily productivity.

- Implemented time-aware baseline models (Linear, Ridge, Random Forest, Gradient Boosting)
- Added regression metrics (MAE, RMSE, R², MAPE)
- Introduced expanding-window cross-validation for temporal consistency
- Generated visual diagnostics (predicted vs true plots, residuals, feature importances)
- Built model card and leaderboard export for reproducibility

## Results & Evaluation

Highlights from baseline modeling and diagnostics:
- Time-aware CV across multiple baselines (Linear/Ridge/RandomForest/GBRT)
- Feature importance + error analysis to surface top behavioral drivers
- Consistent evaluation artifacts (leaderboard, residuals, pred-vs-true)

Baseline snapshot (from `notebooks/reports/04_leaderboard.csv`):

| Model | Val MAE | Val RMSE | Val R² | Test MAE | Test RMSE | Test R² |
|---|---:|---:|---:|---:|---:|---:|
| Ridge | 26.04 | 32.32 | 0.193 | 43.40 | 48.06 | -0.462 |
| GBR | 28.81 | 41.49 | -0.329 | 30.23 | 32.25 | 0.341 |

See `notebooks/03_Baselines.ipynb` and generated artifacts in `notebooks/reports/` (gitignored).

## Key Findings (EDA Snapshot)

From early EDA on the 72-day sample:
- Productivity aligns most with time-of-day self-reports (morning/afternoon/evening).
- Sleep duration and naps show positive relationships with productivity.
- Balanced dinners show higher average productivity vs heavier meals.

See `docs/eda_summary.txt`.

## Limitations & Responsible Use

- Small, single-subject dataset; results are exploratory and not generalizable.
- Trait scoring is heuristic; use as guidance, not as a definitive judgment.
- Best used for self-tracking and reflection, not for high-stakes decisions.

### Reflections (NLP)
Reflections are analyzed with sentiment and lightweight trait signals. Visuals are generated in notebooks and excluded from git.


## Upcoming Work

- Integrate NLP features into modeling
- Introduce XGBoost/LightGBM with hyperparameter tuning
- Add SHAP-based interpretability and stronger model cards
- Provide a multi-user demo path (sample persona + JD + reports)

## Quick Start

### Requirements
- Python **3.11+**
- [Poetry](https://python-poetry.org/) for dependency management

### Install and run (demo with sample data)

```bash
git clone https://github.com/deepakdeo/cte-project.git
cd cte-project
poetry install
```

Run the cleaning pipeline on the included sample:

```bash
poetry run python src/cte/data.py   --in data/sample/cte_sample.csv   --out data/sample/clean_sample.parquet
```

Inspect the processed output:

```bash
poetry run python -c "
import pandas as pd
df = pd.read_parquet('data/sample/clean_sample.parquet')
print(f'Dataset: {df.shape[0]} rows, {df.shape[1]} columns')
print('\nSample columns:', df.columns[:12].tolist())
print('\nHead:\n', df.head(3))
"
```

### Docker (Optional)

Build and run the demo app:

```bash
docker build -t cte-app .
docker run --rm -p 8501:8501 cte-app
```

Or with Docker Compose:

```bash
docker compose up --build
```

## Demo (Streamlit App)

Run the dashboard locally:

```bash
PYTHONPATH=src poetry run streamlit run scripts/cte_app.py
```

Notes:
- The sidebar expects a persona JSON (generated in `notebooks/06_Persona_LLM.ipynb`).
- Set `OPENAI_API_KEY` in a `.env` file if you want LLM-based JD parsing and sentiment.
 
Sample inputs:
- Persona: `data/sample/sample_persona.json`
- JD: `data/sample/sample_jd.txt`

Quick demo flow:
1) Launch the app.
2) In the sidebar, set the persona to `data/sample/sample_persona.json`.
3) Paste the sample JD from `data/sample/sample_jd.txt` into the Evaluate Job tab.

Starter Mode:
- Use the sidebar “Starter Mode” questionnaire to create a low-confidence persona
  and begin logging daily updates immediately.

CLI usage:

```bash
PYTHONPATH=src poetry run python scripts/cte_cli.py --persona path/to/06_profile_persona_llm.json --jd path/to/jd.txt
```

## Supported Data Types (Cleaned Schema)

**Sleep & Timing**
- `sleep_duration_h` (float hours)
- `wakeup_time_minutes`, `bed_time_minutes`, `dinner_time_minutes` (int minutes after midnight)

**Productivity**
- `productivity_pct` (0–100; validated/clipped)
- `when_most_productive_decoded` (categorical)
- `prod_morning`, `prod_afternoon`, `prod_evening`, `prod_none` (one-hot ints)

**Health & Habits**
- `studied_at_home`, `studied_at_school`, `workout_did`, `meditation`,
  `morning_shower`, `played_sports`, `sickness`, `nap_today` (Int64 0/1)
- `water_drank_l`, `breakfast_quality`, `lunch_quality`, `dinner_quality` (floats)

**Social Interactions**
- `{partner|family|friends}_score` (−1=negative, 0=neutral, +1=positive, NaN=unknown)
- `{partner|family|friends}_no_interaction` (Int64 0/1)

**Mood & Text**
- `primary_mood`, `secondary_mood` (normalized strings)
- `reflection` (free text; unmodified)

> The original CSV headers (with newlines/typos) are normalized internally; see `src/cte/data.py::RENAME_MAP`.

## Project Layout

```
cte-project/
├── data/
│ ├── sample/
│ │ ├── cte_sample.csv         # small demo dataset (committed)
│ │ ├── sample_persona.json    # demo persona
│ │ └── sample_jd.txt          # demo job description
│ ├── raw/                     # your private raw data (gitignored)
│ └── interim/                 # cleaned / feature data (gitignored)
├── assets/
│ └── cte_overview.png          # README overview graphic
├── notebooks/
│ ├── 01_Preprocessing.ipynb   # data cleaning
│ ├── 02_Features.ipynb        # feature engineering
│ ├── 03_Baselines.ipynb       # baseline modeling
│ ├── 04_Modeling.ipynb        # advanced modeling
│ ├── 05_TraitScoring.ipynb    # trait scoring
│ ├── 06_Persona_LLM.ipynb     # persona generation
│ └── reports/                 # generated outputs (gitignored)
├── models/                    # saved trained models (.joblib)
├── scripts/                   # Streamlit app + CLI
├── src/
│ └── cte/
│ ├── init.py
│ ├── data.py # cleaning pipeline (MVP)
│ ├── features.py # feature engineering logic
│ ├── nlp.py # sentiment + trait extraction
│ ├── requirements.py # JD parsing
│ ├── scoring.py # job-fit scoring
│ ├── report.py # report writer
│ ├── openai_util.py # OpenAI helper
│ └── persona.py # persona loading
├── .gitignore
├── poetry.lock
├── pyproject.toml
└── README.md
```
> The `Project Layout` above shows the repository structure as of **Phase 3**.  
> Within `src/cte/`, upcoming modules like `api.py` (for deployment) and `insights.py` (for interpretability) will be added as phases progress.

## Pipeline (End-to-End Flow)

```
Raw CSV/Logs
   ↓
Data cleaning (deterministic parsing)
   ↓
Feature engineering (lags, rollings, time encodings)
   ↓
Modeling + evaluation (time-aware CV)
   ↓
Trait profile + job-fit scoring
   ↓
Streamlit dashboard / CLI reports
```


## Development Roadmap

**Phase 1 — Data Foundation** ✅  
- Cleaning pipeline, schema normalization, documented sample

**Phase 2 — Feature Engineering** ✅  
- Rolling windows, trend/volatility, circadian features

**Phase 3 — Baseline Modeling** ✅  
- Linear, Ridge, RandomForest, GradientBoosting regressors
- Time-aware cross-validation and baseline leaderboard

**Phase 4 — Interpretability**  
- SHAP/LIME, feature importances, ablations

**Phase 5 — Advanced Analytics**  
- NLP on reflections, structured insight generation

**Phase 6 — Deployment**  
- FastAPI endpoints, Streamlit demo, Dockerization

## Project Story (Recruiter One-Pager)

See `docs/project_story.md`.

## Tech Stack

- **Core**: Python 3.11, pandas, numpy
- **Tooling**: Poetry, argparse CLI
- **Planned ML**: scikit-learn, XGBoost/LightGBM
- **Planned NLP**: HuggingFace Transformers
- **Planned Deployment**: FastAPI, Streamlit, Docker

## Privacy

CTE is designed for local analysis of personal data. All processing occurs on your machine; raw personal data is excluded from version control by default.

## License

MIT — see [LICENSE](LICENSE).
