# CTE — Character Traits Evaluator

![CI](https://github.com/deepakdeo/cte-project/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.11%2B-blue)
![Poetry](https://img.shields.io/badge/poetry-managed-blue)
![Streamlit](https://img.shields.io/badge/streamlit-app-red)
![License](https://img.shields.io/badge/license-MIT-green)

![CTE Overview](assets/cte_overview.png)

**A reusable ML framework for behavioral self-tracking, personality trait extraction, and job-fit evaluation.**

CTE provides a complete pipeline from raw behavioral data to actionable career insights. Whether you're tracking your own productivity patterns or building a workforce analytics tool, CTE offers the building blocks you need.

---

## What CTE Does

```
Raw Behavioral Data → Clean Features → ML Models → Trait Profile → Job-Fit Score
```

1. **Data Cleaning** — Robust, deterministic parsing for messy self-tracking data
2. **Feature Engineering** — Temporal features, cyclical encodings, rolling statistics
3. **Trait Extraction** — Convert behavioral patterns into personality scores
4. **Job-Fit Scoring** — Match traits against job requirements with explainable verdicts

---

## Quick Start (30 seconds)

### Option A: Try the Demo Instantly

```bash
git clone https://github.com/deepakdeo/cte-project.git
cd cte-project
poetry install

# Launch the dashboard
PYTHONPATH=src poetry run streamlit run scripts/cte_app.py
```

Then click **"🧪 Demo Mode → Load Demo Assets"** in the sidebar to see the full experience.

### Option B: Generate Your Own Demo Data

```bash
# Generate 90 days of synthetic behavioral data
poetry run python src/cte/synthetic.py --days 90 --out data/sample/my_data.csv

# Clean it
poetry run python src/cte/data.py --in data/sample/my_data.csv --out data/sample/my_data_clean.parquet

# Generate a persona
PYTHONPATH=src poetry run python scripts/generate_demo_persona.py
```

### Option C: Use Docker (requires Docker installed)

```bash
docker compose up --build
# Open http://localhost:8501
```

Don't have Docker? Use Option A or B instead.

---

## Why Use CTE?

| Use Case | How CTE Helps |
|----------|---------------|
| **Self-improvement** | Track your patterns, understand what drives productivity |
| **Career planning** | Match your traits to job requirements before applying |
| **Workforce analytics** | Framework for trait-based team composition |
| **Research** | Reproducible pipeline for behavioral studies |
| **Learning** | Well-structured ML project demonstrating end-to-end skills |

---

## Core Features

### Synthetic Data Generator
Generate realistic behavioral data for testing or demos:

```python
from cte.synthetic import generate_synthetic_dataset

# Generate 90 days of data with temporal correlations
df = generate_synthetic_dataset(n_days=90, seed=42)
```

The generator creates realistic patterns including:
- Weekday/weekend differences
- Sleep → productivity correlations
- Mood coherence with reflections
- Social interaction patterns

### Robust Data Cleaning

```python
from cte.data import clean_csv

# Handles messy real-world data
clean_csv("raw_data.csv", "clean.parquet")
```

- Header normalization (handles newlines, typos)
- Deterministic date/time parsing
- Duration parsing (`7h38m` → 7.63 hours)
- Flexible boolean coercion (`yes/y/true/1` → 1)
- Social interaction encoding (positive/neutral/negative → +1/0/-1)

### Feature Engineering

```python
from cte.features import engineer_features

# Add lags, rolling stats, cyclical encodings
df_features = engineer_features(df_clean)
```

- Lag features (t-1, t-2, t-3)
- 7-day rolling mean/std
- Cyclical time encodings (sin/cos)
- Day-of-week one-hots

### Job-Fit Scoring

```python
from cte.scoring import score_requirements

# Compare persona against job requirements
overall, match_ratio, risk, details, criticals = score_requirements(
    per_trait=persona["per_trait"],
    requirements=[
        {"trait": "communication", "required_level": "high"},
        {"trait": "teamwork", "required_level": "medium"},
    ]
)
# Returns: "Strong fit", 0.85, "low-risk", [...], []
```

---

## Project Structure

```
cte-project/
├── src/cte/                    # Core Python package
│   ├── data.py                 # Data cleaning pipeline
│   ├── features.py             # Feature engineering
│   ├── synthetic.py            # Synthetic data generator
│   ├── nlp.py                  # Sentiment analysis
│   ├── requirements.py         # JD parsing (LLM + heuristic)
│   ├── scoring.py              # Job-fit scoring
│   └── report.py               # Report generation
├── scripts/
│   ├── cte_app.py              # Streamlit dashboard
│   ├── cte_cli.py              # CLI tool
│   └── generate_demo_persona.py # Demo persona generator
├── tests/                      # Unit tests (56 tests)
│   ├── test_synthetic.py
│   ├── test_data.py
│   └── test_scoring.py
├── notebooks/                  # Analysis notebooks
│   ├── 01_EDA.ipynb
│   ├── 02_Features.ipynb
│   ├── 03_Baselines.ipynb
│   └── ...
├── data/
│   └── sample/                 # Demo data (committed)
│       ├── synthetic_90d.csv
│       ├── demo_persona.json
│       └── sample_jd.txt
└── docs/                       # Additional documentation
    └── COLLECT_YOUR_DATA.md    # Data collection guide
```

---

## Running Tests

```bash
poetry run pytest tests/ -v

# With coverage
poetry run pytest tests/ --cov=src/cte --cov-report=term-missing
```

---

## Bring Your Own Data

CTE works with any behavioral tracking data that matches the schema. See [docs/COLLECT_YOUR_DATA.md](docs/COLLECT_YOUR_DATA.md) for:

- Required columns and formats
- Optional columns
- Data collection tips
- Integration with tracking apps

---

## The Pipeline in Detail

### 1. Data Cleaning (`data.py`)

Transforms messy self-tracking exports into clean, typed data:

| Raw Input | Cleaned Output |
|-----------|----------------|
| `"Jan 27, 2025"` | `2025-01-27` (datetime) |
| `"7h38m"` or `"7:38"` | `7.63` (float hours) |
| `"yes"`, `"Y"`, `"1"` | `1` (Int64) |
| `"positive"` | `1.0` (interaction score) |

### 2. Feature Engineering (`features.py`)

Adds temporal and behavioral features:

- **Cyclical encoding**: `wakeup_time` → `wakeup_sin`, `wakeup_cos`
- **Lags**: `productivity_lag1`, `sleep_lag2`
- **Rolling stats**: `productivity_roll7_mean`, `productivity_roll7_std`

### 3. Trait Extraction

Maps behavioral patterns to personality traits:

| Behavior Pattern | Trait |
|------------------|-------|
| High productivity + deep work | Focus |
| Consistent routines | Reliability |
| Recovery from low days | Resilience |
| Positive social interactions | Communication |

### 4. Job-Fit Scoring (`scoring.py`)

Matches persona against requirements with configurable thresholds:

```python
thresholds = {"low": 0.50, "medium": 0.60, "high": 0.70}
weights = {"low": 1.0, "medium": 1.2, "high": 1.5}
```

Returns explainable verdicts: **Strong fit**, **Possible fit**, **Leaning no**, **Not a fit**

---

## Streamlit Dashboard

The interactive dashboard provides:

- **Dashboard**: Radar charts, trait breakdown, performance trends
- **Job Evaluation**: Paste a JD, get instant fit analysis
- **Daily Updates**: Log daily performance, build evidence
- **Starter Mode**: Create a persona from a 2-minute questionnaire

---

## CLI Usage

```bash
# Evaluate job fit from command line
PYTHONPATH=src poetry run python scripts/cte_cli.py \
  --persona data/sample/demo_persona.json \
  --jd data/sample/sample_jd.txt
```

---

## Deploy Your Own

### Streamlit Cloud (Free)

1. Fork this repo
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub and select the repo
4. Set:
   - Main file: `scripts/cte_app.py`
   - Python version: 3.11
5. Add secret `OPENAI_API_KEY` (optional, for LLM features)

Your dashboard will be live at `https://your-app.streamlit.app`

### Docker

```bash
docker compose up --build
# Open http://localhost:8501
```

---

## Limitations & Responsible Use

- **Not a hiring tool**: CTE is for self-reflection and career exploration, not employment decisions
- **Small sample sizes**: Trait scores are estimates; treat with appropriate uncertainty
- **Privacy first**: All processing runs locally; no data leaves your machine
- **Bias awareness**: Self-reported data reflects perception, not objective reality

---

## Tech Stack

| Category | Technologies |
|----------|--------------|
| **Core** | Python 3.11+, pandas, numpy, scikit-learn |
| **NLP** | transformers, vaderSentiment, OpenAI API |
| **ML** | XGBoost, statsmodels, SHAP |
| **App** | Streamlit, Plotly |
| **Infra** | Poetry, Docker, pytest |

---

## Roadmap

- [x] Data cleaning pipeline
- [x] Feature engineering
- [x] Baseline modeling (Ridge, RF, GBM)
- [x] NLP sentiment analysis
- [x] Job-fit scoring system
- [x] Streamlit dashboard
- [x] Synthetic data generator
- [x] Test suite (56 tests)
- [x] CI/CD pipeline
- [ ] Public dataset validation
- [ ] FastAPI endpoints

---

## Contributing

Contributions welcome! Areas of interest:

- Additional trait extraction methods
- Integration with more data sources (Oura, Whoop, etc.)
- Public dataset validation
- UI/UX improvements

---

## License

MIT — see [LICENSE](LICENSE)
