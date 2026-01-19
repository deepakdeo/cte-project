# CLAUDE.md - Project Context for Claude Code

## Project Overview

**CTE (Character Traits Evaluator)** is a reusable ML framework for behavioral self-tracking, personality trait extraction, and job-fit evaluation.

**Live Demo:** https://cte-project-deepakdeo.streamlit.app/

## Key Commands

```bash
# Run the Streamlit dashboard
PYTHONPATH=src poetry run streamlit run scripts/cte_app.py

# Run tests
PYTHONPATH=src poetry run pytest tests/ -v

# Generate synthetic data
poetry run python src/cte/synthetic.py --days 90 --out data/sample/synthetic_90d.csv

# Clean data
poetry run python src/cte/data.py --in data/sample/synthetic_90d.csv --out data/sample/clean.parquet

# Generate demo persona
PYTHONPATH=src poetry run python scripts/generate_demo_persona.py

# Docker (requires Docker Desktop)
docker compose up --build
```

## Project Structure

```
src/cte/           # Core Python package
  data.py          # Data cleaning pipeline
  features.py      # Feature engineering (lags, rolling, cyclical)
  synthetic.py     # Synthetic data generator
  nlp.py           # Sentiment analysis (OpenAI + VADER)
  requirements.py  # JD parsing (LLM + heuristic)
  scoring.py       # Job-fit scoring
  openai_util.py   # OpenAI API wrapper (supports runtime API keys)

scripts/
  cte_app.py       # Streamlit dashboard (main app)
  cte_cli.py       # CLI tool
  generate_demo_persona.py

tests/             # 56 unit tests
data/sample/       # Demo data (committed to git)
notebooks/         # Analysis notebooks (01-06)
```

## Architecture Decisions

- **PYTHONPATH=src** required for imports (or use the sys.path hack in cte_app.py)
- **OpenAI API is optional** - app falls back to keyword-based extraction without it
- **Users can add their own API key** in the sidebar (Analysis Settings)
- **Synthetic data generator** creates realistic 90-day behavioral data for demos
- **Privacy-first**: raw personal data is gitignored, all processing is local

## Current Status

**Completed:**
- Data cleaning pipeline
- Feature engineering
- Baseline modeling (Ridge, RF, GBM)
- NLP sentiment analysis
- Job-fit scoring system
- Streamlit dashboard with Demo Mode
- Synthetic data generator
- Test suite (56 tests, CI passing)
- Streamlit Cloud deployment

**Not yet done:**
- Public dataset validation
- FastAPI endpoints

## Testing

Tests are in `tests/` directory. Run with:
```bash
PYTHONPATH=src poetry run pytest tests/ -v
```

CI runs automatically on push via GitHub Actions.

## Deployment

- **Streamlit Cloud**: Auto-deploys from main branch
- **Docker**: `docker compose up --build`
- **Local**: `PYTHONPATH=src poetry run streamlit run scripts/cte_app.py`

## Git Workflow

- Main branch is `main`
- CI must pass before considering changes complete
- Don't commit raw personal data (it's gitignored)
- Sample/demo data in `data/sample/` is committed

## Notes

- The app works without OpenAI API key (uses hybrid/keyword extraction)
- Streamlit Cloud deployment uses minimal requirements.txt (not full poetry deps)
- Poetry is used for local development, requirements.txt for Streamlit Cloud
