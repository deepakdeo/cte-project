# 🧠 Character Traits Evaluator (CTE)

<p align="center">
  <img src="notebooks/reports/figures/cte_readme_overview.png" alt="CTE Overview" width="750">
</p>

---

## 🎯 Project Overview

The **Character Traits Evaluator (CTE)** is a personal data-science and machine learning pipeline  
that analyzes **daily behavioral, physiological, and lifestyle metrics** to uncover how different  
habits influence **day-to-day productivity and well-being**.

This project represents a full **end-to-end ML workflow** — from raw data to predictive modeling —  
built entirely with open-source tools and designed for **reproducibility, interpretability,** and **portfolio readiness**.

---

## 🧩 Concept Pipeline

| Stage | Description | Techniques / Tools |
|:------|:-------------|:------------------|
| **Inputs** | 27 wearable and self-reported daily metrics (sleep, water, mood, study, etc.) | Data ingestion, preprocessing |
| **Analysis** | Exploratory Data Analysis (EDA), NLP on reflections, feature engineering, modeling | `pandas`, `numpy`, `matplotlib`, `scikit-learn` |
| **Outputs** | Insights on factors influencing productivity; correlation heatmaps, trendlines | Model evaluation, visualization |
| **Job-Fit Check** | (Planned) Compare trait patterns to job descriptions using LLMs | Embeddings, similarity modeling |

---

## 🧪 Current Progress

| Notebook | Purpose | Key Outputs |
|-----------|----------|-------------|
| **`01_Preprocessing.ipynb`** | Cleaned and standardized daily logs (sleep, mood, hydration, study sessions) | `data/interim/cleaned.parquet` |
| **`02_Features.ipynb`** | Engineered ~100 quantitative and categorical features | `data/interim/features.parquet` |
| **`03_Baselines.ipynb`** | Built **time-aware regression baselines** predicting `productivity_pct` | `notebooks/reports/baseline_leaderboard.csv`, visual reports, model card |
| **`04_Modeling.ipynb`** *(in progress)* | Advanced models (XGBoost, LightGBM, feature lags, tuning) | — |
| **`05_Insights.ipynb`** *(planned)* | Explainable ML (SHAP) + visualization dashboard | — |

---

## 📊 Summary of Baseline Results

| Model | MAE | RMSE | R² | Notes |
|:------|----:|----:|---:|:------|
| Mean Baseline | 52.5 | 57.9 | -0.93 | Reference |
| Ridge Regression | 43.2 | 50.8 | 0.25 | Improved linear baseline |
| **Gradient Boosting (depth 3)** | **41.0** | **48.5** | **0.42** | Best baseline model |

> *The best baseline model reduced RMSE by ~15–20 % compared to a naive mean predictor,  
> indicating clear predictive signal in the engineered daily features.*

---

## 📈 Outputs Generated

- 📄 **`baseline_leaderboard.csv`** — performance comparison table  
- 📊 **`/notebooks/reports/figures/`** — predicted vs true plots, residuals, feature importance  
- 🧾 **`baseline_modelcard.json`** — model metadata & reproducibility info  
- 💾 **`/models/`** — persisted best baseline model (`.joblib`)

---

## 🧰 Tools & Techniques Demonstrated

| Category | Tools / Concepts |
|-----------|-----------------|
| **Data Wrangling** | `pandas`, `numpy`, datetime parsing, type handling |
| **Feature Engineering** | normalization, encoding, temporal variables |
| **Modeling** | regression (Linear, Ridge, RandomForest, GradientBoosting) |
| **Evaluation** | MAE, RMSE, R², MAPE, expanding time-series CV |
| **Visualization** | `matplotlib`, correlation plots, residual analysis |
| **Automation & Reproducibility** | `pathlib`, modular directories, `Pipeline`, `joblib` |
| **Data Provenance** | JSON model cards, reproducible folder structure |

---

## 📁 Repository Structure

```
cte-project/
├── data/
│   ├── raw/               # Original daily logs
│   ├── interim/           # Clean & feature-engineered data
│   └── processed/         # Modeling-ready data (future)
├── notebooks/
│   ├── 01_Preprocessing.ipynb
│   ├── 02_Features.ipynb
│   ├── 03_Baselines.ipynb
│   ├── reports/
│   │   ├── figures/
│   │   └── baseline_leaderboard.csv
│   └── ...
├── models/                # Saved models (.joblib)
├── src/cte/               # Python modules and helpers
└── pyproject.toml / .gitignore / README.md
```

---

## 🔮 Next Steps

- Add **temporal lag & rolling features** (yesterday’s productivity, 3-day moving averages)  
- Introduce **advanced models**: XGBoost, LightGBM, CatBoost  
- Perform **hyperparameter optimization** with `Optuna` or randomized search  
- Apply **SHAP** and **feature importance visualization** for interpretability  
- Build an interactive **Streamlit dashboard** for daily self-analytics  

---

## 👤 Author

**Deepak Kumar Deo**  
Ph.D. in Physics (Astrophysics) & Curriculum & Instruction  
📍 Kansas City, MO  
💼 Open to Data Scientist / Applied Scientist roles  
🔗 [LinkedIn](https://www.linkedin.com/in/deepakdeo) | [GitHub](https://github.com/deepakdeo)

---

### ⭐ Project Status
| Phase | Status | Description |
|:------|:------:|:------------|
| Data Cleaning & Features | ✅ | Complete |
| Baseline Modeling | ✅ | Complete |
| Advanced Modeling | 🚧 | In progress |
| Insights & Dashboard | ⏳ | Upcoming |

---

> _This project demonstrates end-to-end data-science fluency —  
> from real-world data collection to modeling, interpretation, and reporting —  
> built with clarity, reproducibility, and research-grade rigor._
