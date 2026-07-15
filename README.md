# Continuous Data Profiling for Interactive Data Science

**California Housing Price Analysis** — An end-to-end machine learning pipeline that integrates continuous data profiling, predictive modeling, drift monitoring, and interactive visualization for real-world data science workflows.

![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.0%2B-green)
![Status](https://img.shields.io/badge/status-academic%20research-lightgrey)

---

## Table of Contents

- [Overview](#overview)
- [Research Paper](#research-paper)
- [Features](#features)
- [Project Structure](#project-structure)
- [Dataset](#dataset)
- [Installation](#installation)
- [Usage](#usage)
- [Models & Results](#models--results)
- [Technical Details](#technical-details)
- [Potential Improvements](#potential-improvements)
- [License](#license)
- [Citation](#citation)

---

## Overview

This project demonstrates a complete data science lifecycle on the California Housing dataset. It moves beyond static model training by incorporating **continuous data profiling** — a framework for monitoring data and concept drift over time, simulating how a model would behave in a production environment.

The pipeline covers:

- **Data Profiling** — Automated statistical summaries (shape, dtypes, missing values, describe)
- **Preprocessing** — Robust handling of missing values, feature standardization (without target leakage)
- **Feature Engineering** — Domain-specific ratio features
- **Regression** — Random Forest and Linear Regression for price prediction
- **Classification** — Random Forest for price category prediction (Low / Medium / High)
- **Drift Monitoring** — KS-test for data drift; R²-based concept drift detection
- **Interactive Dashboard** — `ipywidgets`-based UI for data exploration and model insights

The work is accompanied by a research paper that discusses the methodology and implications of continuous data profiling in interactive data science environments.

---

## Research Paper

The file `Dead or Alive Continuous Data Profiling for Interactive Data Science.pdf` contains the accompanying research paper, which provides the theoretical background, experimental design, and discussion of results.

---

## Features

- **Continuous Data Profiler** — A reusable `ContinuousDataProfiler` class that tracks changes across dataset versions
- **Data Leakage Prevention** — All transformations (encoding, imputation) are fit on training data only
- **Drift-Aware Monitoring** — `ModelMonitor` class with KS-test data drift and R² concept drift detection
- **Multi-Model Support** — Random Forest (regressor + classifier) and Linear Regression with isolated variable scopes
- **Interactive Dashboard** — Three-tab `ipywidgets` interface: Data Explorer, Model Insights, Data Profiles
- **Colab-Compatible** — Automatically detects local vs. Google Colab environment

---

## Project Structure

```
.
├── Continuous-Data-Profiling-For-Interactive-Data-Science.ipynb   # Main notebook (13 cells)
├── housing.csv                                                     # California Housing dataset
├── Dead or Alive Continuous Data Profiling for Interactive Data Science.pdf  # Research paper
└── README.md                                                       # Project documentation
```

---

## Dataset

The [California Housing Prices dataset](https://www.kaggle.com/datasets/camnugent/california-housing-prices) contains 20,640 observations of census block groups from the 1990 California census.

| Column | Description |
|--------|-------------|
| `longitude` / `latitude` | Geographic coordinates |
| `housing_median_age` | Median age of houses in the block |
| `total_rooms` / `total_bedrooms` | Room counts |
| `population` / `households` | Demographic counts |
| `median_income` | Median income in the block (in tens of thousands) |
| `ocean_proximity` | Categorical: distance to the ocean |
| `median_house_value` | **Target** — Median house value ($) |

Additional engineered features:
- `RoomsPerHousehold` — `total_rooms / households`
- `BedroomsPerRoom` — `total_bedrooms / total_rooms`
- `PopulationPerHousehold` — `population / households`

---

## Installation

### Prerequisites

- Python 3.8+
- Jupyter Notebook or JupyterLab

### Setup

```bash
# Clone the repository
git clone https://github.com/your-username/Continuous-Data-Profiling-For-Interactive-Data-Science.git
cd Continuous-Data-Profiling-For-Interactive-Data-Science

# Install dependencies (recommended: use a virtual environment)
pip install --upgrade pip
pip install numpy scipy scikit-learn ipywidgets matplotlib seaborn

# Launch the notebook
jupyter notebook Continuous-Data-Profiling-For-Interactive-Data-Science.ipynb
```

> **Note:** The first cell handles package installation. In a Colab environment, simply run the cells in order. Locally, ensure your environment has the required packages pre-installed.

---

## Usage

1. **Open the notebook** — Launch `Continuous-Data-Profiling-For-Interactive-Data-Science.ipynb`
2. **Run cells sequentially** — The notebook is organized into 13 cells, each with a clear purpose:
    - Cell 1: Package installation (skip if pre-installed)
    - Cell 2: Library imports
    - Cell 3: Data loading (local or Google Drive)
    - Cell 4: Continuous Data Profiler initialization
    - Cell 5: Preprocessing (median fill, feature standardization, target unscaled)
    - Cell 6: Feature engineering (ratio features)
    - Cell 7: Debug data flow check
    - Cell 8: Random Forest regression with feature importance
    - Cell 9: Model monitoring (data + concept drift)
    - Cell 10: Classification (Low / Medium / High)
    - Cell 11: Linear Regression baseline
    - Cell 12: Residual analysis
    - Cell 13: Interactive dashboard
3. **Explore the dashboard** — The final cell renders a tabbed UI for interactive exploration

---

## Models & Results

### Regression — Random Forest

| Metric | Value |
|--------|-------|
| RMSE   | ~$49,500 |
| R²     | ~0.81 |

### Regression — Linear Regression

| Metric | Value |
|--------|-------|
| RMSE   | ~$70,000 |
| R²     | ~0.63 |

### Classification — Random Forest

| Metric | Value |
|--------|-------|
| Accuracy | ~0.83 |
| F1 (macro) | ~0.82 |
| F2 (macro) | ~0.81 |

*Results may vary slightly depending on random splits and library versions.*

---

## Technical Details

### Pipeline Architecture

```
┌─────────────┐    ┌──────────────┐    ┌─────────────────┐
│ Raw Data    │───▶│ Preprocess   │───▶│ Feature Eng.    │
│ (housing.csv)│   │ - median fill│    │ - ratio feats  │
│             │    │ - standardize│    │                 │
└─────────────┘    │ (target kept │    └────────┬────────┘
                   │  unscaled)   │             │
                   └──────────────┘             ▼
                                        ┌─────────────────┐
┌────────────────┐    ┌──────────────┐    │ Train/Test     │
│ Interactive    │◀───│ Drift Monitor│◀───│ Split +        │
│ Dashboard      │    │ (KS-test,    │    │ One-Hot Encode │
│                │    │  R² check)   │    │ (no leakage)   │
└────────────────┘    └──────────────┘    └────────┬────────┘
                                                   ▼
                                        ┌─────────────────┐
                                        │ Models          │
                                        │ - RF Regressor  │
                                        │ - Linear Reg.   │
                                        │ - RF Classifier │
                                        └─────────────────┘
```

### Key Design Decisions

- **No target scaling** — Feature standardization excludes the target variable to maintain interpretability
- **Post-split encoding** — One-hot encoding applied after `train_test_split` to prevent data leakage
- **Isolated model scopes** — Each model uses unique variable suffixes (`_rf`, `_lr`, `_cls`) to avoid namespace collisions
- **Test-set baseline for drift** — Concept drift compares against held-out test R² rather than optimistic training R²
- **Fallback data loading** — Attempts local file first, then Google Drive mount for Colab users

---

## Potential Improvements

- Hyperparameter tuning (GridSearchCV / Optuna)
- Advanced models (XGBoost, CatBoost, LightGBM)
- Statistical parity tests for fairness evaluation
- Web deployment (Streamlit / Flask / FastAPI)
- Automated CI/CD pipeline for model retraining
- Real-time drift monitoring with alerting
- SHAP / LIME explanations in the dashboard

---

## License

This project is provided for academic and educational purposes.

---

## Citation

If you use this work in your research, please cite the accompanying paper:

```
@misc{continuous-data-profiling,
  author = {Koushik},
  title = {Dead or Alive: Continuous Data Profiling for Interactive Data Science},
  year = {2024},
  note = {Available as part of the project repository}
}
```

---

*Built with Python, scikit-learn, and Jupyter.*
