<div align="center">

# Continuous Data Profiling for Interactive Data Science

**Explore California housing data, train prediction models, and monitor data drift in one interactive notebook.**

![Python](https://img.shields.io/badge/Python-Jupyter-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/ML-scikit--learn-F7931E?logo=scikitlearn&logoColor=white)

[Run the notebook](#quick-start) · [Workflow](#workflow) · [Results](#results) · [Research](#research-paper)

</div>

A Jupyter-based machine-learning experiment on the **California Housing Prices** dataset. It combines exploratory profiling, feature engineering, regression and classification, distribution-shift checks, and an interactive `ipywidgets` interface.

## Highlights

- Profile data quality, statistics, missing values, and distributions.
- Engineer housing-density ratios and preprocess data for model training.
- Compare **Random Forest** and **Linear Regression** for median house-value prediction.
- Classify properties by value range with **Random Forest**.
- Monitor feature drift with the **Kolmogorov–Smirnov test** and changes in model performance.
- Explore data and model outputs through an `ipywidgets` dashboard.

> [!NOTE]
> This is an educational notebook that simulates monitoring on a fixed dataset. It is not a deployed continuous-monitoring service or a live housing-price prediction API.

## Quick start

Requirements: Python, Jupyter Notebook or JupyterLab, and the packages below.

~~~bash
git clone https://github.com/KOUSHIK1831/Continuous-Data-Profiling-For-Interactive-Data-Science.git
cd Continuous-Data-Profiling-For-Interactive-Data-Science
python -m pip install pandas numpy scipy scikit-learn matplotlib seaborn ipywidgets notebook
jupyter notebook Continuous-Data-Profiling-For-Interactive-Data-Science.ipynb
~~~

Run the notebook's cells in order. You can also upload the notebook and `housing.csv` to Google Colab; adjust file paths to match your Colab session.

## Workflow

```text
housing.csv
    |
    v
Data profiling and visualization
    |
    v
Feature engineering and preprocessing
    |
    +--> Regression (Random Forest / Linear Regression)
    |
    +--> Classification (Random Forest)
    |
    v
Drift and model-performance monitoring
    |
    v
Interactive notebook dashboard
```

The dataset includes geographical coordinates, housing age, room and bedroom counts, population, households, median income, ocean proximity, and the `median_house_value` target. The notebook adds features such as rooms per household, bedrooms per room, and population per household.

## Results

The existing notebook documentation reports the following **approximate experimental results**:

| Task | Model | Reported metric |
| --- | --- | --- |
| Price regression | Random Forest | R² ≈ 0.81 |
| Price regression | Linear Regression | R² ≈ 0.63 |
| Price-category classification | Random Forest | Accuracy ≈ 0.83 |

These figures are illustrative, not a live benchmark; rerun the notebook to reproduce metrics in your own environment. Results can vary by split and package version.

## Repository files

| File | Purpose |
| --- | --- |
| [Notebook](Continuous-Data-Profiling-For-Interactive-Data-Science.ipynb) | Experiments, models, drift checks, interactive views |
| [Dataset](housing.csv) | Input California housing records |
| [Research paper](Dead%20or%20Alive%20Continuous%20Data%20Profiling%20for%20Interactive%20Data%20Science.pdf) | Accompanying research background and discussion |

## Research paper

The included **Dead or Alive: Continuous Data Profiling for Interactive Data Science** paper discusses the methodology and motivation behind combining model evaluation with ongoing data-quality analysis.

> [!TIP]
> Start with the notebook's profiling and baseline model cells before exploring drift and the dashboard. The outputs depend on earlier cells having run successfully.
