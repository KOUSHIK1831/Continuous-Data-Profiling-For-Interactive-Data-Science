# Continuous-Data-Profiling-For-Interactive-Data-Science

# California Housing Price Analysis

This project presents a complete end-to-end machine learning pipeline for analyzing housing prices in California. It includes data profiling, preprocessing, feature engineering, model training (regression and classification), drift detection, and an interactive dashboard.

## Project Structure

```
California_Housing_Project/
│
├── Special_Project.ipynb           # Main Jupyter Notebook
├── housing.csv                     # Dataset (from Google Drive)
├── README.md                       # Project documentation
```

## Objective

To predict California housing prices and classify them into categories based on various features such as location, demographics, and property characteristics. The project also integrates monitoring and interactive tools for data exploration and model evaluation.

## Models Used

**Regression**

* Random Forest Regressor
* Linear Regression

**Classification**

* Random Forest Classifier (to classify housing prices into Low, Medium, High)

## Features Used

* housing\_median\_age
* total\_rooms
* total\_bedrooms
* population
* households
* median\_income
* ocean\_proximity

Additional engineered features:

* RoomsPerHousehold
* BedroomsPerRoom
* PopulationPerHousehold

## Workflow Overview

1. **Data Profiling**

   * Using `ydata-profiling` to generate statistical reports

2. **Preprocessing**

   * Handling missing values
   * Normalizing numerical features
   * Encoding categorical variables

3. **Feature Engineering**

   * Derived metrics from existing features to improve model performance

4. **Model Training & Evaluation**

   * Regression (housing value prediction)
   * Classification (price category prediction)

5. **Drift Monitoring**

   * Data Drift: using KS-test
   * Concept Drift: comparing baseline and current model R²

6. **Interactive Dashboard**

   * Built with `ipywidgets` for feature exploration and model insights

## Evaluation Metrics

### Regression

| Metric   | Value (Approx.) |
| -------- | --------------- |
| RMSE     | 35,000 – 45,000 |
| MAE      | 25,000 – 32,000 |
| R² Score | 0.78 – 0.85     |

### Classification

| Metric   | Value (Approx.) |
| -------- | --------------- |
| Accuracy | \~0.85          |
| F1 Score | \~0.83          |
| F2 Score | \~0.81          |

*Values may vary based on data splits and model tuning.*

## Possible Improvements

* Hyperparameter tuning (GridSearchCV)
* Try advanced models like XGBoost or CatBoost
* Deploy as a web app using Streamlit or Flask
* Automate data monitoring for production usage

## Requirements

Install the required libraries using the following:

```bash
pip install numpy==1.24.4
pip install pandas scikit-learn matplotlib seaborn
pip install ydata-profiling ipywidgets
```

If using Google Colab, also run:

```python
from google.colab import drive
drive.mount('/content/drive')
```

## Running the Notebook

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/California_Housing_Project.git
   cd California_Housing_Project
   ```

2. Launch the notebook:

   ```bash
   jupyter notebook Special_Project.ipynb
   ```



This repository contains the code, research paper for the Project

Kaggle Dataset: https://www.kaggle.com/datasets/camnugent/california-housing-prices
