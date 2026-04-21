# Employee Attrition Analysis & Prediction

Final Assignment — Global Consumer Intelligence (GCI) Program, University of Tokyo (2025)

---

## Overview

This project presents a full end-to-end data analysis and machine learning pipeline built on an HR employee dataset. The objective was to identify the key drivers of employee attrition and build a predictive model to support data-driven business decisions around employee retention.

The work covers data cleaning, exploratory data analysis (EDA), rich visualisation across 13 dimensions, and a tuned LightGBM classification model — culminating in a business proposal with quantified impact.

---

## Dataset

| Property | Value |
|---|---|
| Records | 1,470 employees |
| Features | 44 columns |
| Target | `Attrition` (Yes / No) |
| Null values | None |
| Source | Provided by GCI program |

**Feature categories covered:**
- Demographics: Age, Gender, MaritalStatus, Education, EducationField
- Job attributes: JobRole, JobLevel, Department, BusinessTravel, OverTime
- Satisfaction metrics: JobSatisfaction, EnvironmentSatisfaction, JobInvolvement, WorkLifeBalance, RelationshipSatisfaction
- Compensation: MonthlyIncome, StockOptionLevel, Incentive
- Tenure: YearsAtCompany, YearsInCurrentRole, YearsSinceLastPromotion, TotalWorkingYears
- Wellness & flexibility: RemoteWork, FlexibleWork, WelfareBenefits, InHouseFacility, ExternalFacility, ExtendedLeave, StressRating, StressSelfReported

---

## Project Structure

```
FinalAssignment/
├── FinalAssignment.ipynb   # Main notebook
├── final.md                # This file
└── dataset/
    └── data.csv            # Employee dataset
```

---

## Tech Stack

| Tool | Purpose |
|---|---|
| Python 3 | Core language |
| pandas | Data loading, cleaning, manipulation |
| NumPy | Numerical operations |
| Matplotlib / Seaborn | Data visualisation |
| scikit-learn | Preprocessing, train/test split, metrics, hyperparameter search |
| LightGBM | Gradient boosting classifier |
| SciPy | Statistical utilities, search distributions |
| Google Colab | Execution environment |

---

## Methodology

### 1. Data Cleaning

- Detected **28 duplicate EmployeeNumber** entries
- Resolved by dropping the original column and regenerating a clean sequential ID
- Confirmed zero null values across all 44 columns

### 2. Exploratory Data Analysis

13 structured visualisations examining attrition rates across:

1. Overall attrition distribution (count & percentage)
2. Age (KDE density plot with mean markers)
3. Gender
4. Marital status
5. Education level
6. Education field
7. Overtime status
8. Work-life balance
9. Job involvement
10. Environment satisfaction
11. Job satisfaction
12. Monthly income (median bar + KDE)
13. Job role
14. Business travel frequency
15. Welfare benefits
16. How employees were hired (HowToEmploy)

### 3. Model Training

**Algorithm:** LightGBM (gradient boosting decision tree)

**Data split:** 60% train / 20% validation / 20% test (stratified)

**Preprocessing:**
- Label encoding of all categorical features
- Dropped constant or redundant columns: `EmployeeCount`, `StandardHours`, `Over18`
- Class imbalance addressed via `is_unbalance=True` and threshold optimisation

**Hyperparameter tuning:** RandomizedSearchCV — 30 iterations, 3-fold cross-validation, scored on F1

**Best parameters found:**

| Parameter | Value |
|---|---|
| num_leaves | 70 |
| max_depth | 10 |
| learning_rate | 0.0114 |
| n_estimators | 517 |
| subsample | 0.667 |
| colsample_bytree | 0.944 |
| reg_alpha | 0.269 |
| reg_lambda | 0.244 |
| min_child_samples | 31 |

---

## Results

| Metric | Value |
|---|---|
| AUC (ROC) | **0.834** |
| Accuracy | **88%** |
| F1 Score (default threshold) | 0.486 |
| F1 Score (optimised threshold) | **0.586** |
| Optimised threshold | 0.225 |

**Optimised classification report (test set):**

| Class | Precision | Recall | F1 |
|---|---|---|---|
| No attrition (0) | 0.93 | 0.91 | 0.92 |
| Attrition (1) | 0.56 | 0.62 | 0.59 |
| **Overall accuracy** | | | **86%** |

Threshold optimisation on the validation set improved minority-class recall from 36% to 62%, making the model significantly more useful for proactively flagging at-risk employees.

---

## Feature Importance

The top 25 most predictive features were extracted from the trained model and visualised using a horizontal bar chart. Key drivers identified include monthly income, overtime, job level, years at company, and satisfaction scores.

---

## How to Run

1. Open `FinalAssignment.ipynb` in Google Colab
2. Mount Google Drive and place the `dataset/data.csv` file in the working directory
3. Run all cells in order

**Dependencies** (install if running locally):
```bash
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm scipy
```

---

## References

1. VanderPlas, J. (2016). *Python Data Science Handbook*. O'Reilly Media.
2. Géron, A. (2019). *Hands-On Machine Learning with Scikit-Learn, Keras and TensorFlow* (2nd ed.). O'Reilly Media.
3. James, G., et al. (2021). *An Introduction to Statistical Learning* (2nd ed.). Springer.
4. Ke, G., et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree. *NIPS 2017*.
5. Pedregosa, F., et al. (2011). Scikit-learn: Machine Learning in Python. *JMLR*, 12, 2825–2830.
6. Raschka, S. (2018). Model Evaluation, Model Selection, and Algorithm Selection in Machine Learning. arXiv:1811.12808.
7. Sokolova, M., & Lapalme, G. (2009). A systematic analysis of performance measures for classification tasks. *Information Processing & Management*, 45(4), 427–437.
8. [Kaggle reference](https://www.kaggle.com/code/ayushidhariwal/employee-attrition-analysis)
9. [Towards Data Science reference](https://towardsdatascience.com/cracking-the-employee-attrition-problem-with-machine-learning-6ee751ec4aae/)
