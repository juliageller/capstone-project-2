# Predicting Employee Attrition Using Machine Learning

**Author:** Katherine Geller
**Date:** May 2026
**Course:** Data Science Capstone (Project 2)

---

## BLUF (Bottom Line Up Front)

Employee turnover is expensive. When someone leaves, the company has to spend money recruiting, hiring, and training a replacement. For HR teams, the challenge is figuring out **who is at risk of leaving BEFORE they actually leave**, so they can step in with retention strategies.

This project builds a machine learning model that predicts whether an employee is likely to leave, using the IBM HR Analytics Employee Attrition dataset (1,470 employees, 35 features).

**Final model performance:**

| Metric    | Value |
| --------- | ----- |
| ROC-AUC   | 0.80  |
| Recall    | 0.68  |
| Precision | 0.36  |
| F1 Score  | 0.47  |
| Accuracy  | 0.76  |

**Best model:** Logistic Regression (after hyperparameter tuning with GridSearchCV)

**Top drivers of attrition (in order):**
1. OverTime (working overtime is the #1 risk factor)
2. BusinessTravel (frequent travelers leave more)
3. JobRole (Lab Technicians, Sales Reps)
4. MaritalStatus (Single)
5. YearsSinceLastPromotion

---

## Project Overview

This project follows the **CRISP-DM** framework:

1. **Business Understanding** - Define the problem and stakeholders
2. **Data Understanding** - EDA, distributions, correlations
3. **Data Preparation** - Drop useless columns, encode categoricals, scale numerics
4. **Modeling** - Compare Logistic Regression, Decision Tree, Random Forest
5. **Evaluation** - Cross-validation, hyperparameter tuning, test set evaluation
6. **Deployment** - Save the full pipeline as a pickle file

---


---

## Dataset

**Source:** [IBM HR Analytics Employee Attrition & Performance (Kaggle)](https://www.kaggle.com/datasets/pavansubhasht/ibm-hr-analytics-attrition-dataset)

- 1,470 employee records
- 35 features covering demographics, job details, compensation, experience, and work patterns
- Target: `Attrition` (Yes / No, converted to binary 1 / 0)
- No missing values
- Class imbalance: ~16% Yes (left), ~84% No (stayed)

### Data Usage Guide

To use the dataset:

1. The CSV file is included in the `data/` folder. No download needed.
2. To run the analysis, open `hr_attrition_analysis.ipynb` in Jupyter.
3. To use the trained pipeline on new data, load the pickle file:

```python
import pickle
import pandas as pd

# Load the trained pipeline
with open("models/final_attrition_pipeline.pkl", "rb") as f:
    pipeline = pickle.load(f)

# New employee data must have the same columns as the training data
# (excluding the columns dropped in preprocessing: EmployeeCount, Over18,
# StandardHours, EmployeeNumber, Attrition)
new_data = pd.read_csv("path/to/new_employees.csv")

# Get predictions
predictions = pipeline.predict(new_data)
probabilities = pipeline.predict_proba(new_data)[:, 1]
```



## Key Findings

### What predicts attrition?

The model identified some clear patterns:

1. **OverTime is the biggest red flag.** Employees who work overtime are way more likely to leave than those who don't. The coefficient is roughly 3x larger than the next biggest factor.

2. **Frequent business travelers leave more.** Employees who travel frequently for work have higher attrition than those who travel rarely or not at all.

3. **Some job roles have much higher turnover.** Sales Representatives and Laboratory Technicians have notably higher attrition rates.

4. **Single employees leave more than married ones.** Marital status is a meaningful predictor.

5. **Time since last promotion matters.** Employees who haven't been promoted in a while are at higher risk.

### Recommendations for HR

- **Investigate overtime culture.** Why are people working overtime? Are teams understaffed? Are workloads unrealistic?
- **Review compensation and travel demands** for high-attrition roles like Sales Reps and Lab Techs.
- **Build a promotion review cadence.** Long gaps without promotion are a leading indicator. Make sure promotion conversations happen at least annually.
- **Use the model in practice.** Run this model quarterly on the current employee population. Flag the top 20% riskiest employees for proactive retention check-ins with their managers.

---

## Limitations

- **Synthetic data:** The dataset was created by IBM data scientists, so real-world performance may differ.
- **Small dataset:** With only 1,470 employees, more data would improve reliability.
- **Class imbalance:** Only 16% of employees in the data left. I used `class_weight='balanced'` to handle this, but more advanced techniques (like SMOTE) could be explored.
- **Limited algorithms:** I only compared Logistic Regression, Decision Tree, and Random Forest. More advanced models (like gradient boosting or neural networks) might do better.

---

## Tech Stack

- Python 3.11
- pandas, numpy
- scikit-learn (Pipeline, ColumnTransformer, GridSearchCV)
- matplotlib, seaborn
- Jupyter Notebook

---
