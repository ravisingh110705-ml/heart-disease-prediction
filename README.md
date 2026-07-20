# Heart Disease Prediction

A binary classification project comparing 7 machine learning models to predict the presence of heart disease from patient health data.

## Problem

Given a patient's clinical measurements, predict whether they have heart disease (binary: 0 = No, 1 = Yes).

## Dataset

918 patient records with features including:
- `Age`, `Sex`, `ChestPainType`, `RestingBP`, `Cholesterol`, `FastingBS`, `RestingECG`, `MaxHR`, `ExerciseAngina`, `Oldpeak`, `ST_Slope`
- Target: `HeartDisease`

## Data Cleaning

- Identified `Cholesterol` and `RestingBP` values of 0 as data errors (physiologically implausible) rather than genuine measurements, and imputed them with the median.
- Encoded binary categorical features (`Sex`, `ExerciseAngina`) and one-hot encoded multi-class categoricals (`ChestPainType`, `RestingECG`, `ST_Slope`) with `drop_first=True` to avoid the dummy variable trap.
- Scaled features using `StandardScaler`, fit only on the training set.

## Models Compared

| Model | Accuracy | Precision | Recall | F1 Score |
|---|---|---|---|---|
| Logistic Regression | 0.859 | 0.909 | 0.841 | 0.874 |
| Decision Tree (max_depth=4, tuned) | 0.826 | 0.895 | 0.794 | 0.842 |
| **Random Forest** | **0.864** | **0.887** | **0.879** | **0.883** |
| K-Nearest Neighbors | 0.848 | 0.891 | 0.841 | 0.865 |
| SVM (SVC) | 0.859 | 0.893 | 0.860 | 0.876 |
| Naive Bayes | 0.848 | 0.916 | 0.813 | 0.861 |
| XGBoost | 0.842 | 0.875 | 0.850 | 0.863 |

Random Forest achieved the best overall balance of accuracy, recall, and F1 score, and was selected as the final model. Decision Tree depth was tuned by comparing train/test accuracy across depths 3–6 to control overfitting before final evaluation.

## Tech Stack

Python, Pandas, NumPy, Scikit-learn, XGBoost

## Files

- `main.ipynb` — full notebook: data cleaning, encoding, model training, and evaluation
- `data/` — dataset used for training and testing

## Author

Ravi Singh Bungla — [GitHub](https://github.com/ravisingh110705-ml) | [LinkedIn](your-linkedin-url-here)

## Other Projects
- [Titanic Survival Prediction](https://github.com/ravisingh110705-ml/titanic-survival-prediction)
- [Medical Insurance Cost Prediction](https://github.com/ravisingh110705-ml/medical-insurance-cost-prediction)
- [Telco Customer Churn Prediction](https://github.com/ravisingh110705-ml/telco-churn-prediction)
