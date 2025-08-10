# Salary Classification Analysis

This repository contains an end-to-end machine learning pipeline for predicting whether a person's salary is greater than or less than $50K/year based on census data. The workflow includes data preprocessing, handling imbalanced data, hyperparameter tuning, model training, evaluation, and feature importance analysis.

## Features
- Automatic detection of numeric and categorical features
- Data preprocessing with missing value imputation, scaling, and one-hot encoding
- Handling imbalanced datasets using SMOTE
- Random Forest Classifier with GridSearchCV for hyperparameter optimization
- Model evaluation using ROC AUC, Precision, Recall, F1-score, and Confusion Matrix
- Feature importance extraction for model interpretability
- Model saving with Joblib

## Model Evaluation Results
After training, the model achieved the following metrics:

### ROC AUC:
```
0.9155278858257154
```

### Classification Report:
```
              precision    recall  f1-score   support

       <=50K       0.93      0.84      0.89      4945
        >50K       0.62      0.81      0.71      1568

    accuracy                           0.84      6513
   macro avg       0.78      0.83      0.80      6513
weighted avg       0.86      0.84      0.84      6513
```

### Confusion Matrix:
```
[[4178  767]
 [ 294 1274]]
```

### Top 20 Most Important Features:
```
marital-status_Married-civ-spouse    0.118900
age                                  0.103886
relationship_Husband                 0.088019
education-num                        0.081547
hours-per-week                       0.071009
marital-status_Never-married         0.062609
capital-gain                         0.059773
fnlwgt                               0.029012
relationship_Not-in-family           0.025153
occupation_Exec-managerial           0.024760
relationship_Own-child               0.024271
sex_Male                             0.023476
relationship_Wife                    0.022735
occupation_Prof-specialty            0.021107
sex_Female                           0.016896
occupation_Other-service             0.016489
education_Bachelors                  0.015644
marital-status_Divorced              0.013738
capital-loss                         0.013647
education_HS-grad                    0.012350
```

