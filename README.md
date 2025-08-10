# Salary Classification Analysis

This project predicts whether a person's salary is `<=50K` or `>50K` based on demographic and work-related features from the [Adult Income dataset](https://archive.ics.uci.edu/ml/datasets/adult).

## Features
- Data preprocessing with `pandas` and `scikit-learn`
- Handling missing values
- Encoding categorical variables with OneHotEncoder
- Scaling numerical features
- Balancing dataset using **SMOTE** (Synthetic Minority Over-sampling Technique)
- Model training using **RandomForestClassifier**
- Hyperparameter tuning with **GridSearchCV**
- Model evaluation using **ROC AUC**, precision, recall, F1-score
- Feature importance analysis
- Saving the trained model with `joblib`

## Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/yourusername/salary-classification.git
cd salary-classification
pip install -r requirements.txt
