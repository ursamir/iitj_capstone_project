# Sales Data Analysis and Customer Churn Prediction
## Overview
This project involves analyzing a sales dataset, performing data cleaning and preprocessing, aggregating the data, handling missing values, and building machine learning models to predict customer churn. The following steps are covered in the script:

1. Loading the Data: Reading the sales dataset from a CSV file.
2. Data Cleaning: Dropping unnecessary columns and encoding categorical variables.
3. Data Aggregation: Aggregating the data by CustomerID.
4. Handling Missing Values: Imputing missing values using KNN imputation and random sampling.
5. Comparing Data Distributions: Comparing the distributions before and after imputation.
6. Customer Churn Prediction: Building machine learning models to predict customer churn.
## Dependencies
The following Python libraries are required to run the script:

- pandas
- numpy
- sklearn
- matplotlib
- seaborn
- joblib

You can install the required libraries using the following command:

```
pip install pandas numpy scikit-learn matplotlib seaborn joblib
```

## Instructions
### Step 1: Load the Data
The dataset is loaded using the pandas library.

```
import pandas as pd

# Load the dataset
file_path = 'data\\Sales dataset.csv'
df = pd.read_csv(file_path)
print(df.head())
df.info()
df.drop('Unnamed: 0', inplace=True, axis=1)
df.info()
df.describe()
df.nunique()
```

### Step 2: Encode Categorical Variables
Categorical variables are encoded using predefined mappings.

```
mappings = {
    'Gender': {'Male': 0, 'Female': 1, 'Non-Binary': 2},
    # Add other mappings here
}

def encode_column(series, mapping):
    return series.map(mapping).fillna(series)

def custom_read_csv(filepath):
    df = pd.read_csv(filepath)
    for col, mapping in mappings.items():
        if col in df.columns:
            df[col] = encode_column(df[col], mapping)
    return df

new_df = custom_read_csv(file_path)
new_df.drop('Unnamed: 0', inplace=True, axis=1)
new_df['PurchaseDate'] = pd.to_datetime(new_df['PurchaseDate'], errors='coerce', format='%Y-%m-%d %H:%M:%S')
new_df.head()
new_df.sample(10)
new_df.info()
```

### Step 3: Aggregate Data
Data is aggregated by CustomerID with various aggregation functions.

```
def safe_mode(series):
    mode_value = series.mode()
    return mode_value[0] if not mode_value.empty else None

agg_funcs = {
    'Age': 'max',
    'Gender': 'last',
    # Add other aggregation functions here
}

new_df['TransactionCount'] = 1
agg_df = new_df.groupby('CustomerID').agg(agg_funcs).reset_index()
agg_df.head()
agg_df.info()
agg_df.describe()
```

### Step 4: Handle Missing Values
Missing values are imputed using KNN imputation and random sampling.

```
from sklearn.impute import KNNImputer

numerical_cols = ['LoyaltyScore', 'PurchaseFrequency', 'CustomerLifetimeValue', 'Season',
                  'CustomerType', 'PurchaseChannel', 'SeasonalDiscount', 'HolidayPeriod',
                  'CustomerSatisfactionScore']

impute_df = agg_df.copy()
imputer = KNNImputer()
imputed_data = imputer.fit_transform(impute_df[numerical_cols])
impute_df[numerical_cols] = imputed_data
print(impute_df[numerical_cols].isnull().sum())
impute_df.describe()
impute_df.info()
```

### Step 5: Compare Data Distributions
Distributions are compared before and after imputation.

```
import matplotlib.pyplot as plt
import seaborn as sns

def compare_distributions(df1, df2, cols, categorical=False):
    for col in cols:
        plt.figure(figsize=(12, 6))
        if categorical:
            plt.subplot(1, 2, 1)
            sns.countplot(df1[col])
            plt.title(f'Original {col}')
            plt.subplot(1, 2, 2)
            sns.countplot(df2[col])
            plt.title(f'Imputed {col}')
        else:
            plt.subplot(1, 2, 1)
            sns.histplot(df1[col], kde=True)
            plt.title(f'Original {col}')
            plt.subplot(1, 2, 2)
            sns.histplot(df2[col], kde=True)
            plt.title(f'Imputed {col}')
        plt.show()

compare_distributions(agg_df, impute_df, numerical_cols, categorical=False)
```

### Step 6: Machine Learning Model for Predicting Customer Churn
Logistic Regression and Random Forest models are built to predict customer churn.

```
from sklearn.model_selection import train_test_split

churn_threshold = pd.to_datetime(impute_df['PurchaseDate'].max()) - pd.DateOffset(months=12)
impute_df['Churn'] = (impute_df['PurchaseDate'] < churn_threshold).astype(int)
impute_df['PurchaseDate'] = impute_df['PurchaseDate'].astype(int) / 10**9

X = impute_df.drop(columns=['CustomerID', 'Churn'])
y = impute_df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

log_reg = LogisticRegression(random_state=42)
rf = RandomForestClassifier(random_state=42)

log_reg_param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'solver': ['liblinear', 'lbfgs']}
rf_param_grid = {'n_estimators': [100, 200, 300], 'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]}

log_reg_grid_search = GridSearchCV(log_reg, log_reg_param_grid, cv=5, scoring='accuracy', verbose=1)
log_reg_grid_search.fit(X_train, y_train)

rf_grid_search = GridSearchCV(rf, rf_param_grid, cv=5, scoring='accuracy', verbose=1)
rf_grid_search.fit(X_train, y_train)

log_reg_best = log_reg_grid_search.best_estimator()
rf_best = rf_grid_search.best_estimator()

log_reg_y_pred = log_reg_best.predict(X_test)
rf_y_pred = rf_best.predict(X_test)

print("Logistic Regression:")
print(classification_report(y_test, log_reg_y_pred))

print("Random Forest:")
print(classification_report(y_test, rf_y_pred))

from joblib import dump
model_filename = 'model/logistic_regression_model.pkl'
dump(log_reg_best, model_filename)
```

# License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
