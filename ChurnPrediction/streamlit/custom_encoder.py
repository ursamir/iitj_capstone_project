# custom_encoder.py
from sklearn.base import BaseEstimator, TransformerMixin
import pandas as pd
import numpy as np

# Define mappings for categorical variables
mappings = {
    'Gender': {'Male': 0, 'Female': 1, 'Non-Binary': 2},
    'Location': {'Suburb': 0, 'Rural': 1, 'City': 2},
    'ProductCategory': {'Clothing': 0, 'Home Appliances': 1, 'Electronics': 2, 'Toys': 3, 'Furniture': 4},
    'PaymentMethod': {'Debit Card': 0, 'Cash': 1, 'Credit Card': 2, 'Digital Wallet': 3, 'Bank Transfer': 4},
    'CustomerSegment': {'New': 0, 'Returning': 1, 'Loyal': 2, 'VIP': 3},
    'Region': {'South': 0, 'East': 1, 'West': 2, 'North': 3, 'Central': 4},
    'Season': {'Spring': 0, 'Autumn': 1, 'Winter': 2, 'Summer': 3},
    'CustomerType': {'Individual': 0, 'Business': 1},
    'PurchaseChannel': {'In-Store': 0, 'Online': 1, 'Mobile': 2},
    'HolidayPeriod': {False: 0, True: 1}
}

# Custom transformer to encode categorical columns based on predefined mappings
class CustomLabelEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, mappings):
        self.mappings = mappings

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for col, mapping in self.mappings.items():
            if col in X.columns:
                X[col] = X[col].map(mapping).fillna(X[col])
        return X
# Function to fill missing values using random sampling
def fill_with_random_sampling(series):
    non_missing = series.dropna()
    random_sample = np.random.choice(non_missing, size=series.isnull().sum(), replace=True)
    random_sample_series = pd.Series(random_sample, index=series.index[series.isnull()])
    series_filled = series.combine_first(random_sample_series)
    return series_filled


# Custom transformer for random sampling imputation
class RandomImputer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        for col in self.columns:
            X[col] = fill_with_random_sampling(X[col])
        return X