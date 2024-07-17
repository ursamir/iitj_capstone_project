import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, TransformerMixin
from joblib import dump
from custom_encoder import mappings, CustomLabelEncoder, RandomImputer

# Load and preprocess data
def load_and_preprocess(filepath):
    df = pd.read_csv(filepath)
    df.drop('Unnamed: 0', inplace=True, axis=1)
    df.loc[:999, 'PurchaseDate'] = pd.to_datetime(df.loc[:999, 'PurchaseDate'], format='%Y-%m-%d')
    df.loc[1000:, 'PurchaseDate'] = pd.to_datetime(df.loc[1000:, 'PurchaseDate'], errors='coerce')
    df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'], errors='coerce', format='%Y-%m-%d %H:%M:%S')
    df['TransactionCount'] = 1
    return df

# Aggregation functions for groupby
def safe_mode(series):
    mode_value = series.mode()
    return mode_value[0] if not mode_value.empty else None

agg_funcs = {
    'Age': 'max',
    'Gender': 'last',
    'Location': 'last',
    'ProductCategory': lambda x: x.mode().iloc[0] if not x.mode().empty else None,
    'PurchaseDate': 'last',
    'PurchaseAmount': 'sum',
    'PaymentMethod': 'last',
    'Quantity': 'sum',
    'DiscountPercentage': 'mean',
    'IsReturned': 'sum',
    'Rating': 'mean',
    'IsPromotion': 'sum',
    'CustomerSegment': safe_mode,
    'ShippingDuration': 'mean',
    'Region': 'last',
    'LoyaltyScore': 'last',
    'PurchaseFrequency': 'last',
    'CustomerLifetimeValue': 'last',
    'Season': safe_mode,
    'CustomerType': safe_mode,
    'PurchaseChannel': safe_mode,
    'SeasonalDiscount': 'mean',
    'HolidayPeriod': safe_mode,
    'CustomerSatisfactionScore': 'mean',
    'TransactionCount': 'sum'
}

# Define columns to impute with random sampling or KNN
columns_to_impute = {
    'LoyaltyScore': 'random',
    'PurchaseFrequency': 'random',
    'CustomerLifetimeValue': 'random',
    'Season': 'random',
    'CustomerType': 'random',
    'PurchaseChannel': 'random',
    'SeasonalDiscount': 'random',
    'HolidayPeriod': 'random',
    'CustomerSatisfactionScore': 'random'
}

# Create a pipeline for preprocessing and model training
def create_pipeline(mappings, agg_funcs, columns_to_impute):
    custom_label_encoder = CustomLabelEncoder(mappings)
    random_imputer = RandomImputer(columns_to_impute)

    numeric_features = ['Age', 'PurchaseAmount', 'Quantity', 'DiscountPercentage', 'Rating', 
                        'ShippingDuration', 'LoyaltyScore', 'PurchaseFrequency', 
                        'CustomerLifetimeValue', 'SeasonalDiscount', 'CustomerSatisfactionScore', 
                        'TransactionCount']
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='mean')),
        ('scaler', StandardScaler())])

    categorical_features = ['Gender', 'Location', 'ProductCategory', 'PaymentMethod', 
                            'CustomerSegment', 'Region', 'Season', 'CustomerType', 
                            'PurchaseChannel', 'HolidayPeriod']
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)])

    pipeline = Pipeline(steps=[
        ('custom_label_encoder', custom_label_encoder),
        ('random_imputer', random_imputer),
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(random_state=42))
    ])

    return pipeline

# Load data
file_path = 'ChurnPrediction\\data\\Sales dataset.csv'
df = load_and_preprocess(file_path)

# Aggregate data by CustomerID
agg_df = df.groupby('CustomerID').agg(agg_funcs).reset_index()

# Label churn in agg_df
churn_threshold = pd.to_datetime(agg_df['PurchaseDate'].max()) - pd.DateOffset(months=12)
agg_df['Churn'] = (agg_df['PurchaseDate'] < churn_threshold).astype(int)
agg_df['PurchaseDate'] = agg_df['PurchaseDate'].astype('int64') / 10**9


# Split data into training and testing sets
X = agg_df.drop(columns=['CustomerID', 'Churn'])
y = agg_df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Create pipeline
pipeline = create_pipeline(mappings, agg_funcs, columns_to_impute)

# Define parameter grid for GridSearchCV
param_grid = {
    'classifier__C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization parameter
    'classifier__solver': ['liblinear', 'lbfgs']
}

# Perform GridSearchCV
grid_search = GridSearchCV(pipeline, param_grid, cv=5, scoring='accuracy', verbose=1)
grid_search.fit(X_train, y_train)

# Get best model
best_model = grid_search.best_estimator_

# Evaluate best model on test set
y_pred = best_model.predict(X_test)
print("Logistic Regression:")
print(classification_report(y_test, y_pred))

# Save the pipeline
pipeline_filename = 'ChurnPrediction\\streamlit\\pipeline_with_model.pkl'
dump(best_model, pipeline_filename)
