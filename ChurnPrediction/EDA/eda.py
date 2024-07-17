# %% [markdown]
# Step 1: Load the Data

# %%
import pandas as pd

# Load the dataset
file_path = 'data\\Sales dataset.csv'
df = pd.read_csv(file_path)


# %%
print(df.head())

# %%
df.info()

# %%
df.drop('Unnamed: 0',inplace=True,axis=1)

# %%
df.info()

# %%
df.describe()

# %%
# find unique values in each column
df.nunique()

# %%
# list all unique values for Gender,Location,ProductCategory,PaymentMethod,CustomerSegment,Region,Season,CustomerType,PurchaseChannel,HolidayPeriod

# List of columns to find unique values for
columns_to_check = ['Gender', 'Location', 'ProductCategory', 'PaymentMethod', 
                    'CustomerSegment', 'Region', 'Season', 'CustomerType', 
                    'PurchaseChannel', 'HolidayPeriod']

# Dictionary to hold unique values for each column
unique_values = {col: df[col].unique() for col in columns_to_check}

# Print unique values for each column
for col, values in unique_values.items():
    print(f"Unique values in '{col}':")
    print(values)
    print("\n")


# %% [markdown]
# Read the csv again with encoded values.

# %%
import pandas as pd

# Define the mappings for each column
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

# Define a function to apply the mappings and handle NaN values
def encode_column(series, mapping):
    return series.map(mapping).fillna(series)

# Read the CSV file and apply the mappings
def custom_read_csv(filepath):
    df = pd.read_csv(filepath)
    
    for col, mapping in mappings.items():
        if col in df.columns:
            df[col] = encode_column(df[col], mapping)
    
    return df

# Usage
new_df = custom_read_csv(file_path)

# Drop the 'Unnamed: 0' column
new_df.drop('Unnamed: 0',inplace=True,axis=1)

# Convert first 1000 rows with date-only format
new_df.loc[:999, 'PurchaseDate'] = pd.to_datetime(new_df.loc[:999, 'PurchaseDate'], format='%Y-%m-%d')

# Convert from row 1000 onwards with timestamp format
new_df.loc[1000:, 'PurchaseDate'] = pd.to_datetime(new_df.loc[1000:, 'PurchaseDate'], errors='coerce')  # handle remaining rows

# Handle remaining rows with the correct format
new_df['PurchaseDate'] = pd.to_datetime(new_df['PurchaseDate'], errors='coerce', format='%Y-%m-%d %H:%M:%S')

# Verify the data
new_df.head()

# %%
new_df.sample(10)

# %%
new_df.info()

# %% [markdown]
# Step 2: Aggregate Data

# %%
# Define a safe mode function
def safe_mode(series):
    mode_value = series.mode()
    return mode_value[0] if not mode_value.empty else None

# Define aggregation functions
agg_funcs = {
    'Age': 'max',
    'Gender': 'last',  # Last gender value (assuming it doesn't change)
    'Location': 'last',  # Last location value (assuming it doesn't change)
    'ProductCategory': lambda x: x.mode().iloc[0] if not x.mode().empty else None,  # Most frequent product category
    'PurchaseDate': 'last',  # Last purchase date
    'PurchaseAmount': 'sum',  # Total purchase amount
    'PaymentMethod': 'last',  # Last payment method (assuming it doesn't change)
    'Quantity': 'sum',  # Total quantity
    'DiscountPercentage': 'mean',  # Average discount percentage
    'IsReturned': 'sum',  # Total returns
    'Rating': 'mean',  # Average rating
    'IsPromotion': 'sum',  # Total promotions
    'CustomerSegment': safe_mode,  # Most common customer segment
    'ShippingDuration': 'mean',  # Average shipping duration
    'Region': 'last',  # Last region value (assuming it doesn't change)
    'LoyaltyScore': 'last',  # Last loyalty score (assuming it doesn't change)
    'PurchaseFrequency': 'last',  # Last purchase frequency (assuming it doesn't change)
    'CustomerLifetimeValue': 'last',  # Last customer lifetime value (assuming it doesn't change)
    'Season': safe_mode,  # Most common season
    'CustomerType': safe_mode,  # Most common customer type
    'PurchaseChannel': safe_mode,  # Most common purchase channel
    'SeasonalDiscount': 'mean',  # Average seasonal discount
    'HolidayPeriod': safe_mode,  # Most common holiday period
    'CustomerSatisfactionScore': 'mean',  # Average customer satisfaction score
    'TransactionCount': 'sum'  # Total count of transactions
}

new_df['TransactionCount'] = 1

# Aggregate the data by CustomerID
agg_df = new_df.groupby('CustomerID').agg(agg_funcs).reset_index()


# %%
# Display the first few rows of the aggregated dataframe
agg_df.head()

# %%
print(f"Aggregated rows: {agg_df.shape[0]}")

# %%
agg_df.info()

# %%
agg_df.describe()

# %% [markdown]
# Step 3: Handle Missing Values

# %% [markdown]
# We will use imputation like KNN imputation to fill missing data.

# %%
import pandas as pd
from sklearn.impute import KNNImputer

# Define columns with missing numerical values
numerical_cols = ['LoyaltyScore', 'PurchaseFrequency', 'CustomerLifetimeValue', 'Season', 
                  'CustomerType', 'PurchaseChannel', 'SeasonalDiscount', 'HolidayPeriod', 
                  'CustomerSatisfactionScore']

# Copy the DataFrame to avoid modifying the original data
impute_df = agg_df.copy()

# Initialize KNN imputer
imputer = KNNImputer()

# Fit and transform the data
imputed_data = imputer.fit_transform(impute_df[numerical_cols])

# Convert back to DataFrame
impute_df[numerical_cols] = imputed_data

# Verify no missing values after imputation
print(impute_df[numerical_cols].isnull().sum())


# %%
impute_df.describe()

# %%
impute_df.info()

# %% [markdown]
# 4. Compare change in data distribution

# %%
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

# Numerical columns to compare
numerical_cols = ['LoyaltyScore', 'PurchaseFrequency', 'CustomerLifetimeValue', 'Season', 
                  'CustomerType', 'PurchaseChannel', 'SeasonalDiscount', 'HolidayPeriod', 
                  'CustomerSatisfactionScore']

# Compare distributions of numerical columns
compare_distributions(agg_df, impute_df, numerical_cols, categorical=False)


# %% [markdown]
# Improving the Filling of Missing value based on distribution observation

# %%
import numpy as np
impute_df = agg_df.copy()

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

# Initialize KNN imputer
imputer = KNNImputer()

# Function to fill missing values using random sampling
def fill_with_random_sampling(series):
    # Filter non-missing values
    non_missing = series.dropna()
    # Generate random sample with replacement
    random_sample = np.random.choice(non_missing, size=series.isnull().sum(), replace=True)
    # Create a Series with random values to fill the missing values
    random_sample_series = pd.Series(random_sample, index=series.index[series.isnull()])
    # Combine original and random sample series
    series_filled = series.combine_first(random_sample_series)
    return series_filled

# Fill missing values based on specified methods
for col, method in columns_to_impute.items():
    if method == 'random':
        impute_df[col] = fill_with_random_sampling(impute_df[col])
    elif method == 'knn_round':
        # Use KNN imputer
        imputed_values = imputer.fit_transform(impute_df[[col]])
        impute_df[col] = imputed_values.round().astype(int)

# Verify no missing values after imputation
print(impute_df.isnull().sum())

# %%
# Compare distributions of numerical columns
compare_distributions(agg_df, impute_df, numerical_cols, categorical=False)

# %% [markdown]
# Machine Learning Model for Predicting Customer Churn

# %%
from sklearn.model_selection import train_test_split

# Step 1: Calculate churn threshold
churn_threshold = pd.to_datetime(impute_df['PurchaseDate'].max()) - pd.DateOffset(months=12)

# Step 2: Label churn in impute_df
impute_df['Churn'] = (impute_df['PurchaseDate'] < churn_threshold).astype(int)
impute_df['PurchaseDate'] = impute_df['PurchaseDate'].astype(int) / 10**9

X = impute_df.drop(columns=['CustomerID','Churn'])
y = impute_df['Churn']

# Split data into training and testing sets (adjust test_size and random_state as needed)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Define models
log_reg = LogisticRegression(random_state=42)
rf = RandomForestClassifier(random_state=42)

# Define parameter grids for GridSearchCV
log_reg_param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10, 100],  # Regularization parameter
    'solver': ['liblinear', 'lbfgs'],
}

rf_param_grid = {
    'n_estimators': [100, 200, 300],  # Number of trees in the forest
    'max_depth': [None, 10, 20, 30],  # Maximum depth of the trees
    'min_samples_split': [2, 5, 10],  # Minimum samples required to split an internal node
}

# Perform GridSearchCV for Logistic Regression
log_reg_grid_search = GridSearchCV(log_reg, log_reg_param_grid, cv=5, scoring='accuracy', verbose=1)
log_reg_grid_search.fit(X_train, y_train)

# Perform GridSearchCV for Random Forest
rf_grid_search = GridSearchCV(rf, rf_param_grid, cv=5, scoring='accuracy', verbose=1)
rf_grid_search.fit(X_train, y_train)

# Get best parameters and best score for Logistic Regression
print("Best parameters for Logistic Regression:", log_reg_grid_search.best_params_)
print("Best cross-validation score for Logistic Regression:", log_reg_grid_search.best_score_)

# Get best parameters and best score for Random Forest
print("Best parameters for Random Forest:", rf_grid_search.best_params_)
print("Best cross-validation score for Random Forest:", rf_grid_search.best_score_)

# Evaluate best models on test set
log_reg_best = log_reg_grid_search.best_estimator_
rf_best = rf_grid_search.best_estimator_

log_reg_y_pred = log_reg_best.predict(X_test)
rf_y_pred = rf_best.predict(X_test)

# Evaluate performance on test set
print("Logistic Regression:")
print(classification_report(y_test, log_reg_y_pred))

print("Random Forest:")
print(classification_report(y_test, rf_y_pred))


# %%
from joblib import dump

model_filename = 'model/logistic_regression_model.pkl'
dump(log_reg_best, model_filename)


# %%
# Further analysis or improvement steps
# - Feature importance analysis
# - Hyperparameter tuning
# - Cross-validation
# - Model interpretation


