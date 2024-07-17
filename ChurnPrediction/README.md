# Customer Churn Prediction Project
This project aims to predict customer churn using sales data. The pipeline involves data preprocessing, feature engineering, model training, and evaluation. The best model is saved for deployment in a Streamlit app for visualization.

## Table of Contents
1. Project Overview
2. EDA
3. Data Preprocessing
4. Feature Engineering
5. Model Training and Evaluation
6. Custom Transformers
7. Pipeline and Grid Search
8. Model Saving
9. Streamlit App
10. Deployment

## Project Overview
The goal of this project is to predict customer churn based on historical sales data. The data is processed, aggregated, and fed into machine learning models to identify customers likely to churn.

## EDA
An extensive exploratory data analysis (EDA) was conducted, providing insights into the dataset. The details can be found in the [EDA](EDA/README.md) section of this project.

## Data Preprocessing
Data preprocessing involves the following steps:

1. **Loading Data**: Read the CSV file containing sales data.
2. **Dropping Unnecessary Columns**: Remove the 'Unnamed: 0' column.
3. **Handling Date Columns**: Convert date columns to datetime objects, handling different formats and missing values.
4. **Adding Features**: Add a 'TransactionCount' feature to count the number of transactions per customer.

## Feature Engineering
Data is aggregated by CustomerID using various aggregation functions. The following features are considered:

**Numerical Features**: Age, PurchaseAmount, Quantity, DiscountPercentage, Rating, ShippingDuration, LoyaltyScore, PurchaseFrequency, CustomerLifetimeValue, SeasonalDiscount, CustomerSatisfactionScore, TransactionCount.

**Categorical Features**: Gender, Location, ProductCategory, PaymentMethod, CustomerSegment, Region, Season, CustomerType, PurchaseChannel, HolidayPeriod.

**Label (Churn)**: Calculated based on the time since the last purchase.

## Model Training and Evaluation
The project uses Logistic Regression as the classification model. The data is split into training and testing sets. GridSearchCV is used for hyperparameter tuning to find the best model parameters.

## Custom Transformers
Two custom transformers are implemented:

1. **CustomLabelEncoder**: Encodes categorical variables based on predefined mappings.

2. **RandomImputer**: Fills missing values in selected columns using random sampling from non-missing values.

## Pipeline and Grid Search
A pipeline is created to streamline the preprocessing and model training process. The pipeline includes:

- Custom label encoding
- Random imputation for missing values
- Standard preprocessing for numerical and categorical features
- Logistic Regression classifier

GridSearchCV is used to tune hyperparameters (C and solver).

## Model Saving
The best model from GridSearchCV is saved using joblib for later use in the Streamlit app.

## Streamlit App
The Streamlit app provides an interactive interface for predicting customer churn based on user inputs. It leverages the pre-trained machine learning model saved in the pipeline to make real-time predictions.

### Key Components
1. **Loading the Model**: The pipeline with preprocessing and the trained model is loaded using joblib.
2. **User Input Form**: A form is provided for users to input various features required for churn prediction.
3. **Data Processing**: The user input is processed and mapped to the format expected by the model.
4. **Prediction**: The model predicts whether the customer will churn and displays the result along with prediction probabilities.

### Features
The app allows users to input the following features:

1. **Categorical Features**: Gender, Location, Product Category, Payment Method, Customer Segment, Region, Season, Customer Type, Purchase Channel, Holiday Period.

2. **Numerical Features**: Age, Purchase Amount, Quantity, Discount Percentage, Rating, Is Returned, Is Promotion, Shipping Duration, Loyalty Score, Purchase Frequency, Customer Lifetime Value, Seasonal Discount, Customer Satisfaction Score, Transaction Count.

## Deployment
The app is deployed on huggingface and can be accessed at [here](https://huggingface.co/spaces/g23ai1052/Churn-Prediction)

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.


