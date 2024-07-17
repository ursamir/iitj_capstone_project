import streamlit as st
import joblib
import pandas as pd
import numpy as np
from custom_encoder import mappings, CustomLabelEncoder, RandomImputer

# Load the pipeline with preprocessing and model
pipeline = joblib.load('pipeline_with_model.pkl')

# Function to get user input
def get_user_input():
    gender = st.selectbox('Gender', ['Male', 'Female', 'Non-Binary'])
    location = st.selectbox('Location', ['Suburb', 'Rural', 'City'])
    product_category = st.selectbox('Product Category', ['Clothing', 'Home Appliances', 'Electronics', 'Toys', 'Furniture'])
    payment_method = st.selectbox('Payment Method', ['Debit Card', 'Cash', 'Credit Card', 'Digital Wallet', 'Bank Transfer'])
    customer_segment = st.selectbox('Customer Segment', ['New', 'Returning', 'Loyal', 'VIP'])
    region = st.selectbox('Region', ['South', 'East', 'West', 'North', 'Central'])
    season = st.selectbox('Season', ['Spring', 'Autumn', 'Winter', 'Summer'])
    customer_type = st.selectbox('Customer Type', ['Individual', 'Business'])
    purchase_channel = st.selectbox('Purchase Channel', ['In-Store', 'Online', 'Mobile'])
    holiday_period = st.selectbox('Holiday Period', [False, True])
    age = st.slider('Age', 0, 100, 25)
    purchase_amount = st.number_input('Purchase Amount', min_value=0.0, value=100.0)
    quantity = st.number_input('Quantity', min_value=0, value=1)
    discount_percentage = st.number_input('Discount Percentage', min_value=0.0, max_value=100.0, value=0.0)
    rating = st.slider('Rating', 0.0, 5.0, 4.0)
    is_returned = st.number_input('Is Returned', min_value=0, max_value=1, value=0)
    is_promotion = st.number_input('Is Promotion', min_value=0, max_value=1, value=0)
    shipping_duration = st.number_input('Shipping Duration', min_value=0.0, value=3.0)
    loyalty_score = st.number_input('Loyalty Score', min_value=0, value=50)
    purchase_frequency = st.number_input('Purchase Frequency', min_value=0.0, value=1.0)
    customer_lifetime_value = st.number_input('Customer Lifetime Value', min_value=0.0, value=1000.0)
    seasonal_discount = st.number_input('Seasonal Discount', min_value=0.0, max_value=100.0, value=10.0)
    customer_satisfaction_score = st.number_input('Customer Satisfaction Score', min_value=0.0, max_value=100.0, value=80.0)
    transaction_count = st.number_input('Transaction Count', min_value=0, value=1)

    data = {
        'Gender': gender,
        'Location': location,
        'ProductCategory': product_category,
        'PaymentMethod': payment_method,
        'CustomerSegment': customer_segment,
        'Region': region,
        'Season': season,
        'CustomerType': customer_type,
        'PurchaseChannel': purchase_channel,
        'HolidayPeriod': holiday_period,
        'Age': age,
        'PurchaseAmount': purchase_amount,
        'Quantity': quantity,
        'DiscountPercentage': discount_percentage,
        'Rating': rating,
        'IsReturned': is_returned,
        'IsPromotion': is_promotion,
        'ShippingDuration': shipping_duration,
        'LoyaltyScore': loyalty_score,
        'PurchaseFrequency': purchase_frequency,
        'CustomerLifetimeValue': customer_lifetime_value,
        'SeasonalDiscount': seasonal_discount,
        'CustomerSatisfactionScore': customer_satisfaction_score,
        'TransactionCount': transaction_count
    }

    features = pd.DataFrame(data, index=[0])
    return features

# Main function for Streamlit app
def main():
    st.title("Churn Prediction App")
    st.write("This app predicts whether a customer will churn based on their purchase history.")

    # Get user input
    input_df = get_user_input()

    # Apply mappings
    for col, mapping in mappings.items():
        input_df[col] = input_df[col].map(mapping)

    # Make prediction
    prediction = pipeline.predict(input_df)
    prediction_proba = pipeline.predict_proba(input_df)

    st.subheader('Prediction')
    churn = 'Yes' if prediction[0] == 1 else 'No'
    st.write(f"Will the customer churn? **{churn}**")

    st.subheader('Prediction Probability')
    st.write(f"Probability of churn: **{prediction_proba[0][1]:.2f}**")
    st.write(f"Probability of not churn: **{prediction_proba[0][0]:.2f}**")

if __name__ == '__main__':
    main()
