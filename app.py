import streamlit as st
import pandas as pd
import joblib

# Load trained model
model = joblib.load("payment_anomaly_model.pkl")

st.set_page_config(
    page_title="Payment Transaction Anomaly Detection",
    page_icon="💳",
    layout="centered"
)

st.title("💳 Payment Transaction Anomaly Detection")
st.write(
    "Enter the transaction details below to check whether the transaction "
    "is likely to be normal or anomalous."
)

# Transaction details
amount = st.number_input(
    "Transaction Amount",
    min_value=0.0,
    value=1000.0
)

merchant_category = st.selectbox(
    "Merchant Category",
    ["grocery", "electronics", "travel", "food", "shopping", "utilities"]
)

country = st.selectbox(
    "Transaction Country",
    ["India", "USA", "UK", "Canada", "Australia", "Singapore"]
)

home_country = st.selectbox(
    "Home Country",
    ["India", "USA", "UK", "Canada", "Australia", "Singapore"]
)

channel = st.selectbox(
    "Transaction Channel",
    ["online", "mobile", "pos"]
)

hours_since_prev_txn = st.number_input(
    "Hours Since Previous Transaction",
    min_value=0.0,
    value=12.0
)

transaction_frequency_24h = st.number_input(
    "Transaction Frequency (Last 24 Hours)",
    min_value=0,
    value=2
)

avg_amount_30d = st.number_input(
    "Average Transaction Amount (Last 30 Days)",
    min_value=0.0,
    value=1000.0
)

account_age_days = st.number_input(
    "Account Age (Days)",
    min_value=0,
    value=365
)

distance_from_home_km = st.number_input(
    "Distance From Home (km)",
    min_value=0.0,
    value=10.0
)

new_device = st.selectbox(
    "Is This a New Device?",
    [0, 1],
    format_func=lambda x: "No" if x == 0 else "Yes"
)

transaction_hour = st.number_input(
    "Transaction Hour (0-23)",
    min_value=0,
    max_value=23,
    value=14
)

day_of_week = st.number_input(
    "Day of Week (0=Monday, 6=Sunday)",
    min_value=0,
    max_value=6,
    value=2
)

transaction_month = st.number_input(
    "Transaction Month",
    min_value=1,
    max_value=12,
    value=9
)

# Engineered features
amount_ratio = amount / (avg_amount_30d + 1)

country_changed = int(country != home_country)

high_frequency = int(transaction_frequency_24h >= 5)

# Create input dataframe
input_data = pd.DataFrame([{
    "amount": amount,
    "merchant_category": merchant_category,
    "country": country,
    "home_country": home_country,
    "channel": channel,
    "hours_since_prev_txn": hours_since_prev_txn,
    "transaction_frequency_24h": transaction_frequency_24h,
    "avg_amount_30d": avg_amount_30d,
    "account_age_days": account_age_days,
    "distance_from_home_km": distance_from_home_km,
    "new_device": new_device,
    "amount_ratio": amount_ratio,
    "country_changed": country_changed,
    "high_frequency": high_frequency,
    "transaction_hour": transaction_hour,
    "day_of_week": day_of_week,
    "transaction_month": transaction_month
}])

# Prediction
if st.button("🔍 Check Transaction"):

    prediction = model.predict(input_data)[0]

    probability = model.predict_proba(input_data)[0][1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.error("🚨 Anomalous Transaction")
    else:
        st.success("✅ Normal Transaction")

    st.write(
        f"**Anomaly Probability: {probability * 100:.2f}%**"
    )

    if probability >= 0.5:
        st.warning(
            "The transaction shows a relatively high anomaly probability."
        )
    else:
        st.info(
            "The transaction shows a relatively low anomaly probability."
        )

st.markdown("---")
st.caption(
    "Educational prototype for payment transaction anomaly classification."
)
