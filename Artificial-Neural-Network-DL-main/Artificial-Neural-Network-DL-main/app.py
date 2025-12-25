import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(page_title="Customer Churn Prediction App", layout="centered")
st.title("🏦 Customer Churn Prediction App")
st.write("Enter customer details below to predict whether they will churn or stay.")

# -----------------------------
# Load Saved Model & Scaler
# -----------------------------
model = load_model("ann_churn_model.keras")

scaler = joblib.load("scaler.pkl")

# -----------------------------
# User Inputs
# -----------------------------
credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=600)
geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
gender = st.selectbox("Gender", ["Male", "Female"])
age = st.number_input("Age", min_value=18, max_value=100, value=40)
tenure = st.number_input("Tenure (Years)", min_value=0, max_value=10, value=5)
balance = st.number_input("Balance", min_value=0.0, max_value=250000.0, value=60000.0, step=1000.0)
num_of_products = st.number_input("Number of Products", min_value=1, max_value=4, value=2)
has_credit_card = st.selectbox("Has Credit Card", ["Yes", "No"])
is_active_member = st.selectbox("Is Active Member", ["Yes", "No"])
estimated_salary = st.number_input("Estimated Salary", min_value=0.0, max_value=200000.0, value=50000.0, step=1000.0)

# -----------------------------
# Convert Inputs to Numeric
# -----------------------------
geo_dict = {"France": 0, "Germany": 1, "Spain": 2}
gender_dict = {"Male": 1, "Female": 0}
credit_dict = {"Yes": 1, "No": 0}

input_data = np.array([[
    credit_score,
    geo_dict[geography],
    gender_dict[gender],
    age,
    tenure,
    balance,
    num_of_products,
    credit_dict[has_credit_card],
    credit_dict[is_active_member],
    estimated_salary
]])

# -----------------------------
# Predict Button
# -----------------------------
if st.button("🔍 Predict Churn Probability"):
    input_scaled = scaler.transform(input_data)
    probability = model.predict(input_scaled)[0][0]
    st.markdown(f"### 📊 Churn Probability: **{probability:.2f}**")

    if probability > 0.7:
        st.error("⚠️ High Chance of Customer Churn")
    elif probability > 0.4:
        st.warning("🟠 Moderate Chance of Churn")
    else:
        st.success("🟢 Low Chance of Customer Churn")
