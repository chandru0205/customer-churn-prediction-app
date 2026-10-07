"""
STEP 2: The web app.

Run with:  streamlit run app.py
(Make sure you have already run train_model.py so churn_model.pkl exists.)
"""

import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉")

st.title("📉 Customer Churn Predictor")
st.write("Enter customer details to estimate whether they are likely to leave.")


@st.cache_resource
def load_model():
    return joblib.load("churn_model.pkl")


model = load_model()

# ---- Input form ----
col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Tenure (months with company)", 0, 72, 12)
    monthly = st.number_input("Monthly charges", 18.0, 120.0, 70.0)
    senior = st.selectbox("Senior citizen?", ["No", "Yes"])

with col2:
    contract = st.selectbox("Contract type", ["Month-to-month", "One year", "Two year"])
    internet = st.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
    tech_support = st.selectbox("Tech support", ["Yes", "No", "No internet service"])
    payment = st.selectbox(
        "Payment method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )

# ---- Predict ----
if st.button("Predict"):
    customer = pd.DataFrame([{
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Contract": contract,
        "InternetService": internet,
        "PaymentMethod": payment,
        "TechSupport": tech_support,
    }])

    probability = model.predict_proba(customer)[0][1]

    st.subheader(f"Churn probability: {probability:.0%}")
    st.progress(float(probability))

    if probability >= 0.5:
        st.error("⚠️ This customer is LIKELY to churn.")
        st.write("Suggestion: offer a discount or a longer-term contract.")
    else:
        st.success("✅ This customer is NOT likely to churn.")
