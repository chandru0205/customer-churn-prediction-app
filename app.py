import joblib
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Customer Churn Predictor")

@st.cache_resource
def load_model():
    return joblib.load("churn_model.pkl")

model = load_model()

st.title("Customer Churn Predictor")

col1, col2 = st.columns(2)

with col1:
    tenure = st.slider("Tenure (months)", 0, 72, 12)
    monthly = st.number_input("Monthly Charges", 18.0, 120.0, 70.0)
    senior = st.selectbox("Senior Citizen", ["No", "Yes"])

with col2:
    contract = st.selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"]
    )
    internet = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )
    tech_support = st.selectbox(
        "Tech Support",
        ["Yes", "No", "No internet service"]
    )
    payment = st.selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)"
        ]
    )

if st.button("Predict Churn"):
    customer = pd.DataFrame([{
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "SeniorCitizen": 1 if senior == "Yes" else 0,
        "Contract": contract,
        "InternetService": internet,
        "PaymentMethod": payment,
        "TechSupport": tech_support
    }])

    probability = model.predict_proba(customer)[0][1]

    st.subheader(f"Churn Probability: {probability:.1%}")
    st.progress(float(probability))

    if probability >= 0.5:
        st.error("High churn risk")
        st.write("Consider reviewing this customer's contract and service experience.")
    else:
        st.success("Low churn risk")
