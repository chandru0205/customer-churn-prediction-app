# Customer Churn Prediction System: ML-Powered Web App for Telecom Retention

**Live demo:**https://customer-churn-prediction-app-pvgvrue8fipyztbszm3d8y.streamlit.app/

## Problem
Telecom companies lose revenue when customers leave. This app predicts
whether a customer is likely to churn so the company can act early.

## What I built
- Cleaned and prepared data for 7,043 telecom customers
- Compared 3 models: Logistic Regression, Decision Tree, Random Forest
- Achieved about 79% accuracy on unseen test data
- Deployed a live web app with Streamlit where users enter customer
  details and get a churn probability with a suggested action

## Tech stack
Python, pandas, scikit-learn, Streamlit

## Key insight
Month-to-month contracts, short tenure and no tech support are linked
to higher churn risk.

## Limitations and next steps
The model finds about half of actual churners. Next I plan to handle
class imbalance and add more features.
