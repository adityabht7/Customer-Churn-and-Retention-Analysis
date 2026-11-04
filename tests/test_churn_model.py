"""
Unit tests for Machine Learning model prediction and pipeline.
"""

import pandas as pd
from src.churn_model import (
    load_or_train_default_model, predict_churn_risk, batch_predict,
    prepare_features_and_target
)

def test_pipeline_prediction():
    pipe = load_or_train_default_model()
    assert pipe is not None

    sample_customer = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": 2,
        "PhoneService": "Yes",
        "MultipleLines": "No",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "No",
        "StreamingMovies": "No",
        "Contract": "Month-to-month",
        "PaperlessBilling": "Yes",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 95.0,
        "TotalCharges": 190.0,
        "SupportTickets": 4,
        "UsageFrequency": "Daily",
        "SatisfactionScore": 1
    }

    pred = predict_churn_risk(pipe, sample_customer)
    assert "churn_probability" in pred
    assert 0.0 <= pred["churn_probability"] <= 1.0
    assert pred["risk_tier"] in ["Low Risk", "Moderate Risk", "High Risk"]
    # With low tenure, month-to-month, high tickets, low CSAT, should be high risk
    assert pred["churn_probability"] > 0.50

def test_batch_predict():
    pipe = load_or_train_default_model()
    df = pd.DataFrame([{
        "customerID": f"CUST-{i}",
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "Yes",
        "tenure": 48 if i % 2 == 0 else 3,
        "PhoneService": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "DSL",
        "OnlineSecurity": "Yes",
        "OnlineBackup": "Yes",
        "DeviceProtection": "Yes",
        "TechSupport": "Yes",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Two year" if i % 2 == 0 else "Month-to-month",
        "PaperlessBilling": "No",
        "PaymentMethod": "Bank transfer (automatic)",
        "MonthlyCharges": 50.0,
        "TotalCharges": 2400.0,
        "SupportTickets": 0 if i % 2 == 0 else 4,
        "UsageFrequency": "Daily",
        "SatisfactionScore": 5 if i % 2 == 0 else 1
    } for i in range(4)])

    scored = batch_predict(pipe, df)
    assert "Churn_Probability" in scored.columns
    assert "Predicted_Churn" in scored.columns
    assert "Risk_Tier" in scored.columns
    assert len(scored) == 4
