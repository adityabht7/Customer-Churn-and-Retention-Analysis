"""
Unit tests for data loading, preprocessing, and KPI calculations.
"""

import pandas as pd
import numpy as np
from src.data_loader import load_churn_data, preprocess_data, calculate_business_kpis, get_tenure_cohorts

def test_load_churn_data_returns_dataframe():
    df = load_churn_data()
    assert isinstance(df, pd.DataFrame)
    assert len(df) > 0
    assert "customerID" in df.columns
    assert "Churn" in df.columns
    assert "MonthlyCharges" in df.columns

def test_preprocess_handles_missing_total_charges():
    raw_df = pd.DataFrame({
        "customerID": ["TEST-1", "TEST-2"],
        "tenure": [10, 5],
        "MonthlyCharges": [50.0, 30.0],
        "TotalCharges": [" ", "150.0"],
        "Churn": ["No", "Yes"]
    })
    cleaned = preprocess_data(raw_df)
    assert cleaned["TotalCharges"].iloc[0] == 500.0  # 10 * 50.0
    assert cleaned["TotalCharges"].iloc[1] == 150.0
    assert cleaned["Churn"].tolist() == ["No", "Yes"]

def test_calculate_business_kpis():
    df = pd.DataFrame({
        "customerID": ["1", "2", "3", "4"],
        "MonthlyCharges": [100.0, 50.0, 50.0, 100.0],
        "tenure": [10, 20, 30, 40],
        "TotalCharges": [1000.0, 1000.0, 1500.0, 4000.0],
        "SupportTickets": [1, 2, 0, 3],
        "SatisfactionScore": [4, 3, 5, 2],
        "Churn": ["No", "Yes", "No", "Yes"]
    })
    kpis = calculate_business_kpis(df)
    assert kpis["total_customers"] == 4
    assert kpis["churn_count"] == 2
    assert kpis["churn_rate"] == 50.0
    assert kpis["total_mrr"] == 300.0
    assert kpis["mrr_at_risk"] == 150.0
    assert kpis["mrr_risk_pct"] == 50.0

def test_get_tenure_cohorts():
    df = pd.DataFrame({
        "customerID": ["1", "2", "3", "4"],
        "tenure": [6, 18, 30, 60],
        "MonthlyCharges": [50.0, 60.0, 70.0, 80.0],
        "Churn": ["Yes", "No", "No", "No"]
    })
    cohorts = get_tenure_cohorts(df)
    assert len(cohorts) > 0
    assert "Tenure_Cohort" in cohorts.columns
    assert "Churn_Rate_%" in cohorts.columns
