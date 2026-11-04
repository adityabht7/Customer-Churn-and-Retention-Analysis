"""
Unit tests for customer persona segmentation.
"""

import pandas as pd
from src.segmentation import segment_customers, get_segment_metrics

def test_segment_customers():
    df = pd.DataFrame([
        {
            "customerID": "C-1", "tenure": 2, "MonthlyCharges": 40.0,
            "SupportTickets": 4, "SatisfactionScore": 1, "Contract": "Month-to-month", "Churn": "Yes"
        },
        {
            "customerID": "C-2", "tenure": 50, "MonthlyCharges": 90.0,
            "SupportTickets": 0, "SatisfactionScore": 5, "Contract": "Two year", "Churn": "No"
        },
        {
            "customerID": "C-3", "tenure": 4, "MonthlyCharges": 85.0,
            "SupportTickets": 1, "SatisfactionScore": 3, "Contract": "Month-to-month", "Churn": "Yes"
        }
    ])

    segmented = segment_customers(df)
    assert "Customer_Segment" in segmented.columns
    assert segmented["Customer_Segment"].iloc[0] == "Service-Frustrated"
    assert segmented["Customer_Segment"].iloc[1] == "Champions & Loyalists"
    assert segmented["Customer_Segment"].iloc[2] == "High-Value At-Risk"

def test_get_segment_metrics():
    df = pd.DataFrame([
        {
            "customerID": "C-1", "tenure": 2, "MonthlyCharges": 40.0,
            "SupportTickets": 4, "SatisfactionScore": 1, "Contract": "Month-to-month", "Churn": "Yes"
        },
        {
            "customerID": "C-2", "tenure": 50, "MonthlyCharges": 90.0,
            "SupportTickets": 0, "SatisfactionScore": 5, "Contract": "Two year", "Churn": "No"
        }
    ])
    metrics = get_segment_metrics(df)
    assert "Customer_Segment" in metrics.columns
    assert "Total_Customers" in metrics.columns
    assert "Total_MRR" in metrics.columns
