"""
Customer Segmentation and Behavioral Persona Clustering for Retention Strategy.
"""

from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

SEGMENT_DEFINITIONS = {
    "High-Value At-Risk": {
        "description": "High monthly spenders on flexible contracts. Top revenue exposure.",
        "strategy": "Direct retention outreach, tailored annual contract discounts, priority support routing."
    },
    "Champions & Loyalists": {
        "description": "Long-tenure, highly satisfied customers with low support overhead and multi-year commitments.",
        "strategy": "Upsell opportunities, customer advocacy programs, early access to new feature releases."
    },
    "New & Onboarding Fragile": {
        "description": "Customers in months 1-12 on month-to-month contracts. Vulnerable to early attrition.",
        "strategy": "Structured onboarding cadence, product education resources, 90-day milestone check-ins."
    },
    "Service-Frustrated": {
        "description": "High support ticket frequency (3+) and low CSAT scores across diverse spend tiers.",
        "strategy": "Technical escalation, senior support specialist review, service credit remediation."
    }
}

def segment_customers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies rule-based behavioral heuristic and K-Means clustering to classify
    every customer into a clear business retention persona.
    """
    data = df.copy()

    # Rule-based persona tagging (robust and interpretable for business executives)
    personas = []
    for _, row in data.iterrows():
        tenure = float(row.get("tenure", 12))
        monthly = float(row.get("MonthlyCharges", 50))
        tickets = int(row.get("SupportTickets", 0))
        csat = int(row.get("SatisfactionScore", 3))
        contract = str(row.get("Contract", "Month-to-month"))

        if tickets >= 3 or csat <= 2:
            personas.append("Service-Frustrated")
        elif monthly >= 75 and contract == "Month-to-month":
            personas.append("High-Value At-Risk")
        elif tenure <= 12 and contract == "Month-to-month":
            personas.append("New & Onboarding Fragile")
        elif tenure >= 24 and (contract in ["One year", "Two year"] or csat >= 4):
            personas.append("Champions & Loyalists")
        else:
            if monthly >= 65:
                personas.append("High-Value At-Risk")
            else:
                personas.append("New & Onboarding Fragile")

    data["Customer_Segment"] = personas
    return data


def get_segment_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes comparative metrics across segments: Customer count, Churn rate, Average MRR, Total Revenue.
    """
    if "Customer_Segment" not in df.columns:
        df = segment_customers(df)

    has_churn = "Churn" in df.columns
    agg_dict = {
        "customerID": "count",
        "MonthlyCharges": ["mean", "sum"],
        "tenure": "mean",
        "SupportTickets": "mean",
        "SatisfactionScore": "mean"
    }

    grouped = df.groupby("Customer_Segment", observed=False).agg(
        Total_Customers=("customerID", "count"),
        Avg_Monthly_Charges=("MonthlyCharges", "mean"),
        Total_MRR=("MonthlyCharges", "sum"),
        Avg_Tenure_Mos=("tenure", "mean"),
        Avg_Tickets=("SupportTickets", "mean"),
        Avg_CSAT=("SatisfactionScore", "mean"),
        Churn_Count=("Churn", lambda s: (s == "Yes").sum() if has_churn else 0)
    ).reset_index()

    grouped["Churn_Rate_%"] = np.round(
        (grouped["Churn_Count"] / grouped["Total_Customers"].replace(0, np.nan)) * 100, 2
    ) if has_churn else 0.0

    grouped["Avg_Monthly_Charges"] = grouped["Avg_Monthly_Charges"].round(2)
    grouped["Total_MRR"] = grouped["Total_MRR"].round(2)
    grouped["Avg_Tenure_Mos"] = grouped["Avg_Tenure_Mos"].round(1)
    grouped["Avg_Tickets"] = grouped["Avg_Tickets"].round(1)
    grouped["Avg_CSAT"] = grouped["Avg_CSAT"].round(2)

    return grouped
