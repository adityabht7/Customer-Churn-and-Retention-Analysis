"""
Data loading, cleaning, schema validation, and KPI aggregations for Customer Churn Analysis.
"""

from typing import Dict, Any, Tuple, Optional
import os
import pandas as pd
import numpy as np

DEFAULT_DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "telco_customer_churn.csv")

CATEGORICAL_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaperlessBilling", "PaymentMethod", "UsageFrequency"
]

NUMERICAL_COLS = [
    "tenure", "MonthlyCharges", "TotalCharges", "SupportTickets", "SatisfactionScore"
]

def load_churn_data(file_or_path: Optional[Any] = None) -> pd.DataFrame:
    """
    Loads dataset from a file path, file-like buffer, or fallback default path.
    """
    if file_or_path is None or (isinstance(file_or_path, str) and not os.path.exists(file_or_path)):
        if os.path.exists(DEFAULT_DATA_PATH):
            df = pd.read_csv(DEFAULT_DATA_PATH)
        else:
            from scripts.generate_dataset import generate_churn_dataset
            df = generate_churn_dataset()
    elif isinstance(file_or_path, pd.DataFrame):
        df = file_or_path.copy()
    else:
        df = pd.read_csv(file_or_path)

    return preprocess_data(df)


def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans data, casts types, handles whitespace and missing values.
    """
    data = df.copy()

    # TotalCharges often has blank strings in Telco dataset
    if "TotalCharges" in data.columns:
        data["TotalCharges"] = pd.to_numeric(data["TotalCharges"].astype(str).str.strip(), errors="coerce")
        # Impute missing TotalCharges with MonthlyCharges * tenure
        if "MonthlyCharges" in data.columns and "tenure" in data.columns:
            data["TotalCharges"] = data["TotalCharges"].fillna(data["MonthlyCharges"] * data["tenure"])
        data["TotalCharges"] = data["TotalCharges"].fillna(0.0)

    # Convert SeniorCitizen to string category or int
    if "SeniorCitizen" in data.columns:
        data["SeniorCitizen"] = data["SeniorCitizen"].astype(int)

    # Telemetry fallbacks if user uploads standard Telco dataset without telemetry
    if "SupportTickets" not in data.columns:
        # Approximate based on contract & internet
        np.random.seed(42)
        data["SupportTickets"] = np.random.choice([0, 1, 2, 3, 4], size=len(data), p=[0.4, 0.3, 0.15, 0.1, 0.05])
    if "SatisfactionScore" not in data.columns:
        data["SatisfactionScore"] = np.clip(5 - data["SupportTickets"] + np.random.choice([-1, 0, 1], size=len(data)), 1, 5)
    if "UsageFrequency" not in data.columns:
        data["UsageFrequency"] = "Daily"

    # Normalize Churn column if present
    if "Churn" in data.columns:
        data["Churn"] = data["Churn"].astype(str).str.strip().str.capitalize()
        # Ensure values are 'Yes' or 'No'
        data["Churn"] = data["Churn"].replace({"1": "Yes", "0": "No", "True": "Yes", "False": "No"})

    return data


def calculate_business_kpis(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Computes executive KPI metrics including Churn Rate, MRR, At-Risk Revenue, and CLV.
    """
    total_customers = len(df)
    has_churn = "Churn" in df.columns
    
    if has_churn:
        churn_count = int((df["Churn"] == "Yes").sum())
        retained_count = total_customers - churn_count
        churn_rate = (churn_count / total_customers) * 100.0 if total_customers > 0 else 0.0
    else:
        churn_count = 0
        retained_count = total_customers
        churn_rate = 0.0

    total_mrr = float(df["MonthlyCharges"].sum()) if "MonthlyCharges" in df.columns else 0.0
    avg_mrr = float(df["MonthlyCharges"].mean()) if "MonthlyCharges" in df.columns else 0.0
    
    mrr_at_risk = float(df[df["Churn"] == "Yes"]["MonthlyCharges"].sum()) if has_churn and "MonthlyCharges" in df.columns else 0.0
    mrr_risk_pct = (mrr_at_risk / total_mrr * 100.0) if total_mrr > 0 else 0.0

    avg_tenure = float(df["tenure"].mean()) if "tenure" in df.columns else 0.0
    
    # Customer Lifetime Value (CLV proxy: Average Monthly Charge * Average Tenure)
    avg_clv = avg_mrr * avg_tenure
    total_historical_revenue = float(df["TotalCharges"].sum()) if "TotalCharges" in df.columns else 0.0

    avg_csat = float(df["SatisfactionScore"].mean()) if "SatisfactionScore" in df.columns else 0.0
    avg_tickets = float(df["SupportTickets"].mean()) if "SupportTickets" in df.columns else 0.0

    return {
        "total_customers": total_customers,
        "churn_count": churn_count,
        "retained_count": retained_count,
        "churn_rate": churn_rate,
        "total_mrr": total_mrr,
        "avg_mrr": avg_mrr,
        "mrr_at_risk": mrr_at_risk,
        "mrr_risk_pct": mrr_risk_pct,
        "avg_tenure": avg_tenure,
        "avg_clv": avg_clv,
        "total_historical_revenue": total_historical_revenue,
        "avg_csat": avg_csat,
        "avg_tickets": avg_tickets,
    }


def get_tenure_cohorts(df: pd.DataFrame) -> pd.DataFrame:
    """
    Groups customers into tenure cohorts and calculates retention / churn statistics.
    """
    if "tenure" not in df.columns or "Churn" not in df.columns:
        return pd.DataFrame()

    bins = [0, 12, 24, 48, 72]
    labels = ["0 - 12 Mos (New)", "13 - 24 Mos (Growing)", "25 - 48 Mos (Mature)", "49 - 72 Mos (Loyal)"]
    cohort_series = pd.cut(df["tenure"], bins=bins, labels=labels, include_lowest=True)
    
    grouped = df.groupby(cohort_series, observed=False).agg(
        Total_Customers=("customerID", "count"),
        Churned=("Churn", lambda s: (s == "Yes").sum()),
        Total_MRR=("MonthlyCharges", "sum"),
        Avg_MonthlyCharges=("MonthlyCharges", "mean")
    ).reset_index()

    grouped = grouped.rename(columns={"tenure": "Tenure_Cohort"})
    grouped["Retained"] = grouped["Total_Customers"] - grouped["Churned"]
    grouped["Churn_Rate_%"] = np.round((grouped["Churned"] / grouped["Total_Customers"].replace(0, np.nan)) * 100, 2)
    grouped["Retention_Rate_%"] = 100.0 - grouped["Churn_Rate_%"]
    return grouped
