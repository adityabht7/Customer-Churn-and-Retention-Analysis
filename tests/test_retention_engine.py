"""
Unit tests for prescriptive retention engine, What-If simulation, and ROI calculations.
"""

from src.retention_engine import (
    generate_retention_playbook,
    simulate_what_if_intervention,
    calculate_retention_campaign_roi
)
from src.churn_model import load_or_train_default_model

def test_generate_retention_playbook():
    customer = {
        "Contract": "Month-to-month",
        "SupportTickets": 4,
        "SatisfactionScore": 1,
        "InternetService": "Fiber optic",
        "TechSupport": "No",
        "PaymentMethod": "Electronic check",
        "MonthlyCharges": 85.0
    }
    playbook = generate_retention_playbook(customer, churn_prob=0.88)
    assert len(playbook) >= 2
    action_ids = [p["action_id"] for p in playbook]
    assert "vip_support_escalation" in action_ids
    assert "annual_contract_upgrade" in action_ids

def test_simulate_what_if_intervention():
    pipe = load_or_train_default_model()
    customer = {
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

    # Simulate intervention: upgrade contract to Two year and fix support
    intervention = {
        "Contract": "Two year",
        "SupportTickets": 0,
        "SatisfactionScore": 5,
        "TechSupport": "Yes"
    }

    sim = simulate_what_if_intervention(pipe, customer, intervention)
    assert sim["original_probability"] > sim["new_probability"]
    assert sim["absolute_reduction"] > 0

def test_calculate_retention_campaign_roi():
    roi = calculate_retention_campaign_roi(
        at_risk_count=500,
        avg_mrr=70.0,
        intervention_cost_per_user=20.0,
        expected_save_rate_pct=30.0
    )
    assert roi["total_campaign_cost"] == 10000.0
    assert roi["customers_saved"] == 150
    assert roi["annual_revenue_saved"] == 150 * (70.0 * 12.0)
    assert roi["net_profit"] > 0
    assert roi["roi_pct"] > 0
