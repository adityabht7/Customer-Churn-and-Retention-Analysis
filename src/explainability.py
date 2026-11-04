"""
Explainability engine providing local risk attribution and feature breakdown
for individual customer churn predictions.
"""

from typing import Dict, Any, List, Tuple
import pandas as pd
import numpy as np

def explain_customer_prediction(customer: Dict[str, Any], baseline_prob: float = 0.29) -> Dict[str, Any]:
    """
    Computes local feature risk attribution for an individual customer profile,
    quantifying which specific attributes increase or decrease their churn risk.
    """
    drivers = []
    
    # 1. Contract Analysis
    contract = customer.get("Contract", "Month-to-month")
    if contract == "Month-to-month":
        drivers.append({
            "feature": "Contract: Month-to-Month",
            "impact": 0.28,
            "direction": "Risk Factor",
            "explanation": "No long-term commitment allows easy cancellation with zero penalty."
        })
    elif contract == "Two year":
        drivers.append({
            "feature": "Contract: Two-Year Lock-in",
            "impact": -0.26,
            "direction": "Mitigating Factor",
            "explanation": "Two-year contractual commitment provides strong customer lock-in."
        })
    elif contract == "One year":
        drivers.append({
            "feature": "Contract: One-Year Commitment",
            "impact": -0.12,
            "direction": "Mitigating Factor",
            "explanation": "Annual agreement stabilizes retention."
        })

    # 2. Support Tickets
    tickets = int(customer.get("SupportTickets", 1))
    if tickets >= 3:
        drivers.append({
            "feature": f"High Support Ticket Volume ({tickets} tickets)",
            "impact": 0.07 * tickets,
            "direction": "Risk Factor",
            "explanation": f"Customer had to reach out {tickets} times, signaling recurring friction or service downtime."
        })
    elif tickets == 0:
        drivers.append({
            "feature": "Zero Support Tickets",
            "impact": -0.08,
            "direction": "Mitigating Factor",
            "explanation": "Consistent service experience with no complaints raised."
        })

    # 3. Satisfaction Score (CSAT)
    csat = int(customer.get("SatisfactionScore", 3))
    if csat <= 2:
        drivers.append({
            "feature": f"Low Customer Satisfaction (CSAT {csat}/5)",
            "impact": 0.22 if csat == 1 else 0.14,
            "direction": "Risk Factor",
            "explanation": "Recent customer feedback indicates dissatisfaction."
        })
    elif csat >= 4:
        drivers.append({
            "feature": f"High Customer Satisfaction (CSAT {csat}/5)",
            "impact": -0.16 if csat == 5 else -0.10,
            "direction": "Mitigating Factor",
            "explanation": "Customer is satisfied with service quality."
        })

    # 4. Tenure
    tenure = int(customer.get("tenure", 12))
    if tenure <= 6:
        drivers.append({
            "feature": f"Early Onboarding Phase ({tenure} months tenure)",
            "impact": 0.18,
            "direction": "Risk Factor",
            "explanation": "First 6 months are the highest drop-off window across all cohorts."
        })
    elif tenure >= 48:
        drivers.append({
            "feature": f"Long-term Tenured Customer ({tenure} months tenure)",
            "impact": -0.20,
            "direction": "Mitigating Factor",
            "explanation": "High brand habituation and switching friction."
        })

    # 5. Internet & Tech Support Gap
    internet = customer.get("InternetService", "DSL")
    tech_sup = customer.get("TechSupport", "No")
    if internet == "Fiber optic" and tech_sup != "Yes":
        drivers.append({
            "feature": "Fiber Optic without Dedicated Tech Support",
            "impact": 0.12,
            "direction": "Risk Factor",
            "explanation": "High-bandwidth customers experiencing issues without tech support are prone to switch."
        })
    elif tech_sup == "Yes":
        drivers.append({
            "feature": "Active Tech Support Add-on",
            "impact": -0.09,
            "direction": "Mitigating Factor",
            "explanation": "Access to technical assistance increases product stickiness."
        })

    # 6. Payment Method
    payment = customer.get("PaymentMethod", "Mailed check")
    if payment == "Electronic check":
        drivers.append({
            "feature": "Payment via Manual Electronic Check",
            "impact": 0.10,
            "direction": "Risk Factor",
            "explanation": "Requires active monthly manual payment, giving recurring opportunities to cancel."
        })
    elif "automatic" in payment:
        drivers.append({
            "feature": f"Automated Payment ({payment})",
            "impact": -0.08,
            "direction": "Mitigating Factor",
            "explanation": "Automated recurring billing prevents involuntary or impulsive churn."
        })

    # 7. Pricing Pressure
    monthly = float(customer.get("MonthlyCharges", 60.0))
    if monthly > 85.0:
        drivers.append({
            "feature": f"High Monthly Charges (${monthly:.2f}/mo)",
            "impact": 0.08,
            "direction": "Risk Factor",
            "explanation": "Premium price tier increases price sensitivity when competitor offers appear."
        })
    elif monthly < 35.0:
        drivers.append({
            "feature": f"Affordable Plan (${monthly:.2f}/mo)",
            "impact": -0.07,
            "direction": "Mitigating Factor",
            "explanation": "Budget-friendly tier reduces incentive to shop around."
        })

    # Separate into top risk drivers and retention anchors
    risk_drivers = [d for d in drivers if d["impact"] > 0]
    retention_anchors = [d for d in drivers if d["impact"] < 0]

    risk_drivers.sort(key=lambda x: x["impact"], reverse=True)
    retention_anchors.sort(key=lambda x: abs(x["impact"]), reverse=True)

    return {
        "all_drivers": drivers,
        "top_risk_drivers": risk_drivers[:4],
        "top_retention_anchors": retention_anchors[:4],
        "baseline_prob": baseline_prob
    }
