"""
Prescriptive Retention Engine: Playbook recommendations, What-If intervention simulator,
and financial ROI retention calculator.
"""

from typing import Dict, Any, List, Tuple
import pandas as pd
from sklearn.pipeline import Pipeline
from src.churn_model import predict_churn_risk

def generate_retention_playbook(customer: Dict[str, Any], churn_prob: float) -> List[Dict[str, Any]]:
    """
    Generates prioritized, tailored retention intervention playbooks based on
    the customer's risk profile and specific pain points.
    """
    recommendations = []

    contract = customer.get("Contract", "Month-to-month")
    tickets = int(customer.get("SupportTickets", 0))
    csat = int(customer.get("SatisfactionScore", 3))
    internet = customer.get("InternetService", "DSL")
    tech_sup = customer.get("TechSupport", "No")
    payment = customer.get("PaymentMethod", "")
    monthly = float(customer.get("MonthlyCharges", 50.0))

    # Priority 1: Service Satisfaction Emergency
    if tickets >= 3 or csat <= 2:
        recommendations.append({
            "action_id": "vip_support_escalation",
            "title": "VIP Concierge Support Outreach",
            "urgency": "Immediate (Within 24 Hours)",
            "impact_level": "Critical",
            "cost": "$25 (Staff Time)",
            "description": f"Assign a Level-2 senior technical account specialist to audit recent {tickets} tickets, resolve chronic grievances, and restore trust.",
            "expected_churn_reduction": "25% - 35%",
            "field_overrides": {"SupportTickets": 1, "SatisfactionScore": 4}
        })

    # Priority 2: Contract Lock-in with Incentive
    if contract == "Month-to-month":
        recommendations.append({
            "action_id": "annual_contract_upgrade",
            "title": "Annual Contract Loyalty Incentive (15% Off 6 Mo)",
            "urgency": "High",
            "impact_level": "High",
            "cost": f"${monthly * 0.15 * 6:.2f} (Discount)",
            "description": "Convert month-to-month contract into a 1-year or 2-year plan by offering a 15% discount for the next 6 months.",
            "expected_churn_reduction": "40% - 50%",
            "field_overrides": {"Contract": "One year"}
        })

    # Priority 3: Tech Support & Security Bundle
    if internet == "Fiber optic" and tech_sup != "Yes":
        recommendations.append({
            "action_id": "complimentary_tech_support",
            "title": "Complimentary TechSupport & Online Security (90 Days)",
            "urgency": "Medium",
            "impact_level": "Medium-High",
            "cost": "$15 (Marginal COGS)",
            "description": "Add Premium 24/7 Tech Support and Cyber Protection free for 90 days to eliminate technical friction.",
            "expected_churn_reduction": "15% - 25%",
            "field_overrides": {"TechSupport": "Yes", "OnlineSecurity": "Yes"}
        })

    # Priority 4: Auto-Pay Frictionless Billing
    if payment == "Electronic check":
        recommendations.append({
            "action_id": "autopay_cashback_incentive",
            "title": "Enroll in Auto-Pay with $15 Statement Credit",
            "urgency": "Medium",
            "impact_level": "Moderate",
            "cost": "$15 (One-time credit)",
            "description": "Transition customer to automated credit card or bank debit billing, removing the friction of manual monthly bill paying.",
            "expected_churn_reduction": "10% - 15%",
            "field_overrides": {"PaymentMethod": "Bank transfer (automatic)"}
        })

    # Priority 5: High Bill Re-evaluation
    if monthly > 80.0:
        recommendations.append({
            "action_id": "tier_rightsizing",
            "title": "Value Optimization & Plan Modernization",
            "urgency": "Low-Medium",
            "impact_level": "Moderate",
            "cost": "$0 (Package rebalance)",
            "description": "Offer an updated loyalty bundle that replaces legacy surcharges with modern features, protecting monthly spend while increasing delight.",
            "expected_churn_reduction": "12% - 18%",
            "field_overrides": {"MonthlyCharges": round(monthly * 0.90, 2)}
        })

    # Fallback if already low risk or no trigger
    if not recommendations:
        recommendations.append({
            "action_id": "nps_engagement",
            "title": "Loyalty Appreciation & Anniversary Perk",
            "urgency": "Low",
            "impact_level": "Preventative",
            "cost": "$5 (Digital gift)",
            "description": "Deliver a customer appreciation note with a free speed boost or partner perk to maintain high satisfaction.",
            "expected_churn_reduction": "5% - 8%",
            "field_overrides": {"SatisfactionScore": 5}
        })

    return recommendations


def simulate_what_if_intervention(
    pipeline: Pipeline,
    customer: Dict[str, Any],
    modifications: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Simulates the quantitative impact of taking specific retention actions
    by updating the customer profile and re-evaluating the ML pipeline.
    """
    original_risk = predict_churn_risk(pipeline, customer)
    
    # Apply hypothetical interventions
    modified_customer = dict(customer)
    modified_customer.update(modifications)
    
    new_risk = predict_churn_risk(pipeline, modified_customer)
    
    orig_prob = original_risk["churn_probability"]
    new_prob = new_risk["churn_probability"]
    abs_reduction = orig_prob - new_prob
    rel_reduction = (abs_reduction / orig_prob * 100.0) if orig_prob > 0 else 0.0

    return {
        "original_probability": orig_prob,
        "new_probability": new_prob,
        "absolute_reduction": abs_reduction,
        "relative_reduction_pct": rel_reduction,
        "original_risk_tier": original_risk["risk_tier"],
        "new_risk_tier": new_risk["risk_tier"],
        "original_badge": original_risk["risk_badge"],
        "new_badge": new_risk["risk_badge"],
        "modified_attributes": modifications
    }


def calculate_retention_campaign_roi(
    at_risk_count: int,
    avg_mrr: float,
    intervention_cost_per_user: float,
    expected_save_rate_pct: float
) -> Dict[str, float]:
    """
    Computes financial ROI metrics for targeted retention outreach campaigns.
    """
    total_campaign_cost = at_risk_count * intervention_cost_per_user
    customers_saved = round(at_risk_count * (expected_save_rate_pct / 100.0))
    annual_clv_preserved_per_user = avg_mrr * 12.0
    total_annual_revenue_saved = customers_saved * annual_clv_preserved_per_user
    net_profit = total_annual_revenue_saved - total_campaign_cost
    roi_pct = (net_profit / total_campaign_cost * 100.0) if total_campaign_cost > 0 else 0.0

    return {
        "at_risk_count": at_risk_count,
        "total_campaign_cost": total_campaign_cost,
        "customers_saved": customers_saved,
        "annual_revenue_saved": total_annual_revenue_saved,
        "net_profit": net_profit,
        "roi_pct": roi_pct
    }
