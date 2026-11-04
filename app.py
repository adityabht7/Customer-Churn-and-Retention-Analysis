"""
Customer Churn & Retention Analytics Platform
Interactive Streamlit Application for Churn Prediction & Retention Strategy.
"""

import os
import io
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from src.data_loader import (
    load_churn_data, calculate_business_kpis, get_tenure_cohorts,
    CATEGORICAL_COLS, NUMERICAL_COLS
)
from src.churn_model import (
    load_or_train_default_model, train_and_benchmark_models,
    predict_churn_risk, batch_predict, FEATURE_COLS
)
from src.explainability import explain_customer_prediction
from src.retention_engine import (
    generate_retention_playbook, simulate_what_if_intervention,
    calculate_retention_campaign_roi
)
from src.segmentation import segment_customers, get_segment_metrics, SEGMENT_DEFINITIONS
from src.report_generator import (
    generate_prioritized_outreach_list, generate_executive_text_summary,
    generate_pdf_summary_report
)

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Customer Churn & Retention Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom CSS Styling
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Metric Cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        border: 1px solid #334155;
        padding: 16px 20px;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
    div[data-testid="stMetric"]:hover {
        border-color: #6366F1;
        transition: all 0.2s ease-in-out;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.85rem !important;
        font-weight: 600 !important;
        color: #94A3B8 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.85rem !important;
        font-weight: 700 !important;
        color: #F8FAFC !important;
    }
    /* Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre-wrap;
        border-radius: 8px 8px 0px 0px;
        padding: 10px 18px;
        font-weight: 600;
        font-size: 0.95rem;
    }
    /* Badges & Alerts */
    .badge-card {
        padding: 14px 18px;
        border-radius: 10px;
        margin-bottom: 12px;
        border-left: 5px solid;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# Data & Model Caching
# ---------------------------------------------------------
@st.cache_data
def get_benchmark_dataset():
    return load_churn_data()

@st.cache_resource
def get_champion_pipeline():
    return load_or_train_default_model()

@st.cache_data
def get_benchmark_results(df: pd.DataFrame):
    return train_and_benchmark_models(df)


# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("## **ChurnMetrics**")
    st.markdown("*Customer Retention Analytics*")
    st.divider()

    st.markdown("### Data Source")
    data_source = st.radio(
        "Choose Customer Dataset:",
        ["Telco Benchmark (7,043 Accounts)", "Upload Custom CSV"],
        index=0
    )

    uploaded_file = None
    if data_source == "Upload Custom CSV":
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
        if uploaded_file is not None:
            df = load_churn_data(uploaded_file)
            st.success(f"Loaded {len(df):,} accounts from uploaded CSV!")
        else:
            st.info("Awaiting CSV upload. Falling back to benchmark dataset.")
            df = get_benchmark_dataset()
    else:
        df = get_benchmark_dataset()

    st.divider()
    kpis = calculate_business_kpis(df)
    st.markdown("### Portfolio Summary")
    st.write(f"• **Churn Rate:** `{kpis['churn_rate']:.2f}%`")
    st.write(f"• **MRR at Risk:** `${kpis['mrr_at_risk']:,.0f}`")
    st.write(f"• **Avg Tenure:** `{kpis['avg_tenure']:.1f} months`")
    st.write(f"• **Avg CSAT:** `{kpis['avg_csat']:.2f} / 5.0`")

    st.divider()
    st.markdown("### Export Sample Data")
    csv_bytes = df.head(100).to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download 100-Row Sample CSV",
        data=csv_bytes,
        file_name="sample_customer_churn.csv",
        mime="text/csv",
        use_container_width=True
    )

# ---------------------------------------------------------
# Load Model & Pipeline
# ---------------------------------------------------------
pipeline = get_champion_pipeline()

# ---------------------------------------------------------
# Main App Header
# ---------------------------------------------------------
st.title("Customer Churn & Retention Analysis Platform")
st.markdown(
    "Predictive machine learning pipeline and analytics suite to evaluate attrition drivers, "
    "score customer risk, and simulate targeted retention interventions."
)

# ---------------------------------------------------------
# Navigation Tabs
# ---------------------------------------------------------
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "Dashboard",
    "Model Evaluation",
    "Risk Prediction",
    "Scenario Simulation",
    "Customer Segmentation",
    "Batch Inference",
    "ROI Model"
])


# =========================================================
# TAB 1: Executive KPI Dashboard & Diagnostics
# =========================================================
with tab1:
    st.subheader("Portfolio Health & Revenue Exposure")
    
    # Top KPI Metrics Row
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Customers", f"{kpis['total_customers']:,}")
    m2.metric("Portfolio Churn Rate", f"{kpis['churn_rate']:.2f}%")
    m3.metric("Total Monthly MRR", f"${kpis['total_mrr']:,.0f}")
    m4.metric("MRR Directly At Risk", f"${kpis['mrr_at_risk']:,.0f}", delta=f"-{kpis['mrr_risk_pct']:.1f}% MRR", delta_color="inverse")
    m5.metric("Avg Customer Lifetime Value", f"${kpis['avg_clv']:,.0f}")

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 1: Charts
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("#### Churn Rate by Contract Commitment")
        contract_churn = df.groupby("Contract", observed=False).agg(
            Total=("customerID", "count"),
            Churned=("Churn", lambda s: (s == "Yes").sum())
        ).reset_index()
        contract_churn["Churn_Rate"] = (contract_churn["Churned"] / contract_churn["Total"]) * 100

        fig_contract = px.bar(
            contract_churn,
            x="Contract",
            y="Churn_Rate",
            color="Churn_Rate",
            color_continuous_scale="Reds",
            text=contract_churn["Churn_Rate"].apply(lambda v: f"{v:.1f}%"),
            labels={"Churn_Rate": "Churn Rate (%)"}
        )
        fig_contract.update_layout(
            template="plotly_dark",
            margin=dict(l=20, r=20, t=30, b=20),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_contract, use_container_width=True)
        st.caption("Month-to-month contracts exhibit significantly higher churn compared to 1 or 2-year lock-in agreements.")

    with c2:
        st.markdown("#### Churn by Internet Service & Tech Support")
        tech_churn = df.groupby(["InternetService", "TechSupport"], observed=False).agg(
            Total=("customerID", "count"),
            Churned=("Churn", lambda s: (s == "Yes").sum())
        ).reset_index()
        tech_churn["Churn_Rate"] = (tech_churn["Churned"] / tech_churn["Total"]) * 100

        fig_tech = px.bar(
            tech_churn,
            x="InternetService",
            y="Churn_Rate",
            color="TechSupport",
            barmode="group",
            text=tech_churn["Churn_Rate"].apply(lambda v: f"{v:.1f}%"),
            labels={"Churn_Rate": "Churn Rate (%)", "InternetService": "Internet Type"}
        )
        fig_tech.update_layout(
            template="plotly_dark",
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_tech, use_container_width=True)
        st.caption("Fiber Optic users lacking dedicated Tech Support have the highest churn vulnerability.")

    st.markdown("<br>", unsafe_allow_html=True)

    # Row 2: Charts
    c3, c4 = st.columns(2)

    with c3:
        st.markdown("#### Tenure Cohort Retention Curve")
        cohorts = get_tenure_cohorts(df)
        fig_cohort = px.line(
            cohorts,
            x="Tenure_Cohort",
            y="Retention_Rate_%",
            markers=True,
            text=cohorts["Retention_Rate_%"].apply(lambda v: f"{v:.1f}%"),
            labels={"Retention_Rate_%": "Retention Rate (%)", "Tenure_Cohort": "Tenure Bucket"}
        )
        fig_cohort.update_traces(line_color="#10B981", line_width=4, marker_size=10)
        fig_cohort.update_layout(
            template="plotly_dark",
            margin=dict(l=20, r=20, t=30, b=20),
            yaxis_range=[0, 105]
        )
        st.plotly_chart(fig_cohort, use_container_width=True)
        st.caption("The critical attrition cliff occurs within the first 12 months. Surpassing month 24 establishes durable loyalty.")

    with c4:
        st.markdown("#### Churn Probability across Monthly Spend")
        fig_box = px.box(
            df,
            x="Churn",
            y="MonthlyCharges",
            color="Churn",
            color_discrete_map={"Yes": "#EF4444", "No": "#10B981"},
            points="outliers",
            labels={"MonthlyCharges": "Monthly Charges ($)", "Churn": "Churn Status"}
        )
        fig_box.update_layout(
            template="plotly_dark",
            margin=dict(l=20, r=20, t=30, b=20),
            showlegend=False
        )
        st.plotly_chart(fig_box, use_container_width=True)
        st.caption("Churned accounts have a significantly higher median monthly bill ($80+) without commensurate perceived value.")


# =========================================================
# TAB 2: Machine Learning Models & Benchmarking
# =========================================================
with tab2:
    st.subheader("Machine Learning Classification Benchmark")
    st.markdown(
        "Four candidate algorithms were trained using stratified cross-validation and evaluated on "
        "discriminative ability, recall (catching churners), precision (minimizing false alarms), and ROC-AUC."
    )

    with st.spinner("Benchmarking machine learning classifiers..."):
        benchmark = get_benchmark_results(df)
        bench_summary = benchmark["benchmark_summary"]
        results = benchmark["results"]

    # Model Leaderboard Table
    leaderboard_data = []
    for model_name, m in bench_summary["models"].items():
        leaderboard_data.append({
            "Model Name": model_name,
            "ROC-AUC": f"{m['roc_auc']:.4f}",
            "F1-Score": f"{m['f1_score']:.4f}",
            "Recall (Catch Rate)": f"{m['recall'] * 100:.2f}%",
            "Precision": f"{m['precision'] * 100:.2f}%",
            "Accuracy": f"{m['accuracy'] * 100:.2f}%",
            "Status": "Selected" if model_name == bench_summary["best_model_name"] else "Evaluated"
        })
    df_leaderboard = pd.DataFrame(leaderboard_data)
    st.dataframe(df_leaderboard, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ROC & PR Curves
    col_roc, col_feat = st.columns(2)

    with col_roc:
        st.markdown("#### ROC Curves Comparison")
        fig_roc = go.Figure()
        for name, r in results.items():
            fig_roc.add_trace(go.Scatter(
                x=r["roc_curve"]["fpr"],
                y=r["roc_curve"]["tpr"],
                mode='lines',
                name=f"{name} (AUC = {r['roc_auc']:.3f})"
            ))
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode='lines',
            line=dict(dash='dash', color='gray'),
            name="Random Guess"
        ))
        fig_roc.update_layout(
            template="plotly_dark",
            xaxis_title="False Positive Rate",
            yaxis_title="True Positive Rate (Recall)",
            margin=dict(l=20, r=20, t=30, b=20)
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    with col_feat:
        st.markdown("#### Top Churn Predictive Drivers (Global Feature Importance)")
        feat_df = pd.DataFrame(bench_summary["feature_importances"][:10])
        fig_feat = px.bar(
            feat_df,
            x="importance",
            y="feature",
            orientation="h",
            color="importance",
            color_continuous_scale="Purples",
            labels={"importance": "Predictive Weight", "feature": "Attribute"}
        )
        fig_feat.update_layout(
            template="plotly_dark",
            margin=dict(l=20, r=20, t=30, b=20),
            yaxis=dict(autorange="reversed"),
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_feat, use_container_width=True)


# =========================================================
# TAB 3: Individual Customer Risk Scorer
# =========================================================
with tab3:
    st.subheader("Individual Customer Risk Profiling & Local Attribution")
    st.markdown("Simulate or inspect any individual customer profile to obtain instant risk assessment and root-cause diagnostics.")

    # Preset profiles
    st.markdown("**Load Sample Profiles:**")
    p1, p2, p3 = st.columns(3)

    preset_customer = None
    if p1.button("Load High-Risk Profile", use_container_width=True):
        st.session_state["cust_tenure"] = 3
        st.session_state["cust_contract"] = "Month-to-month"
        st.session_state["cust_internet"] = "Fiber optic"
        st.session_state["cust_techsup"] = "No"
        st.session_state["cust_payment"] = "Electronic check"
        st.session_state["cust_monthly"] = 92.50
        st.session_state["cust_tickets"] = 4
        st.session_state["cust_csat"] = 1

    if p2.button("Load Retained Profile", use_container_width=True):
        st.session_state["cust_tenure"] = 60
        st.session_state["cust_contract"] = "Two year"
        st.session_state["cust_internet"] = "Fiber optic"
        st.session_state["cust_techsup"] = "Yes"
        st.session_state["cust_payment"] = "Bank transfer (automatic)"
        st.session_state["cust_monthly"] = 75.00
        st.session_state["cust_tickets"] = 0
        st.session_state["cust_csat"] = 5

    if p3.button("Load Onboarding Profile", use_container_width=True):
        st.session_state["cust_tenure"] = 6
        st.session_state["cust_contract"] = "Month-to-month"
        st.session_state["cust_internet"] = "DSL"
        st.session_state["cust_techsup"] = "No"
        st.session_state["cust_payment"] = "Credit card (automatic)"
        st.session_state["cust_monthly"] = 45.00
        st.session_state["cust_tickets"] = 1
        st.session_state["cust_csat"] = 3

    # Input Form
    with st.expander("Customer Feature Controls", expanded=True):
        col_in1, col_in2, col_in3 = st.columns(3)
        with col_in1:
            tenure_val = st.slider("Account Tenure (Months)", 1, 72, st.session_state.get("cust_tenure", 4))
            contract_val = st.selectbox("Contract Commitment", ["Month-to-month", "One year", "Two year"], index=["Month-to-month", "One year", "Two year"].index(st.session_state.get("cust_contract", "Month-to-month")))
            monthly_val = st.slider("Monthly Charges ($)", 18.0, 120.0, float(st.session_state.get("cust_monthly", 88.0)), step=0.5)

        with col_in2:
            internet_val = st.selectbox("Internet Service", ["Fiber optic", "DSL", "No"], index=["Fiber optic", "DSL", "No"].index(st.session_state.get("cust_internet", "Fiber optic")))
            techsup_val = st.selectbox("Tech Support Add-on", ["No", "Yes", "No internet service"], index=["No", "Yes", "No internet service"].index(st.session_state.get("cust_techsup", "No")))
            sec_val = st.selectbox("Online Security Add-on", ["No", "Yes", "No internet service"], index=0)

        with col_in3:
            payment_val = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"], index=["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"].index(st.session_state.get("cust_payment", "Electronic check")))
            tickets_val = st.slider("Support Tickets (Last 6 Mos)", 0, 6, st.session_state.get("cust_tickets", 3))
            csat_val = st.slider("Customer Satisfaction Score (1-5)", 1, 5, st.session_state.get("cust_csat", 2))

    # Build input dictionary
    sample_dict = {
        "gender": "Female",
        "SeniorCitizen": 0,
        "Partner": "No",
        "Dependents": "No",
        "tenure": tenure_val,
        "PhoneService": "Yes",
        "MultipleLines": "Yes" if internet_val != "No" else "No",
        "InternetService": internet_val,
        "OnlineSecurity": sec_val,
        "OnlineBackup": "No",
        "DeviceProtection": "No",
        "TechSupport": techsup_val,
        "StreamingTV": "Yes" if internet_val != "No" else "No internet service",
        "StreamingMovies": "Yes" if internet_val != "No" else "No internet service",
        "Contract": contract_val,
        "PaperlessBilling": "Yes",
        "PaymentMethod": payment_val,
        "MonthlyCharges": monthly_val,
        "TotalCharges": monthly_val * tenure_val,
        "SupportTickets": tickets_val,
        "UsageFrequency": "Daily",
        "SatisfactionScore": csat_val
    }

    # Predict Risk
    risk_res = predict_churn_risk(pipeline, sample_dict)
    prob_pct = risk_res["churn_probability"] * 100.0

    st.markdown("<br>", unsafe_allow_html=True)
    gauge_col, explain_col = st.columns([1, 2])

    with gauge_col:
        st.markdown("#### Real-time Churn Gauge")
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob_pct,
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"{risk_res['risk_badge']} {risk_res['risk_tier']}", 'font': {'size': 20, 'color': risk_res['risk_color']}},
            number={'suffix': "%", 'font': {'size': 36}},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': risk_res['risk_color']},
                'steps': [
                    {'range': [0, 30], 'color': "#064E3B"},
                    {'range': [30, 65], 'color': "#78350F"},
                    {'range': [65, 100], 'color': "#7F1D1D"}
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 3},
                    'thickness': 0.75,
                    'value': prob_pct
                }
            }
        ))
        fig_gauge.update_layout(
            template="plotly_dark",
            margin=dict(l=20, r=20, t=40, b=20),
            height=280
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

    with explain_col:
        st.markdown("#### Risk Drivers and Mitigating Factors")
        exp = explain_customer_prediction(sample_dict)

        if exp["top_risk_drivers"]:
            st.markdown("**Primary Risk Factors:**")
            for d in exp["top_risk_drivers"]:
                st.markdown(f"- **{d['feature']}** (+{d['impact']*100:.1f}% risk impact): {d['explanation']}")
        
        if exp["top_retention_anchors"]:
            st.markdown("**Protective Retention Factors:**")
            for a in exp["top_retention_anchors"]:
                st.markdown(f"- **{a['feature']}** ({a['impact']*100:.1f}% risk mitigation): {a['explanation']}")


# =========================================================
# TAB 4: What-If Intervention Simulator
# =========================================================
with tab4:
    st.subheader("Scenario Simulation & Retention Strategy")
    st.markdown(
        "Evaluate proactive retention playbooks on this customer profile. "
        "Test how contract upgrades, technical support escalations, and payment automation mitigate churn probability."
    )

    playbooks = generate_retention_playbook(sample_dict, risk_res["churn_probability"])
    
    st.markdown("### Recommended Interventions")
    for pb in playbooks:
        with st.container():
            st.markdown(f"""
            <div style="background-color: #1E293B; border-left: 5px solid #6366F1; padding: 16px; border-radius: 8px; margin-bottom: 12px;">
                <div style="display: flex; justify-content: space-between; align-items: center;">
                    <h4 style="margin: 0; color: #F8FAFC;">{pb['title']}</h4>
                    <span style="background-color: #312E81; color: #A5B4FC; padding: 4px 10px; border-radius: 12px; font-size: 0.8rem; font-weight: bold;">{pb['urgency']}</span>
                </div>
                <p style="margin: 8px 0 4px 0; color: #CBD5E1;">{pb['description']}</p>
                <div style="display: flex; gap: 20px; font-size: 0.85rem; color: #94A3B8; margin-top: 6px;">
                    <span><b>Cost:</b> {pb['cost']}</span>
                    <span><b>Expected Churn Reduction:</b> <span style="color: #10B981;">{pb['expected_churn_reduction']}</span></span>
                    <span><b>Impact:</b> {pb['impact_level']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### Intervention Simulation")

    s1, s2, s3 = st.columns(3)
    with s1:
        sim_contract = st.selectbox("Intervention: Contract Upgrade", [sample_dict["Contract"], "One year", "Two year"], index=1 if sample_dict["Contract"] == "Month-to-month" else 0)
    with s2:
        sim_tech = st.selectbox("Intervention: Add TechSupport", [sample_dict["TechSupport"], "Yes"], index=1 if sample_dict["TechSupport"] != "Yes" else 0)
    with s3:
        sim_payment = st.selectbox("Intervention: Switch to Auto-Pay", [sample_dict["PaymentMethod"], "Bank transfer (automatic)"], index=1 if "automatic" not in sample_dict["PaymentMethod"] else 0)

    modifications = {
        "Contract": sim_contract,
        "TechSupport": sim_tech,
        "PaymentMethod": sim_payment,
        "SupportTickets": max(0, sample_dict["SupportTickets"] - 2),
        "SatisfactionScore": min(5, sample_dict["SatisfactionScore"] + 2)
    }

    sim_res = simulate_what_if_intervention(pipeline, sample_dict, modifications)

    res_col1, res_col2, res_col3 = st.columns(3)
    res_col1.metric("Original Churn Risk", f"{sim_res['original_probability']*100:.1f}%", f"{sim_res['original_badge']} {sim_res['original_risk_tier']}")
    res_col2.metric("Post-Intervention Risk", f"{sim_res['new_probability']*100:.1f}%", f"{sim_res['new_badge']} {sim_res['new_risk_tier']}")
    res_col3.metric("Net Churn Reduction", f"-{sim_res['absolute_reduction']*100:.1f}%", delta=f"{sim_res['relative_reduction_pct']:.1f}% relative drop", delta_color="normal")


# =========================================================
# TAB 5: Customer Segmentation & Behavioral Personas
# =========================================================
with tab5:
    st.subheader("Customer Personas & Behavioral Clusters")
    st.markdown("Customers are categorized into 4 actionable operational personas to steer targeted retention campaigns.")

    df_segmented = segment_customers(df)
    seg_metrics = get_segment_metrics(df_segmented)

    # Persona Cards
    p_cols = st.columns(4)
    for i, (name, meta) in enumerate(SEGMENT_DEFINITIONS.items()):
        with p_cols[i]:
            st.markdown(f"""
            <div style="background-color: #1E293B; border-top: 4px solid #6366F1; padding: 14px; border-radius: 8px; min-height: 200px;">
                <h4 style="margin: 0; color: #F8FAFC;">{name}</h4>
                <p style="font-size: 0.85rem; color: #94A3B8; margin-top: 8px;">{meta['description']}</p>
                <hr style="border-color: #334155; margin: 8px 0;">
                <p style="font-size: 0.8rem; color: #CBD5E1;"><b>Strategy:</b> {meta['strategy']}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c_seg1, c_seg2 = st.columns([1, 1])

    with c_seg1:
        st.markdown("#### Segment Population Share")
        fig_pie = px.pie(
            seg_metrics,
            names="Customer_Segment",
            values="Total_Customers",
            hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_pie.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=30, b=20))
        st.plotly_chart(fig_pie, use_container_width=True)

    with c_seg2:
        st.markdown("#### Segment Revenue Exposure (MRR)")
        fig_mrr = px.bar(
            seg_metrics,
            x="Customer_Segment",
            y="Total_MRR",
            color="Customer_Segment",
            text=seg_metrics["Total_MRR"].apply(lambda v: f"${v:,.0f}"),
            labels={"Total_MRR": "Total MRR ($)", "Customer_Segment": "Persona"}
        )
        fig_mrr.update_layout(template="plotly_dark", margin=dict(l=20, r=20, t=30, b=20), showlegend=False)
        st.plotly_chart(fig_mrr, use_container_width=True)

    st.markdown("#### Detailed Segment Comparative Analysis")
    st.dataframe(seg_metrics, use_container_width=True, hide_index=True)


# =========================================================
# TAB 6: Batch Prediction & Actionable Outreach Export
# =========================================================
with tab6:
    st.subheader("Batch Customer Scoring & Actionable Outreach Export")
    st.markdown("Run automated ML scoring across all customer accounts to generate prioritized outreach lists for Customer Success teams.")

    scored_df = batch_predict(pipeline, df)

    # Filter Controls
    f_col1, f_col2 = st.columns([1, 2])
    with f_col1:
        selected_risk = st.multiselect("Filter by Risk Tier", ["High Risk", "Moderate Risk", "Low Risk"], default=["High Risk"])
    with f_col2:
        min_monthly = st.slider("Minimum Monthly Spend ($)", 0.0, 120.0, 50.0)

    filtered_scored = scored_df[
        (scored_df["Risk_Tier"].isin(selected_risk)) &
        (scored_df["MonthlyCharges"] >= min_monthly)
    ]

    outreach_df = generate_prioritized_outreach_list(filtered_scored, top_n=200)

    st.write(f"Displaying **{len(outreach_df):,}** high-priority accounts matching criteria:")
    st.dataframe(outreach_df, use_container_width=True, hide_index=True)

    # Downloads
    st.markdown("### Export Reports & Outreach Lists")
    d1, d2, d3 = st.columns(3)

    outreach_csv = outreach_df.to_csv(index=False).encode("utf-8")
    d1.download_button(
        "Download Outreach CSV",
        data=outreach_csv,
        file_name="prioritized_retention_call_list.csv",
        mime="text/csv",
        use_container_width=True
    )

    pdf_bytes = generate_pdf_summary_report(kpis, bench_summary)
    d2.download_button(
        "Download Executive PDF Brief",
        data=pdf_bytes,
        file_name="Customer_Churn_Executive_Brief.pdf",
        mime="application/pdf",
        use_container_width=True
    )

    roi_temp = calculate_retention_campaign_roi(len(filtered_scored), kpis["avg_mrr"], 15.0, 35.0)
    md_summary = generate_executive_text_summary(kpis, bench_summary, roi_temp)
    d3.download_button(
        "Download Executive Markdown",
        data=md_summary.encode("utf-8"),
        file_name="Retention_Executive_Report.md",
        mime="text/markdown",
        use_container_width=True
    )


# =========================================================
# TAB 7: Retention Campaign ROI Calculator
# =========================================================
with tab7:
    st.subheader("Financial ROI & Retention Campaign Budget Model")
    st.markdown("Quantify the net financial return of proactive retention campaigns before deploying budget.")

    r_col1, r_col2 = st.columns([1, 1])

    with r_col1:
        st.markdown("#### Campaign Assumptions")
        target_cohort = st.slider("Targeted At-Risk Customer Count", 50, 2000, 500, step=25)
        spend_per_customer = st.slider("Cost per Retention Intervention Offer ($)", 5.0, 100.0, 20.0, step=5.0)
        expected_save_rate = st.slider("Anticipated Save Rate (%)", 10.0, 70.0, 35.0, step=5.0)
        cohort_mrr = st.number_input("Average Customer MRR ($)", value=round(kpis['avg_mrr'], 2))

    roi_res = calculate_retention_campaign_roi(target_cohort, cohort_mrr, spend_per_customer, expected_save_rate)

    with r_col2:
        st.markdown("#### Projected Campaign Returns")
        st.metric("Total Campaign Budget Required", f"${roi_res['total_campaign_cost']:,.0f}")
        st.metric("Projected Preserved Customers", f"{roi_res['customers_saved']:,} accounts")
        st.metric("Annual Saved Recurring Revenue", f"${roi_res['annual_revenue_saved']:,.0f}")
        st.metric("Net Bottom-Line Profit", f"${roi_res['net_profit']:,.0f}", delta=f"{roi_res['roi_pct']:.1f}% ROI", delta_color="normal")

    st.markdown("<br>", unsafe_allow_html=True)
    st.info(
        f"**Summary:** Investing **${roi_res['total_campaign_cost']:,.0f}** in targeted interventions "
        f"saves **{roi_res['customers_saved']}** high-risk accounts, preserving **${roi_res['annual_revenue_saved']:,.0f}** in annual revenue "
        f"for an estimated net return of **${roi_res['net_profit']:,.0f}** (**{roi_res['roi_pct']:.1f}% ROI**)."
    )

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #64748B; font-size: 0.85rem;'>"
    "Customer Churn & Retention Analytics Platform • Built with Python & Streamlit"
    "</div>",
    unsafe_allow_html=True
)
