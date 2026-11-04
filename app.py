"""AuraVibe IIoT: Plant Asset Health Monitoring & Predictive Maintenance Copilot.
Industrial condition monitoring, ISO 10816 vibration diagnostics, RUL prognostics, and CMMS dispatch.
"""

from __future__ import annotations
import os
import sys
import json
import textwrap
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st

# Configure Streamlit page layout
st.set_page_config(
    page_title="AuraVibe IIoT | Plant Reliability Platform",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Helper to render clean un-indented HTML/CSS when needed
def render_html(html_str: str):
    clean = textwrap.dedent(html_str).strip()
    if hasattr(st, "html"):
        st.html(clean)
    else:
        st.markdown(clean, unsafe_allow_html=True)


# Imports from src
from src.core.constants import (
    FLEET_ASSETS,
    FAILURE_MODES,
    ISO_10816_ZONES,
    FEATURE_COLUMNS,
)
from src.core.schema import TelemetryReading, AssetHealthState
from src.telemetry.generator import TelemetrySimulator
from src.telemetry.signal_processing import compute_spectral_harmonics, calculate_iso_zone
from src.models.pipeline import PredictiveMaintenancePipeline
from src.diagnostics.root_cause import RootCauseAnalyzer
from src.diagnostics.copilot_engine import MaintenanceCopilotEngine
from src.diagnostics.economics import DowntimeEconomicsCalculator
from src.reports.pdf_exporter import generate_work_order_pdf


@st.cache_resource
def load_core_engine():
    """Initializes and caches the predictive pipeline and telemetry simulator."""
    model_path = "models/predictive_maintenance_pipeline.joblib"
    if not os.path.exists(model_path):
        from scripts.train_models import run_training_pipeline
        run_training_pipeline()

    pipeline = PredictiveMaintenancePipeline.load(model_path)
    simulator = TelemetrySimulator(seed=42)
    copilot = MaintenanceCopilotEngine()
    root_cause = RootCauseAnalyzer()
    economics = DowntimeEconomicsCalculator()
    return pipeline, simulator, copilot, root_cause, economics


pipeline, simulator, copilot, root_cause, economics = load_core_engine()

# Initialize session state for real-time simulation
if "simulation_step" not in st.session_state:
    st.session_state.simulation_step = 0
if "active_asset_id" not in st.session_state:
    st.session_state.active_asset_id = "PUMP-301A"
if "injected_fault" not in st.session_state:
    st.session_state.injected_fault = 0
if "operational_stress" not in st.session_state:
    st.session_state.operational_stress = 1.0
if "copilot_history" not in st.session_state:
    st.session_state.copilot_history = []

# --- SIDEBAR: Operational Controls & SCADA Configuration ---
with st.sidebar:
    st.header("Plant Telemetry Controls")
    st.caption("Process Unit: **Bay 3 Hydrocarbon Extraction Circuit**  \nCondition Standard: **ISO 10816-3 Class III**")

    asset_options = [f"{a['asset_id']} — {a['asset_name']}" for a in FLEET_ASSETS]
    current_asset_index = [a["asset_id"] for a in FLEET_ASSETS].index(st.session_state.active_asset_id)
    selected_asset_str = st.selectbox("Active Equipment Target", asset_options, index=current_asset_index)
    selected_asset_id = selected_asset_str.split(" — ")[0]
    st.session_state.active_asset_id = selected_asset_id

    st.divider()
    st.subheader("Scenario & Fault Injection")

    preset = st.selectbox(
        "Load Diagnostic Scenario Preset",
        [
            "Nominal Stable Operation",
            "Bearing Outer Race Spalling (High Shock Kurtosis)",
            "Dynamic Rotor Misalignment (1X/2X Harmonics)",
            "Fluid Cavitation & Impeller Degradation",
            "Lubrication Breakdown & Thermal Runaway",
        ],
        index=0,
    )

    preset_map = {
        "Nominal Stable Operation": (0, 1.0),
        "Bearing Outer Race Spalling (High Shock Kurtosis)": (1, 1.25),
        "Dynamic Rotor Misalignment (1X/2X Harmonics)": (2, 1.20),
        "Fluid Cavitation & Impeller Degradation": (3, 1.30),
        "Lubrication Breakdown & Thermal Runaway": (4, 1.35),
    }

    injected_fault_mode, default_stress = preset_map[preset]
    stress_val = st.slider("Dynamic Operational Stress", min_value=0.8, max_value=1.6, value=float(default_stress), step=0.05)
    st.session_state.injected_fault = injected_fault_mode
    st.session_state.operational_stress = stress_val

    st.divider()
    col_tick, col_reset = st.columns(2)
    with col_tick:
        if st.button("Poll Telemetry ↻", use_container_width=True):
            st.session_state.simulation_step += 1
    with col_reset:
        if st.button("Reset Baseline", use_container_width=True):
            st.session_state.simulation_step = 0
            st.session_state.injected_fault = 0
            st.session_state.operational_stress = 1.0
            st.rerun()

    st.caption(f"SCADA Scan Cycle: #{st.session_state.simulation_step} | ISO 13374 Verified")

# Generate current asset telemetry and assessment
current_reading = simulator.generate_single_reading(
    asset_id=st.session_state.active_asset_id,
    operational_stress=st.session_state.operational_stress,
    injected_fault=st.session_state.injected_fault,
    operational_hours=8420.0 + st.session_state.simulation_step * 2.0,
)
current_health = pipeline.assess_telemetry(current_reading)
current_work_order = copilot.generate_work_order(current_reading, current_health)
current_roi = economics.calculate_intervention_roi(st.session_state.active_asset_id, current_work_order)

# --- TOP NAVIGATION & HEADER ---
col_head1, col_head2 = st.columns([3, 1])
with col_head1:
    st.title("AuraVibe IIoT | Plant Reliability & Machinery Diagnostics")
    st.caption("Continuous Telemetry Acquisition • ISO 10816-3 Vibration Severity • RUL Prognostics • Automated CMMS Dispatch")
with col_head2:
    st.write("")
    if current_health.status_label == "NORMAL":
        st.success(f"**STATUS: NORMAL**  \nTarget: `{current_reading.asset_id}`")
    elif current_health.status_label == "ADVISORY":
        st.info(f"**STATUS: ADVISORY**  \nTarget: `{current_reading.asset_id}`")
    elif current_health.status_label == "WARNING":
        st.warning(f"**STATUS: WARNING**  \nTarget: `{current_reading.asset_id}`")
    else:
        st.error(f"**STATUS: CRITICAL**  \nTarget: `{current_reading.asset_id}`")

# --- TOP KPI METRICS BAR ---
kpi1, kpi2, kpi3, kpi4, kpi5 = st.columns(5)
with kpi1:
    st.metric(
        label="Equipment Health Index",
        value=f"{current_health.health_index:.1f}%",
        delta=f"{current_health.health_index - 100:.1f}% vs baseline",
        delta_color="normal",
    )
with kpi2:
    st.metric(
        label="ISO 10816-3 Severity",
        value=f"Zone {current_health.iso_severity_zone[-1]}",
        delta=f"{current_reading.vibration_de_rms:.2f} mm/s RMS",
        delta_color="off",
    )
with kpi3:
    st.metric(
        label="Forecasted RUL",
        value=f"{current_health.predicted_rul_hrs:.0f} hrs",
        delta=f"[{current_health.rul_confidence_lower_hrs:.0f}–{current_health.rul_confidence_upper_hrs:.0f}h 90% CI]",
        delta_color="off",
    )
with kpi4:
    st.metric(
        label="Diagnosed Mode",
        value=current_health.predicted_failure_mode_name[:18],
        delta=f"{current_health.failure_probability*100:.0f}% confidence",
        delta_color="off",
    )
with kpi5:
    st.metric(
        label="Avoided Outage Loss",
        value=f"${current_roi['net_savings_usd']:,.0f}",
        delta=f"ROI: {current_roi['roi_percentage']:.0f}%",
        delta_color="normal",
    )

st.write("")

# --- APPLICATION TABS ---
tab_fleet, tab_twin, tab_ml, tab_copilot, tab_roi, tab_batch = st.tabs([
    "Fleet SCADA Overview",
    "Digital Twin & Telemetry",
    "Model Diagnostics & Prognostics",
    "Machinery Diagnostics & CMMS",
    "Downtime Economics & ROI",
    "Batch Telemetry Export",
])

# =========================================================================
# TAB 1: FLEET SCADA OVERVIEW
# =========================================================================
with tab_fleet:
    st.subheader("Plant-Wide Asset Condition Matrix")
    st.caption("Real-time synchronized condition monitoring across all monitored rotating assets in Processing Unit 3.")

    # Generate telemetry for all 6 fleet assets
    fleet_rows = []
    for asset in FLEET_ASSETS:
        aid = asset["asset_id"]
        a_fault = st.session_state.injected_fault if aid == st.session_state.active_asset_id else 0
        a_stress = st.session_state.operational_stress if aid == st.session_state.active_asset_id else 1.0
        r = simulator.generate_single_reading(aid, operational_stress=a_stress, injected_fault=a_fault)
        h = pipeline.assess_telemetry(r)
        fleet_rows.append({
            "Asset ID": aid,
            "Equipment Name": asset["asset_name"],
            "Asset Type": asset["asset_type"],
            "Criticality": asset["criticality"],
            "DE Vibe (mm/s)": r.vibration_de_rms,
            "Bearing Temp (°C)": r.bearing_temperature_c,
            "Delta P (bar)": r.pressure_differential_bar,
            "ISO Zone": h.iso_severity_zone,
            "Health Index (%)": h.health_index,
            "Diagnosis": h.predicted_failure_mode_name,
            "RUL (hrs)": h.predicted_rul_hrs,
            "Status": h.status_label,
        })
    df_fleet = pd.DataFrame(fleet_rows)

    # Fleet summary cards
    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        normal_cnt = len(df_fleet[df_fleet["Status"] == "NORMAL"])
        st.metric("Healthy Assets", f"{normal_cnt} / {len(df_fleet)}")
    with fc2:
        advisory_cnt = len(df_fleet[df_fleet["Status"] == "ADVISORY"])
        st.metric("Advisory Alerts", advisory_cnt)
    with fc3:
        warn_cnt = len(df_fleet[df_fleet["Status"].isin(["WARNING", "CRITICAL"])])
        st.metric("Warning / Critical", warn_cnt)
    with fc4:
        avg_fleet_health = df_fleet["Health Index (%)"].mean()
        st.metric("Fleet Average Health", f"{avg_fleet_health:.1f}%")

    # Fleet status table formatted for SCADA display
    df_fleet_display = pd.DataFrame([
        {
            "Asset ID": r["Asset ID"],
            "Equipment Name": r["Equipment Name"],
            "Type": r["Asset Type"],
            "DE Vibe RMS": f"{r['DE Vibe (mm/s)']:.2f} mm/s",
            "Bearing Temp": f"{r['Bearing Temp (°C)']:.1f} °C",
            "Delta-P": f"{r['Delta P (bar)']:.2f} bar",
            "ISO 10816": r["ISO Zone"],
            "Health Index": f"{r['Health Index (%)']:.1f}%",
            "Diagnosed Mode": r["Diagnosis"],
            "RUL Forecast": f"{r['RUL (hrs)']:.0f} hrs",
            "Status": r["Status"],
        }
        for r in fleet_rows
    ])
    st.table(df_fleet_display)

    st.subheader("ISO 10816-3 Machinery Vibration Severity Reference")
    iso_cols = st.columns(4)
    zones = [
        ("Zone A (0.1 – 2.8 mm/s)", "Good / Newly Commissioned", "Permissible for unrestricted continuous long-term operation."),
        ("Zone B (2.8 – 4.5 mm/s)", "Satisfactory Condition", "Acceptable for continuous unrestricted industrial service."),
        ("Zone C (4.5 – 7.1 mm/s)", "Alert / Restricted", "Unsatisfactory for continuous duty. Schedule maintenance window."),
        ("Zone D (> 7.1 mm/s)", "Critical / Damage Imminent", "Vibration severity sufficient to cause mechanical failure. Interlock trip."),
    ]
    for idx, (title, sub, desc) in enumerate(zones):
        with iso_cols[idx]:
            with st.container(border=True):
                st.markdown(f"**{title}**")
                st.caption(f"_{sub}_\n\n{desc}")

# =========================================================================
# TAB 2: DIGITAL TWIN & TELEMETRY
# =========================================================================
with tab_twin:
    st.subheader(f"Digital Twin Sensor Telemetry: {current_reading.asset_id}")
    st.caption("Multi-channel continuous streaming telemetry and fast Fourier transform (FFT) harmonic spectral analysis.")

    twin_col1, twin_col2 = st.columns([1.6, 1])

    with twin_col1:
        trend_df = simulator.generate_degradation_trajectory(
            asset_id=st.session_state.active_asset_id,
            failure_mode=st.session_state.injected_fault,
            duration_hours=72,
        )

        fig_trend = make_subplots(
            rows=2, cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1,
            subplot_titles=("Vibration Velocity RMS (Drive-End vs Non-Drive-End)", "Thermal & Hydraulic Dynamics"),
        )

        fig_trend.add_trace(
            go.Scatter(y=trend_df["vibration_de_rms"], mode="lines", name="DE Vibe RMS (mm/s)", line=dict(color="#38bdf8", width=2)),
            row=1, col=1,
        )
        fig_trend.add_trace(
            go.Scatter(y=trend_df["vibration_nde_rms"], mode="lines", name="NDE Vibe RMS (mm/s)", line=dict(color="#94a3b8", width=1.5, dash="dot")),
            row=1, col=1,
        )
        fig_trend.add_hline(y=4.5, line_dash="dash", line_color="#f59e0b", annotation_text="ISO Zone C (4.5)", row=1, col=1)
        fig_trend.add_hline(y=7.1, line_dash="dash", line_color="#ef4444", annotation_text="ISO Zone D (7.1)", row=1, col=1)

        fig_trend.add_trace(
            go.Scatter(y=trend_df["bearing_temperature_c"], mode="lines", name="Bearing Temp (°C)", line=dict(color="#f97316", width=2)),
            row=2, col=1,
        )
        fig_trend.add_trace(
            go.Scatter(y=trend_df["pressure_differential_bar"], mode="lines", name="Delta-P (bar)", line=dict(color="#a855f7", width=1.5)),
            row=2, col=1,
        )

        fig_trend.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0d131f",
            plot_bgcolor="#0d131f",
            height=420,
            margin=dict(l=40, r=20, t=35, b=30),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_trend, use_container_width=True)

    with twin_col2:
        freqs, spectrum = compute_spectral_harmonics(
            rpm=current_reading.rotational_speed_rpm,
            vibration_rms=current_reading.vibration_de_rms,
        )
        fig_fft = go.Figure()
        fig_fft.add_trace(
            go.Scatter(
                x=freqs[:280],
                y=spectrum[:280],
                mode="lines",
                fill="tozeroy",
                line=dict(color="#34d399", width=1.5),
                fillcolor="rgba(52, 211, 153, 0.15)",
                name="Vibration FFT",
            )
        )
        f0 = current_reading.rotational_speed_rpm / 60.0
        fig_fft.add_vline(x=f0, line_dash="dot", line_color="#38bdf8", annotation_text="1X Running")
        fig_fft.add_vline(x=2 * f0, line_dash="dot", line_color="#f59e0b", annotation_text="2X Harmonic")

        fig_fft.update_layout(
            title=dict(text="Fast Fourier Transform (FFT) Spectral Density", font=dict(size=13, color="#f8fafc")),
            xaxis_title="Frequency (Hz)",
            yaxis_title="Amplitude (mm/s)",
            template="plotly_dark",
            paper_bgcolor="#0d131f",
            plot_bgcolor="#0d131f",
            height=420,
            margin=dict(l=40, r=20, t=45, b=40),
        )
        st.plotly_chart(fig_fft, use_container_width=True)

    st.subheader("Real-Time Sensor Telemetry Snapshot")
    s_col1, s_col2, s_col3, s_col4 = st.columns(4)
    with s_col1:
        st.metric("Rotational Speed", f"{current_reading.rotational_speed_rpm:.0f} RPM")
        st.metric("Acoustic Emission", f"{current_reading.acoustic_emission_db:.1f} dB")
    with s_col2:
        st.metric("Vibration Kurtosis", f"{current_reading.vibration_kurtosis:.2f}", delta=f"{current_reading.vibration_kurtosis - 3.0:+.2f} vs norm")
        st.metric("Motor Current", f"{current_reading.motor_current_amps:.1f} A")
    with s_col3:
        st.metric("Crest Factor", f"{current_reading.vibration_crest_factor:.2f}")
        st.metric("Hydraulic Line Pressure", f"{current_reading.hydraulic_pressure_bar:.1f} bar")
    with s_col4:
        st.metric("Motor Winding Temp", f"{current_reading.motor_winding_temperature_c:.1f} °C")
        st.metric("Filter Delta-P", f"{current_reading.pressure_differential_bar:.2f} bar")

# =========================================================================
# TAB 3: ML DIAGNOSTICS & PROGNOSTICS
# =========================================================================
with tab_ml:
    st.subheader("Machine Learning Model Diagnostics & Prognostics")
    st.caption("Multi-class failure mode classification probabilities, RUL regression with confidence bounds, and benchmark comparisons.")

    ml_col1, ml_col2 = st.columns(2)

    with ml_col1:
        df_single = pd.DataFrame([current_reading.to_dict()])[FEATURE_COLUMNS]
        fault_probs = pipeline.classifier.predict_proba(df_single)[0]

        prob_data = []
        for mode_id, prob in enumerate(fault_probs):
            prob_data.append({
                "Failure Mode": FAILURE_MODES[mode_id]["name"],
                "Probability": prob * 100.0,
                "Category": FAILURE_MODES[mode_id]["category"],
            })
        df_probs = pd.DataFrame(prob_data)

        fig_prob = px.bar(
            df_probs,
            x="Probability",
            y="Failure Mode",
            orientation="h",
            color="Probability",
            color_continuous_scale="Tealgrn",
            text="Probability",
            title="Failure Mode Posterior Probabilities (%)",
        )
        fig_prob.update_traces(texttemplate="%{text:.1f}%", textposition="outside")
        fig_prob.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0d131f",
            plot_bgcolor="#0d131f",
            height=320,
            margin=dict(l=10, r=20, t=40, b=20),
            coloraxis_showscale=False,
            xaxis=dict(range=[0, 115]),
        )
        st.plotly_chart(fig_prob, use_container_width=True)

    with ml_col2:
        traj = simulator.generate_degradation_trajectory(
            asset_id=st.session_state.active_asset_id,
            failure_mode=current_health.predicted_failure_mode_id,
            duration_hours=120,
        )

        future_steps = np.arange(0, int(current_health.predicted_rul_hrs) + 24, 6)
        future_rul = np.maximum(0, current_health.predicted_rul_hrs - future_steps)
        future_lower = np.maximum(0, current_health.rul_confidence_lower_hrs - future_steps)
        future_upper = current_health.rul_confidence_upper_hrs - future_steps

        fig_rul = go.Figure()
        fig_rul.add_trace(
            go.Scatter(x=future_steps, y=future_upper, mode="lines", line=dict(width=0), showlegend=False)
        )
        fig_rul.add_trace(
            go.Scatter(
                x=future_steps,
                y=future_lower,
                mode="lines",
                fill="tonexty",
                fillcolor="rgba(56, 189, 248, 0.15)",
                line=dict(width=0),
                name="90% Confidence Interval",
            )
        )
        fig_rul.add_trace(
            go.Scatter(
                x=future_steps,
                y=future_rul,
                mode="lines+markers",
                line=dict(color="#38bdf8", width=2.5),
                name="RUL Degradation Forecast",
            )
        )
        fig_rul.add_hline(y=48.0, line_dash="dash", line_color="#ef4444", annotation_text="Critical PM Threshold (48h)")

        fig_rul.update_layout(
            title=dict(text=f"RUL Prognostic Trajectory (Forecast: {current_health.predicted_rul_hrs:.0f} hrs)", font=dict(size=13, color="#f8fafc")),
            xaxis_title="Operating Hours Forward",
            yaxis_title="Estimated RUL (Hours)",
            template="plotly_dark",
            paper_bgcolor="#0d131f",
            plot_bgcolor="#0d131f",
            height=320,
            margin=dict(l=40, r=20, t=40, b=40),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        )
        st.plotly_chart(fig_rul, use_container_width=True)

    st.subheader("Model Architecture Benchmark & Cross-Validation Results")
    metrics_path = "models/benchmark_metrics.json"
    if os.path.exists(metrics_path):
        with open(metrics_path, "r") as f:
            bench_data = json.load(f)

        bench_rows = []
        for cand, metrics in bench_data.get("classification_benchmarks", {}).items():
            bench_rows.append({
                "Model Pipeline": cand.replace("_", " ").title(),
                "Task": "Multi-Class Failure Classification",
                "Macro F1": f"{metrics['macro_f1_mean']:.4f}",
                "Balanced Accuracy": f"{metrics['balanced_acc_mean']:.4f}",
                "ROC-AUC (OVR)": f"{metrics['roc_auc_mean']:.4f}",
                "Status": "Selected (Production)" if cand == "hist_gradient_boosting" else "Evaluated",
            })
        for cand, metrics in bench_data.get("rul_benchmarks", {}).items():
            bench_rows.append({
                "Model Pipeline": cand.replace("_", " ").title(),
                "Task": "Remaining Useful Life (RUL) Regression",
                "Macro F1": f"RMSE: {metrics['rmse_hours']:.1f}h",
                "Balanced Accuracy": f"MAE: {metrics['mae_hours']:.1f}h",
                "ROC-AUC (OVR)": f"R²: {metrics['r2_score']:.4f}",
                "Status": "Selected (Production)" if cand == "gradient_boosting" else "Evaluated",
            })
        st.table(pd.DataFrame(bench_rows))

# =========================================================================
# TAB 4: MACHINERY DIAGNOSTICS & CMMS
# =========================================================================
with tab_copilot:
    st.subheader("Field Machinery Diagnostics & CMMS Dispatch")
    st.caption("Condition-based prescriptive guidance, root-cause attribution, and formal ISO 13374 work order dispatch.")

    cop_col1, cop_col2 = st.columns([1.1, 1])

    with cop_col1:
        st.subheader("Diagnostic Query Console")
        prompt_options = [
            f"What is the root cause diagnosis for {current_reading.asset_id}?",
            f"What is the forecasted remaining useful life and time-to-trip?",
            f"What OEM spare parts and BOM must be staged for repair?",
            f"What are the mandatory Lockout/Tagout (LOTO) safety protocols?",
        ]
        preset_prompt = st.selectbox("Select Engineering Query", prompt_options)
        custom_prompt = st.text_input("Or enter custom maintenance inquiry:", value=preset_prompt)

        if st.button("Evaluate Machinery Diagnosis ➔", use_container_width=True):
            response_text = copilot.ask_copilot(custom_prompt, current_reading, current_health)
            st.session_state.copilot_history.insert(0, {"query": custom_prompt, "response": response_text})

        # Display latest responses
        if st.session_state.copilot_history:
            latest = st.session_state.copilot_history[0]
            with st.container(border=True):
                st.markdown(f"**Diagnostic Query:** `{latest['query']}`")
                st.markdown(latest["response"])

        st.subheader("Primary Sensor Attribution Breakdown")
        deviations = root_cause.attribute_deviations(current_reading)
        df_dev_display = pd.DataFrame([
            {
                "Sensor Channel": d["sensor"],
                "Current Reading": f"{d['current_value']:.2f} {d['unit']}",
                "Baseline Nominal": f"{d['baseline_nominal']:.2f} {d['unit']}",
                "Deviation (%)": f"{d['pct_change']:+.1f}%",
                "Z-Score": f"{d['z_score']:+.2f} σ",
                "Severity": d["direction"],
            }
            for d in deviations[:6]
        ])
        st.table(df_dev_display)

    with cop_col2:
        st.subheader("Automated CMMS Work Order Dispatch")
        with st.container(border=True):
            wo_head_col, wo_badge_col = st.columns([2, 1])
            with wo_head_col:
                st.markdown(f"### `{current_work_order.work_order_id}`")
                st.caption(f"Dispatched: {current_work_order.created_at}")
            with wo_badge_col:
                st.info(f"**{current_work_order.priority}**")

            st.markdown(f"**Asset Target:** {current_work_order.asset_id} — {current_work_order.asset_name}")
            st.markdown(f"**Location:** {current_work_order.location}  \n**Trade Craft:** {current_work_order.required_craft}")
            st.markdown(f"**FMEA Failure Code:** {current_work_order.fmea_failure_mode} (Severity: {current_work_order.fmea_severity}/10)")
            
            if current_work_order.loto_required:
                st.error(f"**Mandatory LOTO Protocol:** {current_work_order.loto_procedure}")
            else:
                st.success("**LOTO Status:** Non-invasive monitoring")

            st.markdown(
                f"**Labor Estimate:** {current_work_order.est_labor_hours} hrs @ ${current_work_order.est_labor_rate_usd_hr}/hr (${current_work_order.total_labor_cost:.2f})  \n"
                f"**Parts Requisition:** ${current_work_order.total_parts_cost:,.2f}  \n"
                f"**Total Work Order Estimate:** **${current_work_order.total_estimated_cost:,.2f}**"
            )

        st.markdown("**Staged OEM Spare Parts BOM**")
        if current_work_order.recommended_parts:
            parts_df = pd.DataFrame([
                {
                    "Part Number": p.part_no,
                    "Description": p.desc,
                    "Qty": p.qty,
                    "Unit Cost": f"${p.unit_cost_usd:.2f}",
                    "Total": f"${p.total_cost:.2f}",
                }
                for p in current_work_order.recommended_parts
            ])
            st.table(parts_df)
        else:
            st.info("No OEM replacement parts required for nominal status.")

        # Download PDF Work Order button
        pdf_bytes = generate_work_order_pdf(
            work_order=current_work_order,
            health_state=current_health,
            reading=current_reading,
            economic_data=current_roi,
        )
        st.download_button(
            label="📄 Export Official CMMS Work Order (PDF)",
            data=pdf_bytes,
            file_name=f"{current_work_order.work_order_id}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )

# =========================================================================
# TAB 5: DOWNTIME ECONOMICS & ROI
# =========================================================================
with tab_roi:
    st.subheader("Downtime Cost Avoidance & Reliability Financial Model")
    st.caption("Financial comparison of run-to-failure emergency breakdown versus early predictive maintenance turnaround.")

    roi_input_col, roi_chart_col = st.columns([1, 1.4])

    with roi_input_col:
        st.markdown("**Scenario Cost Parameters**")
        asset_hourly_loss = st.number_input(
            "Production Loss Rate ($/Hour)",
            min_value=500.0,
            max_value=25000.0,
            value=float(current_roi["downtime_rate_hourly"]),
            step=500.0,
        )
        unplanned_hours = st.slider("Unplanned Outage Duration (Hours)", min_value=4.0, max_value=48.0, value=14.0, step=1.0)
        planned_hours = st.slider("Predictive Turnaround Duration (Hours)", min_value=1.0, max_value=12.0, value=3.5, step=0.5)
        ot_multiplier = st.slider("Emergency Overtime Multiplier", min_value=1.5, max_value=3.0, value=2.0, step=0.25)
        collateral_mult = st.slider("Collateral Secondary Damage Multiplier", min_value=1.0, max_value=4.0, value=2.4, step=0.2)

        custom_roi = economics.calculate_intervention_roi(
            asset_id=st.session_state.active_asset_id,
            work_order=current_work_order,
            unplanned_downtime_hours=unplanned_hours,
            planned_turnaround_hours=planned_hours,
            emergency_overtime_multiplier=ot_multiplier,
            collateral_damage_multiplier=collateral_mult,
        )

    with roi_chart_col:
        st.markdown("**Cost Comparison: Breakdown vs Predictive Intervention**")

        cost_breakdown_df = pd.DataFrame([
            {"Category": "Production Loss", "Scenario": "Unplanned Breakdown", "Amount": custom_roi["unplanned"]["production_loss_usd"]},
            {"Category": "Emergency Labor", "Scenario": "Unplanned Breakdown", "Amount": custom_roi["unplanned"]["emergency_labor_usd"]},
            {"Category": "Collateral Parts", "Scenario": "Unplanned Breakdown", "Amount": custom_roi["unplanned"]["collateral_parts_usd"]},
            {"Category": "Production Loss", "Scenario": "Predictive Turnaround", "Amount": custom_roi["planned"]["production_loss_usd"]},
            {"Category": "Scheduled Labor", "Scenario": "Predictive Turnaround", "Amount": custom_roi["planned"]["scheduled_labor_usd"]},
            {"Category": "OEM Parts", "Scenario": "Predictive Turnaround", "Amount": custom_roi["planned"]["oem_parts_usd"]},
        ])

        fig_cost = px.bar(
            cost_breakdown_df,
            x="Scenario",
            y="Amount",
            color="Category",
            title=f"Total: ${custom_roi['unplanned']['total_cost_usd']:,.0f} (Unplanned) vs ${custom_roi['planned']['total_cost_usd']:,.0f} (Predictive)",
            barmode="stack",
            color_discrete_sequence=["#ef4444", "#f97316", "#f59e0b", "#38bdf8", "#10b981", "#6366f1"],
        )
        fig_cost.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0d131f",
            plot_bgcolor="#0d131f",
            height=360,
            margin=dict(l=30, r=20, t=40, b=30),
        )
        st.plotly_chart(fig_cost, use_container_width=True)

        with st.container(border=True):
            st.metric(
                label="Estimated Net Capital Savings",
                value=f"${custom_roi['net_savings_usd']:,.2f}",
                delta=f"ROI: {custom_roi['roi_percentage']:.1f}% vs run-to-failure",
                delta_color="normal",
            )

# =========================================================================
# TAB 6: BATCH TELEMETRY EXPORT
# =========================================================================
with tab_batch:
    st.subheader("Batch Telemetry Diagnostics & Prioritized Dispatch Export")
    st.caption("Process historical telemetry records in bulk, execute automated fleet triage, and export high-risk maintenance backlogs.")

    batch_csv_path = "data/industrial_iot_telemetry.csv"
    if os.path.exists(batch_csv_path):
        df_batch = pd.read_csv(batch_csv_path)
        st.markdown(f"**Loaded Telemetry Store:** `{batch_csv_path}` ({len(df_batch):,} rows across {df_batch['asset_id'].nunique()} machines)")

        filter_col1, filter_col2 = st.columns(2)
        with filter_col1:
            sel_filter_asset = st.multiselect("Filter Asset IDs", options=list(df_batch["asset_id"].unique()), default=list(df_batch["asset_id"].unique())[:2])
        with filter_col2:
            sel_filter_iso = st.multiselect("Filter ISO 10816 Zones", options=["ZONE_A", "ZONE_B", "ZONE_C", "ZONE_D"], default=["ZONE_C", "ZONE_D"])

        filtered = df_batch[
            (df_batch["asset_id"].isin(sel_filter_asset)) &
            (df_batch["iso_zone"].isin(sel_filter_iso))
        ]

        st.table(
            filtered[[
                "timestamp", "asset_id", "vibration_de_rms", "bearing_temperature_c",
                "vibration_kurtosis", "pressure_differential_bar", "health_index", "iso_zone", "remaining_useful_life_hrs"
            ]].head(25)
        )

        csv_data = filtered.to_csv(index=False).encode("utf-8")
        st.download_button(
            label=f"⬇ Download Filtered Dispatch Backlog ({len(filtered)} Records CSV)",
            data=csv_data,
            file_name=f"maintenance_dispatch_backlog_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )
