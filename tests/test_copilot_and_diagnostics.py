"""Unit tests for root-cause sensor attribution, copilot reasoning, and downtime economics."""

import pytest
from src.telemetry.generator import TelemetrySimulator
from src.diagnostics.root_cause import RootCauseAnalyzer
from src.diagnostics.copilot_engine import MaintenanceCopilotEngine
from src.diagnostics.economics import DowntimeEconomicsCalculator
from src.models.pipeline import PredictiveMaintenancePipeline


@pytest.fixture(scope="module")
def sample_pipeline():
    pipeline_path = "models/predictive_maintenance_pipeline.joblib"
    return PredictiveMaintenancePipeline.load(pipeline_path)


def test_root_cause_attribution():
    sim = TelemetrySimulator(seed=42)
    # Inject bearing spall fault
    reading = sim.generate_single_reading("PUMP-301A", injected_fault=1)
    analyzer = RootCauseAnalyzer()
    deviations = analyzer.attribute_deviations(reading)

    assert len(deviations) > 0
    # Top deviation should reflect elevated kurtosis, crest factor, or vibration
    top_sensors = [d["sensor"] for d in deviations[:4]]
    assert any(s in top_sensors for s in ["vibration_kurtosis", "vibration_crest_factor", "vibration_de_rms", "acoustic_emission_db"])


def test_copilot_work_order_generation(sample_pipeline):
    sim = TelemetrySimulator(seed=42)
    reading = sim.generate_single_reading("COMP-104", injected_fault=2)
    health_state = sample_pipeline.assess_telemetry(reading)

    copilot = MaintenanceCopilotEngine()
    wo = copilot.generate_work_order(reading, health_state)

    assert wo.asset_id == "COMP-104"
    assert "WO-" in wo.work_order_id
    assert len(wo.recommended_parts) > 0
    assert wo.est_labor_hours > 0
    assert wo.total_estimated_cost > 0
    assert "OPERATIONAL PROCEDURE" in wo.action_instructions


def test_copilot_conversational_queries(sample_pipeline):
    sim = TelemetrySimulator(seed=42)
    reading = sim.generate_single_reading("FAN-202", injected_fault=1)
    health_state = sample_pipeline.assess_telemetry(reading)

    copilot = MaintenanceCopilotEngine()

    ans_fault = copilot.ask_copilot("What is wrong with this fan?", reading, health_state)
    assert "Root-Cause Diagnostic Assessment" in ans_fault or "Bearing" in ans_fault

    ans_rul = copilot.ask_copilot("What is the remaining useful life?", reading, health_state)
    assert "Operating Hours" in ans_rul

    ans_parts = copilot.ask_copilot("What spare parts do I need?", reading, health_state)
    assert "BOM" in ans_parts or "SKF" in ans_parts or "Part" in ans_parts

    ans_loto = copilot.ask_copilot("What are the lockout tagout safety procedures?", reading, health_state)
    assert "LOTO" in ans_loto or "Isolation" in ans_loto


def test_downtime_economics_calculator():
    calc = DowntimeEconomicsCalculator()
    sim = TelemetrySimulator(seed=42)
    reading = sim.generate_single_reading("PRESS-401", injected_fault=4)

    copilot = MaintenanceCopilotEngine()
    # Create mock health state for work order
    class MockHealth:
        predicted_failure_mode_id = 4
        iso_severity_zone = "ZONE_D"
        predicted_rul_hrs = 12.0
        health_index = 28.0
        failure_probability = 0.95

    wo = copilot.generate_work_order(reading, MockHealth())
    roi_data = calc.calculate_intervention_roi(
        asset_id="PRESS-401",
        work_order=wo,
        unplanned_downtime_hours=16.0,
        planned_turnaround_hours=3.0,
    )

    assert roi_data["net_savings_usd"] > 10000.0
    assert roi_data["roi_percentage"] > 100.0
    assert roi_data["unplanned"]["total_cost_usd"] > roi_data["planned"]["total_cost_usd"]
