"""Unit test for PDF report and CMMS dispatch document generator."""

import pytest
from src.telemetry.generator import TelemetrySimulator
from src.diagnostics.copilot_engine import MaintenanceCopilotEngine
from src.diagnostics.economics import DowntimeEconomicsCalculator
from src.models.pipeline import PredictiveMaintenancePipeline
from src.reports.pdf_exporter import generate_work_order_pdf


def test_pdf_work_order_generation():
    pipeline = PredictiveMaintenancePipeline.load("models/predictive_maintenance_pipeline.joblib")
    sim = TelemetrySimulator(seed=42)
    reading = sim.generate_single_reading("PUMP-301A", injected_fault=1)
    health_state = pipeline.assess_telemetry(reading)

    copilot = MaintenanceCopilotEngine()
    wo = copilot.generate_work_order(reading, health_state)

    econ = DowntimeEconomicsCalculator().calculate_intervention_roi("PUMP-301A", wo)
    pdf_bytes = generate_work_order_pdf(wo, health_state, reading, econ)

    assert isinstance(pdf_bytes, bytes)
    assert len(pdf_bytes) > 2000
    assert pdf_bytes.startswith(b"%PDF")
