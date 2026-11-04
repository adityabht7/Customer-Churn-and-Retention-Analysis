"""Integration tests for app components and Streamlit caching functions."""

import pytest
from app import load_core_engine
from src.core.constants import FLEET_ASSETS


def test_app_core_engine_initialization():
    pipeline, simulator, copilot, root_cause, economics = load_core_engine()
    assert pipeline is not None
    assert simulator is not None
    assert copilot is not None
    assert root_cause is not None
    assert economics is not None


def test_app_fleet_diagnostics_flow():
    pipeline, simulator, copilot, root_cause, economics = load_core_engine()
    for asset in FLEET_ASSETS:
        aid = asset["asset_id"]
        reading = simulator.generate_single_reading(aid, operational_stress=1.1)
        health = pipeline.assess_telemetry(reading)
        wo = copilot.generate_work_order(reading, health)
        roi = economics.calculate_intervention_roi(aid, wo)

        assert health.health_index >= 0.0
        assert health.predicted_rul_hrs >= 0.0
        assert wo.total_estimated_cost > 0.0
        assert roi["net_savings_usd"] >= 0.0
