"""Unit tests for core constants, schema validations, and asset registry."""

import pytest
from src.core.constants import (
    ISO_10816_ZONES,
    FAILURE_MODES,
    FLEET_ASSETS,
    FEATURE_COLUMNS,
    SENSOR_BOUNDS,
)
from src.core.schema import TelemetryReading, WorkOrder, WorkOrderItem


def test_iso_10816_zones_structure():
    assert "ZONE_A" in ISO_10816_ZONES
    assert "ZONE_B" in ISO_10816_ZONES
    assert "ZONE_C" in ISO_10816_ZONES
    assert "ZONE_D" in ISO_10816_ZONES
    assert ISO_10816_ZONES["ZONE_A"]["max_rms"] == 2.8
    assert ISO_10816_ZONES["ZONE_B"]["max_rms"] == 4.5
    assert ISO_10816_ZONES["ZONE_C"]["max_rms"] == 7.1


def test_failure_modes_completeness():
    assert len(FAILURE_MODES) == 5
    for mode_id, info in FAILURE_MODES.items():
        assert "name" in info
        assert "code" in info
        assert "category" in info
        assert "fmea_severity" in info
        assert 1 <= info["fmea_severity"] <= 10
        assert "recommended_action" in info
        assert "loto_required" in info


def test_fleet_assets_registry():
    assert len(FLEET_ASSETS) >= 6
    asset_ids = [a["asset_id"] for a in FLEET_ASSETS]
    assert len(asset_ids) == len(set(asset_ids)), "Asset IDs must be unique"
    for asset in FLEET_ASSETS:
        assert asset["rated_power_kw"] > 0
        assert asset["rated_rpm"] > 0
        assert asset["downtime_cost_per_hr_usd"] > 0


def test_telemetry_reading_feature_vector():
    reading = TelemetryReading(
        timestamp="2026-11-04T08:00:00",
        asset_id="PUMP-301A",
        asset_type="Centrifugal Pump",
        rotational_speed_rpm=1780.0,
        vibration_de_rms=2.1,
        vibration_nde_rms=1.8,
        vibration_kurtosis=3.1,
        vibration_crest_factor=2.7,
        bearing_temperature_c=52.0,
        motor_winding_temperature_c=66.0,
        hydraulic_pressure_bar=12.5,
        pressure_differential_bar=2.3,
        acoustic_emission_db=49.0,
        motor_current_amps=145.0,
        operational_hours=8500.0,
    )
    vec = reading.to_feature_vector(FEATURE_COLUMNS)
    assert len(vec) == len(FEATURE_COLUMNS)
    assert vec[0] == 1780.0
    assert vec[1] == 2.1


def test_work_order_cost_calculations():
    items = [
        WorkOrderItem(part_no="BRG-01", desc="Deep Groove Bearing", qty=2, unit_cost_usd=150.0),
        WorkOrderItem(part_no="SEAL-01", desc="Shaft Seal", qty=1, unit_cost_usd=50.0),
    ]
    wo = WorkOrder(
        work_order_id="WO-TEST-001",
        created_at="2026-11-04 08:00:00",
        asset_id="PUMP-301A",
        asset_name="Booster Pump",
        location="Bay 1",
        priority="P2-Urgent",
        fmea_failure_mode="Bearing Race Spalling",
        fmea_severity=8,
        health_index=62.5,
        predicted_rul_hrs=96.0,
        loto_required=True,
        loto_procedure="LOTO-01",
        required_craft="Mechanical Millwright",
        est_labor_hours=4.0,
        est_labor_rate_usd_hr=80.0,
        recommended_parts=items,
        action_instructions="Inspect and replace bearing assembly.",
    )
    assert wo.total_parts_cost == 350.0  # (2*150) + (1*50)
    assert wo.total_labor_cost == 320.0  # 4 * 80
    assert wo.total_estimated_cost == 670.0  # 350 + 320
