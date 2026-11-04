"""Unit tests for telemetry signal processing, FFT harmonic generation, and physics simulation."""

import numpy as np
import pytest
from src.telemetry.signal_processing import (
    calculate_iso_zone,
    calculate_health_index,
    compute_spectral_harmonics,
)
from src.telemetry.generator import TelemetrySimulator


def test_iso_zone_boundaries():
    assert calculate_iso_zone(1.5) == "ZONE_A"
    assert calculate_iso_zone(2.8) == "ZONE_A"
    assert calculate_iso_zone(2.81) == "ZONE_B"
    assert calculate_iso_zone(4.5) == "ZONE_B"
    assert calculate_iso_zone(4.51) == "ZONE_C"
    assert calculate_iso_zone(7.1) == "ZONE_C"
    assert calculate_iso_zone(7.11) == "ZONE_D"
    assert calculate_iso_zone(15.0) == "ZONE_D"


def test_calculate_health_index_degradation():
    healthy_score = calculate_health_index(
        vibration_rms=1.8,
        bearing_temp_c=50.0,
        vibration_kurtosis=3.0,
        pressure_diff_bar=2.0,
        nominal_vibe=2.0,
        nominal_temp=50.0,
    )
    assert healthy_score >= 95.0

    degraded_score = calculate_health_index(
        vibration_rms=6.5,
        bearing_temp_c=82.0,
        vibration_kurtosis=8.5,
        pressure_diff_bar=9.5,
        nominal_vibe=2.0,
        nominal_temp=50.0,
    )
    assert degraded_score < 45.0
    assert 0.0 <= degraded_score <= 100.0


def test_spectral_harmonics_generation():
    freqs, spectrum = compute_spectral_harmonics(rpm=1800, vibration_rms=4.0)
    assert len(freqs) == len(spectrum)
    assert len(freqs) > 0
    # 1800 RPM = 30 Hz fundamental
    idx_30hz = np.argmin(np.abs(freqs - 30.0))
    assert spectrum[idx_30hz] > 1.0


def test_telemetry_simulator_single_reading():
    sim = TelemetrySimulator(seed=123)
    reading = sim.generate_single_reading("PUMP-301A", operational_stress=1.0)
    assert reading.asset_id == "PUMP-301A"
    assert reading.vibration_de_rms > 0
    assert reading.bearing_temperature_c > 0
    assert reading.failure_mode == 0

    fault_reading = sim.generate_single_reading("PUMP-301A", injected_fault=1)
    assert fault_reading.failure_mode == 1
    assert fault_reading.vibration_kurtosis > 4.0


def test_telemetry_degradation_trajectory():
    sim = TelemetrySimulator(seed=123)
    traj_df = sim.generate_degradation_trajectory("COMP-104", failure_mode=1, duration_hours=24)
    assert len(traj_df) == 24
    assert "remaining_useful_life_hrs" in traj_df.columns
    assert "health_index" in traj_df.columns
    # RUL should decrease over time
    assert traj_df["remaining_useful_life_hrs"].iloc[0] > traj_df["remaining_useful_life_hrs"].iloc[-1]
