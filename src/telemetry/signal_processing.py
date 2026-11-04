"""Digital signal processing and feature extraction for vibration and process telemetry."""

from __future__ import annotations
import numpy as np
from typing import Dict, Any, Tuple
from src.core.constants import ISO_10816_ZONES


def calculate_iso_zone(vibration_rms: float) -> str:
    """Classifies vibration velocity RMS into ISO 10816-3 severity zones.
    
    Args:
        vibration_rms: Velocity RMS in mm/s.
        
    Returns:
        Zone key string: 'ZONE_A', 'ZONE_B', 'ZONE_C', or 'ZONE_D'.
    """
    if vibration_rms <= 2.8:
        return "ZONE_A"
    elif vibration_rms <= 4.5:
        return "ZONE_B"
    elif vibration_rms <= 7.1:
        return "ZONE_C"
    else:
        return "ZONE_D"


def calculate_health_index(
    vibration_rms: float,
    bearing_temp_c: float,
    vibration_kurtosis: float,
    pressure_diff_bar: float,
    nominal_vibe: float = 2.0,
    nominal_temp: float = 50.0,
) -> float:
    """Computes a multi-criteria equipment Health Index (0.0 to 100.0%).
    
    Penalties are computed according to mechanical, thermal, and dynamic deviations.
    """
    health = 100.0

    # Vibration penalty (ISO 10816 weighting)
    vibe_ratio = vibration_rms / max(nominal_vibe, 0.5)
    if vibe_ratio > 1.2:
        health -= min(40.0, (vibe_ratio - 1.2) * 22.0)

    # Thermal penalty
    temp_excess = bearing_temp_c - nominal_temp
    if temp_excess > 5.0:
        health -= min(30.0, (temp_excess / 35.0) * 30.0)

    # Kurtosis / Shock pulse penalty (Normal kurtosis is ~3.0)
    if vibration_kurtosis > 3.8:
        kurt_excess = vibration_kurtosis - 3.8
        health -= min(20.0, kurt_excess * 3.5)

    # Pressure differential penalty
    if pressure_diff_bar > 8.0:
        health -= min(15.0, (pressure_diff_bar - 8.0) * 2.0)

    return float(np.clip(health, 0.0, 100.0))


def compute_spectral_harmonics(
    rpm: float,
    vibration_rms: float,
    noise_level: float = 0.05,
    sample_rate_hz: int = 2000,
    num_samples: int = 1024,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates a realistic synthetic frequency spectrum (FFT) centered around running speed harmonics.
    
    Returns:
        Tuple of (frequencies_hz, amplitudes_mm_s)
    """
    f0 = rpm / 60.0  # Fundamental frequency 1X in Hz
    frequencies = np.fft.rfftfreq(num_samples, d=1.0 / sample_rate_hz)
    spectrum = np.zeros_like(frequencies)

    # Add Gaussian background noise
    spectrum += np.random.normal(loc=noise_level * 0.2, scale=noise_level * 0.1, size=len(frequencies))

    # Add 1X harmonic (Rotor fundamental)
    idx_1x = np.argmin(np.abs(frequencies - f0))
    spectrum[max(0, idx_1x - 1) : min(len(frequencies), idx_1x + 2)] += vibration_rms * 0.7

    # Add 2X harmonic (Misalignment/bow)
    idx_2x = np.argmin(np.abs(frequencies - (2.0 * f0)))
    spectrum[max(0, idx_2x - 1) : min(len(frequencies), idx_2x + 2)] += vibration_rms * 0.35

    # Add 3X harmonic
    idx_3x = np.argmin(np.abs(frequencies - (3.0 * f0)))
    spectrum[max(0, idx_3x - 1) : min(len(frequencies), idx_3x + 2)] += vibration_rms * 0.15

    # High frequency bearing defect band (~5X to 8X f0)
    idx_brg = np.argmin(np.abs(frequencies - (6.4 * f0)))
    spectrum[max(0, idx_brg - 2) : min(len(frequencies), idx_brg + 3)] += vibration_rms * 0.2

    return frequencies, np.clip(spectrum, 0.0, None)
