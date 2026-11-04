"""Root cause diagnosis and sensor attribution engine."""

from __future__ import annotations
import numpy as np
from typing import Dict, Any, List
from src.core.constants import FEATURE_COLUMNS, FLEET_ASSETS, FAILURE_MODES
from src.core.schema import TelemetryReading


class RootCauseAnalyzer:
    """Attributes operational anomalies and fault probabilities to specific sensor channels."""

    def __init__(self):
        self.asset_registry = {a["asset_id"]: a for a in FLEET_ASSETS}

    def attribute_deviations(self, reading: TelemetryReading) -> List[Dict[str, Any]]:
        """Calculates normalized z-score and percentage deviations from nominal baseline."""
        meta = self.asset_registry.get(
            reading.asset_id,
            {"nominal_vibe_rms": 2.0, "nominal_temp_c": 50.0, "nominal_pressure_bar": 10.0, "rated_rpm": 1800},
        )

        nominal_vibe = float(meta.get("nominal_vibe_rms", 2.0))
        nominal_temp = float(meta.get("nominal_temp_c", 50.0))
        nominal_press = float(meta.get("nominal_pressure_bar", 10.0))
        nominal_rpm = float(meta.get("rated_rpm", 1800.0))

        # Expected baseline reference table
        baselines = {
            "rotational_speed_rpm": (nominal_rpm, 20.0, "RPM"),
            "vibration_de_rms": (nominal_vibe, 0.4, "mm/s"),
            "vibration_nde_rms": (nominal_vibe * 0.85, 0.35, "mm/s"),
            "vibration_kurtosis": (3.0, 0.3, "kurt"),
            "vibration_crest_factor": (2.8, 0.25, "ratio"),
            "bearing_temperature_c": (nominal_temp, 3.5, "°C"),
            "motor_winding_temperature_c": (nominal_temp + 14.0, 4.0, "°C"),
            "hydraulic_pressure_bar": (nominal_press, 0.5, "bar"),
            "pressure_differential_bar": (2.2, 0.3, "bar"),
            "acoustic_emission_db": (48.0, 3.0, "dB"),
            "motor_current_amps": (120.0, 10.0, "A"),
            "operational_hours": (10000.0, 1000.0, "hrs"),
        }

        readings_dict = reading.to_dict()
        deviations: List[Dict[str, Any]] = []

        for feat in FEATURE_COLUMNS:
            val = float(readings_dict.get(feat, 0.0))
            nominal_val, std_dev, unit = baselines.get(feat, (1.0, 1.0, ""))
            z_score = (val - nominal_val) / max(std_dev, 0.001)
            pct_change = ((val - nominal_val) / max(abs(nominal_val), 0.001)) * 100.0

            # Absolute severity score
            severity = abs(z_score)

            deviations.append({
                "sensor": feat,
                "current_value": round(val, 2),
                "baseline_nominal": round(nominal_val, 2),
                "unit": unit,
                "z_score": round(z_score, 2),
                "pct_change": round(pct_change, 1),
                "severity_index": round(severity, 2),
                "direction": "HIGH" if z_score > 0 else "LOW",
            })

        # Sort by deviation severity descending
        deviations.sort(key=lambda x: x["severity_index"], reverse=True)
        return deviations

    def explain_failure_mode(self, failure_mode_id: int) -> Dict[str, Any]:
        """Provides mechanistic explanation and physics of the diagnosed failure mode."""
        info = FAILURE_MODES.get(failure_mode_id, FAILURE_MODES[0])
        return {
            "mode_id": failure_mode_id,
            "name": info["name"],
            "category": info["category"],
            "description": info["description"],
            "fmea_severity": info["fmea_severity"],
            "standard_action": info["recommended_action"],
        }
