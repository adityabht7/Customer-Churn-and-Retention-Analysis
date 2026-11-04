"""Composite multi-task inference pipeline orchestrating classification, RUL, and anomaly detection."""

from __future__ import annotations
import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from src.core.constants import FEATURE_COLUMNS, FAILURE_MODES, FLEET_ASSETS
from src.core.schema import TelemetryReading, AssetHealthState
from src.models.failure_classifier import FailureModeClassifier
from src.models.rul_regressor import RULRegressor
from src.models.anomaly_detector import SensorAnomalyDetector
from src.telemetry.signal_processing import calculate_iso_zone, calculate_health_index


class PredictiveMaintenancePipeline:
    """End-to-end industrial model pipeline integrating classification, prognostics, and anomaly scoring."""

    def __init__(
        self,
        classifier: Optional[FailureModeClassifier] = None,
        rul_regressor: Optional[RULRegressor] = None,
        anomaly_detector: Optional[SensorAnomalyDetector] = None,
    ):
        self.classifier = classifier or FailureModeClassifier()
        self.rul_regressor = rul_regressor or RULRegressor()
        self.anomaly_detector = anomaly_detector or SensorAnomalyDetector()
        self.asset_registry = {a["asset_id"]: a for a in FLEET_ASSETS}

    def fit(self, X: pd.DataFrame, y_fault: pd.Series, y_rul: pd.Series) -> PredictiveMaintenancePipeline:
        """Trains all three predictive models synchronously."""
        X_feats = X[FEATURE_COLUMNS]
        self.classifier.fit(X_feats, y_fault)
        self.rul_regressor.fit(X_feats, y_rul)
        self.anomaly_detector.fit(X_feats)
        return self

    def assess_telemetry(self, reading: TelemetryReading) -> AssetHealthState:
        """Evaluates a live single-sample telemetry scan and synthesizes the asset health condition."""
        asset_meta = self.asset_registry.get(
            reading.asset_id,
            {
                "asset_name": reading.asset_id,
                "asset_type": reading.asset_type,
                "location": "Plant Floor",
                "nominal_vibe_rms": 2.0,
                "nominal_temp_c": 50.0,
            },
        )

        df_single = pd.DataFrame([reading.to_dict()])[FEATURE_COLUMNS]

        # Multi-class failure prediction
        fault_probs = self.classifier.predict_proba(df_single)[0]
        pred_fault_id = int(np.argmax(fault_probs))
        pred_fault_prob = float(fault_probs[pred_fault_id])
        fault_info = FAILURE_MODES.get(pred_fault_id, FAILURE_MODES[0])

        # RUL prognostics
        rul_pred, rul_low, rul_high = self.rul_regressor.predict_with_intervals(df_single)
        pred_rul = float(rul_pred[0])
        pred_rul_low = float(rul_low[0])
        pred_rul_high = float(rul_high[0])

        # Anomaly scoring
        anomaly_score = float(self.anomaly_detector.score_anomaly(df_single)[0])

        # ISO 10816 zone
        iso_zone = calculate_iso_zone(reading.vibration_de_rms)

        # Health index computation
        health_idx = calculate_health_index(
            vibration_rms=reading.vibration_de_rms,
            bearing_temp_c=reading.bearing_temperature_c,
            vibration_kurtosis=reading.vibration_kurtosis,
            pressure_diff_bar=reading.pressure_differential_bar,
            nominal_vibe=float(asset_meta.get("nominal_vibe_rms", 2.0)),
            nominal_temp=float(asset_meta.get("nominal_temp_c", 50.0)),
        )

        # Determine alarms
        alarms: List[str] = []
        if iso_zone in ["ZONE_C", "ZONE_D"]:
            alarms.append(f"ISO 10816 Severity Alert: Vibration RMS ({reading.vibration_de_rms} mm/s) in {iso_zone}")
        if reading.bearing_temperature_c > 80.0:
            alarms.append(f"Thermal Critical: Bearing Temperature at {reading.bearing_temperature_c}°C (Threshold 80°C)")
        elif reading.bearing_temperature_c > 70.0:
            alarms.append(f"Thermal Advisory: Bearing Temperature at {reading.bearing_temperature_c}°C")
        if reading.vibration_kurtosis > 5.5:
            alarms.append(f"Impact Shock Warning: Kurtosis elevated at {reading.vibration_kurtosis} (Shock pulse fault)")
        if reading.pressure_differential_bar > 8.0:
            alarms.append(f"Hydraulic Delta-P Warning: Differential pressure at {reading.pressure_differential_bar} bar")

        # Determine overall asset status label
        if pred_fault_id != 0 or iso_zone == "ZONE_D" or health_idx < 40.0:
            status_label = "CRITICAL" if (iso_zone == "ZONE_D" or pred_fault_id in [3, 4] or health_idx < 30.0) else "WARNING"
        elif iso_zone == "ZONE_C" or anomaly_score > 0.65 or health_idx < 70.0:
            status_label = "ADVISORY"
        else:
            status_label = "NORMAL"

        return AssetHealthState(
            asset_id=reading.asset_id,
            asset_name=asset_meta.get("asset_name", reading.asset_id),
            asset_type=asset_meta.get("asset_type", reading.asset_type),
            location=asset_meta.get("location", "Plant Floor"),
            health_index=round(health_idx, 1),
            iso_severity_zone=iso_zone,
            predicted_failure_mode_id=pred_fault_id,
            predicted_failure_mode_name=fault_info["name"],
            failure_probability=round(pred_fault_prob, 3),
            predicted_rul_hrs=round(pred_rul, 1),
            rul_confidence_lower_hrs=round(pred_rul_low, 1),
            rul_confidence_upper_hrs=round(pred_rul_high, 1),
            anomaly_score=round(anomaly_score, 3),
            active_alarms=alarms,
            status_label=status_label,
        )

    def save(self, filepath: str) -> None:
        """Serializes pipeline bundle to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)

    @classmethod
    def load(cls, filepath: str) -> PredictiveMaintenancePipeline:
        """Loads serialized pipeline bundle."""
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Serialized pipeline not found at {filepath}")
        return joblib.load(filepath)
