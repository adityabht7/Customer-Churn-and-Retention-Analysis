"""Unsupervised multivariate anomaly detection for condition monitoring."""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Optional
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


class SensorAnomalyDetector:
    """Detects zero-day multivariate sensor drifts and operating condition anomalies."""

    def __init__(self, contamination: float = 0.08, random_state: int = 42):
        self.contamination = contamination
        self.random_state = random_state
        self.pipeline: Optional[Pipeline] = None

    def fit(self, X: pd.DataFrame | np.ndarray) -> SensorAnomalyDetector:
        iso = IsolationForest(
            contamination=self.contamination,
            n_estimators=100,
            max_samples="auto",
            random_state=self.random_state,
            n_jobs=-1,
        )
        self.pipeline = Pipeline([("scaler", StandardScaler()), ("iso", iso)])
        self.pipeline.fit(X)
        return self

    def score_anomaly(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Returns normalized anomaly score between 0.0 (strictly normal) and 1.0 (highly anomalous)."""
        if self.pipeline is None:
            raise RuntimeError("Anomaly detector is not fitted yet.")

        # decision_function yields negative for anomalies, positive for inliers
        raw_scores = self.pipeline.named_steps["iso"].decision_function(
            self.pipeline.named_steps["scaler"].transform(X)
        )
        # Normalize into a 0.0 to 1.0 risk spectrum
        # Typical raw scores range from -0.3 to +0.3
        normalized = 1.0 / (1.0 + np.exp(raw_scores * 8.0))
        return np.clip(normalized, 0.0, 1.0)
