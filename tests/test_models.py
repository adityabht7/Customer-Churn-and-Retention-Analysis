"""Unit and integration tests for ML models, regressors, and end-to-end inference pipeline."""

import os
import pytest
import pandas as pd
import numpy as np
from src.core.constants import FEATURE_COLUMNS
from src.telemetry.generator import TelemetrySimulator
from src.models.failure_classifier import FailureModeClassifier
from src.models.rul_regressor import RULRegressor
from src.models.anomaly_detector import SensorAnomalyDetector
from src.models.pipeline import PredictiveMaintenancePipeline


@pytest.fixture(scope="module")
def sample_training_data():
    sim = TelemetrySimulator(seed=99)
    df = sim.generate_fleet_dataset(samples_per_asset=100, failure_rate=0.30)
    return df


def test_failure_classifier_train_predict(sample_training_data):
    X = sample_training_data[FEATURE_COLUMNS]
    y = sample_training_data["failure_mode"]
    clf = FailureModeClassifier(model_type="hist_gradient_boosting", random_state=42)
    clf.fit(X, y)

    preds = clf.predict(X[:10])
    probs = clf.predict_proba(X[:10])
    assert len(preds) == 10
    assert probs.shape == (10, 5)
    assert np.allclose(probs.sum(axis=1), 1.0)

    importances = clf.get_feature_importances()
    assert len(importances) == len(FEATURE_COLUMNS)


def test_rul_regressor_intervals(sample_training_data):
    X = sample_training_data[FEATURE_COLUMNS]
    y = sample_training_data["remaining_useful_life_hrs"]
    reg = RULRegressor(model_type="gradient_boosting", random_state=42)
    reg.fit(X, y)

    preds, lower, upper = reg.predict_with_intervals(X[:15], confidence=0.90)
    assert len(preds) == 15
    assert len(lower) == 15
    assert len(upper) == 15
    assert np.all(lower <= preds)
    assert np.all(preds <= upper)

    eval_dict = reg.evaluate(X, y)
    assert eval_dict["rmse_hours"] > 0
    assert eval_dict["r2_score"] > 0.80


def test_anomaly_detector_scoring(sample_training_data):
    X = sample_training_data[FEATURE_COLUMNS]
    detector = SensorAnomalyDetector(contamination=0.08, random_state=42)
    detector.fit(X)

    scores = detector.score_anomaly(X[:20])
    assert len(scores) == 20
    assert np.all(scores >= 0.0)
    assert np.all(scores <= 1.0)


def test_pipeline_assessment_and_serialization(tmp_path, sample_training_data):
    X = sample_training_data[FEATURE_COLUMNS]
    y_fault = sample_training_data["failure_mode"]
    y_rul = sample_training_data["remaining_useful_life_hrs"]

    pipeline = PredictiveMaintenancePipeline()
    pipeline.fit(X, y_fault, y_rul)

    sim = TelemetrySimulator(seed=42)
    reading = sim.generate_single_reading("PUMP-301A", injected_fault=1)
    health_state = pipeline.assess_telemetry(reading)

    assert health_state.asset_id == "PUMP-301A"
    assert health_state.predicted_failure_mode_id == 1
    assert health_state.failure_probability > 0.5
    assert health_state.health_index < 80.0
    assert health_state.status_label in ["WARNING", "CRITICAL"]

    # Test serialization and reload
    save_path = str(tmp_path / "test_pipeline.joblib")
    pipeline.save(save_path)
    assert os.path.exists(save_path)

    loaded = PredictiveMaintenancePipeline.load(save_path)
    reloaded_health = loaded.assess_telemetry(reading)
    assert reloaded_health.predicted_failure_mode_id == health_state.predicted_failure_mode_id
