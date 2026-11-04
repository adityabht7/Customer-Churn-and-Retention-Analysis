"""Offline model training, evaluation, cross-validation benchmarking, and artifact serialization."""

from __future__ import annotations
import os
import sys

# Ensure repository root is on sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import json
import numpy as np
import pandas as pd
from src.core.constants import FEATURE_COLUMNS
from src.telemetry.generator import TelemetrySimulator
from src.models.failure_classifier import benchmark_classifiers
from src.models.rul_regressor import RULRegressor
from src.models.pipeline import PredictiveMaintenancePipeline


def run_training_pipeline():
    print("[1/5] Initializing Physics-Based Industrial IoT Telemetry Generator...")
    sim = TelemetrySimulator(seed=42)
    df = sim.generate_fleet_dataset(samples_per_asset=1500, failure_rate=0.30)

    os.makedirs("data", exist_ok=True)
    csv_path = "data/industrial_iot_telemetry.csv"
    df.to_csv(csv_path, index=False)
    print(f"      Saved {len(df):,} telemetry samples to {csv_path}")

    X = df[FEATURE_COLUMNS]
    y_fault = df["failure_mode"]
    y_rul = df["remaining_useful_life_hrs"]

    print("\n[2/5] Benchmarking Multi-Class Failure Classifiers (Stratified 5-Fold CV)...")
    clf_benchmarks = benchmark_classifiers(X, y_fault, n_splits=5, random_state=42)
    for model_name, metrics in clf_benchmarks.items():
        print(f"      - {model_name:24s} | Macro F1: {metrics['macro_f1_mean']:.4f} (±{metrics['macro_f1_std']:.4f}) | Balanced Acc: {metrics['balanced_acc_mean']:.4f} | ROC-AUC: {metrics['roc_auc_mean']:.4f}")

    print("\n[3/5] Benchmarking Remaining Useful Life (RUL) Regressors...")
    rul_gb = RULRegressor(model_type="gradient_boosting", random_state=42)
    rul_gb.fit(X, y_rul)
    gb_eval = rul_gb.evaluate(X, y_rul)
    print(f"      - GradientBoostingRegressor | RMSE: {gb_eval['rmse_hours']:.2f} hrs | MAE: {gb_eval['mae_hours']:.2f} hrs | R²: {gb_eval['r2_score']:.4f}")

    rul_rf = RULRegressor(model_type="random_forest", random_state=42)
    rul_rf.fit(X, y_rul)
    rf_eval = rul_rf.evaluate(X, y_rul)
    print(f"      - RandomForestRegressor     | RMSE: {rf_eval['rmse_hours']:.2f} hrs | MAE: {rf_eval['mae_hours']:.2f} hrs | R²: {rf_eval['r2_score']:.4f}")

    print("\n[4/5] Training Production Predictive Maintenance Pipeline...")
    pipeline = PredictiveMaintenancePipeline()
    pipeline.fit(X, y_fault, y_rul)

    os.makedirs("models", exist_ok=True)
    model_path = "models/predictive_maintenance_pipeline.joblib"
    pipeline.save(model_path)
    print(f"      Saved serialized pipeline bundle to {model_path}")

    # Save benchmark metrics
    metrics_summary = {
        "dataset_samples": len(df),
        "fleet_assets_count": len(df["asset_id"].unique()),
        "features": FEATURE_COLUMNS,
        "classification_benchmarks": clf_benchmarks,
        "rul_benchmarks": {
            "gradient_boosting": gb_eval,
            "random_forest": rf_eval,
        },
    }
    metrics_path = "models/benchmark_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics_summary, f, indent=2)
    print(f"      Saved metrics summary to {metrics_path}")

    print("\n[5/5] Pipeline training and artifact export completed successfully.")


if __name__ == "__main__":
    run_training_pipeline()
