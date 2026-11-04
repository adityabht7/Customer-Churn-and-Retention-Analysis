"""
Machine Learning Pipeline for Customer Churn Prediction and Model Benchmarking.
Supports Random Forest, HistGradientBoosting, Gradient Boosting, and Logistic Regression.
"""

from typing import Dict, Any, Tuple, List, Optional
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, roc_curve, precision_recall_curve,
    average_precision_score
)

MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models")
MODEL_FILE = os.path.join(MODEL_DIR, "churn_pipeline.joblib")

FEATURE_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService", "OnlineSecurity",
    "OnlineBackup", "DeviceProtection", "TechSupport", "StreamingTV",
    "StreamingMovies", "Contract", "PaperlessBilling", "PaymentMethod",
    "MonthlyCharges", "TotalCharges", "SupportTickets", "UsageFrequency",
    "SatisfactionScore"
]

NUM_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "SupportTickets", "SatisfactionScore"]
CAT_COLS = [c for c in FEATURE_COLS if c not in NUM_COLS]


def build_preprocessor() -> ColumnTransformer:
    """Creates a ColumnTransformer to scale numerics and one-hot encode categoricals."""
    numeric_transformer = Pipeline(steps=[
        ("scaler", StandardScaler())
    ])
    categorical_transformer = Pipeline(steps=[
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUM_COLS),
            ("cat", categorical_transformer, CAT_COLS)
        ]
    )
    return preprocessor


def get_model_instances() -> Dict[str, Any]:
    """Returns candidate classification algorithms."""
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=150, max_depth=8, class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=150, max_depth=6, class_weight="balanced", random_state=42
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=120, max_depth=4, learning_rate=0.08, random_state=42
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        )
    }


def prepare_features_and_target(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """Validates and extracts feature matrix X and target y."""
    data = df.copy()
    for col in FEATURE_COLS:
        if col not in data.columns:
            if col in NUM_COLS:
                data[col] = 0.0
            else:
                data[col] = "No"

    X = data[FEATURE_COLS]
    y = None
    if "Churn" in data.columns:
        y = (data["Churn"].astype(str).str.strip().str.capitalize() == "Yes").astype(int)
    return X, y


def train_and_benchmark_models(df: pd.DataFrame, test_size: float = 0.2) -> Dict[str, Any]:
    """
    Trains all candidate models, evaluates their performance, and selects the best performing model.
    """
    X, y = prepare_features_and_target(df)
    if y is None:
        raise ValueError("Dataset does not contain a 'Churn' target column.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    models = get_model_instances()
    results = {}
    best_model_name = None
    best_roc_auc = -1.0
    best_pipeline = None

    for name, clf in models.items():
        preprocessor = build_preprocessor()
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("classifier", clf)
        ])
        pipe.fit(X_train, y_train)

        # Predictions
        y_pred = pipe.predict(X_test)
        y_prob = pipe.predict_proba(X_test)[:, 1]

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        roc_auc = roc_auc_score(y_test, y_prob)
        pr_auc = average_precision_score(y_test, y_prob)
        cm = confusion_matrix(y_test, y_pred).tolist()

        fpr, tpr, _ = roc_curve(y_test, y_prob)
        precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_prob)

        results[name] = {
            "pipeline": pipe,
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "roc_auc": float(roc_auc),
            "pr_auc": float(pr_auc),
            "confusion_matrix": cm,
            "roc_curve": {"fpr": fpr.tolist(), "tpr": tpr.tolist()},
            "pr_curve": {"precision": precision_curve.tolist(), "recall": recall_curve.tolist()}
        }

        if roc_auc > best_roc_auc:
            best_roc_auc = roc_auc
            best_model_name = name
            best_pipeline = pipe

    # Feature importances from best tree model
    feature_names = []
    encoder = best_pipeline.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
    cat_feature_names = encoder.get_feature_names_out(CAT_COLS).tolist()
    feature_names = NUM_COLS + cat_feature_names

    feature_importances = []
    clf_step = best_pipeline.named_steps["classifier"]
    if hasattr(clf_step, "feature_importances_"):
        raw_importances = clf_step.feature_importances_
        feature_importances = [
            {"feature": f, "importance": float(imp)}
            for f, imp in sorted(zip(feature_names, raw_importances), key=lambda x: x[1], reverse=True)
        ]
    elif hasattr(clf_step, "coef_"):
        raw_coefs = np.abs(clf_step.coef_[0])
        feature_importances = [
            {"feature": f, "importance": float(imp)}
            for f, imp in sorted(zip(feature_names, raw_coefs), key=lambda x: x[1], reverse=True)
        ]

    benchmark_summary = {
        "models": {k: {m: v for m, v in vals.items() if m != "pipeline"} for k, vals in results.items()},
        "best_model_name": best_model_name,
        "best_roc_auc": best_roc_auc,
        "feature_importances": feature_importances[:15],
        "total_test_samples": len(y_test)
    }

    return {
        "benchmark_summary": benchmark_summary,
        "best_pipeline": best_pipeline,
        "results": results,
        "X_test": X_test,
        "y_test": y_test
    }


def save_pipeline(pipeline: Pipeline, filepath: str = MODEL_FILE) -> None:
    """Serializes pipeline to disk."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    joblib.dump(pipeline, filepath)


def load_or_train_default_model() -> Pipeline:
    """Loads existing pipeline from disk, or trains default from benchmark data."""
    if os.path.exists(MODEL_FILE):
        try:
            return joblib.load(MODEL_FILE)
        except Exception:
            pass

    from src.data_loader import load_churn_data
    df = load_churn_data()
    train_res = train_and_benchmark_models(df)
    pipe = train_res["best_pipeline"]
    save_pipeline(pipe)
    return pipe


def predict_churn_risk(pipeline: Pipeline, customer_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Computes churn probability and risk tier for a single customer dictionary.
    """
    df_single = pd.DataFrame([customer_dict])
    X, _ = prepare_features_and_target(df_single)
    
    prob = float(pipeline.predict_proba(X)[0][1])
    is_churn = int(prob >= 0.50)

    if prob < 0.30:
        risk_tier = "Low Risk"
        risk_color = "#10B981"
        badge = ""
    elif prob < 0.65:
        risk_tier = "Moderate Risk"
        risk_color = "#F59E0B"
        badge = ""
    else:
        risk_tier = "High Risk"
        risk_color = "#EF4444"
        badge = ""

    return {
        "churn_probability": prob,
        "predicted_churn": is_churn,
        "risk_tier": risk_tier,
        "risk_color": risk_color,
        "risk_badge": badge
    }


def batch_predict(pipeline: Pipeline, df: pd.DataFrame) -> pd.DataFrame:
    """Scores an entire DataFrame with churn probability and risk tier."""
    X, _ = prepare_features_and_target(df)
    probs = pipeline.predict_proba(X)[:, 1]
    
    res = df.copy()
    res["Churn_Probability"] = np.round(probs, 4)
    res["Predicted_Churn"] = ["Yes" if p >= 0.50 else "No" for p in probs]
    
    def assign_tier(p):
        if p < 0.30:
            return "Low Risk"
        elif p < 0.65:
            return "Moderate Risk"
        return "High Risk"

    res["Risk_Tier"] = [assign_tier(p) for p in probs]
    return res
