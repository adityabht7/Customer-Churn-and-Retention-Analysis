"""Remaining Useful Life (RUL) regression and prognostic forecasting models."""

from __future__ import annotations
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class RULRegressor:
    """Prognostic model estimating remaining operating hours before threshold failure."""

    def __init__(self, model_type: str = "gradient_boosting", random_state: int = 42):
        self.model_type = model_type
        self.random_state = random_state
        self.pipeline: Optional[Pipeline] = None
        self.residual_std_: float = 24.0  # Estimated standard error in hours

    def _build_estimator(self, model_type: str) -> Pipeline:
        if model_type == "gradient_boosting":
            reg = GradientBoostingRegressor(
                n_estimators=140,
                learning_rate=0.07,
                max_depth=5,
                subsample=0.85,
                random_state=self.random_state,
            )
            return Pipeline([("scaler", StandardScaler()), ("reg", reg)])
        elif model_type == "random_forest":
            reg = RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_leaf=4,
                n_jobs=-1,
                random_state=self.random_state,
            )
            return Pipeline([("scaler", StandardScaler()), ("reg", reg)])
        else:
            raise ValueError(f"Unsupported model type: {model_type}")

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> RULRegressor:
        self.pipeline = self._build_estimator(self.model_type)
        self.pipeline.fit(X, y)
        preds = self.pipeline.predict(X)
        self.residual_std_ = float(np.std(np.array(y) - preds))
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        if self.pipeline is None:
            raise RuntimeError("Model is not fitted yet.")
        preds = self.pipeline.predict(X)
        return np.clip(preds, 0.0, None)

    def predict_with_intervals(
        self, X: pd.DataFrame | np.ndarray, confidence: float = 0.90
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculates point RUL forecast along with lower and upper confidence intervals.
        
        Returns:
            Tuple of (point_estimates, lower_bounds, upper_bounds) in hours.
        """
        point_preds = self.predict(X)
        z_multiplier = 1.645 if confidence <= 0.90 else 1.960
        margin = z_multiplier * self.residual_std_

        lower = np.clip(point_preds - margin, 0.0, None)
        upper = point_preds + margin
        return point_preds, lower, upper

    def evaluate(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> Dict[str, float]:
        preds = self.predict(X)
        rmse = float(np.sqrt(mean_squared_error(y, preds)))
        mae = float(mean_absolute_error(y, preds))
        r2 = float(r2_score(y, preds))
        return {"rmse_hours": rmse, "mae_hours": mae, "r2_score": r2}
