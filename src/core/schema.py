"""Data schemas and type models for telemetry, health states, and work orders."""

from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, List, Optional
from datetime import datetime


@dataclass
class TelemetryReading:
    """Represents a discrete multi-channel sensor scan from an industrial asset."""
    timestamp: str
    asset_id: str
    asset_type: str
    rotational_speed_rpm: float
    vibration_de_rms: float
    vibration_nde_rms: float
    vibration_kurtosis: float
    vibration_crest_factor: float
    bearing_temperature_c: float
    motor_winding_temperature_c: float
    hydraulic_pressure_bar: float
    pressure_differential_bar: float
    acoustic_emission_db: float
    motor_current_amps: float
    operational_hours: float
    # Ground truth or injected simulation states
    failure_mode: int = 0
    remaining_useful_life_hrs: float = 1200.0

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_feature_vector(self, feature_columns: List[str]) -> List[float]:
        data = asdict(self)
        return [float(data[col]) for col in feature_columns]


@dataclass
class AssetHealthState:
    """Composite health condition and diagnostic summary for an asset."""
    asset_id: str
    asset_name: str
    asset_type: str
    location: str
    health_index: float  # 0.0 to 100.0%
    iso_severity_zone: str  # ZONE_A, ZONE_B, ZONE_C, ZONE_D
    predicted_failure_mode_id: int
    predicted_failure_mode_name: str
    failure_probability: float
    predicted_rul_hrs: float
    rul_confidence_lower_hrs: float
    rul_confidence_upper_hrs: float
    anomaly_score: float  # -1 to 1, or probability
    active_alarms: List[str] = field(default_factory=list)
    top_sensor_deviations: List[Dict[str, Any]] = field(default_factory=list)
    status_label: str = "NORMAL"  # NORMAL, ADVISORY, WARNING, CRITICAL


@dataclass
class WorkOrderItem:
    """Spare part or consumable required for maintenance intervention."""
    part_no: str
    desc: str
    qty: int
    unit_cost_usd: float

    @property
    def total_cost(self) -> float:
        return self.qty * self.unit_cost_usd


@dataclass
class WorkOrder:
    """Formal CMMS maintenance work order dispatch object."""
    work_order_id: str
    created_at: str
    asset_id: str
    asset_name: str
    location: str
    priority: str
    fmea_failure_mode: str
    fmea_severity: int
    health_index: float
    predicted_rul_hrs: float
    loto_required: bool
    loto_procedure: Optional[str]
    required_craft: str
    est_labor_hours: float
    est_labor_rate_usd_hr: float
    recommended_parts: List[WorkOrderItem]
    action_instructions: str
    dispatched_by: str = "AuraVibe Condition Monitoring & Reliability System"

    @property
    def total_parts_cost(self) -> float:
        return sum(item.total_cost for item in self.recommended_parts)

    @property
    def total_labor_cost(self) -> float:
        return self.est_labor_hours * self.est_labor_rate_usd_hr

    @property
    def total_estimated_cost(self) -> float:
        return self.total_parts_cost + self.total_labor_cost

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["total_parts_cost"] = self.total_parts_cost
        d["total_labor_cost"] = self.total_labor_cost
        d["total_estimated_cost"] = self.total_estimated_cost
        return d
