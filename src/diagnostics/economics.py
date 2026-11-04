"""Industrial economic financial model and downtime cost avoidance calculator."""

from __future__ import annotations
from typing import Dict, Any
from src.core.constants import FLEET_ASSETS, FAILURE_MODES
from src.core.schema import WorkOrder


class DowntimeEconomicsCalculator:
    """Calculates unplanned outage financial losses versus predictive maintenance turnaround savings."""

    def __init__(self):
        self.asset_registry = {a["asset_id"]: a for a in FLEET_ASSETS}

    def calculate_intervention_roi(
        self,
        asset_id: str,
        work_order: WorkOrder,
        unplanned_downtime_hours: float = 14.0,
        planned_turnaround_hours: float = 3.5,
        emergency_overtime_multiplier: float = 2.0,
        collateral_damage_multiplier: float = 2.4,
    ) -> Dict[str, Any]:
        """Calculates granular financial impact of early predictive intervention versus run-to-failure outage."""
        asset = self.asset_registry.get(asset_id, {})
        downtime_cost_per_hr = float(asset.get("downtime_cost_per_hr_usd", 4500.0))

        # 1. Unplanned Failure Scenario
        unplanned_prod_loss = unplanned_downtime_hours * downtime_cost_per_hr
        unplanned_labor_cost = (
            work_order.est_labor_hours * work_order.est_labor_rate_usd_hr * emergency_overtime_multiplier
        )
        unplanned_parts_cost = work_order.total_parts_cost * collateral_damage_multiplier
        total_unplanned_cost = unplanned_prod_loss + unplanned_labor_cost + unplanned_parts_cost

        # 2. Predictive Planned Turnaround Scenario
        planned_prod_loss = planned_turnaround_hours * downtime_cost_per_hr * 0.35  # Often mitigated via buffer/shift
        planned_labor_cost = work_order.est_labor_hours * work_order.est_labor_rate_usd_hr
        planned_parts_cost = work_order.total_parts_cost
        total_planned_cost = planned_prod_loss + planned_labor_cost + planned_parts_cost

        # 3. Financial Delta & ROI
        net_savings_usd = max(0.0, total_unplanned_cost - total_planned_cost)
        roi_ratio = (net_savings_usd / max(1.0, total_planned_cost)) * 100.0

        return {
            "asset_id": asset_id,
            "downtime_rate_hourly": downtime_cost_per_hr,
            "unplanned_outage_hours": unplanned_downtime_hours,
            "planned_turnaround_hours": planned_turnaround_hours,
            "unplanned": {
                "production_loss_usd": round(unplanned_prod_loss, 2),
                "emergency_labor_usd": round(unplanned_labor_cost, 2),
                "collateral_parts_usd": round(unplanned_parts_cost, 2),
                "total_cost_usd": round(total_unplanned_cost, 2),
            },
            "planned": {
                "production_loss_usd": round(planned_prod_loss, 2),
                "scheduled_labor_usd": round(planned_labor_cost, 2),
                "oem_parts_usd": round(planned_parts_cost, 2),
                "total_cost_usd": round(total_planned_cost, 2),
            },
            "net_savings_usd": round(net_savings_usd, 2),
            "roi_percentage": round(roi_ratio, 1),
        }
