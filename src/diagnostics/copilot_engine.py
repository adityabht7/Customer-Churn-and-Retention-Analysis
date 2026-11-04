"""Industrial reliability copilot and automated CMMS work order generation engine."""

from __future__ import annotations
import uuid
from datetime import datetime
from typing import Dict, Any, List, Optional
from src.core.constants import FLEET_ASSETS, FAILURE_MODES, ISO_10816_ZONES
from src.core.schema import TelemetryReading, AssetHealthState, WorkOrder, WorkOrderItem
from src.diagnostics.root_cause import RootCauseAnalyzer


class MaintenanceCopilotEngine:
    """Intelligent reliability engineering copilot delivering prescriptive maintenance actions and CMMS work orders."""

    def __init__(self):
        self.root_cause_analyzer = RootCauseAnalyzer()
        self.asset_registry = {a["asset_id"]: a for a in FLEET_ASSETS}

    def generate_work_order(
        self,
        reading: TelemetryReading,
        health_state: AssetHealthState,
        labor_rate_usd: float = 85.0,
    ) -> WorkOrder:
        """Dispatches an actionable, ISO-compliant Computerized Maintenance Management System (CMMS) work order."""
        mode_id = health_state.predicted_failure_mode_id
        fault_info = FAILURE_MODES.get(mode_id, FAILURE_MODES[0])
        asset_meta = self.asset_registry.get(reading.asset_id, {})

        now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        wo_id = f"WO-{datetime.now().strftime('%Y%m%d')}-{reading.asset_id.replace('-', '')}-{uuid.uuid4().hex[:4].upper()}"

        # Assign priority based on ISO zone and RUL
        if health_state.iso_severity_zone == "ZONE_D" or health_state.predicted_rul_hrs < 48.0 or mode_id in [3, 4]:
            priority = "P1-Immediate Action (Safety Interlock Required)"
        elif health_state.iso_severity_zone == "ZONE_C" or health_state.predicted_rul_hrs < 120.0 or mode_id in [1, 2]:
            priority = "P2-Urgent Turnaround (< 72h)"
        elif health_state.health_index < 75.0:
            priority = "P3-Scheduled Preventative Maintenance"
        else:
            priority = "P4-Routine Inspection Round"

        # Map parts
        parts_data = fault_info.get("recommended_parts", [])
        recommended_items = [
            WorkOrderItem(
                part_no=p["part_no"],
                desc=p["desc"],
                qty=p["qty"],
                unit_cost_usd=p["unit_cost_usd"],
            )
            for p in parts_data
        ]

        # Detailed step-by-step field instructions
        instructions = (
            f"DIAGNOSTIC SUMMARY: {fault_info['name']} detected with {health_state.failure_probability*100:.1f}% confidence. "
            f"ISO 10816 Vibration Severity: {health_state.iso_severity_zone} ({reading.vibration_de_rms} mm/s RMS). "
            f"Forecasted Remaining Useful Life: {health_state.predicted_rul_hrs:.1f} operating hours.\n\n"
            f"OPERATIONAL PROCEDURE:\n"
            f"1. Mandatory Safety: Implement {fault_info.get('loto_procedure', 'Standard Mechanical Isolation')}. Verify zero kinetic energy before cowl disassembly.\n"
            f"2. Sensor Validation: Connect portable FFT datalogger on Drive-End and Non-Drive-End bearing blocks to confirm spectral energy peaks.\n"
            f"3. Core Intervention: {fault_info['recommended_action']}\n"
            f"4. Re-commissioning & Acceptance: Rotate rotor by hand to verify smooth clearance. Perform dynamic spin-up to rated {asset_meta.get('rated_rpm', 1800)} RPM. "
            f"Verify Drive-End vibration remains below 2.8 mm/s RMS (ISO Zone A) for continuous 30-minute run."
        )

        return WorkOrder(
            work_order_id=wo_id,
            created_at=now_str,
            asset_id=reading.asset_id,
            asset_name=asset_meta.get("asset_name", reading.asset_id),
            location=asset_meta.get("location", "Plant Floor"),
            priority=priority,
            fmea_failure_mode=fault_info["name"],
            fmea_severity=fault_info["fmea_severity"],
            health_index=health_state.health_index,
            predicted_rul_hrs=health_state.predicted_rul_hrs,
            loto_required=fault_info.get("loto_required", False),
            loto_procedure=fault_info.get("loto_procedure"),
            required_craft=fault_info.get("required_craft", "Mechanical Maintenance Technician"),
            est_labor_hours=float(fault_info.get("est_labor_hours", 2.0)),
            est_labor_rate_usd_hr=labor_rate_usd,
            recommended_parts=recommended_items,
            action_instructions=instructions,
        )

    def ask_copilot(
        self,
        query: str,
        reading: TelemetryReading,
        health_state: AssetHealthState,
    ) -> str:
        """Answers maintenance and reliability engineering queries with technical precision."""
        q = query.lower()
        asset_id = reading.asset_id
        meta = self.asset_registry.get(asset_id, {})
        deviations = self.root_cause_analyzer.attribute_deviations(reading)
        top_dev = deviations[0] if deviations else None

        # 1. Root cause / What is wrong queries
        if any(w in q for w in ["wrong", "issue", "fault", "root cause", "problem", "diagnos"]):
            if health_state.predicted_failure_mode_id == 0 and health_state.iso_severity_zone in ["ZONE_A", "ZONE_B"]:
                return (
                    f"**Asset Health Status: Healthy Nominal Operation**\n\n"
                    f"{asset_id} ({meta.get('asset_name')}) is operating within standard operating limits. "
                    f"Drive-End vibration velocity is {reading.vibration_de_rms} mm/s RMS (ISO 10816 Zone {health_state.iso_severity_zone[-1]}). "
                    f"Bearing temperature is steady at {reading.bearing_temperature_c}°C against a nominal baseline of {meta.get('nominal_temp_c')}°C. "
                    f"No anomalous shock pulses or harmonic misalignment signatures are currently detected."
                )

            return (
                f"**Root-Cause Diagnostic Assessment: {health_state.predicted_failure_mode_name}**\n\n"
                f"- **Primary Defect Mechanism**: {health_state.predicted_failure_mode_name} (Confidence: {health_state.failure_probability*100:.1f}%)\n"
                f"- **ISO 10816-3 Status**: {health_state.iso_severity_zone} ({reading.vibration_de_rms} mm/s RMS)\n"
                f"- **Critical Sensor Attribution**: `{top_dev['sensor']}` is deviating by {top_dev['pct_change']:+.1f}% "
                f"({top_dev['current_value']} {top_dev['unit']} vs nominal {top_dev['baseline_nominal']} {top_dev['unit']}, z-score {top_dev['z_score']:+.2f}).\n\n"
                f"**Engineering Diagnosis**: The signature is characterized by {FAILURE_MODES[health_state.predicted_failure_mode_id]['description']}"
            )

        # 2. RUL / Prognostics queries
        elif any(w in q for w in ["rul", "remaining", "life", "time", "hours", "trip", "fail", "when"]):
            return (
                f"**Prognostics & Remaining Useful Life (RUL) Forecast**\n\n"
                f"- **Point Estimate**: **{health_state.predicted_rul_hrs:.1f} Operating Hours**\n"
                f"- **90% Confidence Interval**: [{health_state.rul_confidence_lower_hrs:.1f} hrs — {health_state.rul_confidence_upper_hrs:.1f} hrs]\n"
                f"- **Health Index**: {health_state.health_index:.1f}% / 100%\n\n"
                f"**Recommendation**: Plan maintenance intervention within **{max(1.0, health_state.predicted_rul_hrs*0.65):.0f} hours** "
                f"to maintain a 35% safety margin before boundary lubrication breakdown or mechanical seizure occurs."
            )

        # 3. Parts / Spares queries
        elif any(w in q for w in ["part", "spare", "kit", "replace", "bom", "stage"]):
            parts = FAILURE_MODES[health_state.predicted_failure_mode_id].get("recommended_parts", [])
            if not parts:
                return f"No replacement parts currently required for {asset_id}. Standard consumables (grease/oil) suffice for scheduled PM."

            lines = [f"- **{p['part_no']}**: {p['desc']} (Qty: {p['qty']}, Est. Unit Cost: ${p['unit_cost_usd']:.2f})" for p in parts]
            total_parts = sum(p["qty"] * p["unit_cost_usd"] for p in parts)
            return (
                f"**OEM Spare Parts Requisition for {asset_id}**\n\n"
                + "\n".join(lines)
                + f"\n\n**Total Estimated Parts Cost**: ${total_parts:,.2f}\n"
                f"**Assigned Trade Craft**: {FAILURE_MODES[health_state.predicted_failure_mode_id].get('required_craft')}"
            )

        # 4. Safety / LOTO queries
        elif any(w in q for w in ["loto", "safety", "lockout", "permit", "procedure"]):
            fault = FAILURE_MODES[health_state.predicted_failure_mode_id]
            loto_req = fault.get("loto_required", False)
            proc = fault.get("loto_procedure", "Standard Mechanical/Electrical Isolation")
            return (
                f"**Environmental Health & Safety (EHS) / LOTO Requirements**\n\n"
                f"- **LOTO Required**: {'YES (Mandatory before opening casing)' if loto_req else 'No - Non-invasive monitoring'}\n"
                f"- **Protocol ID**: `{proc}`\n"
                f"- **Safety Notice**: Confirm zero-pressure differential across manifolds and zero electrical potential at drive inverter before touching rotating assembly."
            )

        # 5. General fallback / Comprehensive overview
        else:
            return (
                f"**Asset Intelligence Summary for {asset_id} ({meta.get('asset_name')})**\n\n"
                f"- **Condition State**: {health_state.status_label} (Health Index: {health_state.health_index}%)\n"
                f"- **Diagnosis**: {health_state.predicted_failure_mode_name} (ISO 10816 Severity: {health_state.iso_severity_zone})\n"
                f"- **Forecasted RUL**: {health_state.predicted_rul_hrs} hrs\n"
                f"- **Vibration RMS**: {reading.vibration_de_rms} mm/s | **Bearing Temp**: {reading.bearing_temperature_c}°C\n"
                f"- **Prescriptive Action**: {FAILURE_MODES[health_state.predicted_failure_mode_id]['recommended_action']}"
            )
