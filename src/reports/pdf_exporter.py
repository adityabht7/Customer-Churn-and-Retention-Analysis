"""PDF Work Order and Reliability Audit Report Exporter using ReportLab."""

from __future__ import annotations
import io
from typing import Dict, Any, Optional
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    HRFlowable,
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from src.core.schema import WorkOrder, AssetHealthState, TelemetryReading


def generate_work_order_pdf(
    work_order: WorkOrder,
    health_state: AssetHealthState,
    reading: TelemetryReading,
    economic_data: Optional[Dict[str, Any]] = None,
) -> bytes:
    """Renders an ISO-compliant CMMS Maintenance Dispatch PDF into an in-memory byte buffer."""
    buffer = io.BytesIO()
    doc = SimpleDocTemplate(
        buffer,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        "DocTitle",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=4,
    )
    subtitle_style = ParagraphStyle(
        "DocSub",
        parent=styles["Normal"],
        fontName="Helvetica-Bold",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#3b82f6"),
        spaceAfter=12,
    )
    section_heading = ParagraphStyle(
        "SecHead",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=10,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "BodyDark",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#334155"),
    )
    code_style = ParagraphStyle(
        "CodeText",
        parent=styles["Normal"],
        fontName="Courier",
        fontSize=8,
        leading=11,
        textColor=colors.HexColor("#0f172a"),
    )

    story = []

    # Title Banner
    story.append(Paragraph("AuraVibe IIoT Condition Monitoring & CMMS Dispatch", title_style))
    story.append(
        Paragraph(
            f"ISO 13374 Condition-Based Maintenance Directive | Work Order Ref: {work_order.work_order_id}",
            subtitle_style,
        )
    )
    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#cbd5e1"), spaceAfter=10))

    # Meta Summary Table
    meta_table_data = [
        [
            Paragraph("<b>Asset ID:</b>", body_style),
            Paragraph(work_order.asset_id, body_style),
            Paragraph("<b>Priority:</b>", body_style),
            Paragraph(f"<font color='red'><b>{work_order.priority}</b></font>", body_style),
        ],
        [
            Paragraph("<b>Equipment:</b>", body_style),
            Paragraph(work_order.asset_name, body_style),
            Paragraph("<b>Date Dispatched:</b>", body_style),
            Paragraph(work_order.created_at, body_style),
        ],
        [
            Paragraph("<b>Plant Location:</b>", body_style),
            Paragraph(work_order.location, body_style),
            Paragraph("<b>Assigned Trade:</b>", body_style),
            Paragraph(work_order.required_craft, body_style),
        ],
        [
            Paragraph("<b>Health Index:</b>", body_style),
            Paragraph(f"{work_order.health_index:.1f}%", body_style),
            Paragraph("<b>Forecasted RUL:</b>", body_style),
            Paragraph(f"{work_order.predicted_rul_hrs:.1f} Operating Hours", body_style),
        ],
    ]
    meta_table = Table(meta_table_data, colWidths=[90, 180, 90, 180])
    meta_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
            ("BOX", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("INNERGRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
            ("TOPPADDING", (0, 0), (-1, -1), 4),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ])
    )
    story.append(meta_table)
    story.append(Spacer(1, 10))

    # Safety Notice & LOTO
    story.append(Paragraph("Mandatory EHS & Lockout/Tagout (LOTO) Compliance", section_heading))
    loto_banner_color = colors.HexColor("#fee2e2") if work_order.loto_required else colors.HexColor("#f0fdf4")
    loto_text = (
        f"<b>MANDATORY ISOLATION:</b> {work_order.loto_procedure}."
        if work_order.loto_required
        else "Standard physical safety precautions apply. No mechanical lock required."
    )
    loto_table = Table([[Paragraph(loto_text, body_style)]], colWidths=[540])
    loto_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), loto_banner_color),
            ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#ef4444") if work_order.loto_required else colors.HexColor("#22c55e")),
            ("PADDING", (0, 0), (-1, -1), 6),
        ])
    )
    story.append(loto_table)
    story.append(Spacer(1, 10))

    # Telemetry Snapshot Table
    story.append(Paragraph("Physical Telemetry & ISO 10816 Diagnostic Baseline", section_heading))
    telemetry_table_data = [
        ["Channel / Sensor", "Measured Value", "Engineering Unit", "Diagnostic Assessment"],
        ["Drive-End Vibration RMS", f"{reading.vibration_de_rms:.3f}", "mm/s", f"ISO Zone: {health_state.iso_severity_zone}"],
        ["Non-Drive Vibration RMS", f"{reading.vibration_nde_rms:.3f}", "mm/s", "Within relative balance envelope"],
        ["Vibration Kurtosis", f"{reading.vibration_kurtosis:.2f}", "dimensionless", "Shock pulse impact marker" if reading.vibration_kurtosis > 4.5 else "Nominal"],
        ["Bearing Housing Temp", f"{reading.bearing_temperature_c:.1f}", "°C", "Thermal overload" if reading.bearing_temperature_c > 80 else "Acceptable"],
        ["Motor Winding Temp", f"{reading.motor_winding_temperature_c:.1f}", "°C", "Stator thermal envelope"],
        ["Differential Pressure", f"{reading.pressure_differential_bar:.2f}", "bar", "Flow restriction marker" if reading.pressure_differential_bar > 8.0 else "Nominal"],
        ["Acoustic Emission", f"{reading.acoustic_emission_db:.1f}", "dB", "High frequency stress wave"],
    ]
    t_table = Table(telemetry_table_data, colWidths=[150, 90, 110, 190])
    t_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e293b")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8fafc")]),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ("PADDING", (0, 0), (-1, -1), 4),
        ])
    )
    story.append(t_table)
    story.append(Spacer(1, 10))

    # Prescriptive Action Instructions
    story.append(Paragraph("Prescriptive Maintenance Action & Overhaul Protocol", section_heading))
    clean_instructions = work_order.action_instructions.replace("\n", "<br/>")
    story.append(Paragraph(clean_instructions, body_style))
    story.append(Spacer(1, 10))

    # Required OEM Spare Parts
    story.append(Paragraph("Required OEM Parts & Consumables Bill of Materials (BOM)", section_heading))
    parts_data = [["OEM Part Number", "Component Description", "Qty", "Unit Cost (USD)", "Subtotal (USD)"]]
    for item in work_order.recommended_parts:
        parts_data.append([
            item.part_no,
            item.desc,
            str(item.qty),
            f"${item.unit_cost_usd:.2f}",
            f"${item.total_cost:.2f}",
        ])
    if len(work_order.recommended_parts) == 0:
        parts_data.append(["N/A", "No OEM replacement parts required for this task", "-", "$0.00", "$0.00"])

    parts_data.append(["", "", "", "Total Parts Cost:", f"${work_order.total_parts_cost:.2f}"])
    parts_data.append(["", "", "", f"Labor ({work_order.est_labor_hours}h @ ${work_order.est_labor_rate_usd_hr}/h):", f"${work_order.total_labor_cost:.2f}"])
    parts_data.append(["", "", "", "Total Work Order Estimate:", f"${work_order.total_estimated_cost:.2f}"])

    p_table = Table(parts_data, colWidths=[120, 200, 40, 90, 90])
    p_table.setStyle(
        TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#334155")),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("GRID", (0, 0), (-1, -4), 0.5, colors.HexColor("#cbd5e1")),
            ("LINEABOVE", (3, -3), (-1, -1), 1, colors.HexColor("#0f172a")),
            ("FONTNAME", (3, -1), (-1, -1), "Helvetica-Bold"),
            ("PADDING", (0, 0), (-1, -1), 3),
        ])
    )
    story.append(p_table)
    story.append(Spacer(1, 12))

    # Economic impact note
    if economic_data:
        savings = economic_data.get("net_savings_usd", 0.0)
        story.append(
            Paragraph(
                f"<b>Economic Avoidance:</b> Predictive turnaround avoids an estimated "
                f"<b>${savings:,.2f}</b> in unplanned downtime production loss.",
                body_style,
            )
        )
        story.append(Spacer(1, 10))

    # Sign-Off Blocks
    story.append(Spacer(1, 15))
    signoff_data = [
        ["Field Reliability Engineer", "Safety / EHS Officer", "Maintenance Supervisor"],
        ["\n\n__________________________\nSignature & Date", "\n\n__________________________\nSignature & Date", "\n\n__________________________\nSignature & Date"],
    ]
    s_table = Table(signoff_data, colWidths=[180, 180, 180])
    s_table.setStyle(
        TableStyle([
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("FONTSIZE", (0, 0), (-1, -1), 8),
            ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ])
    )
    story.append(s_table)

    doc.build(story)
    pdf_bytes = buffer.getvalue()
    buffer.close()
    return pdf_bytes
