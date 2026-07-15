from __future__ import annotations

from pathlib import Path
from typing import Any

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle
from PIL import Image as PILImage

from iaei.contracts import ContractError, validate_report_payload
from iaei.paths import ROOT


PAGE_TITLES = (
    "1. Executive decision",
    "2. Data and Databricks architecture",
    "3. Machine learning and chronological validation",
    "4. Structural drift and constrained optimization",
    "5. Governed agents and business impact",
)


def _as_rows(section: dict[str, Any]) -> list[list[str]]:
    rows: list[list[str]] = []
    for key, value in section.items():
        if isinstance(value, (dict, list)):
            continue
        rows.append([str(key).replace("_", " ").title(), str(value)])
    return rows[:6]


def _figure(path_value: str) -> Image:
    chart_path = Path(path_value)
    if not chart_path.is_absolute():
        chart_path = ROOT / chart_path
    if not chart_path.exists() or chart_path.stat().st_size < 30_000:
        raise ContractError(f"Required report figure is missing or below size gate: {chart_path}")
    with PILImage.open(chart_path) as image:
        width, height = image.size
    if width < 2_400 or height < 1_200:
        raise ContractError(f"Report figure is below the required dimensions: {chart_path} ({width}x{height})")
    target_width = 180 * mm
    target_height = target_width * height / width
    if target_height > 91 * mm:
        target_height = 91 * mm
        target_width = target_height * width / height
    return Image(str(chart_path), width=target_width, height=target_height)


def build_technical_brief(payload_path: Path, output_path: Path) -> Path:
    payload = validate_report_payload(payload_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    styles = getSampleStyleSheet()
    title = ParagraphStyle(
        "BriefTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=21,
        alignment=TA_LEFT,
        spaceAfter=4 * mm,
    )
    body = ParagraphStyle(
        "BriefBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=8.8,
        leading=11.5,
        spaceAfter=2.5 * mm,
    )
    small = ParagraphStyle("Small", parent=body, fontSize=7.1, leading=9)

    doc = SimpleDocTemplate(
        str(output_path),
        pagesize=A4,
        rightMargin=15 * mm,
        leftMargin=15 * mm,
        topMargin=12 * mm,
        bottomMargin=12 * mm,
        title="Industrial Adaptive Energy Intelligence - Technical Brief",
        author="Rodolfo Pereira",
    )

    sections = [
        payload["executive"],
        payload["data"],
        payload["models"],
        {**payload["drift"], **payload["optimization"]},
        {**payload["agents"], **payload["impact"]},
    ]

    story: list[Any] = []
    for index, (page_title, section) in enumerate(zip(PAGE_TITLES, sections, strict=True), start=1):
        story.append(Paragraph(page_title, title))
        narrative = section.get("narrative") if isinstance(section, dict) else None
        if narrative:
            story.append(Paragraph(str(narrative), body))
        story.append(_figure(payload["visuals"][f"page_{index}"]))
        story.append(Spacer(1, 2.5 * mm))
        rows = _as_rows(section if isinstance(section, dict) else {})
        if rows:
            table = Table(rows, colWidths=[45 * mm, 125 * mm], repeatRows=0)
            table.setStyle(
                TableStyle(
                    [
                        ("FONTNAME", (0, 0), (0, -1), "Helvetica-Bold"),
                        ("FONTNAME", (1, 0), (1, -1), "Helvetica"),
                        ("FONTSIZE", (0, 0), (-1, -1), 7.3),
                        ("LEADING", (0, 0), (-1, -1), 8.7),
                        ("VALIGN", (0, 0), (-1, -1), "TOP"),
                        ("GRID", (0, 0), (-1, -1), 0.25, colors.HexColor("#C8D0D7")),
                        ("BACKGROUND", (0, 0), (0, -1), colors.HexColor("#F1F4F6")),
                        ("LEFTPADDING", (0, 0), (-1, -1), 4),
                        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
                        ("TOPPADDING", (0, 0), (-1, -1), 3.5),
                        ("BOTTOMPADDING", (0, 0), (-1, -1), 3.5),
                    ]
                )
            )
            story.append(table)
        story.append(Spacer(1, 2.5 * mm))
        story.append(
            Paragraph(
                "Independent industrial energy analysis. Licensed real industrial energy data only. "
                "No proprietary company data or unsupported savings claim.",
                small,
            )
        )
        if index < 5:
            story.append(PageBreak())

    doc.build(story)
    if not output_path.exists() or output_path.stat().st_size < 10_000:
        raise ContractError("PDF build did not produce a valid artifact")
    return output_path
