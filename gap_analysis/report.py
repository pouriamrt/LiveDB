"""Phase 5: Report generation -- PDF + HTML dashboard."""

from __future__ import annotations

import os
from pathlib import Path
from xml.sax.saxutils import escape

from loguru import logger as log
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import (
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)

from gap_analysis.models import GapReport

TEMPLATE_DIR = Path(__file__).parent / "templates"

GAP_TYPE_COLORS = {
    "contradiction": "#e74c3c",
    "under_explored": "#f39c12",
    "methodological": "#3498db",
    "population": "#9b59b6",
    "missing_comparison": "#1abc9c",
    "future_direction": "#2ecc71",
}

SEVERITY_COLORS = {
    "high": "#e74c3c",
    "medium": "#f39c12",
    "low": "#27ae60",
}


def generate_pdf(report: GapReport, output_path: str) -> str:
    """Generate a PDF report from a GapReport."""
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    doc = SimpleDocTemplate(output_path, pagesize=letter)
    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "CustomTitle", parent=styles["Title"], fontSize=18, spaceAfter=20
    )
    heading_style = ParagraphStyle(
        "CustomHeading", parent=styles["Heading1"], fontSize=14, spaceAfter=10
    )
    body_style = ParagraphStyle(
        "CustomBody", parent=styles["Normal"], fontSize=10, spaceAfter=6, leading=14
    )

    elements: list = []

    # Title
    elements.append(Paragraph("Research Gap Analysis Report", title_style))
    elements.append(Paragraph(f"Topic: {report.query}", body_style))
    elements.append(
        Paragraph(
            f"Date: {report.created_at.strftime('%Y-%m-%d')} | "
            f"Papers analyzed: {report.scope} | "
            f"Period: {report.date_range}",
            body_style,
        )
    )
    elements.append(Spacer(1, 0.3 * inch))

    # Executive Summary
    elements.append(Paragraph("1. Executive Summary", heading_style))
    elements.append(Paragraph(escape(report.executive_summary or "N/A"), body_style))
    elements.append(Spacer(1, 0.2 * inch))

    # Research Landscape
    elements.append(Paragraph("2. Research Landscape", heading_style))
    elements.append(Paragraph("<b>Methodology Overview</b>", body_style))
    elements.append(Paragraph(escape(report.methodology_overview or "N/A"), body_style))
    elements.append(Paragraph("<b>Population Coverage</b>", body_style))
    elements.append(Paragraph(escape(report.population_overview or "N/A"), body_style))

    # Theme table
    if report.themes:
        theme_data = [["Theme", "Papers", "Description"]]
        for t in report.themes:
            theme_data.append([t.label, str(len(t.papers)), t.description[:80]])
        theme_table = Table(theme_data, colWidths=[2 * inch, 0.8 * inch, 4 * inch])
        theme_table.setStyle(
            TableStyle(
                [
                    ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#2c3e50")),
                    ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                    ("FONTSIZE", (0, 0), (-1, -1), 9),
                    ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                    ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ]
            )
        )
        elements.append(Spacer(1, 0.1 * inch))
        elements.append(theme_table)
    elements.append(Spacer(1, 0.2 * inch))

    # Gaps
    elements.append(Paragraph("3. Identified Research Gaps", heading_style))
    for gap in report.gaps:
        color = SEVERITY_COLORS.get(gap.severity, "#999")
        elements.append(
            Paragraph(
                f'<font color="{color}">[{escape(gap.severity.upper())}]</font> '
                f"<b>{escape(gap.gap_type)}</b>: {escape(gap.title)}",
                body_style,
            )
        )
        elements.append(Paragraph(escape(gap.description), body_style))
        if gap.evidence:
            evidence_text = ", ".join(escape(e) for e in gap.evidence[:5])
            elements.append(Paragraph(f"<i>Evidence: {evidence_text}</i>", body_style))
        if gap.suggested_research:
            elements.append(
                Paragraph(
                    f"<i>Suggested research: {escape(gap.suggested_research)}</i>",
                    body_style,
                )
            )
        elements.append(Spacer(1, 0.1 * inch))

    # Conclusion
    elements.append(Paragraph("4. Conclusion &amp; Priorities", heading_style))
    elements.append(Paragraph(escape(report.conclusion or "N/A"), body_style))

    # Appendix
    elements.append(Spacer(1, 0.3 * inch))
    elements.append(Paragraph("Appendix: Analyzed Papers", heading_style))
    for theme in report.themes:
        for p in theme.papers:
            elements.append(
                Paragraph(
                    f"<b>{p.paper.title}</b> — {p.paper.journal or 'N/A'} "
                    f"({p.paper.publication_date or 'N/A'})",
                    body_style,
                )
            )

    doc.build(elements)
    log.info(f"PDF report saved to {output_path}")
    return output_path


def generate_dashboard_html(report: GapReport) -> str:
    """Generate an HTML dashboard string from a GapReport."""
    from jinja2 import Environment, FileSystemLoader

    env = Environment(
        loader=FileSystemLoader(str(TEMPLATE_DIR)),
        autoescape=True,
    )
    template = env.get_template("dashboard.html")
    return template.render(
        report=report,
        gap_type_colors=GAP_TYPE_COLORS,
        severity_colors=SEVERITY_COLORS,
    )
