"""PDF health-advisory report (PRD BR-009 / TR-011)."""
import io
from datetime import datetime

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.platypus import HRFlowable, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from dashboard.core import ALL_POLLUTANTS, MODEL_LABELS, SAFE_LIMITS, UNITS
from dashboard.tiers import ALL_TIERS

INK = colors.HexColor("#16181D")
MUTED = colors.HexColor("#6B7280")
LINE = colors.HexColor("#E3E0D8")
SCOPE = ("Decision-support estimate from a machine-learning model trained on historical CPCB data (2015–2020). "
         "It is not a certified regulatory measurement, is not connected to live sensor networks, and must not be "
         "used to issue legal or regulatory orders or for medical decisions.")


def _styles():
    s = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("t", parent=s["Title"], fontName="Helvetica-Bold", fontSize=20, leading=24,
                                textColor=INK, alignment=TA_LEFT, spaceAfter=2),
        "sub": ParagraphStyle("s", parent=s["Normal"], fontSize=9, textColor=MUTED, leading=12),
        "h": ParagraphStyle("h", parent=s["Heading2"], fontName="Helvetica-Bold", fontSize=12, textColor=INK,
                            spaceBefore=12, spaceAfter=6),
        "body": ParagraphStyle("b", parent=s["Normal"], fontSize=10, leading=14, textColor=INK),
        "small": ParagraphStyle("sm", parent=s["Normal"], fontSize=8, leading=11, textColor=MUTED),
    }


def _table(rows, widths, header=True):
    t = Table(rows, colWidths=widths)
    style = [("FONT", (0, 0), (-1, -1), "Helvetica", 9), ("TEXTCOLOR", (0, 0), (-1, -1), INK),
             ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINE), ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
             ("TOPPADDING", (0, 0), (-1, -1), 4), ("BOTTOMPADDING", (0, 0), (-1, -1), 4)]
    if header:
        style += [("FONT", (0, 0), (-1, 0), "Helvetica-Bold", 8), ("TEXTCOLOR", (0, 0), (-1, 0), MUTED)]
    t.setStyle(TableStyle(style))
    return t


def prediction_report(job: dict, batch_summary: dict | None = None) -> bytes:
    st_ = _styles()
    buf = io.BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=18 * mm, rightMargin=18 * mm, topMargin=16 * mm,
                            bottomMargin=16 * mm, title="AQI Health Advisory Report")
    tier = job["tier"]
    story = [
        Paragraph("AQI Health Advisory Report", st_["title"]),
        Paragraph(f"Job {job['job_id']} · generated {datetime.now():%d %b %Y, %H:%M} · "
                  f"model {job['model_version']}", st_["sub"]),
        Spacer(1, 8), HRFlowable(width="100%", color=LINE), Spacer(1, 10),
    ]

    loc = " · ".join(x for x in [job.get("city"), str(job["date"]) if job.get("date") else None] if x) or "User-supplied reading"
    hero = Table([[
        Paragraph(f"<font size=9 color='#6B7280'>PREDICTED AQI</font><br/><font size=34><b>{job['predicted_aqi']:.0f}</b></font>",
                  st_["body"]),
        Paragraph(f"<font size=9>HEALTH TIER (CPCB)</font><br/><font size=16><b>{tier.label}</b></font>"
                  f"<br/><font size=9>AQI range {tier.range_text}</font>",
                  ParagraphStyle("tier", parent=st_["body"], textColor=colors.HexColor(tier.text_color))),
        Paragraph(f"<font size=9 color='#6B7280'>STATUS</font><br/><b>{job['status'].replace('_', ' ')}</b><br/>"
                  f"<font size=9>80% error band: {job['interval'][0]:.0f}–{job['interval'][1]:.0f}</font><br/>"
                  f"<font size=9>{loc}</font>", st_["body"]),
    ]], colWidths=[50 * mm, 55 * mm, 69 * mm], rowHeights=[30 * mm])
    hero.setStyle(TableStyle([("BACKGROUND", (1, 0), (1, 0), colors.HexColor(tier.color)),
                              ("TEXTCOLOR", (1, 0), (1, 0), colors.HexColor(tier.text_color)),
                              ("VALIGN", (0, 0), (-1, -1), "MIDDLE"), ("LEFTPADDING", (0, 0), (-1, -1), 8),
                              ("BOX", (0, 0), (-1, -1), 0.5, LINE), ("INNERGRID", (0, 0), (-1, -1), 0.5, LINE)]))
    story += [hero]

    story += [Paragraph("Health advisory", st_["h"]), Paragraph(job["advisory"], st_["body"])]
    if job.get("review_reasons"):
        story += [Paragraph("Manual review required", st_["h"])]
        story += [Paragraph(f"• {r}", st_["body"]) for r in job["review_reasons"]]

    story += [Paragraph("Pollutant readings", st_["h"])]
    rows = [["Pollutant", "Entered", "Used by model", "Safe limit (24 h)", "vs limit"]]
    for p in ALL_POLLUTANTS:
        entered = job["inputs"].get(p)
        used = job.get("model_input", {}).get(p)
        used_txt = "not a model input" if p not in job.get("model_input", {}) else (
            f"{used:.2f}" + (" (median)" if p in job.get("imputed", []) else " (capped)" if entered is not None and used < entered - 1e-9 else ""))
        ratio = (entered / SAFE_LIMITS[p]) if entered is not None else None
        rows.append([f"{p} ({UNITS[p]})", "—" if entered is None else f"{entered:g}", used_txt, f"{SAFE_LIMITS[p]}",
                     "—" if ratio is None else f"{ratio:.1f}×"])
    story.append(_table(rows, [38 * mm, 26 * mm, 40 * mm, 34 * mm, 36 * mm]))

    story += [Paragraph("Top contributing pollutants (SHAP)", st_["h"]),
              Paragraph(f"Starting from the model's average prediction of {job['shap_base']:.0f}, each pollutant pushes "
                        f"the estimate up or down:", st_["body"]), Spacer(1, 4)]
    shap_rows = [["Pollutant", "Contribution (AQI points)"]] + [
        [k, f"{v:+.1f}"] for k, v in sorted(job["shap"].items(), key=lambda kv: -abs(kv[1]))]
    story.append(_table(shap_rows, [60 * mm, 60 * mm]))

    if batch_summary:
        story += [Paragraph("Batch CSV summary", st_["h"]),
                  Paragraph(f"{batch_summary['rows_scored']:,} of {batch_summary['rows_total']:,} rows scored · "
                            f"{batch_summary['review_required']:,} flagged for review", st_["body"]), Spacer(1, 4)]
        story.append(_table([["Tier", "Rows"]] + [[t.label, f"{batch_summary['tiers'].get(t.label, 0):,}"] for t in ALL_TIERS],
                            [60 * mm, 40 * mm]))

    story += [Spacer(1, 14), HRFlowable(width="100%", color=LINE), Spacer(1, 6),
              Paragraph(f"Model: {MODEL_LABELS.get(job['model_key'], job['model_key'])}. Tier thresholds: CPCB National AQI "
                        f"(Good 0–50, Satisfactory 51–100, Moderate 101–200, Poor 201–300, Very Poor 301–400, Severe 401+).",
                        st_["small"]),
              Spacer(1, 4), Paragraph(SCOPE, st_["small"])]
    doc.build(story)
    return buf.getvalue()
