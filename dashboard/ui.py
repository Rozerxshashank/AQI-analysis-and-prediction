"""Design system: CSS, HTML components and the Plotly theme used across all screens."""
import math

import plotly.graph_objects as go
import streamlit as st

from dashboard.tiers import ALL_TIERS, HealthTier, get_health_tier

ACCENT = "#3B5BDB"
INK = "#16181D"
INK2 = "#4A4F5A"
MUTED = "#8A8F99"
LINE = "#E3E0D8"
SURFACE2 = "#F0EEE8"
LIB = "#A3A8B3"

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;600&display=swap');
.stApp { font-family: 'Inter', system-ui, sans-serif; }
.stApp h1, .stApp h2, .stApp h3 { font-family: 'Inter', system-ui, sans-serif; letter-spacing: -0.025em; }
.block-container { padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1240px; }
[data-testid="stSidebar"] { border-right: 1px solid #E3E0D8; }
[data-testid="stSidebarNav"] a span { font-weight: 500; }

/* bordered containers become cards */
[data-testid="stVerticalBlockBorderWrapper"]:has(> div > [data-testid="stVerticalBlock"]) {
  background: #FFFFFF; border-radius: 16px !important; box-shadow: 0 1px 2px rgba(20,22,28,.04), 0 8px 24px rgba(20,22,28,.05);
}
[data-testid="stMetric"] { background: #FFFFFF; border: 1px solid #E3E0D8; border-radius: 14px; padding: 14px 16px;
  container-type: inline-size; }
/* Size the number to the tile's own width (cqi = 1% of the tile), capped at Streamlit's normal 2.25rem,
   and let values/labels wrap instead of being cut off with "…" on narrow screens. */
[data-testid="stMetricValue"] { font-family: 'JetBrains Mono', monospace; font-weight: 600; letter-spacing: -0.02em;
  font-size: clamp(1.15rem, 22cqi, 2.25rem); line-height: 1.15; }
[data-testid="stMetricValue"] *, [data-testid="stMetricLabel"] * {
  white-space: normal !important; overflow: visible !important; text-overflow: clip !important; }
[data-testid="stMetricLabel"] p { color: #6B7280; font-size: 0.82rem; line-height: 1.3; }
.stTabs [data-baseweb="tab-list"] { gap: 4px; }
.stTabs [data-baseweb="tab"] { padding: 8px 14px; border-radius: 10px 10px 0 0; font-weight: 600; }
div[data-testid="stDownloadButton"] button, div[data-testid="stButton"] button { border-radius: 10px; font-weight: 600; }

.aq-eyebrow { font-family: 'JetBrains Mono', monospace; font-size: 12px; text-transform: uppercase; letter-spacing: .12em; color: #3B5BDB; margin-bottom: 6px; }
.aq-title { font-size: clamp(28px, 3.6vw, 42px); line-height: 1.08; font-weight: 700; letter-spacing: -0.03em; color: #16181D; margin: 0; }
.aq-lede { color: #4A4F5A; font-size: 16px; max-width: 720px; margin: 10px 0 6px; line-height: 1.55; }
.aq-mono { font-family: 'JetBrains Mono', monospace; font-variant-numeric: tabular-nums; }
.aq-muted { color: #6B7280; font-size: 13px; }

.aq-badge { display: inline-flex; align-items: center; gap: 8px; border-radius: 999px; font-weight: 700; padding: 6px 14px 6px 12px; font-size: 15px; line-height: 1; }
.aq-badge .ic { font-size: 11px; }
.aq-badge.sm { font-size: 12px; padding: 4px 10px 4px 9px; gap: 6px; }

.aq-hero { display: grid; grid-template-columns: minmax(260px, 380px) 1fr; gap: 28px; align-items: center; }
@media (max-width: 860px) { .aq-hero { grid-template-columns: 1fr; } }
.aq-big { font-family: 'JetBrains Mono', monospace; font-size: 72px; font-weight: 600; line-height: .95; letter-spacing: -0.05em; color: #16181D; }
.aq-k { font-size: 12px; color: #6B7280; text-transform: uppercase; letter-spacing: .08em; font-family: 'JetBrains Mono', monospace; }

.aq-steps { display: flex; flex-wrap: wrap; gap: 6px; align-items: center; margin: 4px 0 2px; }
.aq-step { font-family: 'JetBrains Mono', monospace; font-size: 11.5px; padding: 5px 10px; border-radius: 8px; background: #F0EEE8; color: #8A8F99; }
.aq-step.done { background: #E7ECFB; color: #3B5BDB; }
.aq-step.final-ok { background: #2E9E5B; color: #fff; }
.aq-step.final-review { background: #E8B62C; color: #16181D; }
.aq-step.final-fail { background: #D9443A; color: #fff; }
.aq-arrow { color: #C4C1B8; font-size: 12px; }

.aq-callout { border-radius: 12px; padding: 14px 16px; font-size: 14.5px; line-height: 1.55; }
.aq-callout.review { background: #FDF6E3; border: 1px solid #F0D58A; }
.aq-callout.ok { background: #EDF7F0; border: 1px solid #B9DFC6; }
.aq-callout.scope { background: #F0EEE8; border: 1px solid #E3E0D8; color: #4A4F5A; font-size: 13.5px; }
.aq-callout b { color: #16181D; }

.aq-flow { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }
.aq-flow span.n { background: #FFFFFF; border: 1px solid #E3E0D8; border-radius: 10px; padding: 8px 12px; font-size: 13.5px; font-weight: 600; }
.aq-flow span.n.hl { border-color: #3B5BDB; color: #3B5BDB; background: #F2F5FE; }
.aq-flow span.a { color: #A3A8B3; }

.aq-scale { display: flex; border-radius: 10px; overflow: hidden; }
.aq-scale div { flex: 1; padding: 8px 6px; font-size: 11.5px; line-height: 1.3; }
.aq-scale div b { display: block; font-size: 12.5px; }
</style>
"""


def inject_css():
    st.markdown(CSS, unsafe_allow_html=True)


def header(eyebrow: str, title: str, lede: str | None = None):
    st.markdown(f"<div class='aq-eyebrow'>{eyebrow}</div><h1 class='aq-title'>{title}</h1>"
                + (f"<p class='aq-lede'>{lede}</p>" if lede else ""), unsafe_allow_html=True)


def tier_badge(tier: HealthTier | str, small=False) -> str:
    t = tier if isinstance(tier, HealthTier) else next(x for x in ALL_TIERS if x.label == tier)
    return (f"<span class='aq-badge{' sm' if small else ''}' style='background:{t.color};color:{t.text_color}' "
            f"role='status' aria-label='Health tier: {t.label}'><span class='ic'>{t.icon}</span>{t.label}</span>")


def tier_scale() -> str:
    cells = "".join(f"<div style='background:{t.color};color:{t.text_color}'><b>{t.icon} {t.label}</b>{t.range_text}</div>"
                    for t in ALL_TIERS)
    return f"<div class='aq-scale' role='list' aria-label='CPCB AQI tiers'>{cells}</div>"


def gauge(aqi: float, actual: float | None = None, interval=None) -> str:
    """Semicircle gauge with CPCB bands, prediction marker, error band and optional actual marker."""
    cx, cy, R, r = 200, 200, 176, 146

    def pt(v, rad):
        a = math.pi * (1 - max(0, min(500, v)) / 500)
        return cx + rad * math.cos(a), cy - rad * math.sin(a)

    parts, lo = [], 0
    for t, hi in zip(ALL_TIERS, [50, 100, 200, 300, 400, 500]):
        a1, a2, b1, b2 = pt(lo + .8, R), pt(hi - .8, R), pt(hi - .8, r), pt(lo + .8, r)
        large = 1 if (hi - lo) > 250 else 0
        parts.append(f"<path d='M{a1[0]:.1f},{a1[1]:.1f} A{R},{R} 0 {large} 1 {a2[0]:.1f},{a2[1]:.1f} "
                     f"L{b1[0]:.1f},{b1[1]:.1f} A{r},{r} 0 {large} 0 {b2[0]:.1f},{b2[1]:.1f}Z' fill='{t.color}'>"
                     f"<title>{t.label} {t.range_text}</title></path>")
        lo = hi
    for tick in [0, 50, 100, 200, 300, 400, 500]:
        x, y = pt(tick, R + 14)
        parts.append(f"<text x='{x:.1f}' y='{y + 4:.1f}' text-anchor='middle' font-size='11' font-family='JetBrains Mono' fill='#8A8F99'>{tick}</text>")
    if interval:
        a, b = interval
        p1, p2 = pt(a, r - 10), pt(b, r - 10)
        large = 1 if (b - a) > 250 else 0
        parts.append(f"<path d='M{p1[0]:.1f},{p1[1]:.1f} A{r - 10},{r - 10} 0 {large} 1 {p2[0]:.1f},{p2[1]:.1f}' "
                     f"stroke='#16181D' stroke-opacity='.28' stroke-width='6' fill='none' stroke-linecap='round'>"
                     f"<title>80% error band {a:.0f}–{b:.0f}</title></path>")
    if actual is not None:
        x1, y1 = pt(actual, R + 2)
        x2, y2 = pt(actual, R + 16)
        parts.append(f"<line x1='{x1:.1f}' y1='{y1:.1f}' x2='{x2:.1f}' y2='{y2:.1f}' stroke='#16181D' stroke-width='3'/>"
                     f"<text x='{x2:.1f}' y='{y2 - 6:.1f}' text-anchor='middle' font-size='10' font-weight='700' fill='#16181D'>actual</text>")
    mx, my = pt(aqi, (R + r) / 2)
    parts.append(f"<circle cx='{mx:.1f}' cy='{my:.1f}' r='19' fill='#fff' stroke='#16181D' stroke-width='3'/>"
                 f"<circle cx='{mx:.1f}' cy='{my:.1f}' r='7' fill='#16181D'/>")
    t = get_health_tier(aqi)
    parts.append(f"<text x='200' y='178' text-anchor='middle' font-size='64' font-weight='600' font-family='JetBrains Mono' "
                 f"fill='#16181D' letter-spacing='-3'>{aqi:.0f}</text>"
                 f"<text x='200' y='202' text-anchor='middle' font-size='12' font-family='JetBrains Mono' fill='#6B7280'>PREDICTED AQI</text>")
    return (f"<svg viewBox='-6 0 412 214' width='100%' role='img' aria-label='Predicted AQI {aqi:.0f}, {t.label}'>"
            + "".join(parts) + "</svg>")


STATES = ["SUBMITTED", "PROCESSING", "PREDICTED", "TIER_MAPPED"]


def stepper(states: list[str]) -> str:
    final = states[-1] if states else None
    html = []
    for s in STATES:
        html.append(f"<span class='aq-step {'done' if s in states else ''}'>{s}</span><span class='aq-arrow'>→</span>")
    cls = {"COMPLETED": "final-ok", "REVIEW_REQUIRED": "final-review", "PROCESSING_FAILED": "final-fail"}.get(final, "")
    end = final if final in ("COMPLETED", "REVIEW_REQUIRED", "PROCESSING_FAILED") else "COMPLETED / REVIEW_REQUIRED"
    html.append(f"<span class='aq-step {cls}'>{end}</span>")
    return f"<div class='aq-steps' aria-label='Job status'>{''.join(html)}</div>"


def callout(html: str, kind="scope"):
    st.markdown(f"<div class='aq-callout {kind}'>{html}</div>", unsafe_allow_html=True)


SCOPE_TEXT = ("<b>Scope boundary.</b> Predictions are decision-support estimates from historical CPCB data (2015–2020). "
              "This is not a certified regulatory monitor, is not connected to live sensor networks, and does not issue "
              "legal or regulatory orders.")


def style_fig(fig: go.Figure, height=360, legend=False) -> go.Figure:
    fig.update_layout(
        height=height, margin=dict(l=8, r=8, t=36, b=8), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, system-ui, sans-serif", size=12, color=INK2), showlegend=legend,
        title=dict(font=dict(size=14, color=INK), x=0, xanchor="left"),
        hoverlabel=dict(bgcolor=INK, font=dict(color="#fff", family="Inter")),
        legend=dict(orientation="h", y=1.08, x=0, font=dict(size=11)),
    )
    fig.update_xaxes(gridcolor="#ECE9E1", zerolinecolor="#D6D2C8", linecolor=LINE, tickfont=dict(family="JetBrains Mono", size=10))
    fig.update_yaxes(gridcolor="#ECE9E1", zerolinecolor="#D6D2C8", linecolor=LINE, tickfont=dict(family="JetBrains Mono", size=10))
    return fig


CHART_CONFIG = {"displayModeBar": False}
