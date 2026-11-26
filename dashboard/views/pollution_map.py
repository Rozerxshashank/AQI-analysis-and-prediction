"""4. Pollution Map — Folium heatmap (PRD: Pollution heatmap, UC-6)."""
import folium
import plotly.graph_objects as go
import streamlit as st
from folium.plugins import HeatMap

from dashboard import core, ui
from dashboard.tiers import ALL_TIERS, get_health_tier

SEASONS = ["Winter", "Pre-monsoon", "Monsoon", "Post-monsoon"]


@st.cache_data(show_spinner=False)
def _city_values(metric: str, years: tuple, seasons: tuple):
    df = core.dataset()
    d = df[df["Date"].dt.year.between(*years) & df["Date"].dt.month.map(core.season_of).isin(seasons)]
    g = d.groupby("City")[metric].agg(["mean", "count"]).dropna()
    return g[g["count"] > 0].sort_values("mean", ascending=False)


def _map(values, metric, highlight=None):
    m = folium.Map(location=[22.5, 80.5], zoom_start=5, control_scale=True,
                   tiles="https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}",
                   attr="Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ")
    vmax = values["mean"].max()
    HeatMap([[*core.CITY_COORDS[c], float(v / vmax)] for c, v in values["mean"].items()],
            radius=38, blur=30, min_opacity=0.35,
            gradient={0.2: "#2E9E5B", 0.4: "#E8B62C", 0.65: "#E8812C", 0.85: "#D9443A", 1: "#8E2C48"}).add_to(m)
    for c, row in values.iterrows():
        v = row["mean"]
        if metric == "AQI":
            t = get_health_tier(v)
            color, label = t.color, f"{t.icon} {t.label}"
        else:
            ratio = v / core.SAFE_LIMITS[metric]
            color = "#2E9E5B" if ratio <= 0.5 else "#E8B62C" if ratio <= 1 else "#D9443A"
            label = f"{ratio:.1f}× the CPCB safe limit"
        unit = "" if metric == "AQI" else f" {core.UNITS[metric]}"
        folium.CircleMarker(
            location=core.CITY_COORDS[c], radius=7 + 9 * (v / vmax), color="#16181D" if c == highlight else "#FFFFFF",
            weight=3 if c == highlight else 1.5, fill=True, fill_color=color, fill_opacity=0.95,
            tooltip=folium.Tooltip(f"<b>{c}</b><br>{metric}: {v:.1f}{unit}<br>{label}<br>{int(row['count']):,} days"),
        ).add_to(m)
    return m


def render():
    ui.header("Pollution Map", "Where the air is worst",
              "City averages from the historical dataset. Heat shows relative intensity; markers carry the value and tier label.")
    st.write("")
    c1, c2, c3 = st.columns([1, 1.4, 1.6])
    metric = c1.selectbox("Measure", ["AQI"] + core.ALL_POLLUTANTS)
    years = c2.slider("Years", 2015, 2020, (2015, 2020))
    seasons = c3.multiselect("Seasons", SEASONS, default=SEASONS)
    if not seasons:
        st.warning("Select at least one season.")
        return
    values = _city_values(metric, years, tuple(seasons))
    if values.empty:
        st.info(f"No {metric} readings for that selection.")
        return
    highlight = (st.session_state.get("result") or {}).get("city")

    left, right = st.columns([1.7, 1], gap="medium")
    with left.container(border=True):
        st.iframe(_map(values, metric, highlight).get_root().render(), height=600)
        if metric == "AQI":
            st.markdown(ui.tier_scale(), unsafe_allow_html=True)
        if highlight:
            st.caption(f"Outlined marker: {highlight}, the city of your latest prediction.")
    with right.container(border=True):
        unit = "" if metric == "AQI" else f" ({core.UNITS[metric]})"
        colors = ([get_health_tier(v).color for v in values["mean"]] if metric == "AQI"
                  else ["#D9443A" if v > core.SAFE_LIMITS[metric] else "#3B5BDB" for v in values["mean"]])
        fig = go.Figure(go.Bar(y=values.index[::-1], x=values["mean"][::-1], orientation="h", marker_color=colors[::-1],
                               text=[f"{v:.0f}" if v >= 10 else f"{v:.2f}" for v in values["mean"][::-1]], textposition="outside",
                               hovertemplate="%{y}: %{x:.1f}<extra></extra>"))
        if metric != "AQI":
            fig.add_vline(x=core.SAFE_LIMITS[metric], line_dash="dash", line_color="#16181D",
                          annotation_text="safe limit", annotation_position="top")
        st.plotly_chart(ui.style_fig(fig, 640).update_layout(title=f"Average {metric}{unit} by city")
                        .update_xaxes(range=[0, values["mean"].max() * 1.18]), width="stretch", config=ui.CHART_CONFIG)
        if metric != "AQI":
            st.caption("Red bars exceed the CPCB 24-hour safe limit on average.")
        else:
            st.caption("Recorded history, before cleaning: these averages include the AQI readings above 500 "
                       "(mostly Ahmedabad) that were removed before training the models.")
