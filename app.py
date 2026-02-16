# app.py

import streamlit as st
import pandas as pd
from datetime import timedelta
import plotly.express as px

from utils.scheduler import schedule_loads
from utils.prediction_pipeline import predict_next_day_production

# Configuration

st.set_page_config(page_title="Strathmore EMS Dashboard", layout="wide")
st.title("🔋 Strathmore University Energy Management Dashboard")

LATITUDE = -1.2921
LONGITUDE = 36.8219
NBO_TZ = "Africa/Nairobi"

# Computation of "tomorrow" in Africa/Nairobi to show on the dashboard
now_nbo = pd.Timestamp.now(tz=NBO_TZ)
day_start = now_nbo.normalize() + pd.Timedelta(days=1)   
day_end   = day_start + pd.Timedelta(days=1)             
st.caption(f"Prediction date: **{day_start.strftime('%A, %d %B %Y')}** (Africa/Nairobi)")

# Cache the Predictions

@st.cache_data(ttl=60 * 15)  # cache for 15 minutes
def _predict_cached(lat: float, lon: float) -> pd.DataFrame:
    """Call the pipeline and cache the DataFrame."""
    return predict_next_day_production(lat=lat, lon=lon)

# Forecast + Solar Predictions

irradiance_data = _predict_cached(LATITUDE, LONGITUDE)

# Quick sanity metrics (computed in Nairobi time)
dfp = irradiance_data.sort_values("timestamp").copy()
gti = pd.to_numeric(dfp["Global Tilted Irradiation"], errors="coerce").fillna(0)
pv  = pd.to_numeric(dfp["predicted_solar_production"], errors="coerce").fillna(0)

def _to_nairobi(ts: pd.Timestamp) -> pd.Timestamp:
    ts = pd.to_datetime(ts)
    if ts.tz is None:
        return ts.tz_localize(NBO_TZ)
    return ts.tz_convert(NBO_TZ)

peak_gti_ts = _to_nairobi(dfp.loc[gti.idxmax(), "timestamp"])
peak_pv_ts  = _to_nairobi(dfp.loc[pv.idxmax(),  "timestamp"])

total_pv_kwh = float(pv.sum()) / 1000.0                 # Wh/slot to kWh
daily_irr_kwh_m2 = float((gti * 0.5).sum() / 1000.0)    # W/m² * 0.5h to Wh/m² then to kWh/m²

m1, m2, m3, m4 = st.columns(4)
m1.metric("Peak GTI time", peak_gti_ts.strftime("%I:%M %p"))
m2.metric("Peak PV time",  peak_pv_ts.strftime("%I:%M %p"))
m3.metric("Total PV (kWh)", f"{total_pv_kwh:,.0f}")
m4.metric("Daily Irr (kWh/m²)", f"{daily_irr_kwh_m2:,.2f}")

# Building a plotting frame in *local* time to align charts with metrics
plot_df = irradiance_data.copy()
# Ensuring tz-aware in Nairobi, then dropping tz so Vega-Lite won't convert to UTC
plot_df["ts_local"] = (
    pd.to_datetime(plot_df["timestamp"])
      .dt.tz_convert(NBO_TZ)     # Nairobi tz-aware
      .dt.tz_localize(None)      # making naive to prevent UTC auto-shift in Streamlit chart
)
plot_df = plot_df.set_index("ts_local")

# The Charts row
col1, col2 = st.columns(2)

with col1:
    st.subheader("🌤️ Forecasted Irradiance (Next Day)")
    st.line_chart(plot_df["Global Tilted Irradiation"], use_container_width=True)
    st.caption("Series: Global Tilted Irradiation (GTI), units: W/m² — times shown in Africa/Nairobi")

with col2:
    st.subheader("☀️ Predicted Solar Production (Next Day)")
    st.line_chart(plot_df["predicted_solar_production"], use_container_width=True)
    st.caption("Series: Predicted PV output, units: Wh per 30-min slot — times shown in Africa/Nairobi")

# Building an Optimal Load Schedule 

load_data = pd.read_csv("data/load_data.csv", parse_dates=["timestamp"])
scheduled_df = schedule_loads(load_data, irradiance_data)

def _build_timeline_from_schedule(df: pd.DataFrame, device_cols):
    """
    Convert a 30-min schedule dataframe into contiguous (Start, End) intervals per device.
    Returns a frame with tz-aware Start/End in Africa/Nairobi plus naive local copies
    for plotting (prevents Plotly from auto-converting to UTC).
    """
    # Ensuring that tz-aware in Nairobi
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True).dt.tz_convert(NBO_TZ)
    df = df.sort_values("timestamp").set_index("timestamp")

    rows = []

    for col in device_cols:
        s = df[col].fillna(0)
        if (s > 0).sum() == 0:
            continue

        in_run = False
        start_ts = None
        prev_ts = None

        for ts, val in s.items():
            on = val > 0
            if on and not in_run:
                start_ts = ts
                in_run = True
            elif (not on) and in_run:
                end_ts = prev_ts + pd.Timedelta(minutes=30)
                rows.append({
                    "Device": col.replace("_kW", "").replace("_", " "),
                    "Start": start_ts,     # tz-aware
                    "End": end_ts,         # tz-aware (right-exclusive)
                })
                in_run = False
            prev_ts = ts

        # If we ended while still running, close the interval
        if in_run and prev_ts is not None:
            end_ts = prev_ts + pd.Timedelta(minutes=30)
            rows.append({
                "Device": col.replace("_kW", "").replace("_", " "),
                "Start": start_ts,
                "End": end_ts,
            })

    timeline = pd.DataFrame(rows)
    if timeline.empty:
        return timeline

    # Restriction to tomorrow for safety
    mask = (timeline["Start"] >= day_start) & (timeline["Start"] < day_end)
    timeline = timeline.loc[mask].copy()


    timeline["StartLocal"] = timeline["Start"].dt.tz_convert(NBO_TZ).dt.tz_localize(None)
    timeline["EndLocal"]   = timeline["End"].dt.tz_convert(NBO_TZ).dt.tz_localize(None)


    timeline["Duration (min)"] = (timeline["End"] - timeline["Start"]).dt.total_seconds() / 60.0

    return timeline

# Keep only tomorrow and devices
sched_plot = scheduled_df.copy()
sched_plot["timestamp"] = pd.to_datetime(sched_plot["timestamp"], utc=True).dt.tz_convert(NBO_TZ)
sched_plot = sched_plot[(sched_plot["timestamp"] >= day_start) & (sched_plot["timestamp"] < day_end)].copy()

device_columns = [
    c for c in sched_plot.columns
    if c.endswith("_kW") and c not in ("base_load_kW", "total_load_kW")
]

timeline_df = _build_timeline_from_schedule(sched_plot, device_columns)

st.subheader(f"🗓️ Optimal Load Schedule — {day_start.strftime('%d %b %Y')} (30-min slots)")
if timeline_df.empty:
    st.info("No controllable loads were scheduled for tomorrow.")
else:
    # Using the local naive columns so Plotly doesn't auto-convert timezones
    fig_timeline = px.timeline(
        timeline_df,
        x_start="StartLocal",
        x_end="EndLocal",
        y="Device",
        color="Device",
        hover_data={
            "StartLocal": True,
            "EndLocal": True,
            "Duration (min)": True,
            "Start": False,   # hides tz-aware versions from hover
            "End": False,
        },
        title=f"Device Run Windows (Africa/Nairobi) — {day_start.strftime('%d %b %Y')}",
        template="plotly_dark"
    )
    # Showing devices from top to bottom
    fig_timeline.update_yaxes(autorange="reversed")
    # Fixing x-axis to tomorrow (local, naive)
    x0 = day_start.tz_convert(NBO_TZ).tz_localize(None)
    x1 = day_end.tz_convert(NBO_TZ).tz_localize(None)
    fig_timeline.update_xaxes(
        range=[x0, x1],
        tickformat="%H:%M",
        tick0=x0,
        dtick=2 * 60 * 60 * 1000  # every 2 hours
    )
    fig_timeline.update_traces(opacity=0.95, selector=dict(type="bar"))
    fig_timeline.update_layout(
        height=460,
        legend_title_text="Device",
        xaxis_title="Time of Day (Africa/Nairobi)",
        yaxis_title=""
    )
    st.plotly_chart(fig_timeline, use_container_width=True)

# Export scheduled loads

csv = scheduled_df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download Load Schedule CSV", csv, file_name="scheduled_loads.csv", mime="text/csv")

