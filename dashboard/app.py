"""Live dashboard for the Delhi very short-term load forecaster.

Reads the readings and forecasts written to Firestore by the hourly job and shows:
  - the current load, weather and the next-hour forecast with its confidence band,
  - a chart of recent actual load vs forecast with the 95% band,
  - recent forecast accuracy (MAPE/MAE over the last days),
  - the model's held-out test metrics.

Run locally:   streamlit run dashboard/app.py
Deploy free:   Streamlit Community Cloud (see README).
"""
import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src import firebase_client as fb
from src import predict

st.set_page_config(page_title="Delhi Load Forecast", page_icon="⚡", layout="wide")
st.title("⚡ Delhi very short-term load forecast")
st.caption("Sequential MNM–GPR hybrid · one-hour-ahead · live data stored on Firebase")


@st.cache_data(ttl=300)
def load_data():
    readings = pd.DataFrame(fb.get_recent_readings(limit=240))
    forecasts = pd.DataFrame(fb.get_recent_forecasts(limit=240))
    return readings, forecasts


try:
    readings, forecasts = load_data()
except Exception as e:
    st.error(f"Could not read from Firestore: {e}")
    st.stop()

if readings.empty:
    st.warning("No data yet. Run the seed script and let the hourly job run once.")
    st.stop()

readings["ts"] = pd.to_datetime(readings["ts"])
readings = readings.sort_values("ts")
latest = readings.iloc[-1]

# upcoming forecast = the forecast row whose target is after the latest reading
upcoming = None
if not forecasts.empty:
    forecasts["target_ts"] = pd.to_datetime(forecasts["target_ts"])
    forecasts = forecasts.sort_values("target_ts")
    future = forecasts[forecasts["target_ts"] > latest["ts"]]
    if not future.empty:
        upcoming = future.iloc[0]

# ---- top metric cards ----
c1, c2, c3, c4 = st.columns(4)
c1.metric("Current load", f"{latest['Load']:.0f} MW")
c2.metric("Temperature", f"{latest['Temperature']:.1f} °C")
c3.metric("Humidity (sp.)", f"{latest['Humidity']:.1f} g/kg")
if upcoming is not None:
    delta = upcoming["forecast_mw"] - latest["Load"]
    c4.metric("Next-hour forecast", f"{upcoming['forecast_mw']:.0f} MW", f"{delta:+.0f} MW")
    st.info(f"Next hour ({upcoming['target_ts']:%H:%M}): "
            f"**{upcoming['forecast_mw']:.0f} MW**  "
            f"(95% band {upcoming['lower_mw']:.0f}–{upcoming['upper_mw']:.0f} MW)")

# ---- actual vs forecast chart ----
st.subheader("Actual load vs forecast")
hist = readings[["ts", "Load"]].rename(columns={"Load": "Actual"})
if not forecasts.empty:
    fc = forecasts[["target_ts", "forecast_mw", "lower_mw", "upper_mw"]].rename(
        columns={"target_ts": "ts", "forecast_mw": "Forecast"})
    merged = pd.merge(hist, fc, on="ts", how="outer").sort_values("ts")
else:
    merged = hist
st.line_chart(merged.set_index("ts")[[c for c in ["Actual", "Forecast"] if c in merged]])

# ---- accuracy over recent forecasts ----
st.subheader("Recent forecast accuracy")
if not forecasts.empty and "actual_mw" in forecasts:
    ev = forecasts.dropna(subset=["actual_mw", "forecast_mw"]).copy()
    if not ev.empty:
        ev["abs_err"] = (ev["actual_mw"] - ev["forecast_mw"]).abs()
        ev["ape"] = 100 * ev["abs_err"] / ev["actual_mw"]
        a1, a2, a3 = st.columns(3)
        a1.metric("Live MAE", f"{ev['abs_err'].mean():.1f} MW")
        a2.metric("Live MAPE", f"{ev['ape'].mean():.2f} %")
        a3.metric("Forecasts evaluated", f"{len(ev)}")
    else:
        st.caption("Accuracy will appear once forecasts can be compared with actuals.")
else:
    st.caption("Accuracy will appear after the first full hour cycle.")

# ---- model card ----
with st.expander("Model details (held-out test set)"):
    try:
        info = predict.model_info()
        m = info.get("test_metrics", {})
        if m:
            st.write(pd.DataFrame([m]).T.rename(columns={0: "value"}))
        st.caption(f"Trained on {info.get('n_train')} hourly samples · lags {info.get('lags')}")
    except Exception as e:
        st.caption(f"Model card unavailable: {e}")

st.caption("Data: Delhi SLDC (load) · OpenWeather (weather) · stored on Firebase Firestore.")
