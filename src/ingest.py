"""Hourly job: fetch live weather + load, store them, forecast the next hour, store it.

Run once per hour (see .github/workflows/hourly.yml). Each run:
  1. fetches the current Delhi load (SLDC) and weather (OpenWeather),
  2. writes this hour's reading to Firestore,
  3. pulls the last 48+ hours of load from Firestore,
  4. forecasts the next hour with a 95% confidence band and writes it,
  5. back-fills the 'actual' value of the forecast made one hour ago (for accuracy).
"""
import sys
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import config
from src import firebase_client as fb
from src import weather as wx
from src import sldc
from src import predict

IST = ZoneInfo(config.TIMEZONE)


def run():
    now = datetime.now(IST).replace(minute=0, second=0, microsecond=0)
    this_id = fb.hour_id(now)
    next_hour = now + timedelta(hours=1)
    next_id = fb.hour_id(next_hour)

    # 1. fetch live data
    w = wx.fetch_weather()
    l = sldc.fetch_load()
    reading = {
        "ts": now.isoformat(),
        "hour": this_id,
        "Load": l["Load"],
        "Temperature": w["Temperature"],
        "DewPoint": w["DewPoint"],
        "Humidity": w["Humidity"],
        "rh_percent": w["rh_percent"],
        "weather_desc": w["weather_desc"],
        "sldc_slot": l["slot"],
    }
    fb.save_reading(this_id, reading)
    print(f"[{this_id}] load={l['Load']} MW  temp={w['Temperature']}C  "
          f"dew={w['DewPoint']}C  q={w['Humidity']} g/kg")

    # 5. back-fill the actual for the forecast that targeted THIS hour
    fb.save_forecast(this_id, {"actual_mw": l["Load"]})

    # 3 + 4. forecast next hour if we have enough history
    readings = fb.get_recent_readings(limit=80)
    loads = [r["Load"] for r in readings if r.get("Load") is not None]
    if len(loads) < 48:
        print(f"history has {len(loads)} hours; need 48 to forecast. "
              "Run seed_history.py once to bootstrap.")
        return

    fc = predict.forecast_next_hour(loads, w)   # weather persistence: reuse current wx
    fc_doc = {
        "target_ts": next_hour.isoformat(),
        "target_hour": next_id,
        "issued_ts": now.isoformat(),
        **fc,
    }
    fb.save_forecast(next_id, fc_doc)
    print(f"[{next_id}] forecast={fc['forecast_mw']} MW  "
          f"95% band=[{fc['lower_mw']}, {fc['upper_mw']}]")


if __name__ == "__main__":
    try:
        run()
    except Exception as e:                      # fail loudly in CI logs
        print("INGEST ERROR:", repr(e), file=sys.stderr)
        sys.exit(1)
