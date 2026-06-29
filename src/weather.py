"""Fetch live weather from OpenWeather and convert to the model's features.

The model was trained on three weather variables:
    Temperature (degrees C), DewPoint (degrees C), Humidity (specific humidity, g/kg)

OpenWeather's current-weather endpoint returns temperature, relative humidity (%)
and pressure (hPa). We derive dew point (Magnus formula) and specific humidity from
these so the live features match the training data.
"""
import math
import requests

import config


def _dew_point(temp_c, rh_pct):
    a, b = 17.625, 243.04
    gamma = math.log(max(rh_pct, 1e-3) / 100.0) + (a * temp_c) / (b + temp_c)
    return (b * gamma) / (a - gamma)


def _specific_humidity(dew_c, pressure_hpa):
    e = 6.112 * math.exp((17.67 * dew_c) / (dew_c + 243.5))   # vapour pressure (hPa)
    w = 0.622 * e / max(pressure_hpa - e, 1e-3)                # mixing ratio (kg/kg)
    q = w / (1.0 + w)                                          # specific humidity
    return q * 1000.0                                          # g/kg


def fetch_weather():
    """Return dict with Temperature, DewPoint, Humidity (model features)."""
    if not config.OPENWEATHER_API_KEY:
        raise RuntimeError("OPENWEATHER_API_KEY is not set.")
    params = {"lat": config.DELHI_LAT, "lon": config.DELHI_LON,
              "appid": config.OPENWEATHER_API_KEY, "units": "metric"}
    r = requests.get(config.OPENWEATHER_URL, params=params, timeout=20)
    r.raise_for_status()
    j = r.json()
    temp = float(j["main"]["temp"])
    rh = float(j["main"]["humidity"])
    pressure = float(j["main"].get("pressure", 1013.0))
    dew = _dew_point(temp, rh)
    q = _specific_humidity(dew, pressure)
    return {
        "Temperature": round(temp, 2),
        "DewPoint": round(dew, 2),
        "Humidity": round(q, 2),
        "rh_percent": round(rh, 1),
        "weather_desc": j.get("weather", [{}])[0].get("description", ""),
    }


if __name__ == "__main__":
    print(fetch_weather())
