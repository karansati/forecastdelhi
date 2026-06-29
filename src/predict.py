"""Load the trained artifact and forecast the next hour's load with a 95% band."""
import numpy as np
import joblib

import config
from model.features import build_one

_ART = None


def _artifact():
    global _ART
    if _ART is None:
        _ART = joblib.load(config.MODEL_PATH)
    return _ART


def forecast_next_hour(load_history, target_weather):
    """Forecast the load for the next hour.

    load_history:   list of recent hourly load values, oldest..newest (>= 48 values).
    target_weather: dict with Temperature, DewPoint, Humidity for the hour to forecast.
    Returns dict: point forecast (MW) and 95% lower/upper bounds.
    """
    art = _artifact()
    scaler = art["scaler"]
    model = art["model"]
    wcols = art["weather_cols"]

    tw = [float(target_weather[c]) for c in wcols]
    X = build_one(load_history, tw)
    Xs = scaler.transform_X(X)
    ys, std = model.predict(Xs, return_std=True)

    point = float(scaler.inverse_y(ys)[0])
    span = (scaler.ymax - scaler.ymin) / (art["tgt_hi"] - art["tgt_lo"])
    half = 1.96 * float(std[0]) * span
    return {
        "forecast_mw": round(point, 1),
        "lower_mw": round(point - half, 1),
        "upper_mw": round(point + half, 1),
        "band_mw": round(half, 1),
    }


def model_info():
    art = _artifact()
    return {"test_metrics": art.get("test_metrics", {}),
            "n_train": art.get("n_train"), "lags": art.get("lags")}
