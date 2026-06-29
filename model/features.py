"""Feature construction and scaling for very short-term load forecasting.

Input record schema (one row per hour, chronological):
    Temperature, DewPoint, Humidity, Load

The target is the NEXT hour's load. Features are six load lags
[1, 2, 3, 4, 24, 48] hours plus the three weather variables of the target hour.
"""
import numpy as np

LAGS = [1, 2, 3, 4, 24, 48]
MAX_LAG = max(LAGS)               # 48 hours of history needed to make one forecast
WEATHER_COLS = ["Temperature", "DewPoint", "Humidity"]
TGT_LO, TGT_HI = 0.10, 0.90       # target is rescaled into this band (sigmoid-friendly)


def build_supervised(load, weather):
    """Turn aligned hourly arrays into (X, y) for next-hour forecasting.

    load:    1-D array of hourly load (MW), length N
    weather: 2-D array (N, 3) of [Temperature, DewPoint, Humidity] aligned to load
    Returns X (M, 9) and y (M,) where M = N - MAX_LAG - 1.
    Row i predicts the load at time (MAX_LAG + 1 + i); its weather features are the
    weather AT the target hour (known from the forecast issue time in operation).
    """
    load = np.asarray(load, dtype=float)
    weather = np.asarray(weather, dtype=float)
    N = len(load)
    rows_X, rows_y = [], []
    for t in range(MAX_LAG, N - 1):          # t is the issue time; predict t+1
        lagvals = [load[t + 1 - L] for L in LAGS]   # lag L before the TARGET hour t+1
        feats = lagvals + list(weather[t + 1])      # weather at target hour
        rows_X.append(feats)
        rows_y.append(load[t + 1])
    return np.array(rows_X), np.array(rows_y)


def build_one(load_history, target_weather):
    """Build a single feature vector to forecast the next hour in live operation.

    load_history: list/array of the most recent load values, oldest..newest,
                  with at least MAX_LAG values (index -1 is the current hour).
    target_weather: [Temperature, DewPoint, Humidity] for the hour being forecast.
    """
    h = np.asarray(load_history, dtype=float)
    if len(h) < MAX_LAG:
        raise ValueError(f"need at least {MAX_LAG} hours of load history, got {len(h)}")
    # lag L before the target hour == h[-L] (h[-1] is current hour = lag 1 of target)
    lagvals = [h[-L] for L in LAGS]
    return np.array(lagvals + list(target_weather), dtype=float).reshape(1, -1)


class MinMaxScaler:
    """Simple per-column min-max scaler fitted on the training set only."""

    def __init__(self):
        self.xmin = self.xmax = None
        self.ymin = self.ymax = None

    def fit(self, X, y):
        self.xmin, self.xmax = X.min(0), X.max(0)
        self.ymin, self.ymax = float(y.min()), float(y.max())
        return self

    def transform_X(self, X):
        rng = np.where((self.xmax - self.xmin) == 0, 1.0, self.xmax - self.xmin)
        return (X - self.xmin) / rng

    def transform_y(self, y):
        rng = (self.ymax - self.ymin) or 1.0
        return TGT_LO + (np.asarray(y, float) - self.ymin) / rng * (TGT_HI - TGT_LO)

    def inverse_y(self, ys):
        rng = (self.ymax - self.ymin) or 1.0
        return (np.asarray(ys, float) - TGT_LO) / (TGT_HI - TGT_LO) * rng + self.ymin
