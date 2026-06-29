"""Train the Sequential MNM-GPR forecaster and save a deployable artifact.

Usage:
    python -m model.train --csv path/to/history.csv --out artifacts/model.joblib

The CSV must have columns: Temperature, DewPoint, Humidity, Load  (one row per hour,
in chronological order). This is the same format produced by the ingestion job, so
the model can be retrained periodically on the data collected in the cloud.
"""
import argparse
import csv
import json
import os
import numpy as np
import joblib

from .features import build_supervised, MinMaxScaler, LAGS, WEATHER_COLS, TGT_LO, TGT_HI
from .mnm_gpr import SequentialMNMGPR


def read_csv(path):
    rows = []
    with open(path) as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append([float(row["Temperature"]), float(row["DewPoint"]),
                         float(row["Humidity"]), float(row["Load"])])
    arr = np.array(rows, dtype=float)
    return arr[:, :3], arr[:, 3]


def metrics(y, yhat):
    err = y - yhat
    return {
        "MAE": float(np.mean(np.abs(err))),
        "RMSE": float(np.sqrt(np.mean(err ** 2))),
        "MAPE": float(100 * np.mean(np.abs(err / y))),
        "R2": float(1 - np.sum(err ** 2) / np.sum((y - y.mean()) ** 2)),
    }


def train(csv_path, out_path, gpr_subset=3000, seed=1, eval_split=True):
    weather, load = read_csv(csv_path)
    X, y = build_supervised(load, weather)

    if eval_split:
        n = len(X); a = int(0.8 * n); b = int(0.9 * n)
        Xtr, ytr, Xte, yte = X[:a], y[:a], X[b:], y[b:]
    else:
        Xtr, ytr = X, y
        Xte = yte = None

    scaler = MinMaxScaler().fit(Xtr, ytr)
    model = SequentialMNMGPR(gpr_subset=gpr_subset, seed=seed)
    model.fit(scaler.transform_X(Xtr), scaler.transform_y(ytr))

    report = {}
    if eval_split:
        yhat = scaler.inverse_y(model.predict(scaler.transform_X(Xte)))
        report = metrics(yte, yhat)
        print("Hold-out test metrics:", json.dumps(report, indent=2))

    # retrain on ALL data for deployment
    scaler_full = MinMaxScaler().fit(X, y)
    model_full = SequentialMNMGPR(gpr_subset=gpr_subset, seed=seed)
    model_full.fit(scaler_full.transform_X(X), scaler_full.transform_y(y))

    artifact = {
        "model": model_full,
        "scaler": scaler_full,
        "lags": LAGS,
        "weather_cols": WEATHER_COLS,
        "tgt_lo": TGT_LO, "tgt_hi": TGT_HI,
        "test_metrics": report,
        "n_train": int(len(X)),
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    joblib.dump(artifact, out_path, compress=3)
    print(f"Saved artifact -> {out_path}  (trained on {len(X)} samples)")
    return artifact


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="historical hourly CSV")
    ap.add_argument("--out", default="artifacts/model.joblib")
    ap.add_argument("--gpr-subset", type=int, default=3000)
    args = ap.parse_args()
    train(args.csv, args.out, gpr_subset=args.gpr_subset)
