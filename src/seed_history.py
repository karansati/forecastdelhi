"""Bootstrap Firestore with recent history so the model can forecast immediately.

Uploads the last N hours of a historical CSV (Temperature, DewPoint, Humidity, Load)
as hourly readings ending at the most recent whole hour. Run this ONCE after setting
up Firebase, before the hourly job has had time to accumulate 48 hours of live data.

    python -m src.seed_history --csv artifacts/history_seed.csv --hours 60
"""
import argparse
import csv
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import config
from src import firebase_client as fb

IST = ZoneInfo(config.TIMEZONE)


def seed(csv_path, hours=60):
    rows = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    tail = rows[-hours:]
    end = datetime.now(IST).replace(minute=0, second=0, microsecond=0) - timedelta(hours=1)
    start = end - timedelta(hours=len(tail) - 1)
    for i, row in enumerate(tail):
        ts = start + timedelta(hours=i)
        doc_id = fb.hour_id(ts)
        fb.save_reading(doc_id, {
            "ts": ts.isoformat(), "hour": doc_id,
            "Load": float(row["Load"]),
            "Temperature": float(row["Temperature"]),
            "DewPoint": float(row["DewPoint"]),
            "Humidity": float(row["Humidity"]),
            "seed": True,
        })
    print(f"Seeded {len(tail)} hours ending {fb.hour_id(end)}.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="artifacts/history_seed.csv")
    ap.add_argument("--hours", type=int, default=60)
    args = ap.parse_args()
    seed(args.csv, args.hours)
