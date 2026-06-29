"""Build the full training CSV = historical seed data + everything collected live.

The seed CSV (artifacts/history_seed.csv) is the historical record. Firestore holds
the hourly readings collected since deployment. We concatenate the historical rows
with the genuinely-collected live rows (skipping the bootstrap rows that were just
copies of the seed tail), de-duplicate by hour, and write a chronological CSV that
the trainer consumes.
"""
import argparse
import csv
import os

import config
from src import firebase_client as fb

COLS = ["Temperature", "DewPoint", "Humidity", "Load"]


def _read_seed(path):
    rows = []
    if os.path.exists(path):
        with open(path) as f:
            for r in csv.DictReader(f):
                rows.append({c: float(r[c]) for c in COLS})
    return rows


def build_training_csv(seed_csv, out_csv):
    base = _read_seed(seed_csv)

    live = []
    seen_hours = set()
    for d in fb.get_all_readings():
        if d.get("seed"):                       # skip bootstrap copies of the seed tail
            continue
        if any(d.get(c) is None for c in COLS):
            continue
        h = d.get("hour") or d.get("ts")
        if h in seen_hours:
            continue
        seen_hours.add(h)
        live.append({c: float(d[c]) for c in COLS})

    allrows = base + live
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS)
        w.writeheader()
        w.writerows(allrows)
    print(f"Wrote {len(allrows)} rows to {out_csv} "
          f"({len(base)} historical + {len(live)} collected live).")
    return out_csv


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", default="artifacts/history_seed.csv")
    ap.add_argument("--out", default="artifacts/training_data.csv")
    args = ap.parse_args()
    build_training_csv(args.seed, args.out)
