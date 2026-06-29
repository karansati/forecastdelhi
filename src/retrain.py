"""Retrain the model on the full (historical + live) data, with a safety check.

Steps:
  1. export historical + live readings to a training CSV,
  2. train a candidate model and measure its hold-out accuracy,
  3. accept it ONLY if the accuracy is sane (guards against corrupt data),
  4. replace artifacts/model.joblib with the accepted candidate.

Exit code 0 = model updated; non-zero = kept the existing model.
"""
import os
import shutil
import sys

import config
from src.export_history import build_training_csv
from model.train import train

# acceptance thresholds: a healthy retrain should be well within these
MAX_ACCEPT_MAPE = 6.0     # percent
MIN_ACCEPT_R2 = 0.90
MIN_ROWS = 1000           # refuse to train on too little data

SEED_CSV = "artifacts/history_seed.csv"
TRAIN_CSV = "artifacts/training_data.csv"
CANDIDATE = "artifacts/model_candidate.joblib"
LIVE_MODEL = config.MODEL_PATH


def main():
    build_training_csv(SEED_CSV, TRAIN_CSV)

    with open(TRAIN_CSV) as f:
        n_rows = sum(1 for _ in f) - 1
    if n_rows < MIN_ROWS:
        print(f"Only {n_rows} rows; need >= {MIN_ROWS}. Keeping existing model.")
        return 1

    artifact = train(TRAIN_CSV, CANDIDATE, gpr_subset=1800)
    m = artifact.get("test_metrics", {})
    mape, r2 = m.get("MAPE", 1e9), m.get("R2", -1e9)
    print(f"Candidate hold-out: MAPE={mape:.2f}%  R2={r2:.3f}")

    if mape <= MAX_ACCEPT_MAPE and r2 >= MIN_ACCEPT_R2:
        shutil.move(CANDIDATE, LIVE_MODEL)
        print(f"ACCEPTED. Updated {LIVE_MODEL} (trained on {n_rows} rows).")
        return 0

    if os.path.exists(CANDIDATE):
        os.remove(CANDIDATE)
    print(f"REJECTED (MAPE>{MAX_ACCEPT_MAPE} or R2<{MIN_ACCEPT_R2}). "
          "Kept the existing model.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
