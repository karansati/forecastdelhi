"""Firestore access. Stores hourly readings and forecasts in the cloud (free tier).

Document id = hour timestamp in UTC ISO form 'YYYY-MM-DDTHH' so each hour is unique
and writes are idempotent.
"""
import json
import os

import firebase_admin
from firebase_admin import credentials, firestore

import config

_db = None


def _init():
    global _db
    if _db is not None:
        return _db
    if not firebase_admin._apps:
        if config.FIREBASE_CREDENTIALS_JSON.strip():
            cred = credentials.Certificate(json.loads(config.FIREBASE_CREDENTIALS_JSON))
        elif os.path.exists(config.FIREBASE_CREDENTIALS_FILE):
            cred = credentials.Certificate(config.FIREBASE_CREDENTIALS_FILE)
        else:
            raise RuntimeError(
                "No Firebase credentials. Set FIREBASE_CREDENTIALS_JSON or provide "
                f"{config.FIREBASE_CREDENTIALS_FILE}.")
        firebase_admin.initialize_app(cred)
    _db = firestore.client()
    return _db


def hour_id(dt):
    """Stable per-hour document id from a timezone-aware datetime (UTC)."""
    return dt.astimezone().strftime("%Y-%m-%dT%H")


def save_reading(doc_id, data):
    _init().collection(config.COL_READINGS).document(doc_id).set(data, merge=True)


def save_forecast(doc_id, data):
    _init().collection(config.COL_FORECASTS).document(doc_id).set(data, merge=True)


def get_recent_readings(limit=120):
    """Return the most recent readings, oldest..newest."""
    docs = (_init().collection(config.COL_READINGS)
            .order_by("ts").limit_to_last(limit).get())
    return [d.to_dict() for d in docs]


def get_recent_forecasts(limit=120):
    docs = (_init().collection(config.COL_FORECASTS)
            .order_by("target_ts").limit_to_last(limit).get())
    return [d.to_dict() for d in docs]


def get_all_readings():
    """Return every stored reading, oldest..newest (used for retraining export)."""
    docs = _init().collection(config.COL_READINGS).order_by("ts").stream()
    return [d.to_dict() for d in docs]
