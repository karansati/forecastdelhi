"""Central configuration. All secrets come from environment variables.

Copy .env.example to .env and fill in your own keys for local runs. In GitHub
Actions and Streamlit Cloud these are provided as secrets (see README).
"""
import os

# --- Location: Delhi ---
DELHI_LAT = float(os.getenv("DELHI_LAT", "28.6139"))
DELHI_LON = float(os.getenv("DELHI_LON", "77.2090"))
TIMEZONE = "Asia/Kolkata"

# --- OpenWeather ---
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
OPENWEATHER_URL = "https://api.openweathermap.org/data/2.5/weather"

# --- SLDC Delhi ---
SLDC_LOADDATA_URL = "https://www.delhisldc.org/Loaddata.aspx"

# --- Firebase / Firestore ---
# The service-account JSON is passed as a single environment variable (string).
FIREBASE_CREDENTIALS_JSON = os.getenv("FIREBASE_CREDENTIALS_JSON", "")
# Or a path to the JSON file (used if the variable above is empty).
FIREBASE_CREDENTIALS_FILE = os.getenv("FIREBASE_CREDENTIALS_FILE", "serviceAccount.json")

# Firestore collections
COL_READINGS = "readings"      # one document per hour: load + weather
COL_FORECASTS = "forecasts"    # one document per hour: next-hour forecast + band

# Model artifact
MODEL_PATH = os.getenv("MODEL_PATH", "artifacts/model.joblib")
