# Delhi Very Short-Term Load Forecast ⚡

A free, self-running web application that every hour:

1. pulls the **live Delhi system load** from Delhi SLDC and the **live weather** from OpenWeather,
2. stores both in the cloud (**Firebase Firestore**),
3. forecasts the **next hour's load** with a **95% confidence band** using the best model from the study (a **Sequential MNM–GPR hybrid**), and
4. shows everything on a live **Streamlit dashboard**.

Once a week it also **retrains itself automatically** on all the data collected so far (historical + live), validates the new model, and starts using it — so it keeps up with changing demand without any manual work.

Everything runs on **free tiers**: GitHub Actions (hourly job + weekly retrain), Firebase Spark plan (storage), OpenWeather free plan (weather), and Streamlit Community Cloud (dashboard). No servers, no credit card.

---

## The model

The deployed model is the best performer from the study, the **Sequential MNM–GPR hybrid**: a fundamental multiplicative neuron makes a first forecast, and a Gaussian process regression models its residuals, correcting the error and producing the confidence band. Inputs are six load lags (1, 2, 3, 4, 24, 48 h) plus temperature, dew point and specific humidity.

Held-out test accuracy on the four-year Delhi dataset (retrained in Python):

| Metric | Value |
|---|---|
| MAE | ~84 MW |
| RMSE | ~117 MW |
| MAPE | ~2.7 % |
| R² | ~0.98 |
| 95% interval coverage | ~95 % |

A pre-trained model ships in `artifacts/model.joblib`, so you do **not** need to train anything to get started.

---

## Architecture

```
            ┌───────────────────────────── every hour (GitHub Actions cron) ──────────────────────────────┐
            │                                                                                              │
  OpenWeather API ──► weather.py ─┐                                                                        │
                                  ├─► ingest.py ──► Firestore (readings) ──► predict.py (model.joblib) ──► Firestore (forecasts)
  Delhi SLDC site ──► sldc.py ────┘                                                                        │
            └──────────────────────────────────────────────────────────────────────────────────────────────┘
                                                          │
                                       Streamlit Cloud ◄──┘  dashboard/app.py  (reads Firestore, shows live charts)

  ┌──────────── once a week (GitHub Actions cron) ────────────┐
  │  Firestore readings + history_seed.csv ──► retrain.py     │
  │  ──► validate (safety check) ──► commit new model.joblib  │
  └───────────────────────────────────────────────────────────┘
```

---

## What you need (all free)

- A **GitHub** account (hosts the code + runs the hourly job).
- An **OpenWeather** account → free API key.
- A **Google/Firebase** account → free Firestore database + a service-account key.
- A **Streamlit Community Cloud** account (sign in with GitHub) → free dashboard hosting.

---

## Setup — step by step

### 1. Get an OpenWeather API key
1. Sign up at https://openweathermap.org/api (free "Current Weather Data" plan).
2. Go to **API keys** and copy your key. (New keys can take ~1–2 hours to activate.)

### 2. Create the Firebase database
1. Go to https://console.firebase.google.com → **Add project** (disable Analytics to keep it simple).
2. In the project, open **Build → Firestore Database → Create database** → start in **production mode** → pick a location.
3. Create a service account key: **Project settings (gear) → Service accounts → Generate new private key**. A JSON file downloads. Keep it safe — this is your `serviceAccount.json`.

### 3. Put the code on GitHub
1. Create a new **private** repository on GitHub.
2. Push this project to it:
   ```bash
   git init
   git add .
   git commit -m "Delhi load forecast app"
   git branch -M main
   git remote add origin https://github.com/<you>/<repo>.git
   git push -u origin main
   ```

### 4. Add your secrets to GitHub
In the repo: **Settings → Secrets and variables → Actions → New repository secret**. Add two:
- `OPENWEATHER_API_KEY` → your OpenWeather key.
- `FIREBASE_CREDENTIALS_JSON` → paste the **entire contents** of the `serviceAccount.json` file (open it in a text editor and copy everything).

### 5. Seed the history (one time)
The model needs 48 hours of past load to make its first forecast. Bootstrap it:
- In GitHub, open the **Actions** tab → enable workflows if prompted → choose **seed-history** → **Run workflow**.
- This uploads the last 60 hours from `artifacts/history_seed.csv` to Firestore.

### 6. Turn on the hourly job
- In the **Actions** tab, the **hourly-ingest-forecast** workflow runs automatically at :05 past every hour. You can also click **Run workflow** to test it immediately.
- After it runs once, check Firestore — you should see `readings` and `forecasts` collections filling up.

### 7. Deploy the dashboard (free)
1. Go to https://share.streamlit.io → **New app** → connect your GitHub repo.
2. Set **Main file path** to `dashboard/app.py`.
3. Under **Advanced settings → Secrets**, add the same secret so the dashboard can read Firestore:
   ```toml
   FIREBASE_CREDENTIALS_JSON = '''<paste the full serviceAccount.json here>'''
   ```
4. Deploy. Your dashboard gets a public URL and refreshes as new hours arrive.

That's it — the app now collects data, forecasts, and displays everything on its own.

---

## Run it locally (optional)

```bash
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env          # then edit .env with your key
# put your service-account file at ./serviceAccount.json

python -m src.seed_history    # one-time history bootstrap
python -m src.ingest          # one hourly cycle (fetch + store + forecast)
streamlit run dashboard/app.py
```

Local env loading: the app reads plain environment variables. Either `export` them, or use a tool like `python-dotenv`/`direnv`, or set `FIREBASE_CREDENTIALS_FILE=serviceAccount.json` (already the default).

---

## Automatic weekly retraining

The model **retrains itself once a week** — you don't have to do anything. A separate
workflow (`.github/workflows/weekly_retrain.yml`) runs every **Sunday 18:30 UTC
(Monday 00:00 IST)** and:

1. exports all stored readings from Firestore and combines them with the historical
   data in `artifacts/history_seed.csv` (`src/export_history.py`),
2. retrains the Sequential MNM–GPR model on the full series and measures its
   hold-out accuracy (`src/retrain.py`),
3. **accepts the new model only if it is healthy** (hold-out MAPE ≤ 6% and R² ≥ 0.90);
   a corrupt data batch is rejected and the previous model is kept,
4. commits the updated `artifacts/model.joblib` back to the repo, so the next hourly
   run automatically uses the fresher model.

You can also trigger a retrain on demand: **Actions → weekly-retrain → Run workflow**.

> Why weekly and not hourly? One new hour barely changes a model trained on tens of
> thousands of hours, while retraining the GPR is the expensive O(N³) step. Weekly
> retraining keeps the model current with demand growth at almost no cost, and the
> safety check keeps an automated retrain from ever locking in a bad model.

### Manual one-off retrain (optional)

```bash
python -m src.export_history          # build artifacts/training_data.csv from the cloud
python -m model.train --csv artifacts/training_data.csv --out artifacts/model.joblib
```

---

## Project structure

```
.
├── README.md
├── requirements.txt
├── config.py                  # central config (reads env vars)
├── artifacts/
│   ├── model.joblib           # pre-trained Sequential MNM–GPR model (ships ready)
│   └── history_seed.csv       # historical hourly data used to seed + retrain
├── model/
│   ├── features.py            # lag features, scaling
│   ├── mnm_gpr.py             # the Sequential MNM–GPR model
│   └── train.py               # training → saves the artifact
├── src/
│   ├── weather.py             # OpenWeather fetch + feature conversions
│   ├── sldc.py                # Delhi SLDC load scraper
│   ├── predict.py             # load artifact, forecast next hour + band
│   ├── ingest.py              # hourly job: fetch → store → forecast → store
│   ├── seed_history.py        # one-time Firestore bootstrap
│   ├── export_history.py      # combine historical + live readings → training CSV
│   ├── retrain.py             # weekly retrain with safety validation
│   └── firebase_client.py     # Firestore read/write
├── dashboard/
│   └── app.py                 # Streamlit dashboard
└── .github/workflows/
    ├── hourly.yml             # hourly cron (ingest + forecast)
    ├── seed.yml               # manual one-time history seed
    └── weekly_retrain.yml     # weekly automatic retrain + commit
```

---

## Costs and limits

- **GitHub Actions**: free tier gives generous monthly minutes; one short hourly job is well within it (private repos have a monthly minute cap — public repos are unlimited).
- **Firebase Firestore (Spark)**: free quota is ~50k reads / 20k writes per day; this app uses a few writes per hour and a few reads per dashboard view — far below the limit.
- **OpenWeather (free)**: 60 calls/min, 1M/month; this app makes one call per hour.
- **Streamlit Community Cloud**: free public app hosting.

---

## Troubleshooting

- **"history has N hours; need 48 to forecast"** — run the **seed-history** workflow once (step 5), then trigger the hourly job.
- **SLDC scraper returns an error** — Delhi SLDC has no official API and occasionally changes its page or is briefly unreachable. The scraper fails loudly (it never stores a wrong number). If the layout changed, adjust `parse_latest()` in `src/sldc.py`; the data lives in the table on `Loaddata.aspx`.
- **OpenWeather 401** — the key is wrong or not yet active (new keys take up to ~2 hours).
- **Firestore permission errors** — make sure you pasted the full service-account JSON and that Firestore is created in the project.
- **Dashboard empty** — it needs at least one completed hourly cycle (and the Firebase secret set in Streamlit).

---

## Notes and honest caveats

- The forecast uses the **current** weather as the next hour's weather (weather persistence). At a one-hour horizon this is a good approximation, and the study showed the load lags dominate the forecast anyway. For more accuracy you can switch `weather.py` to OpenWeather's hourly forecast endpoint.
- The Python model is a faithful reimplementation of the study's best model and reproduces its reported accuracy; small differences from the thesis come from the GPR training-subset size and random seeds.
- Delhi SLDC scraping depends on a public web page; treat it as best-effort and respect the site's terms and reasonable request rates (one request per hour here).
- Keep your `serviceAccount.json` and API key private — never commit them (`.gitignore` already excludes them).
