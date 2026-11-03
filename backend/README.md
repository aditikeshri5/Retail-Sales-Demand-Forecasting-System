# RetailPulse AI — Backend

FastAPI backend for the RetailPulse AI retail sales forecasting project.

## 1. Expected project layout

Place this `backend/` folder directly inside the existing project root:

```text
Retail-Sales-Demand-Forecasting-System/
├── backend/
│   ├── app/
│   ├── requirements.txt
│   └── README.md
├── data/
│   └── processed/
│       ├── cleaned_extended_sales.csv
│       └── feature_engineered_sales.csv
├── models/
│   ├── random_forest_model.pkl
│   ├── arima_model.pkl
│   ├── sarima_model.pkl
│   └── improved_sarima_model.pkl
└── outputs/
    └── metrics/
        ├── model_comparison.csv
        └── 30_day_model_comparison.csv
```

The ZIP contains backend source code only. It does not duplicate your trained `.pkl` files or datasets; keep your existing project folders in place.

## 2. Install dependencies

Open a terminal at the project root, activate your existing virtual environment, then run:

```powershell
pip install -r backend/requirements.txt
```

## 3. Start the API

From the project root:

```powershell
python -m uvicorn app.main:app --app-dir backend --reload
```

Open:
- API: http://127.0.0.1:8000
- Interactive API documentation: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/health

## 4. API endpoints

### Status

`GET /api/v1/status`

Checks whether the four expected model files exist.

### Metadata

`GET /api/v1/metadata`

Returns store/item IDs and dataset date range.

### Aggregate time-series forecast

`POST /api/v1/forecast/time-series`

Example request:

```json
{
  "model": "arima",
  "horizon": 7
}
```

Allowed model values: `arima`, `sarima`, `improved_sarima`.

**Important:** Your ARIMA/SARIMA notebooks train on the aggregate daily sales series across all stores and items. These models cannot provide a forecast for an individual store or item. Their API response explicitly identifies this aggregate level.

### Random Forest forecast

`POST /api/v1/forecast/random-forest`

Example request:

```json
{
  "store": 1,
  "item": 1,
  "horizon": 7
}
```

The Random Forest endpoint loads the existing feature-engineered CSV and saved model. It attempts to derive common calendar, lag, and rolling features recursively. If your model uses additional engineered feature names, the endpoint returns a clear error; align `_feature_value()` in `app/model_service.py` with the exact columns from notebooks 03 and 04 before relying on RF forecasts.

### Saved metrics

`GET /api/v1/metrics`

Reads the CSV evaluation files from `outputs/metrics/` when available.

## 5. Test the API

With the server running, open `/docs` and try the endpoints.

You can also test from PowerShell:

```powershell
Invoke-RestMethod -Method Post `
  -Uri "http://127.0.0.1:8000/api/v1/forecast/time-series" `
  -ContentType "application/json" `
  -Body '{"model":"arima","horizon":7}'
```

## 6. Important model/data notes

- The saved time-series `.pkl` artifacts are the notebook's 2013–2016 fitted benchmark models used for 2017 evaluation. For a genuine forecast after the latest available date, this API refits the same ARIMA/SARIMA orders on the complete `cleaned_extended_sales.csv` series each time a request is made. This avoids incorrectly labeling a 2017 holdout forecast as a 2026 forecast.
- The backend aggregates the complete extended dataset and refits the selected time-series specification on that series. Since 2018–2025 records are synthetic, these forecasts are partly influenced by simulated data and must be described as such.
- Synthetic 2018–2025 data must not be presented as real observed sales.
- Random Forest is a separate store-item model. Its exact feature engineering must match the model's training notebook. Review its feature names and recursive forecast behavior before presenting it as production-ready.
- Restrict CORS origins before public deployment.
- Only load trusted `.pkl` model files. Pickle/joblib files can execute code during loading if malicious.
