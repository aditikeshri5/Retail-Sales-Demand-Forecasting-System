from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from .schemas import TimeSeriesForecastRequest, RandomForestForecastRequest
from .data_service import load_extended_data
from .model_service import forecast_time_series, forecast_random_forest, TIME_SERIES_FILES
from .config import MODELS_DIR

app = FastAPI(
    title="RetailPulse AI Forecasting API",
    description="API for retail demand forecasting using Random Forest, ARIMA, and SARIMA models.",
    version="1.0.0",
)

# During local development, allow the frontend dev server.
# Restrict allow_origins to your deployed frontend domain before production.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
def root():
    return {"message": "RetailPulse AI API is running", "docs": "/docs"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/api/v1/status")
def status():
    model_files = {
        "random_forest": (MODELS_DIR / "random_forest_model.pkl").exists(),
        **{name: (MODELS_DIR / filename).exists() for name, filename in TIME_SERIES_FILES.items()},
    }
    return {"models": model_files, "all_models_present": all(model_files.values())}

@app.get("/api/v1/metadata")
def metadata():
    df = load_extended_data()
    return {
        "stores": sorted(int(x) for x in df["store"].dropna().unique()),
        "items": sorted(int(x) for x in df["item"].dropna().unique()),
        "date_min": str(df["date"].min().date()),
        "date_max": str(df["date"].max().date()),
        "rows": int(len(df)),
        "note": "Time-series models forecast aggregate daily sales. Random Forest forecasts one store-item combination.",
    }

@app.post("/api/v1/forecast/time-series")
def time_series_forecast(request: TimeSeriesForecastRequest):
    forecasts = forecast_time_series(request.model, request.horizon)
    return {
        "model": request.model,
        "horizon": request.horizon,
        "forecast_level": "daily aggregate across all stores and items",
        "forecasts": forecasts,
        "note": "ARIMA/SARIMA models were trained on aggregate daily sales from the original historical period. Forecasts are aggregate, not store/item-specific.",
    }

@app.post("/api/v1/forecast/random-forest")
def random_forest_forecast(request: RandomForestForecastRequest):
    forecasts = forecast_random_forest(request.store, request.item, request.horizon)
    return {
        "model": "random_forest",
        "horizon": request.horizon,
        "forecast_level": f"store {request.store}, item {request.item}",
        "forecasts": forecasts,
        "note": "Recursive forecasts use previous predictions as lag history. Verify that backend-derived feature names match the exact feature engineering in notebook 03/04.",
    }

@app.get("/api/v1/metrics")
def metrics():
    # Read saved evaluation files if present.
    from .config import OUTPUTS_DIR
    import pandas as pd
    result = {}
    for key, filename in [
        ("full_year", "model_comparison.csv"),
        ("30_day", "30_day_model_comparison.csv"),
    ]:
        path = OUTPUTS_DIR / "metrics" / filename
        if path.exists():
            result[key] = pd.read_csv(path).to_dict(orient="records")
        else:
            result[key] = None
    return result
