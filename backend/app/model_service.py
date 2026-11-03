import joblib
import numpy as np
import pandas as pd
from fastapi import HTTPException
from .config import MODELS_DIR
from .data_service import daily_total_series, load_feature_data

TIME_SERIES_FILES = {
    "arima": "arima_model.pkl",
    "sarima": "sarima_model.pkl",
    "improved_sarima": "improved_sarima_model.pkl",
}


def forecast_time_series(name: str, horizon: int):
    # The saved notebook artifacts were fitted only through 2016 for 2017 evaluation.
    # For a genuine forecast beyond the latest available date, refit the same model
    # specification on the complete available series (2013-2025, including simulated years).
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from .data_service import daily_total_series

    series = daily_total_series()
    if len(series) < 100:
        raise HTTPException(status_code=422, detail="Not enough daily observations to fit a time-series model.")

    try:
        if name == "arima":
            fitted = ARIMA(series, order=(1, 0, 1)).fit()
        elif name == "sarima":
            fitted = SARIMAX(
                series,
                order=(1, 0, 1),
                seasonal_order=(1, 0, 1, 7),
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False)
        elif name == "improved_sarima":
            fitted = SARIMAX(
                series,
                order=(1, 0, 1),
                seasonal_order=(1, 1, 1, 7),
                enforce_stationarity=False,
                enforce_invertibility=False,
            ).fit(disp=False, maxiter=200)
        else:
            raise HTTPException(status_code=400, detail=f"Unsupported time-series model: {name}")

        values = fitted.get_forecast(steps=horizon).predicted_mean
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Time-series model fitting/forecast failed: {exc}")

    start = series.index.max() + pd.Timedelta(days=1)
    dates = pd.date_range(start=start, periods=horizon, freq="D")
    arr = np.maximum(np.asarray(values, dtype=float).reshape(-1), 0)
    return [
        {"date": d.strftime("%Y-%m-%d"), "predicted_sales": round(float(v), 2)}
        for d, v in zip(dates, arr)
    ]

def _feature_value(row, name, date, store, item, history):
    # Support common time, lag, and rolling features used in the project's feature engineering.
    aliases = {
        "store": store, "item": item,
        "year": date.year, "month": date.month, "day": date.day,
        "day_of_week": date.dayofweek, "dayofweek": date.dayofweek,
        "day_of_month": date.day, "week": int(date.isocalendar().week),
        "week_of_year": int(date.isocalendar().week),
        "quarter": date.quarter, "is_weekend": int(date.dayofweek >= 5),
        "is_month_start": int(date.is_month_start),
        "is_month_end": int(date.is_month_end),
    }
    if name in aliases:
        return aliases[name]
    if name in row.index:
        return row[name]
    import re
    m = re.fullmatch(r"lag_(\d+)", name)
    if m:
        k = int(m.group(1))
        return float(history[-k]) if len(history) >= k else float(history[-1])
    m = re.fullmatch(r"rolling_(mean|std)_(\d+)", name)
    if m:
        kind, n = m.group(1), int(m.group(2))
        vals = np.asarray(history[-n:] if len(history) >= n else history, dtype=float)
        if len(vals) == 0:
            return 0.0
        if kind == "mean":
            return float(np.mean(vals))
        return float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0
    raise HTTPException(
        status_code=422,
        detail=f"Random Forest requires feature '{name}', but backend cannot derive it. Inspect the feature columns used in notebook 04."
    )

def forecast_random_forest(store: int, item: int, horizon: int):
    model_path = MODELS_DIR / "random_forest_model.pkl"
    if not model_path.exists():
        raise HTTPException(status_code=503, detail=f"Model file not found: {model_path}")
    try:
        model = joblib.load(model_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not load Random Forest model: {exc}")

    df, _ = load_feature_data()
    required = {"date", "store", "item", "sales"}
    if not required.issubset(df.columns):
        raise HTTPException(status_code=500, detail="Feature dataset must contain date, store, item, and sales columns.")

    subset = df[(df["store"] == store) & (df["item"] == item)].copy()
    subset = subset.sort_values("date")
    if subset.empty:
        raise HTTPException(status_code=404, detail=f"No history found for store {store}, item {item}.")

    # Use the last observed sales as the initial history, then recursively feed predictions.
    sales_history = subset["sales"].astype(float).tolist()
    last_date = pd.Timestamp(subset["date"].max())
    feature_names = list(getattr(model, "feature_names_in_", []))
    if not feature_names:
        raise HTTPException(
            status_code=500,
            detail="The saved Random Forest model has no feature_names_in_. The exact training feature list is needed."
        )

    last_row = subset.iloc[-1]
    results = []
    for step in range(1, horizon + 1):
        future_date = last_date + pd.Timedelta(days=step)
        values = {}
        for feature in feature_names:
            values[feature] = _feature_value(last_row, feature, future_date, store, item, sales_history)
        X = pd.DataFrame([values], columns=feature_names)
        try:
            pred = float(model.predict(X)[0])
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Random Forest prediction failed: {exc}")
        pred = max(0.0, pred)
        sales_history.append(pred)
        results.append({"date": future_date.strftime("%Y-%m-%d"), "predicted_sales": round(pred, 2)})

    return results
