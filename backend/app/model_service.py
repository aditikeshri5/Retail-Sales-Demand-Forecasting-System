
import json
import joblib
import numpy as np
import pandas as pd

from fastapi import HTTPException

from .config import MODELS_DIR
from .data_service import daily_total_series, load_extended_data


# --------------------------------------------------
# MODEL FILE CONFIGURATION
# --------------------------------------------------

TIME_SERIES_FILES = {
    "arima": "arima_model.pkl",
    "sarima": "sarima_model.pkl",
    "improved_sarima": "improved_sarima_model.pkl",
}

RANDOM_FOREST_MODEL = MODELS_DIR / "random_forest_extended_model.pkl"

RANDOM_FOREST_FEATURES = (
    MODELS_DIR / "random_forest_extended_features.json"
)


# --------------------------------------------------
# ARIMA / SARIMA FORECASTING
# --------------------------------------------------

def forecast_time_series(name: str, horizon: int):

    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.statespace.sarimax import SARIMAX

    series = daily_total_series()

    if len(series) < 100:
        raise HTTPException(
            status_code=422,
            detail="Not enough daily observations to fit a time-series model."
        )

    try:

        if name == "arima":

            fitted = ARIMA(
                series,
                order=(1, 0, 1)
            ).fit()

        elif name == "sarima":

            fitted = SARIMAX(
                series,
                order=(1, 0, 1),
                seasonal_order=(1, 0, 1, 7),
                enforce_stationarity=False,
                enforce_invertibility=False
            ).fit(disp=False)

        elif name == "improved_sarima":

            fitted = SARIMAX(
                series,
                order=(1, 0, 1),
                seasonal_order=(1, 1, 1, 7),
                enforce_stationarity=False,
                enforce_invertibility=False
            ).fit(disp=False, maxiter=200)

        else:

            raise HTTPException(
                status_code=400,
                detail=f"Unsupported time-series model: {name}"
            )

        values = fitted.get_forecast(
            steps=horizon
        ).predicted_mean

    except HTTPException:
        raise

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Time-series model fitting/forecast failed: {exc}"
        )

    start = series.index.max() + pd.Timedelta(days=1)

    dates = pd.date_range(
        start=start,
        periods=horizon,
        freq="D"
    )

    predictions = np.maximum(
        np.asarray(values, dtype=float).reshape(-1),
        0
    )

    return [
        {
            "date": date.strftime("%Y-%m-%d"),
            "predicted_sales": round(float(value), 2)
        }
        for date, value in zip(dates, predictions)
    ]


# --------------------------------------------------
# RANDOM FOREST MODEL LOADING
# --------------------------------------------------

def load_extended_random_forest():

    if not RANDOM_FOREST_MODEL.exists():

        raise HTTPException(
            status_code=503,
            detail=f"Extended Random Forest model not found: {RANDOM_FOREST_MODEL}"
        )

    if not RANDOM_FOREST_FEATURES.exists():

        raise HTTPException(
            status_code=503,
            detail=f"Random Forest feature configuration not found: {RANDOM_FOREST_FEATURES}"
        )

    try:

        model = joblib.load(RANDOM_FOREST_MODEL)

        with open(
            RANDOM_FOREST_FEATURES,
            "r",
            encoding="utf-8"
        ) as file:

            feature_names = json.load(file)

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Could not load extended Random Forest model: {exc}"
        )

    if not isinstance(feature_names, list) or not feature_names:

        raise HTTPException(
            status_code=500,
            detail="Random Forest feature configuration is invalid."
        )

    model_features = list(
        getattr(model, "feature_names_in_", [])
    )

    if model_features and model_features != feature_names:

        raise HTTPException(
            status_code=500,
            detail="Feature order in JSON does not match the trained Random Forest model."
        )

    if not model_features:

        raise HTTPException(
            status_code=500,
            detail="Trained Random Forest model does not contain feature_names_in_."
        )

    return model, feature_names


# --------------------------------------------------
# RANDOM FOREST FEATURE ENGINEERING
# --------------------------------------------------

def create_future_features(
    date,
    store,
    item,
    history,
    feature_names
):

    history = np.asarray(history, dtype=float)

    if len(history) < 7:

        raise ValueError(
            "At least seven historical sales observations are required."
        )

    feature_values = {

        "store": store,

        "item": item,

        "day_of_week": date.dayofweek,

        "day_of_month": date.day,

        "month": date.month,

        "year": date.year,

        "is_weekend": int(date.dayofweek >= 5),

        "lag_1": float(history[-1]),

        "lag_7": float(history[-7]),

        "rolling_mean_7": float(np.mean(history[-7:]))

    }

    missing_features = [
        feature
        for feature in feature_names
        if feature not in feature_values
    ]

    if missing_features:

        raise ValueError(
            f"Backend cannot calculate these model features: {missing_features}"
        )

    return {
        feature: feature_values[feature]
        for feature in feature_names
    }


# --------------------------------------------------
# EXTENDED RANDOM FOREST FORECASTING
# --------------------------------------------------

def forecast_random_forest(store: int, item: int, horizon: int):

    # Load the newly trained extended model.

    model, feature_names = load_extended_random_forest()

    # Load the extended historical dataset (2013–2025).

    df = load_extended_data()

    required_columns = {
        "date",
        "store",
        "item",
        "sales"
    }

    if not required_columns.issubset(df.columns):

        raise HTTPException(
            status_code=500,
            detail="Extended dataset must contain date, store, item, and sales columns."
        )

    df["date"] = pd.to_datetime(df["date"])

    # Select the requested store and item.

    subset = df[
        (df["store"] == store) &
        (df["item"] == item)
    ].copy()

    subset = subset.sort_values("date")

    if subset.empty:

        raise HTTPException(
            status_code=404,
            detail=f"No history found for store {store}, item {item}."
        )

    # Ensure the selected series contains daily historical observations.

    if subset["date"].duplicated().any():

        raise HTTPException(
            status_code=500,
            detail="Duplicate dates found for the selected store-item combination."
        )

    # Initialize historical sales.

    sales_history = subset["sales"].astype(float).tolist()

    last_date = pd.Timestamp(
        subset["date"].max()
    )

    # Ensure the model has enough historical observations.

    if len(sales_history) < 7:

        raise HTTPException(
            status_code=422,
            detail="At least seven days of historical sales are required."
        )

    # Generate predictions recursively.

    results = []

    for step in range(1, horizon + 1):

        future_date = last_date + pd.Timedelta(days=step)

        try:

            # Create features in the exact training order.

            future_features = create_future_features(
                date=future_date,
                store=store,
                item=item,
                history=sales_history,
                feature_names=feature_names
            )

            X_future = pd.DataFrame(
                [future_features],
                columns=feature_names
            )

            # Predict the next day's demand.

            prediction = float(
                model.predict(X_future)[0]
            )

            prediction = max(0.0, prediction)

        except Exception as exc:

            raise HTTPException(
                status_code=500,
                detail=f"Random Forest prediction failed: {exc}"
            )

        # Add the prediction to history for the next forecast step.

        sales_history.append(prediction)

        results.append({
            "date": future_date.strftime("%Y-%m-%d"),
            "predicted_sales": round(prediction, 2)
        })

    return results
