# RetailPulse AI Frontend

Responsive HTML/CSS/JavaScript dashboard connected to the existing FastAPI forecasting backend.

## Run
1. Extract the ZIP.
2. Start your backend using the same Uvicorn command that already works in your project.
3. Open a terminal in this folder and run:
   `python -m http.server 5500`
4. Visit `http://127.0.0.1:5500`.

The API base URL is `http://127.0.0.1:8000` in `js/app.js`; change `API` if needed.

## Connected endpoints
- GET `/health`
- GET `/api/v1/status`
- GET `/api/v1/metadata`
- GET `/api/v1/metrics`
- POST `/api/v1/forecast/time-series` with `{ "model": "arima" | "sarima", "horizon": 30 }`
- POST `/api/v1/forecast/random-forest` with `{ "horizon": 30 }`, plus optional `store` and `item` if entered.

The ARIMA/SARIMA payload matches the successful Swagger test. Random Forest starts with the horizon and optional store/item fields; if the `RandomForestForecastRequest` schema in `/docs` uses different names, adjust the payload builder in `js/app.js`.

Time-series forecasts are aggregate daily sales across all stores and items, as reported by the API. They are not store/item-specific. Chart.js and Google Fonts are loaded from CDNs, so an internet connection is needed for those visual assets.
