from typing import Literal, Optional
from pydantic import BaseModel, Field

ModelName = Literal["arima", "sarima", "improved_sarima"]

class TimeSeriesForecastRequest(BaseModel):
    model: ModelName = "arima"
    horizon: int = Field(default=7, ge=1, le=90)

class RandomForestForecastRequest(BaseModel):
    store: int = Field(ge=1, le=10)
    item: int = Field(ge=1, le=50)
    horizon: int = Field(default=7, ge=1, le=30)

class ForecastPoint(BaseModel):
    date: str
    predicted_sales: float

class ForecastResponse(BaseModel):
    model: str
    horizon: int
    forecast_level: str
    forecasts: list[ForecastPoint]
    note: Optional[str] = None
