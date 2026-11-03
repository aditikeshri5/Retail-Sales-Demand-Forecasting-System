import pandas as pd
from fastapi import HTTPException
from .config import EXTENDED_DATA, FEATURE_DATA_CANDIDATES

def load_extended_data() -> pd.DataFrame:
    if not EXTENDED_DATA.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Dataset not found: {EXTENDED_DATA}. Run notebook 05 and confirm the processed CSV exists."
        )
    df = pd.read_csv(EXTENDED_DATA, parse_dates=["date"])
    required = {"date", "store", "item", "sales"}
    missing = required - set(df.columns)
    if missing:
        raise HTTPException(status_code=500, detail=f"Dataset is missing columns: {sorted(missing)}")
    return df

def load_feature_data():
    for path in FEATURE_DATA_CANDIDATES:
        if path.exists():
            return pd.read_csv(path, parse_dates=["date"]), path
    raise HTTPException(
        status_code=503,
        detail=("Random Forest feature dataset was not found. Expected one of: " +
                ", ".join(str(p) for p in FEATURE_DATA_CANDIDATES))
    )

def daily_total_series() -> pd.Series:
    df = load_extended_data()
    daily = df.groupby("date", sort=True)["sales"].sum().astype(float)
    daily.index = pd.DatetimeIndex(daily.index)
    daily = daily.asfreq("D")
    if daily.isna().any():
        daily = daily.fillna(0.0)
    return daily
