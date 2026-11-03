from pathlib import Path

# backend/ is expected to sit directly inside the project root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELS_DIR = PROJECT_ROOT / "models"
DATA_PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

EXTENDED_DATA = DATA_PROCESSED_DIR / "cleaned_extended_sales.csv"
FEATURE_DATA_CANDIDATES = [
    DATA_PROCESSED_DIR / "feature_engineered_sales.csv",
    DATA_PROCESSED_DIR / "cleaned_sales.csv",
]
