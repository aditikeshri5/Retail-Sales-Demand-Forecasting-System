
import kagglehub
import os

# Authenticate with Kaggle
kagglehub.login()

# Create the dataset folder
os.makedirs("data/raw", exist_ok=True)

print("Downloading Retail Sales dataset...")

path = kagglehub.competition_download(
    "demand-forecasting-kernels-only",
    path="train.csv",
    output_dir="data/raw"
)

print("\nDownload completed!")
print("Dataset location:", path)
