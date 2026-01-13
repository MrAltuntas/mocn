from src.data_processing import load_data
from config import DATASET_NAME

print(f"Checking class distribution for {DATASET_NAME} with 10,000 samples...")
X, y = load_data(DATASET_NAME)
print("\nDone.")
