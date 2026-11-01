# Registers the dataset for the pipeline: confirms the CSV is in the repo,
# has every column we expect, and prints a quick summary of what's in it.
import sys
import pandas as pd

DATA_PATH = "tourism_project/data/tourism.csv"

EXPECTED_COLUMNS = [
    "CustomerID", "ProdTaken", "Age", "TypeofContact", "CityTier",
    "DurationOfPitch", "Occupation", "Gender", "NumberOfPersonVisiting",
    "NumberOfFollowups", "ProductPitched", "PreferredPropertyStar",
    "MaritalStatus", "NumberOfTrips", "Passport", "PitchSatisfactionScore",
    "OwnCar", "NumberOfChildrenVisiting", "Designation", "MonthlyIncome",
]

df = pd.read_csv(DATA_PATH)

missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
if missing:
    sys.exit(f"Dataset check failed, missing columns: {missing}")

print(f"Dataset registered: {DATA_PATH}")
print(f"Rows: {df.shape[0]}, columns: {df.shape[1]}")
print("\nColumn types:")
print(df.dtypes.to_string())
print("\nColumns with missing values:")
nulls = df.isna().sum()
print(nulls[nulls > 0].to_string() if nulls.any() else "none")
print("\nTarget (ProdTaken) split:")
print(df["ProdTaken"].value_counts(normalize=True).round(3).to_string())
