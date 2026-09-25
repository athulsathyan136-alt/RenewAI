import pandas as pd
from pathlib import Path

# File paths
input_file = Path("data/processed/dubai_solar_clean.csv")
output_file = Path("data/processed/dubai_solar_features.csv")

print("Loading cleaned Dubai solar dataset...")

# Load data
df = pd.read_csv(input_file)

print("Rows loaded:", len(df))

# Convert PVGIS timestamp
df["timestamp"] = pd.to_datetime(
    df["timestamp"],
    format="%Y%m%d:%H%M"
)

# Create time-based features
df["year"] = df["timestamp"].dt.year
df["month"] = df["timestamp"].dt.month
df["day"] = df["timestamp"].dt.day
df["hour"] = df["timestamp"].dt.hour
df["day_of_year"] = df["timestamp"].dt.dayofyear

# Check timestamp ordering
df = df.sort_values("timestamp").reset_index(drop=True)

# Check duplicates
duplicates = df["timestamp"].duplicated().sum()

# Check missing values
missing = df.isna().sum().sum()

# Save feature dataset
output_file.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_file, index=False)

print()
print("Feature engineering complete!")
print("Final shape:", df.shape)
print("Duplicate timestamps:", duplicates)
print("Total missing values:", missing)

print()
print("Columns:")
print(df.columns.tolist())

print()
print("First 5 rows:")
print(df.head().to_string())

print()
print("Last 5 rows:")
print(df.tail().to_string())

print()
print("Saved to:")
print(output_file)