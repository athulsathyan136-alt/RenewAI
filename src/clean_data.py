import pandas as pd
from pathlib import Path

# File paths
input_file = Path("data/raw/dubai_pvgis_hourly.csv")
output_file = Path("data/processed/dubai_solar_clean.csv")

print("Loading PVGIS dataset...")

# Read the CSV
df = pd.read_csv(
    input_file,
    skiprows=10,
    low_memory=False
)

print("Original shape:", df.shape)

# Remove rows where the timestamp is not a real data timestamp.
# Real PVGIS timestamps look like: 20050101:0008
df = df[df["time"].astype(str).str.match(r"^\d{8}:\d{4}$")].copy()

print("After removing metadata rows:", df.shape)

# Rename columns
df = df.rename(columns={
    "time": "timestamp",
    "P": "pv_power_w",
    "Gb(i)": "beam_irradiance_wm2",
    "Gd(i)": "diffuse_irradiance_wm2",
    "Gr(i)": "reflected_irradiance_wm2",
    "H_sun": "sun_height_deg",
    "T2m": "temperature_c",
    "WS10m": "wind_speed_ms",
    "Int": "intensity"
})

# Convert numeric columns
numeric_columns = [
    "pv_power_w",
    "beam_irradiance_wm2",
    "diffuse_irradiance_wm2",
    "reflected_irradiance_wm2",
    "sun_height_deg",
    "temperature_c",
    "wind_speed_ms",
    "intensity"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Save processed dataset
output_file.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_file, index=False)

print()
print("Cleaning complete!")
print("Final shape:", df.shape)
print("Saved to:", output_file)

print()
print("Columns:")
print(df.columns.tolist())

print()
print("Missing values:")
print(df.isna().sum())

print()
print("Data types:")
print(df.dtypes)

print()
print("First 5 cleaned rows:")
print(df.head())