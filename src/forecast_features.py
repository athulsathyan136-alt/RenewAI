import pandas as pd
from pathlib import Path

input_file = Path("data/processed/dubai_solar_features.csv")
output_file = Path("data/processed/dubai_solar_forecast_features.csv")

print("Loading RenewAI dataset...")

df = pd.read_csv(input_file)

df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.sort_values("timestamp").reset_index(drop=True)

print("Original rows:", len(df))

# --------------------------------------------------
# Create historical lag features
# --------------------------------------------------

df["pv_power_lag_1h"] = df["pv_power_w"].shift(1)
df["pv_power_lag_2h"] = df["pv_power_w"].shift(2)
df["pv_power_lag_3h"] = df["pv_power_w"].shift(3)

df["irradiance_lag_1h"] = (
    df["beam_irradiance_wm2"].shift(1)
)

df["temperature_lag_1h"] = (
    df["temperature_c"].shift(1)
)

df["wind_lag_1h"] = (
    df["wind_speed_ms"].shift(1)
)

# --------------------------------------------------
# Remove rows with missing lag values
# --------------------------------------------------

df = df.dropna().reset_index(drop=True)

print("Rows after lag features:", len(df))

# --------------------------------------------------
# Save forecasting dataset
# --------------------------------------------------

output_file.parent.mkdir(parents=True, exist_ok=True)

df.to_csv(output_file, index=False)

print()
print("Forecast feature engineering complete!")

print()
print("New forecasting features:")
print(
    [
        "pv_power_lag_1h",
        "pv_power_lag_2h",
        "pv_power_lag_3h",
        "irradiance_lag_1h",
        "temperature_lag_1h",
        "wind_lag_1h",
    ]
)

print()
print("Final shape:", df.shape)

print()
print("First 5 rows:")
print(df.head().to_string())

print()
print("Saved to:")
print(output_file)