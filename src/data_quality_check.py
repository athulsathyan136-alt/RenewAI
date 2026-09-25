import pandas as pd
from pathlib import Path

# Load feature dataset
input_file = Path("data/processed/dubai_solar_features.csv")

print("Loading RenewAI dataset...")
df = pd.read_csv(input_file)

print("=" * 60)
print("RENEWAI DATA QUALITY CHECK")
print("=" * 60)

# Basic information
print("\nDataset shape:")
print(df.shape)

print("\nDate range:")
print("Start:", df["timestamp"].iloc[0])
print("End:  ", df["timestamp"].iloc[-1])

# PV power statistics
print("\nPV POWER STATISTICS")
print("-" * 60)
print("Minimum:", df["pv_power_w"].min(), "W")
print("Maximum:", df["pv_power_w"].max(), "W")
print("Mean:   ", round(df["pv_power_w"].mean(), 2), "W")
print("Median: ", round(df["pv_power_w"].median(), 2), "W")

# Zero generation
zero_power = (df["pv_power_w"] == 0).sum()
zero_percentage = zero_power / len(df) * 100

print("\nZero PV generation:")
print("Hours:", zero_power)
print("Percentage:", round(zero_percentage, 2), "%")

# Negative values
negative_power = (df["pv_power_w"] < 0).sum()

print("\nNegative PV power values:")
print(negative_power)

# Environmental statistics
print("\nENVIRONMENTAL STATISTICS")
print("-" * 60)

print(
    "Beam irradiance:",
    df["beam_irradiance_wm2"].min(),
    "to",
    df["beam_irradiance_wm2"].max(),
    "W/m²"
)

print(
    "Diffuse irradiance:",
    df["diffuse_irradiance_wm2"].min(),
    "to",
    df["diffuse_irradiance_wm2"].max(),
    "W/m²"
)

print(
    "Temperature:",
    df["temperature_c"].min(),
    "to",
    df["temperature_c"].max(),
    "°C"
)

print(
    "Wind speed:",
    df["wind_speed_ms"].min(),
    "to",
    df["wind_speed_ms"].max(),
    "m/s"
)

# Solar-producing hours
solar_hours = df[df["pv_power_w"] > 0]

print("\nSOLAR-PRODUCING HOURS")
print("-" * 60)

print("Total:", len(solar_hours))

if len(solar_hours) > 0:
    print(
        "Earliest hour:",
        solar_hours["hour"].min()
    )

    print(
        "Latest hour:",
        solar_hours["hour"].max()
    )

# Hourly average generation
hourly_average = (
    df.groupby("hour")["pv_power_w"]
    .mean()
    .round(2)
)

print("\nAVERAGE PV POWER BY HOUR")
print("-" * 60)
print(hourly_average)

# Check missing values
print("\nMISSING VALUES")
print("-" * 60)
print(df.isna().sum())

# Check duplicate timestamps
print("\nDUPLICATE TIMESTAMPS")
print("-" * 60)
print(df["timestamp"].duplicated().sum())

print("\n" + "=" * 60)
print("DATA QUALITY CHECK COMPLETE")
print("=" * 60)