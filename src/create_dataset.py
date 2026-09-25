import pandas as pd
import numpy as np

# Make results reproducible
np.random.seed(42)

# Create hourly timestamps for 2024-2025
dates = pd.date_range(
    start="2024-01-01",
    end="2025-12-31 23:00:00",
    freq="h"
)

hours = dates.hour
days = dates.dayofyear

# -----------------------------
# 1. Solar irradiance
# -----------------------------
solar_irradiance = np.maximum(
    0,
    800 * np.sin(np.pi * (hours - 6) / 12)
)

solar_irradiance += np.random.normal(
    0,
    50,
    len(dates)
)

solar_irradiance = np.clip(
    solar_irradiance,
    0,
    None
)

# -----------------------------
# 2. Temperature
# -----------------------------
temperature = (
    25
    + 7 * np.sin(2 * np.pi * days / 365)
    + np.random.normal(0, 2, len(dates))
)

# -----------------------------
# 3. Wind speed
# -----------------------------
wind_speed = np.clip(
    5 + np.random.normal(0, 2, len(dates)),
    0,
    None
)

# -----------------------------
# 4. Humidity
# -----------------------------
humidity = np.clip(
    70 - solar_irradiance / 20
    + np.random.normal(0, 5, len(dates)),
    20,
    100
)

# -----------------------------
# 5. Cloud cover
# -----------------------------
cloud_cover = np.clip(
    np.random.normal(40, 25, len(dates)),
    0,
    100
)

# -----------------------------
# 6. Solar generation
# -----------------------------
solar_generation = (
    solar_irradiance
    * (1 - cloud_cover / 150)
    * 0.005
)

solar_generation += np.random.normal(
    0,
    0.1,
    len(dates)
)

solar_generation = np.clip(
    solar_generation,
    0,
    None
)

# -----------------------------
# Create DataFrame
# -----------------------------
df = pd.DataFrame({
    "date": dates,
    "solar_irradiance": solar_irradiance,
    "temperature": temperature,
    "wind_speed": wind_speed,
    "humidity": humidity,
    "cloud_cover": cloud_cover,
    "solar_generation": solar_generation
})

# -----------------------------
# Save dataset
# -----------------------------
output_path = "data/raw/renewable_energy.csv"

df.to_csv(
    output_path,
    index=False
)

print("RenewAI dataset created successfully!")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")
print(f"Saved to: {output_path}")
print()
print(df.head())