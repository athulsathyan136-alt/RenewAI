import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load dataset
input_file = Path("data/processed/dubai_solar_features.csv")
df = pd.read_csv(input_file)

# Convert timestamp
df["timestamp"] = pd.to_datetime(df["timestamp"])

# --------------------------------------------------
# Plot 1: First 7 days of PV generation
# --------------------------------------------------

first_week = df.iloc[:24 * 7]

plt.figure(figsize=(12, 5))
plt.plot(
    first_week["timestamp"],
    first_week["pv_power_w"]
)

plt.title("Dubai Solar PV Generation — First 7 Days")
plt.xlabel("Time")
plt.ylabel("PV Power (W)")
plt.xticks(rotation=45)
plt.tight_layout()

plt.savefig(
    "data/processed/pv_generation_first_week.png",
    dpi=150
)

plt.show()

# --------------------------------------------------
# Plot 2: Average PV generation by hour
# --------------------------------------------------

hourly_average = (
    df.groupby("hour")["pv_power_w"]
    .mean()
)

plt.figure(figsize=(10, 5))
plt.plot(
    hourly_average.index,
    hourly_average.values,
    marker="o"
)

plt.title("Average Dubai PV Generation by Hour")
plt.xlabel("Hour of Day")
plt.ylabel("Average PV Power (W)")
plt.xticks(range(24))
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "data/processed/average_pv_by_hour.png",
    dpi=150
)

plt.show()

# --------------------------------------------------
# Plot 3: Average PV generation by month
# --------------------------------------------------

monthly_average = (
    df.groupby("month")["pv_power_w"]
    .mean()
)

plt.figure(figsize=(10, 5))
plt.plot(
    monthly_average.index,
    monthly_average.values,
    marker="o"
)

plt.title("Average Dubai PV Generation by Month")
plt.xlabel("Month")
plt.ylabel("Average PV Power (W)")
plt.xticks(range(1, 13))
plt.grid(True)
plt.tight_layout()

plt.savefig(
    "data/processed/average_pv_by_month.png",
    dpi=150
)

plt.show()

print("Visualization complete.")
print("Saved:")
print("1. data/processed/pv_generation_first_week.png")
print("2. data/processed/average_pv_by_hour.png")
print("3. data/processed/average_pv_by_month.png")