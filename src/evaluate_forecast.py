import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


input_file = Path(
    "data/processed/forecast_predictions_2023.csv"
)


print("Loading forecast predictions...")

df = pd.read_csv(input_file)

df["timestamp"] = pd.to_datetime(
    df["timestamp"]
)

print("Rows:", len(df))


# --------------------------------------------------
# First 7 days
# --------------------------------------------------

first_week = df.iloc[:24 * 7]


plt.figure(figsize=(12, 5))

plt.plot(
    first_week["timestamp"],
    first_week["pv_power_w"],
    label="Actual"
)

plt.plot(
    first_week["timestamp"],
    first_week["predicted_next_hour_pv_power_w"],
    label="1-Hour Forecast"
)

plt.title(
    "RenewAI — 1-Hour-Ahead Solar PV Forecast"
)

plt.xlabel("Time")

plt.ylabel(
    "PV Power (W)"
)

plt.legend()

plt.xticks(
    rotation=45
)

plt.tight_layout()


output1 = Path(
    "data/processed/forecast_first_week.png"
)

plt.savefig(
    output1,
    dpi=150
)

plt.show()


# --------------------------------------------------
# Scatter plot
# --------------------------------------------------

plt.figure(figsize=(7, 7))

plt.scatter(
    df["pv_power_w"],
    df["predicted_next_hour_pv_power_w"],
    alpha=0.3
)


max_value = max(
    df["pv_power_w"].max(),
    df["predicted_next_hour_pv_power_w"].max()
)


plt.plot(
    [0, max_value],
    [0, max_value],
    linestyle="--"
)


plt.title(
    "RenewAI — Actual vs 1-Hour Forecast"
)

plt.xlabel(
    "Actual PV Power (W)"
)

plt.ylabel(
    "Forecast PV Power (W)"
)

plt.tight_layout()


output2 = Path(
    "data/processed/forecast_scatter.png"
)

plt.savefig(
    output2,
    dpi=150
)

plt.show()


print()
print("Forecast evaluation visualization complete!")

print()
print("Saved:")

print(output1)

print(output2)