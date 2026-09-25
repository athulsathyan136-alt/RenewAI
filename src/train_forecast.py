import pandas as pd
import numpy as np
from pathlib import Path
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


input_file = Path(
    "data/processed/dubai_solar_forecast_features.csv"
)

output_file = Path(
    "data/processed/forecast_predictions_2023.csv"
)


print("Loading RenewAI forecasting dataset...")

df = pd.read_csv(input_file)

df["timestamp"] = pd.to_datetime(df["timestamp"])

df = df.sort_values("timestamp").reset_index(drop=True)

print("Dataset:", df.shape)


# --------------------------------------------------
# Create 1-hour-ahead target
# --------------------------------------------------

df["target_next_hour"] = df["pv_power_w"].shift(-1)

df = df.dropna().reset_index(drop=True)


# --------------------------------------------------
# Forecasting features
# --------------------------------------------------

features = [
    "pv_power_lag_1h",
    "pv_power_lag_2h",
    "pv_power_lag_3h",
    "irradiance_lag_1h",
    "temperature_lag_1h",
    "wind_lag_1h",
    "month",
    "day",
    "hour",
    "day_of_year"
]

target = "target_next_hour"


# --------------------------------------------------
# Time-based train/test split
# --------------------------------------------------

train_df = df[df["year"] < 2023].copy()

test_df = df[df["year"] == 2023].copy()


X_train = train_df[features]
y_train = train_df[target]

X_test = test_df[features]
y_test = test_df[target]


print()
print("TRAINING DATA")
print("Rows:", len(train_df))
print(
    "Period:",
    train_df["timestamp"].min(),
    "→",
    train_df["timestamp"].max()
)


print()
print("TEST DATA")
print("Rows:", len(test_df))
print(
    "Period:",
    test_df["timestamp"].min(),
    "→",
    test_df["timestamp"].max()
)


# --------------------------------------------------
# Train model
# --------------------------------------------------

print()
print("Training 1-hour-ahead Random Forest...")


model = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)


model.fit(X_train, y_train)

print("Training complete.")

# --------------------------------------------------
# Save trained model
# --------------------------------------------------

model_file = Path(
    "models/renewai_forecast_model.joblib"
)

joblib.dump(
    model,
    model_file
)

print()
print("Model saved to:")
print(model_file)



# --------------------------------------------------
# Predictions
# --------------------------------------------------

predictions = model.predict(X_test)


# --------------------------------------------------
# Evaluation
# --------------------------------------------------

mae = mean_absolute_error(
    y_test,
    predictions
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        predictions
    )
)

r2 = r2_score(
    y_test,
    predictions
)


print()
print("=" * 60)
print("RENEWAI 1-HOUR-AHEAD FORECAST RESULTS")
print("=" * 60)

print(f"MAE : {mae:.2f} W")
print(f"RMSE: {rmse:.2f} W")
print(f"R²  : {r2:.4f}")

print("=" * 60)


# --------------------------------------------------
# Feature importance
# --------------------------------------------------

importance = pd.DataFrame({
    "feature": features,
    "importance": model.feature_importances_
})

importance = importance.sort_values(
    "importance",
    ascending=False
)


print()
print("FEATURE IMPORTANCE")
print("-" * 60)

print(
    importance.to_string(
        index=False
    )
)


# --------------------------------------------------
# Save predictions
# --------------------------------------------------

results = test_df[
    ["timestamp", "pv_power_w"]
].copy()

results["predicted_next_hour_pv_power_w"] = predictions


results.to_csv(
    output_file,
    index=False
)


print()
print("Forecast predictions saved to:")

print(output_file)