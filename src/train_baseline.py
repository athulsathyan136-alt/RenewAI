import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# --------------------------------------------------
# 1. Load dataset
# --------------------------------------------------

input_file = Path("data/processed/dubai_solar_features.csv")

print("Loading RenewAI dataset...")

df = pd.read_csv(input_file)

df["timestamp"] = pd.to_datetime(df["timestamp"])

print("Dataset:", df.shape)


# --------------------------------------------------
# 2. Define features and target
# --------------------------------------------------

features = [
    "beam_irradiance_wm2",
    "diffuse_irradiance_wm2",
    "reflected_irradiance_wm2",
    "sun_height_deg",
    "temperature_c",
    "wind_speed_ms",
    "month",
    "day",
    "hour",
    "day_of_year"
]

target = "pv_power_w"


# --------------------------------------------------
# 3. Time-based train/test split
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
print("Period:", train_df["timestamp"].min(), "→", train_df["timestamp"].max())

print()
print("TEST DATA")
print("Rows:", len(test_df))
print("Period:", test_df["timestamp"].min(), "→", test_df["timestamp"].max())


# --------------------------------------------------
# 4. Train Random Forest
# --------------------------------------------------

print()
print("Training Random Forest...")

model = RandomForestRegressor(
    n_estimators=100,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)

model.fit(X_train, y_train)

print("Training complete.")


# --------------------------------------------------
# 5. Make predictions
# --------------------------------------------------

predictions = model.predict(X_test)


# --------------------------------------------------
# 6. Evaluate model
# --------------------------------------------------

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print()
print("=" * 60)
print("RENEWAI BASELINE MODEL RESULTS")
print("=" * 60)

print(f"MAE : {mae:.2f} W")
print(f"RMSE: {rmse:.2f} W")
print(f"R²  : {r2:.4f}")

print("=" * 60)


# --------------------------------------------------
# 7. Feature importance
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
print(importance.to_string(index=False))


# --------------------------------------------------
# 8. Save results
# --------------------------------------------------

results = test_df[
    ["timestamp", "pv_power_w"]
].copy()

results["predicted_pv_power_w"] = predictions

output_file = Path(
    "data/processed/baseline_predictions_2023.csv"
)

results.to_csv(output_file, index=False)

print()
print("Predictions saved to:")
print(output_file)