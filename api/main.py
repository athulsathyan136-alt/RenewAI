from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel, Field


# --------------------------------------------------
# Load trained model
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = BASE_DIR / "models" / "renewai_forecast_model.joblib"

model = joblib.load(MODEL_PATH)


# --------------------------------------------------
# FastAPI application
# --------------------------------------------------

app = FastAPI(
    title="RenewAI Solar Forecast API",
    description="1-hour-ahead solar PV generation forecasting API for Dubai",
    version="1.0.0"
)


# --------------------------------------------------
# Request schema
# --------------------------------------------------

class ForecastRequest(BaseModel):

    pv_power_lag_1h: float = Field(
        ge=0,
        description="PV power from the previous hour in watts"
    )

    pv_power_lag_2h: float = Field(
        ge=0,
        description="PV power from two hours ago in watts"
    )

    pv_power_lag_3h: float = Field(
        ge=0,
        description="PV power from three hours ago in watts"
    )

    irradiance_lag_1h: float = Field(
        ge=0,
        description="Previous-hour solar irradiance in W/m²"
    )

    temperature_lag_1h: float = Field(
        ge=-50,
        le=70,
        description="Previous-hour temperature in °C"
    )

    wind_lag_1h: float = Field(
        ge=0,
        description="Previous-hour wind speed in m/s"
    )

    month: int = Field(
        ge=1,
        le=12,
        description="Month number"
    )

    day: int = Field(
        ge=1,
        le=31,
        description="Day of month"
    )

    hour: int = Field(
        ge=0,
        le=23,
        description="Hour of day"
    )

    day_of_year: int = Field(
        ge=1,
        le=366,
        description="Day number within the year"
    )


# --------------------------------------------------
# Health endpoint
# --------------------------------------------------

@app.get("/")
def root():
    return {
        "message": "RenewAI Solar Forecast API",
        "status": "running",
        "forecast_horizon": "1 hour ahead"
    }

@app.get("/health")
def health():
    return {
        "status": "healthy",
        "model": "renewai_forecast_model",
        "forecast_horizon": "1 hour ahead"
    }

# --------------------------------------------------
# Forecast endpoint
# --------------------------------------------------

@app.post("/predict")
def predict(request: ForecastRequest):

    input_data = pd.DataFrame([
        {
            "pv_power_lag_1h": request.pv_power_lag_1h,
            "pv_power_lag_2h": request.pv_power_lag_2h,
            "pv_power_lag_3h": request.pv_power_lag_3h,
            "irradiance_lag_1h": request.irradiance_lag_1h,
            "temperature_lag_1h": request.temperature_lag_1h,
            "wind_lag_1h": request.wind_lag_1h,
            "month": request.month,
            "day": request.day,
            "hour": request.hour,
            "day_of_year": request.day_of_year
        }
    ])

    prediction = model.predict(input_data)[0]

    prediction = max(0.0, float(prediction))

    return {
        "forecast_horizon": "1 hour ahead",
        "predicted_pv_power_w": round(prediction, 2),
        "predicted_pv_power_kw": round(prediction / 1000, 3)
    }