from fastapi.testclient import TestClient

from api.main import app


client = TestClient(app)


def test_root():
    response = client.get("/")

    assert response.status_code == 200
    assert response.json()["status"] == "running"


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_predict():
    payload = {
        "pv_power_lag_1h": 694.46,
        "pv_power_lag_2h": 642.76,
        "pv_power_lag_3h": 481.10,
        "irradiance_lag_1h": 850.0,
        "temperature_lag_1h": 30.0,
        "wind_lag_1h": 4.0,
        "month": 1,
        "day": 1,
        "hour": 9,
        "day_of_year": 1
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert "predicted_pv_power_w" in response.json()
    assert "predicted_pv_power_kw" in response.json()


def test_invalid_month():
    payload = {
        "pv_power_lag_1h": 694.46,
        "pv_power_lag_2h": 642.76,
        "pv_power_lag_3h": 481.10,
        "irradiance_lag_1h": 850.0,
        "temperature_lag_1h": 30.0,
        "wind_lag_1h": 4.0,
        "month": 0,
        "day": 1,
        "hour": 9,
        "day_of_year": 1
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 422
