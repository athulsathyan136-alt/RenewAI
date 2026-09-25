# RenewAI ☀️

## Dubai Solar Energy Forecasting & Renewable Intelligence Platform

RenewAI is an end-to-end renewable-energy machine learning platform designed to forecast solar photovoltaic (PV) power generation for Dubai.

The project combines real solar-resource and meteorological data from the European Commission's **Photovoltaic Geographical Information System (PVGIS)** with machine learning, data engineering, FastAPI, Docker, automated testing, and cloud-ready architecture.

The primary machine-learning objective is **1-hour-ahead solar PV power forecasting** using historical generation, irradiance, weather, and calendar features.

> **Project status:** Working ML forecasting pipeline + REST API + Docker + automated API tests
> **Dataset:** PVGIS 5.3, Dubai, 2005–2023
> **Forecast horizon:** 1 hour ahead
> **Primary model:** Random Forest Regressor
> **Author:** Athul Sathyan

---

## 🌍 Why RenewAI?

Solar energy production changes continuously with sunlight, weather, seasonality, and time of day.

Accurate short-term solar forecasting can support:

* Renewable-energy planning
* Solar power monitoring
* Battery-storage planning
* Grid integration
* Energy-management systems
* Renewable-energy decision support
* Future smart-grid applications

RenewAI explores how machine learning can transform historical solar and weather information into short-term PV generation forecasts.

The project is designed as a practical demonstration of how **AI/ML + renewable energy + cloud engineering** can work together.

---

## 🎯 Project Objectives

RenewAI has five main objectives:

1. Collect a long-term solar-energy dataset for Dubai.
2. Clean and engineer the data for machine learning.
3. Build a baseline PV power estimation model.
4. Build a genuine 1-hour-ahead forecasting model.
5. Expose the trained model through a production-style REST API and Docker container.

---

## 📊 Dataset

RenewAI uses the **PVGIS 5.3** hourly dataset for Dubai.

PVGIS is developed by the European Commission Joint Research Centre (JRC). PVGIS 5.3 provides updated solar-resource and meteorological datasets extending through 2023. The PVGIS hourly service can provide hourly PV output and solar/weather variables.

Official documentation:

* [PVGIS 5.3 — European Commission JRC](https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/using-pvgis-5_en)
* [PVGIS API documentation](https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/using-pvgis-5/api-non-interactive-service_en)
* [PVGIS hourly radiation documentation](https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/using-pvgis-5/pvgis-5-tools/hourly-radiation_en)

### Location

**Dubai, United Arab Emirates**

Approximate coordinates used:

```text
Latitude:  25.2048° N
Longitude: 55.2708° E
```

### PV system configuration

```text
PV technology:        Crystalline silicon
Nominal power:        1 kWp
System losses:        14%
Mounting:             Free-standing
Slope:                25°
Azimuth:              0°
Radiation database:   PVGIS-SARAH3
Period:               2005–2023
```

PVGIS-SARAH3 provides satellite-based solar radiation data for 2005–2023. PVGIS also provides meteorological variables such as temperature and wind speed.

### Important dataset note

The PVGIS dataset should **not** be described as direct measurements from a DEWA solar plant.

It is a location-specific PVGIS dataset containing modeled/calculated solar-resource, meteorological, and PV-output information.

RenewAI therefore does **not** claim that its training data represents measurements from a specific Dubai solar power plant.

---

## 📁 Dataset Pipeline

The original PVGIS CSV contains metadata followed by hourly observations.

RenewAI processes the dataset through the following pipeline:

```text
PVGIS Raw CSV
      │
      ▼
Metadata Removal
      │
      ▼
Data Cleaning
      │
      ▼
Timestamp Conversion
      │
      ▼
Calendar Feature Engineering
      │
      ▼
Lag Feature Engineering
      │
      ▼
Train / Test Split
      │
      ▼
Machine Learning
      │
      ▼
Forecast Evaluation
```

---

## 🧹 Data Cleaning

The raw dataset contains metadata rows in addition to the hourly observations.

The cleaning pipeline:

* Removes PVGIS metadata rows
* Converts numeric columns to numeric types
* Renames columns to clearer machine-learning names
* Checks missing values
* Saves the cleaned dataset

Cleaned dataset:

```text
data/processed/dubai_solar_clean.csv
```

### Cleaned dataset size

```text
Rows: 166,536
Columns: 9
Missing values: 0
Duplicate timestamps: 0
```

---

## 🧠 Feature Engineering

RenewAI creates calendar features from the timestamp:

```text
year
month
day
hour
day_of_year
```

The forecasting pipeline additionally creates historical lag features:

```text
pv_power_lag_1h
pv_power_lag_2h
pv_power_lag_3h

irradiance_lag_1h
temperature_lag_1h
wind_lag_1h
```

These features allow the model to use historical conditions when predicting the next hourly PV-power value.

---

## 🤖 Machine Learning Models

RenewAI currently contains two modeling stages.

### 1. Baseline PV Power Estimation

The first model estimates PV power using information from the same observation period, including irradiance and weather variables.

This model achieved:

```text
MAE : 1.12 W
RMSE: 2.49 W
R²  : 0.9999
```

However, this should **not** be interpreted as a genuine future forecast.

The extremely high performance is expected because current irradiance information is directly related to current PV generation.

The baseline model is therefore treated as a reference/estimation model.

---

### 2. 1-Hour-Ahead Forecast Model

The main RenewAI model predicts the next hourly PV power value.

The target is created by shifting PV power forward:

```python
df["target_next_hour"] = df["pv_power_w"].shift(-1)
```

The model uses historical information such as:

```text
Previous PV power
Previous irradiance
Previous temperature
Previous wind speed
Calendar information
```

### Model

```text
Algorithm: Random Forest Regressor
Trees: 100
Maximum depth: 20
Random state: 42
```

### Train/Test Strategy

```text
Training period: 2005–2022
Testing period: 2023
```

### Results

```text
MAE : 17.40 W
RMSE: 49.63 W
R²  : 0.9659
```

These results are from the current experimental forecasting pipeline and should not be interpreted as proof of deployment-level performance.

---

## 📈 Forecasting Features

The current model uses:

| Feature              | Description                                |
| -------------------- | ------------------------------------------ |
| `pv_power_lag_1h`    | PV power from the previous hourly record   |
| `pv_power_lag_2h`    | PV power from two hourly records earlier   |
| `pv_power_lag_3h`    | PV power from three hourly records earlier |
| `irradiance_lag_1h`  | Previous-hour irradiance                   |
| `temperature_lag_1h` | Previous-hour temperature                  |
| `wind_lag_1h`        | Previous-hour wind speed                   |
| `month`              | Month of year                              |
| `day`                | Day of month                               |
| `hour`               | Hour of day                                |
| `day_of_year`        | Day number within the year                 |

---

## 🔎 Feature Importance

The current Random Forest model produced the following feature-importance values:

```text
pv_power_lag_1h       0.521213
hour                  0.436569
day_of_year           0.010317
irradiance_lag_1h     0.009843
temperature_lag_1h    0.007309
wind_lag_1h           0.004658
pv_power_lag_2h       0.003644
pv_power_lag_3h       0.003008
day                   0.002899
month                 0.000540
```

This indicates that recent PV power and time-of-day information are currently the strongest predictors in the model.

Feature importance from a tree model should be interpreted as model-specific importance, not as a causal explanation of solar generation.

---

## 📊 Dataset Statistics

The processed Dubai dataset contains:

```text
Total observations:      166,536
Period:                  2005–2023
PV power minimum:        0 W
PV power maximum:        885.47 W
Average PV power:        196.27 W
Zero-generation records: 87,159
```

Environmental ranges:

```text
Beam irradiance:    0 – 977.19 W/m²
Diffuse irradiance: 0 – 486.17 W/m²
Temperature:        7.25 – 47.06 °C
Wind speed:         0 – 14.97 m/s
```

The model operates on a **1 kWp reference PV system**, so the PV-power values should not be interpreted as the total output of Dubai's entire solar fleet.

---

## 📊 Visualizations

RenewAI generates several visualizations during the analysis pipeline.

### Historical PV generation

```text
data/processed/pv_generation_first_week.png
```

### Average PV generation by hour

```text
data/processed/average_pv_by_hour.png
```

### Average PV generation by month

```text
data/processed/average_pv_by_month.png
```

### Baseline actual vs predicted

```text
data/processed/baseline_actual_vs_predicted_first_week.png
data/processed/baseline_actual_vs_predicted_scatter.png
```

### Forecast actual vs predicted

```text
data/processed/forecast_first_week.png
data/processed/forecast_scatter.png
```

---

## 🚀 FastAPI

RenewAI exposes the trained model through a REST API using FastAPI.

### Start the API

From the project root:

```powershell
.venv\Scripts\python.exe -m uvicorn api.main:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

Interactive Swagger documentation:

```text
http://127.0.0.1:8000/docs
```

---

## 🔌 API Endpoints

### Root

```http
GET /
```

Example response:

```json
{
  "message": "RenewAI Solar Forecast API",
  "status": "running",
  "forecast_horizon": "1 hour ahead"
}
```

---

### Health Check

```http
GET /health
```

Example:

```json
{
  "status": "healthy",
  "model": "renewai_forecast_model",
  "forecast_horizon": "1 hour ahead"
}
```

---

### Solar Forecast

```http
POST /predict
```

Example request:

```json
{
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
```

Example response:

```json
{
  "forecast_horizon": "1 hour ahead",
  "predicted_pv_power_w": 565.39,
  "predicted_pv_power_kw": 0.565
}
```

---

## 🧪 API Validation

The API validates incoming requests using Pydantic.

For example:

```text
month = 0
```

is rejected because valid months are:

```text
1–12
```

An invalid request returns:

```text
HTTP 422
```

---

## 🧪 Automated Testing

RenewAI uses `pytest` and FastAPI's `TestClient`.

Current tests cover:

```text
✓ Root endpoint
✓ Health endpoint
✓ Prediction endpoint
✓ Invalid month validation
```

Run tests:

```powershell
.venv\Scripts\python.exe -m pytest -q
```

Current result:

```text
4 passed
```

---

## 🐳 Docker

RenewAI can run as a containerized API.

Build the Docker image:

```powershell
docker build -t renewai-api .
```

Run the container:

```powershell
docker run --rm -p 8000:8000 renewai-api
```

Then open:

```text
http://127.0.0.1:8000
```

Swagger:

```text
http://127.0.0.1:8000/docs
```

The Docker image contains:

```text
Python 3.12
FastAPI
Uvicorn
Pandas
NumPy
SciPy
Scikit-learn
Joblib
RenewAI API
Trained model
```

---

## 📦 Project Structure

```text
RenewAI/
│
├── api/
│   └── main.py
│
├── data/
│   ├── raw/
│   │   └── dubai_pvgis_hourly.csv
│   │
│   └── processed/
│       ├── dubai_solar_clean.csv
│       ├── dubai_solar_features.csv
│       ├── dubai_solar_forecast_features.csv
│       ├── baseline_predictions_2023.csv
│       ├── forecast_predictions_2023.csv
│       └── visualization outputs
│
├── models/
│   └── renewai_forecast_model.joblib
│
├── src/
│   ├── create_dataset.py
│   ├── clean_data.py
│   ├── feature_engineering.py
│   ├── forecast_features.py
│   ├── data_quality_check.py
│   ├── visualize_data.py
│   ├── train_baseline.py
│   ├── evaluate_baseline.py
│   ├── train_forecast.py
│   └── evaluate_forecast.py
│
├── tests/
│   └── test_api.py
│
├── Dockerfile
├── .dockerignore
├── .gitignore
├── requirements.txt
├── requirements-dev.txt
└── README.md
```

---

## ⚙️ Installation

### 1. Clone the repository

```bash
git clone https://github.com/athulsathyan136-alt/RenewAI.git
cd RenewAI
```

### 2. Create a virtual environment

Windows:

```powershell
py -3.12 -m venv .venv
```

### 3. Install dependencies

```powershell
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For development and testing:

```powershell
.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
```

### 4. Run the API

```powershell
.venv\Scripts\python.exe -m uvicorn api.main:app --reload
```

---

## 🔬 Reproducing the ML Pipeline

The project scripts can be executed in stages.

### Clean the raw dataset

```powershell
.venv\Scripts\python.exe src/clean_data.py
```

### Create calendar features

```powershell
.venv\Scripts\python.exe src/feature_engineering.py
```

### Check data quality

```powershell
.venv\Scripts\python.exe src/data_quality_check.py
```

### Generate visualizations

```powershell
.venv\Scripts\python.exe src/visualize_data.py
```

### Create forecasting features

```powershell
.venv\Scripts\python.exe src/forecast_features.py
```

### Train baseline model

```powershell
.venv\Scripts\python.exe src/train_baseline.py
```

### Evaluate baseline

```powershell
.venv\Scripts\python.exe src/evaluate_baseline.py
```

### Train 1-hour-ahead model

```powershell
.venv\Scripts\python.exe src/train_forecast.py
```

### Evaluate forecast

```powershell
.venv\Scripts\python.exe src/evaluate_forecast.py
```

---

## ☁️ Cloud & MLOps Roadmap

RenewAI is designed to evolve from a local machine-learning project into a cloud-based renewable-energy forecasting platform.

Planned components include:

```text
AWS S3
   │
   ▼
Data Storage
   │
   ▼
ML Training
   │
   ▼
Model Artifact
   │
   ▼
Model Registry / Object Storage
   │
   ▼
Docker
   │
   ▼
FastAPI
   │
   ▼
Cloud Deployment
   │
   ▼
Monitoring
```

Potential future AWS components:

* Amazon S3
* Amazon ECR
* Amazon ECS
* AWS Lambda
* IAM
* CloudWatch
* CI/CD
* Model artifact storage
* Automated retraining

---

## 🔮 Future Improvements

RenewAI is an evolving project.

Planned improvements include:

### Forecasting

* Persistence baseline comparison
* Better temporal validation
* Daylight-only evaluation
* Capacity-normalized metrics
* Additional regression models
* Gradient boosting
* XGBoost/LightGBM experimentation
* Deep-learning forecasting
* LSTM/GRU models
* Transformer-based time-series models

### Weather-aware forecasting

The current model primarily uses historical weather/solar variables.

Future versions can incorporate:

* Numerical weather prediction data
* Cloud-cover forecasts
* Temperature forecasts
* Wind forecasts
* Solar irradiance forecasts

This would make the forecasting pipeline more suitable for genuine operational forecasting scenarios.

### MLOps

Future versions may include:

* Automated model training
* Experiment tracking
* Model versioning
* Data versioning
* CI/CD
* Automated testing
* Model monitoring
* Data-drift detection
* Model-performance monitoring

### Cloud

Planned deployment architecture:

```text
GitHub
   │
   ▼
CI/CD
   │
   ▼
Docker
   │
   ▼
AWS
   │
   ├── S3
   ├── ECR
   ├── ECS
   └── CloudWatch
```

---

## ⚠️ Current Limitations

RenewAI is currently a research/portfolio prototype rather than a production grid-control system.

Important limitations include:

1. The PVGIS dataset is not direct measurements from a specific Dubai PV plant.
2. The current PV system represents a 1 kWp reference system.
3. The current forecasting model uses historical lag features rather than future weather forecasts.
4. The current evaluation uses a single calendar-year holdout: 2023.
5. The current Random Forest model has not been compared against a full set of operational forecasting baselines.
6. The current API expects manually supplied feature values.
7. The trained model file is too large for normal GitHub repository storage and is intentionally excluded from Git tracking.
8. Additional validation is required before any real-world energy-management or grid-control use.

---

## 🗂️ Model Artifact

The trained model is generated locally as:

```text
models/renewai_forecast_model.joblib
```

The model is intentionally excluded from Git because of its large file size.

The GitHub repository contains the code required to understand and reproduce the modeling pipeline.

A future version will use dedicated model-artifact storage such as:

```text
AWS S3
```

or another model-hosting solution.

---

## 🔐 Repository Practices

RenewAI excludes:

```text
.venv/
data/raw/
data/processed/
*.joblib
.env
__pycache__/
```

This keeps the repository focused on source code and documentation while avoiding large generated artifacts and local environment files.

---

## 🌱 Renewable-Energy Impact

RenewAI is designed around a practical renewable-energy problem:

> **How can machine learning help anticipate solar PV generation before it occurs?**

Short-term solar forecasting can contribute to better planning for:

* Solar integration
* Energy storage
* Grid balancing
* Renewable-energy scheduling
* Energy management
* Distributed solar systems
* Smart-grid applications

The project is intentionally structured so that the machine-learning component can later become part of a larger renewable-energy intelligence platform.

---

## 🌍 IRENA-Oriented Development

RenewAI is also being developed as a technical project exploring the intersection of:

```text
Artificial Intelligence
        +
Renewable Energy
        +
Cloud Computing
        +
Data Engineering
        +
MLOps
```

The long-term objective is to demonstrate practical technical skills that can contribute to renewable-energy innovation and digital energy systems.

The project does not claim to represent an official IRENA system, project, partnership, or endorsement.

---

## 🧑‍💻 Author

**Athul Sathyan**

B.Tech Computer Engineering

Focus areas:

```text
AI / Machine Learning
Renewable Energy AI
Cloud Engineering
MLOps
Generative AI
Data Engineering
```

---

## 📌 Repository

GitHub:

https://github.com/athulsathyan136-alt/RenewAI

---

## 📚 Data & Technical References

### PVGIS

European Commission Joint Research Centre:

https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis_en

PVGIS 5.3 documentation:

https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/using-pvgis-5_en

PVGIS API:

https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/using-pvgis-5/api-non-interactive-service_en

PVGIS hourly radiation:

https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/using-pvgis-5/pvgis-5-tools/hourly-radiation_en

---

## 📄 License

This project is intended for educational, research, and portfolio purposes.

See the repository `LICENSE` file for the applicable license.

---

## ⭐ Project Status

```text
[████████████████████████████████████████] Core prototype complete

Data pipeline       ✅
ML baseline         ✅
1-hour forecasting  ✅
Evaluation          ✅
FastAPI             ✅
Validation          ✅
Docker              ✅
GitHub              ✅

Cloud deployment    🔄 Planned
MLOps pipeline      🔄 Planned
Advanced models     🔄 Planned
Monitoring          🔄 Planned
```

---

**RenewAI — Using AI and data engineering to explore smarter renewable-energy forecasting. ☀️**
