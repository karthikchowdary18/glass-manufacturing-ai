# Glass Manufacturing AI

Glass Manufacturing AI is an Industry 4.0 portfolio project that simulates a multi-plant glass factory and packages the result like a small analytics product instead of a one-off notebook. It combines synthetic manufacturing data generation, SQL analytics, defect-risk prediction, a FastAPI backend, a Streamlit dashboard, tests, and Docker support in one repository.

The main idea is simple: show how data, software, and operations thinking can work together in a believable manufacturing use case. The project is aimed at data, AI, analytics, and industrial software roles where quality control, process optimization, yield improvement, and operational visibility matter.

## Table of Contents

- [Why This Project Exists](#why-this-project-exists)
- [What The Platform Does](#what-the-platform-does)
- [Architecture](#architecture)
- [Dataset At A Glance](#dataset-at-a-glance)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Running The API](#running-the-api)
- [Running The Dashboard](#running-the-dashboard)
- [API Endpoints](#api-endpoints)
- [Example Prediction Request](#example-prediction-request)
- [Testing And CI](#testing-and-ci)
- [How The Data Flow Works](#how-the-data-flow-works)
- [Why This Repo Is Strong On GitHub](#why-this-repo-is-strong-on-github)
- [Known Limitations](#known-limitations)
- [Roadmap](#roadmap)
- [Documentation](#documentation)

## Why This Project Exists

Many portfolio repositories show an ML model in isolation. This project is deliberately broader:

- It solves a believable factory problem instead of a generic classification exercise.
- It demonstrates product thinking, not just model training.
- It separates pipelines, shared logic, services, apps, and tests into clear layers.
- It frames ML as decision support with alerts and recommendations, not as "magic AI".
- It is easy to demo locally because the synthetic dataset and SQLite database are included.

If someone opens this repo on GitHub, they should be able to understand the problem, run the system, inspect the architecture, and see evidence of software engineering discipline within a few minutes.

## What The Platform Does

The repository simulates a glass manufacturing environment with multiple plants, machines, product families, and operator shifts. On top of that data, it provides:

- Executive KPI summaries for throughput, defects, yield, on-time delivery, scrap cost, and energy intensity
- SQL-backed operational analyses for plants, machines, shifts, glass types, and monthly performance
- A defect-risk model that predicts likely quality issues for a proposed production batch
- Action-oriented alerts and recommended corrective actions based on process conditions
- A Streamlit dashboard for plant-level monitoring and scenario exploration
- A FastAPI backend for programmatic access to analytics and predictions
- A lightweight analytics helper for curated manufacturing questions

## Architecture

```text
pipelines/generate_data.py
        |
        v
data/raw/glass_production.csv
        |
        v
pipelines/build_sqlite.py
        |
        v
data/processed/glass_factory.db
        |
        v
glass_ai/
  - config.py      environment-aware settings
  - data.py        CSV and SQLite access
  - analytics.py   KPIs, executive insights, SQL analyses
  - ml.py          model training, metrics, alerts, recommendations
  - chatbot.py     curated question routing
        |
        +----------------------------+
        |                            |
        v                            v
services/api/main.py         apps/dashboard/app.py
FastAPI service              Streamlit dashboard
```

### Layer Responsibilities

- `pipelines/` creates reproducible demo assets: the CSV dataset and the SQLite analytics store.
- `glass_ai/` holds the shared business logic so the API and dashboard do not duplicate behavior.
- `services/api/` exposes analytics, health, model diagnostics, and prediction endpoints.
- `apps/dashboard/` turns the same shared logic into a visual control-tower style interface.
- `tests/` validates both API behavior and analytics outputs.

## Dataset At A Glance

The dataset is synthetic, but it is structured to feel like an actual manufacturing environment rather than a toy table.

- 1,800 simulated production batches
- 26 production and quality columns
- 3 plants: `Plant_A`, `Plant_B`, `Plant_C`
- 5 machines: `M1` to `M5`
- 4 glass families: `Float`, `Tempered`, `Laminated`, `Solar`
- 3 shifts: `Day`, `Night`, `Weekend`
- Roughly 18 months of production activity

### Example Signals Included

- furnace temperature
- cooling time
- pressure
- line speed
- raw material quality
- operator experience
- ambient humidity
- produced units
- scrap units
- packing delay
- downtime
- estimated scrap cost
- CO2 emissions

The data generator intentionally injects realistic bias patterns such as tougher night and weekend conditions, machine-level quality variation, and humidity sensitivity for some product families.

## Tech Stack

- Python
- Pandas and NumPy
- scikit-learn
- SQLite
- FastAPI
- Streamlit
- Altair
- Pytest
- Docker and Docker Compose
- GitHub Actions

## Repository Structure

```text
glass-manufacturing-ai/
|-- .github/workflows/ci.yml        # CI pipeline for rebuild + test
|-- apps/dashboard/                 # Streamlit UI
|-- data/raw/                       # synthetic CSV dataset
|-- data/processed/                 # SQLite analytics database
|-- docs/                           # architecture and positioning notes
|-- glass_ai/                       # shared domain, data, analytics, and ML logic
|-- infra/docker/                   # Dockerfiles for API and dashboard
|-- pipelines/                      # reproducible data generation scripts
|-- services/api/                   # FastAPI application
|-- tests/                          # API and analytics tests
|-- .env.example                    # example configuration
|-- docker-compose.yml              # local multi-service startup
|-- requirements.txt                # Python dependencies
`-- README.md
```

## Quick Start

### Prerequisites

- Python 3.11 or newer recommended
- `pip`
- Docker Desktop optional, only if you want container-based startup

### 1. Clone The Repository

```bash
git clone <your-repo-url>
cd glass-manufacturing-ai
```

### 2. Create And Activate A Virtual Environment

```bash
python -m venv .venv
```

Linux or macOS:

```bash
source .venv/bin/activate
```

Windows PowerShell:

```powershell
.venv\Scripts\Activate.ps1
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Add Your Local Environment File

Linux or macOS:

```bash
cp .env.example .env
```

Windows PowerShell:

```powershell
Copy-Item .env.example .env
```

### 5. Use The Bundled Demo Data Or Rebuild It

This repository already includes:

- `data/raw/glass_production.csv`
- `data/processed/glass_factory.db`

That means you can start the app immediately. If you want to regenerate both assets:

```bash
python -m pipelines.generate_data
python -m pipelines.build_sqlite
```

## Configuration

Configuration lives in `.env` and is loaded by [`glass_ai/config.py`](glass_ai/config.py).

| Variable | Purpose | Default |
| --- | --- | --- |
| `APP_ENV` | environment label | `development` |
| `PROJECT_NAME` | API and app display name | `Glass Manufacturing AI` |
| `CSV_PATH` | input dataset location | `data/raw/glass_production.csv` |
| `DB_PATH` | SQLite database location | `data/processed/glass_factory.db` |
| `SQLITE_TABLE` | analytics table name | `production` |
| `API_HOST` | FastAPI bind host | `0.0.0.0` |
| `API_PORT` | FastAPI port | `8000` |
| `DASHBOARD_HOST` | Streamlit bind host | `0.0.0.0` |
| `DASHBOARD_PORT` | Streamlit port | `8501` |

## Running The API

Start the backend service:

```bash
uvicorn services.api.main:app --reload
```

The API will be available at:

- `http://127.0.0.1:8000`
- interactive docs: `http://127.0.0.1:8000/docs`

The service ensures the SQLite database exists on startup. Model training is cached, so the first model-related request may take a little longer than later calls.

## Running The Dashboard

Start the Streamlit application:

```bash
streamlit run apps/dashboard/app.py
```

The dashboard will be available at:

- `http://127.0.0.1:8501`

### Dashboard Pages

- `Overview` shows KPI cards, monthly trends, plant benchmarks, shift performance, and delayed batches.
- `Prediction` lets you simulate a batch and inspect risk, alerts, recommendations, and model diagnostics.
- `Explorer` lets you run named analyses and use curated example questions against the analytics layer.

## API Endpoints

### Platform

- `GET /`
- `GET /health`

### Analytics

- `GET /api/v1/summary`
- `GET /api/v1/analytics`
- `GET /api/v1/analytics/{analysis_slug}`
- `GET /api/v1/insights/executive-summary`

### Machine Learning

- `GET /api/v1/model/performance`
- `POST /api/v1/predictions/defect-risk`

### Available Analysis Slugs

- `plant-throughput-and-yield`
- `machine-defect-hotspots`
- `shift-quality-performance`
- `glass-type-cost-risk`
- `monthly-quality-trend`
- `energy-efficiency-by-plant`

## Example Prediction Request

```bash
curl -X POST http://127.0.0.1:8000/api/v1/predictions/defect-risk \
  -H "Content-Type: application/json" \
  -d '{
    "plant_id": "Plant_C",
    "machine_id": "M4",
    "glass_type": "Laminated",
    "thickness_mm": 8.5,
    "furnace_temperature_c": 1548.0,
    "pressure_bar": 4.2,
    "line_speed_mps": 6.9,
    "cooling_time_sec": 66.0,
    "raw_material_quality": 0.79,
    "operator_shift": "Night",
    "operator_experience_years": 2.0,
    "ambient_humidity_pct": 73.0,
    "furnace_zone": "Zone_3"
  }'
```

Typical response fields:

- `prediction`
- `probability`
- `risk_level`
- `alerts`
- `recommendations`

## Testing And CI

Run the local test suite:

```bash
pytest
```

The repository also includes a GitHub Actions workflow at [`.github/workflows/ci.yml`](.github/workflows/ci.yml) that:

1. installs dependencies
2. rebuilds the synthetic dataset and SQLite database
3. runs the test suite

This helps keep the repo push-friendly and gives the project a cleaner public engineering story.

## Docker Startup

If you want to run both services with containers:

```bash
docker compose up --build
```

Services exposed locally:

- API: `http://127.0.0.1:8000`
- Dashboard: `http://127.0.0.1:8501`

The Docker setup mounts the repository into the containers, which makes local iteration easier during development.

## How The Data Flow Works

1. `pipelines/generate_data.py` creates a synthetic production dataset.
2. `pipelines/build_sqlite.py` loads that CSV into SQLite.
3. `glass_ai/data.py` provides shared access helpers for the CSV and database.
4. `glass_ai/analytics.py` calculates KPIs and executes SQL-backed analysis views.
5. `glass_ai/ml.py` trains a defect-risk model and produces alerts and recommendations.
6. `services/api/main.py` exposes the logic as HTTP endpoints.
7. `apps/dashboard/app.py` exposes the same logic as a visual interface.

## Why This Repo Is Strong On GitHub

This project is designed to communicate well to recruiters, hiring managers, and collaborators because it shows more than one skill area at once:

- Business framing: quality control, scrap cost, energy intensity, and operational excellence
- Data engineering basics: reproducible synthetic data and a queryable SQLite store
- Analytics: KPI generation, SQL analysis, filtering, and executive storytelling
- Machine learning: supervised classification, preprocessing, threshold tuning, and evaluation metrics
- Backend engineering: FastAPI routes, response models, and service layering
- Frontend/data app work: a usable Streamlit dashboard with scenario exploration
- Software engineering: tests, CI, Docker, config management, and repo structure

## Known Limitations

- The dataset is synthetic and intended for demonstration, not production decision-making.
- The analytics helper is a curated question router, not a free-form LLM assistant.
- SQLite is a good demo choice, but not the final storage layer for a real factory platform.
- The model trains from local data inside the application process; there is no separate training pipeline or model registry yet.

## Roadmap

- Add time-series anomaly detection for near-real-time process monitoring
- Replace SQLite with PostgreSQL or DuckDB
- Add scheduled pipeline runs and model retraining
- Add deployment automation and hosted demos
- Add richer observability for API health and data drift
- Replace synthetic inputs with anonymized real-world production data when available

## Documentation

- [Architecture Notes](docs/architecture.md)
- [Resume Positioning Guide](docs/resume-positioning.md)

## License

This project is distributed under the terms of the [LICENSE](LICENSE) file included in the repository.
