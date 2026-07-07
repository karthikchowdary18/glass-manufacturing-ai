# Architecture Notes

## Goal

The project is intentionally structured like a small production-grade analytics product instead of a notebook demo. The main design idea is simple: keep manufacturing logic in one place and let both the API and dashboard reuse it.

## Layers

### `pipelines/`

Creates the synthetic dataset and rebuilds the SQLite database. This makes the repo reproducible and gives the project a proper data-ingestion story.

### `glass_ai/`

Shared domain logic:

- `data.py` handles CSV and SQLite access
- `analytics.py` computes KPIs, business insights, and SQL-driven analysis views
- `ml.py` trains the defect-risk model, exposes evaluation metrics, and generates process recommendations
- `chatbot.py` maps curated questions to analytics outputs
- `config.py` centralizes environment-aware settings

### `services/api/`

FastAPI wraps the shared logic with stable HTTP contracts. This shows backend thinking, schema design, and service-oriented reuse.

### `apps/dashboard/`

Streamlit presents the same business logic as a control-tower style UI for plant managers, quality engineers, and operations analysts.

## Why SQLite

SQLite is small enough for a self-contained portfolio repo but still demonstrates:

- SQL-based KPI analysis
- clean separation between raw data and queryable analytics store
- realistic migration path to PostgreSQL or a warehouse later

## Why The ML Setup Matters

The model is intentionally framed as a quality-alert assistant, not a magic predictor. The repo includes:

- leakage-aware feature selection
- categorical + numerical preprocessing
- precision / recall / F1 / ROC-AUC reporting
- decision-threshold tuning
- action-oriented recommendations after inference

This makes the ML part easier to defend in interviews because the project shows evaluation discipline instead of only reporting accuracy.

## Suggested Evolution Path

1. Replace SQLite with PostgreSQL or DuckDB.
2. Add scheduled pipelines and model retraining.
3. Add a deployment target such as Azure, Render, or Fly.io.
4. Add observability for data drift and API health.
