from __future__ import annotations

from contextlib import asynccontextmanager
from functools import lru_cache

from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field

from glass_ai.analytics import (
    get_executive_insights,
    get_summary_metrics,
    list_analyses,
    run_named_analysis,
)
from glass_ai.config import settings
from glass_ai.data import ensure_database, load_production_data
from glass_ai.ml import predict_defect_risk, train_defect_model


@asynccontextmanager
async def lifespan(_: FastAPI):
    ensure_database()
    yield


app = FastAPI(
    title=f"{settings.project_name} API",
    version="0.1.0",
    summary="Industrial analytics and defect-risk APIs for a simulated glass factory.",
    description=(
        "A portfolio-friendly Industry 4.0 backend that exposes manufacturing KPIs, "
        "SQL analytics, model evaluation metrics, and defect-risk predictions."
    ),
    lifespan=lifespan,
    openapi_tags=[
        {"name": "platform", "description": "Service discovery and health endpoints."},
        {"name": "analytics", "description": "Manufacturing KPI and SQL analytics endpoints."},
        {"name": "ml", "description": "Defect prediction and model-performance endpoints."},
    ],
)


class HealthResponse(BaseModel):
    status: str
    environment: str
    database_path: str


class SummaryResponse(BaseModel):
    total_batches: int
    total_defects: int
    average_produced_units: float
    defect_rate_pct: float
    on_time_delivery_rate_pct: float
    average_yield_rate_pct: float
    total_scrap_cost_eur: float
    energy_intensity_kwh_per_unit: float
    scrap_share_pct: float
    model_accuracy: float
    model_recall: float
    model_roc_auc: float


class AnalysisDefinition(BaseModel):
    slug: str
    label: str
    description: str


class ModelPerformanceResponse(BaseModel):
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float
    decision_threshold: float
    train_rows: int
    test_rows: int
    confusion_matrix: list[list[int]]


class ExecutiveInsightsResponse(BaseModel):
    insights: list[str]


class PredictionRequest(BaseModel):
    plant_id: str
    machine_id: str
    glass_type: str
    thickness_mm: float = Field(gt=0.0)
    furnace_temperature_c: float
    pressure_bar: float
    line_speed_mps: float
    cooling_time_sec: float
    raw_material_quality: float = Field(ge=0.0, le=1.0)
    operator_shift: str
    operator_experience_years: float = Field(ge=0.0)
    ambient_humidity_pct: float = Field(ge=0.0, le=100.0)
    furnace_zone: str


class PredictionResponse(BaseModel):
    prediction: int
    probability: float
    risk_level: str
    alerts: list[str]
    recommendations: list[str]


@lru_cache(maxsize=1)
def get_dataframe():
    ensure_database()
    return load_production_data()


@lru_cache(maxsize=1)
def get_model_bundle():
    return train_defect_model(get_dataframe())


@app.get("/", tags=["platform"])
def read_root() -> dict[str, object]:
    return {
        "project": settings.project_name,
        "environment": settings.app_env,
        "docs_url": "/docs",
        "focus": [
            "industrial analytics",
            "quality intelligence",
            "defect prediction",
        ],
        "available_routes": [
            "/health",
            "/api/v1/summary",
            "/api/v1/analytics",
            "/api/v1/model/performance",
            "/api/v1/insights/executive-summary",
            "/api/v1/predictions/defect-risk",
        ],
    }


@app.get("/health", response_model=HealthResponse, tags=["platform"])
def health_check() -> HealthResponse:
    ensure_database()
    return HealthResponse(
        status="ok",
        environment=settings.app_env,
        database_path=str(settings.db_path),
    )


@app.get("/api/v1/summary", response_model=SummaryResponse, tags=["analytics"])
def get_summary() -> SummaryResponse:
    summary_metrics = get_summary_metrics()
    model_bundle = get_model_bundle()

    return SummaryResponse(
        **summary_metrics,
        model_accuracy=model_bundle.accuracy,
        model_recall=model_bundle.metrics["recall"],
        model_roc_auc=model_bundle.metrics["roc_auc"],
    )


@app.get("/api/v1/analytics", response_model=list[AnalysisDefinition], tags=["analytics"])
def get_analytics_catalog() -> list[AnalysisDefinition]:
    return [AnalysisDefinition(**analysis) for analysis in list_analyses()]


@app.get("/api/v1/analytics/{analysis_slug}", tags=["analytics"])
def get_analysis_result(analysis_slug: str) -> dict[str, object]:
    try:
        result = run_named_analysis(analysis_slug)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error

    return {
        "analysis_slug": analysis_slug,
        "records": jsonable_encoder(result.to_dict(orient="records")),
    }


@app.get(
    "/api/v1/model/performance",
    response_model=ModelPerformanceResponse,
    tags=["ml"],
)
def get_model_performance() -> ModelPerformanceResponse:
    model_bundle = get_model_bundle()
    return ModelPerformanceResponse(
        **model_bundle.metrics,
        train_rows=model_bundle.train_rows,
        test_rows=model_bundle.test_rows,
        confusion_matrix=model_bundle.confusion_matrix,
    )


@app.get(
    "/api/v1/insights/executive-summary",
    response_model=ExecutiveInsightsResponse,
    tags=["analytics"],
)
def get_executive_summary() -> ExecutiveInsightsResponse:
    return ExecutiveInsightsResponse(insights=get_executive_insights())


@app.post(
    "/api/v1/predictions/defect-risk",
    response_model=PredictionResponse,
    tags=["ml"],
)
def create_defect_prediction(request: PredictionRequest) -> PredictionResponse:
    prediction_result = predict_defect_risk(get_model_bundle(), request.model_dump())
    return PredictionResponse(**prediction_result)
