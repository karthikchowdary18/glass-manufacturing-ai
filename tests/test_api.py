from fastapi.testclient import TestClient

from services.api.main import app


client = TestClient(app)


def test_health_check():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_summary_endpoint():
    response = client.get("/api/v1/summary")
    assert response.status_code == 200
    payload = response.json()
    assert payload["total_batches"] > 0
    assert "model_accuracy" in payload
    assert "model_recall" in payload
    assert "model_roc_auc" in payload


def test_analytics_catalog_endpoint():
    response = client.get("/api/v1/analytics")
    assert response.status_code == 200
    payload = response.json()
    assert len(payload) >= 4
    assert "description" in payload[0]


def test_model_performance_endpoint():
    response = client.get("/api/v1/model/performance")
    assert response.status_code == 200
    payload = response.json()
    assert payload["roc_auc"] > 0.6
    assert payload["decision_threshold"] >= 0.2
    assert len(payload["confusion_matrix"]) == 2


def test_prediction_endpoint_returns_actions():
    response = client.post(
        "/api/v1/predictions/defect-risk",
        json={
            "plant_id": "Plant_A",
            "machine_id": "M2",
            "glass_type": "Laminated",
            "thickness_mm": 8.0,
            "furnace_temperature_c": 1548.0,
            "pressure_bar": 4.2,
            "line_speed_mps": 6.9,
            "cooling_time_sec": 66.0,
            "raw_material_quality": 0.79,
            "operator_shift": "Night",
            "operator_experience_years": 2.0,
            "ambient_humidity_pct": 74.0,
            "furnace_zone": "Zone_3",
        },
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["risk_level"] in {"low", "medium", "high"}
    assert len(payload["recommendations"]) >= 1
    assert len(payload["alerts"]) >= 1


def test_unknown_analysis_returns_404():
    response = client.get("/api/v1/analytics/not-a-real-analysis")
    assert response.status_code == 404
