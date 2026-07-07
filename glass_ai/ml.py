from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


NUMERICAL_FEATURES = [
    "thickness_mm",
    "furnace_temperature_c",
    "cooling_time_sec",
    "pressure_bar",
    "line_speed_mps",
    "raw_material_quality",
    "operator_experience_years",
    "ambient_humidity_pct",
]

CATEGORICAL_FEATURES = [
    "plant_id",
    "machine_id",
    "glass_type",
    "operator_shift",
    "furnace_zone",
]

FEATURE_COLUMNS = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

TEMPERATURE_TARGETS = {
    "Float": 1485.0,
    "Tempered": 1525.0,
    "Laminated": 1505.0,
    "Solar": 1538.0,
}


@dataclass
class ModelBundle:
    pipeline: Pipeline
    feature_columns: list[str]
    metrics: dict[str, float]
    train_rows: int
    test_rows: int
    confusion_matrix: list[list[int]]
    decision_threshold: float

    @property
    def accuracy(self) -> float:
        return self.metrics["accuracy"]


def build_model() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERICAL_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

    classifier = GradientBoostingClassifier(
        n_estimators=220,
        learning_rate=0.05,
        max_depth=3,
        min_samples_leaf=6,
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def train_defect_model(dataframe: pd.DataFrame) -> ModelBundle:
    features = dataframe[FEATURE_COLUMNS]
    target = dataframe["defect_flag"]

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.2,
        random_state=42,
        stratify=target,
    )

    pipeline = build_model()
    pipeline.fit(X_train, y_train)

    probabilities = pipeline.predict_proba(X_test)[:, 1]
    candidate_thresholds = [0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
    threshold_scores = [
        (
            threshold,
            f1_score(
                y_test,
                (probabilities >= threshold).astype(int),
                zero_division=0,
            ),
        )
        for threshold in candidate_thresholds
    ]
    decision_threshold = max(threshold_scores, key=lambda item: item[1])[0]
    predictions = (probabilities >= decision_threshold).astype(int)

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probabilities)), 4),
        "decision_threshold": round(float(decision_threshold), 2),
    }

    return ModelBundle(
        pipeline=pipeline,
        feature_columns=FEATURE_COLUMNS,
        metrics=metrics,
        train_rows=int(len(X_train)),
        test_rows=int(len(X_test)),
        confusion_matrix=confusion_matrix(y_test, predictions).tolist(),
        decision_threshold=float(decision_threshold),
    )


def _get_temperature_reference(glass_type: str) -> float:
    return TEMPERATURE_TARGETS.get(glass_type, 1505.0)


def build_process_alerts(features: dict[str, float | str]) -> list[str]:
    alerts: list[str] = []

    glass_type = str(features["glass_type"])
    target_temp = _get_temperature_reference(glass_type)
    furnace_temperature = float(features["furnace_temperature_c"])

    if abs(furnace_temperature - target_temp) > 25:
        alerts.append("Furnace temperature is far from the expected setpoint for this glass type.")
    if float(features["raw_material_quality"]) < 0.84:
        alerts.append("Raw material quality is below the preferred threshold for stable production.")
    if float(features["pressure_bar"]) > 4.0:
        alerts.append("Pressure is elevated and can increase the probability of defects.")
    if float(features["ambient_humidity_pct"]) > 70 and glass_type == "Laminated":
        alerts.append("High ambient humidity is a known risk factor for laminated glass quality.")
    if str(features["operator_shift"]) in {"Night", "Weekend"}:
        alerts.append("The selected shift historically shows higher quality volatility in this dataset.")
    if float(features["operator_experience_years"]) < 3:
        alerts.append("Low operator experience increases process supervision requirements.")

    return alerts


def build_recommendations(
    features: dict[str, float | str],
    alerts: list[str],
    risk_level: str,
) -> list[str]:
    recommendations: list[str] = []

    if float(features["raw_material_quality"]) < 0.84:
        recommendations.append("Tighten incoming raw material inspection before releasing the batch.")
    if float(features["ambient_humidity_pct"]) > 70:
        recommendations.append("Increase humidity monitoring and stabilise conditioning around the line.")
    if float(features["pressure_bar"]) > 4.0:
        recommendations.append("Review pressure calibration and reduce line pressure before the next run.")
    if abs(float(features["furnace_temperature_c"]) - _get_temperature_reference(str(features["glass_type"]))) > 25:
        recommendations.append("Re-align the furnace setpoint to the product-specific target temperature.")
    if float(features["line_speed_mps"]) > 8.5:
        recommendations.append("Reduce line speed slightly and validate the next batches for surface quality.")
    if float(features["operator_experience_years"]) < 3:
        recommendations.append("Assign senior operator supervision for the current shift.")
    if risk_level == "high" and not recommendations:
        recommendations.append("Perform a first-piece quality check before scaling the batch.")
    if risk_level == "medium" and not recommendations:
        recommendations.append("Track the next batches closely and keep SPC limits under review.")
    if not recommendations and not alerts:
        recommendations.append("Current operating conditions look stable for this batch.")

    return recommendations


def predict_defect_risk(
    model_bundle: ModelBundle,
    features: dict[str, float | str],
) -> dict[str, float | int | str | list[str]]:
    input_frame = pd.DataFrame([{column: features[column] for column in model_bundle.feature_columns}])
    probability = float(model_bundle.pipeline.predict_proba(input_frame)[0][1])
    prediction = int(probability >= model_bundle.decision_threshold)

    if probability >= 0.6:
        risk_level = "high"
    elif probability >= 0.35:
        risk_level = "medium"
    else:
        risk_level = "low"

    alerts = build_process_alerts(features)
    recommendations = build_recommendations(features, alerts, risk_level)

    return {
        "prediction": prediction,
        "probability": round(probability, 4),
        "risk_level": risk_level,
        "alerts": alerts,
        "recommendations": recommendations,
    }


def build_feature_importance_frame(model_bundle: ModelBundle) -> pd.DataFrame:
    preprocessor = model_bundle.pipeline.named_steps["preprocessor"]
    classifier = model_bundle.pipeline.named_steps["classifier"]
    raw_feature_names = preprocessor.get_feature_names_out()

    rows: list[dict[str, float | str]] = []
    for feature_name, importance in zip(raw_feature_names, classifier.feature_importances_):
        cleaned_name = feature_name.replace("num__", "").replace("cat__", "")
        parent_feature = cleaned_name
        for categorical_feature in sorted(CATEGORICAL_FEATURES, key=len, reverse=True):
            prefix = f"{categorical_feature}_"
            if cleaned_name.startswith(prefix):
                parent_feature = categorical_feature
                break

        rows.append(
            {
                "feature": parent_feature,
                "feature_variant": cleaned_name,
                "importance": float(importance),
            }
        )

    aggregated = (
        pd.DataFrame(rows)
        .groupby("feature", as_index=False)["importance"]
        .sum()
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    aggregated["importance"] = aggregated["importance"].round(4)
    return aggregated
