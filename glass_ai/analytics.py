from __future__ import annotations

import pandas as pd

from glass_ai.data import ensure_database, load_production_data, run_query


ANALYSIS_DEFINITIONS: dict[str, dict[str, str]] = {
    "plant-throughput-and-yield": {
        "label": "Plant throughput and yield",
        "description": "Compare total throughput, yield, and scrap cost across plants.",
        "query": """
            SELECT
                plant_id,
                SUM(produced_units) AS total_units,
                ROUND((1 - (SUM(scrap_units) * 1.0 / SUM(produced_units))) * 100, 2) AS yield_rate_pct,
                ROUND(SUM(estimated_scrap_cost_eur), 2) AS scrap_cost_eur
            FROM production
            GROUP BY plant_id
            ORDER BY total_units DESC
        """,
    },
    "machine-defect-hotspots": {
        "label": "Machine defect hotspots",
        "description": "Identify machines with the highest defect pressure and downtime.",
        "query": """
            SELECT
                machine_id,
                ROUND(AVG(defect_flag) * 100, 2) AS defect_rate_pct,
                ROUND(AVG(downtime_min), 2) AS avg_downtime_min,
                ROUND(SUM(estimated_scrap_cost_eur), 2) AS scrap_cost_eur
            FROM production
            GROUP BY machine_id
            ORDER BY defect_rate_pct DESC, scrap_cost_eur DESC
        """,
    },
    "shift-quality-performance": {
        "label": "Shift quality performance",
        "description": "Measure how day, night, and weekend shifts compare on quality and delivery.",
        "query": """
            SELECT
                operator_shift,
                ROUND(AVG(defect_flag) * 100, 2) AS defect_rate_pct,
                ROUND(AVG(produced_units), 2) AS avg_produced_units,
                ROUND(AVG(packing_delay_min), 2) AS avg_packing_delay_min
            FROM production
            GROUP BY operator_shift
            ORDER BY defect_rate_pct DESC
        """,
    },
    "glass-type-cost-risk": {
        "label": "Glass type cost risk",
        "description": "Show which product families create the biggest quality and cost pressure.",
        "query": """
            SELECT
                glass_type,
                ROUND(AVG(defect_flag) * 100, 2) AS defect_rate_pct,
                ROUND(AVG(estimated_scrap_cost_eur), 2) AS avg_scrap_cost_eur,
                ROUND(AVG(energy_consumption_kwh * 1.0 / produced_units), 3) AS kwh_per_unit
            FROM production
            GROUP BY glass_type
            ORDER BY avg_scrap_cost_eur DESC
        """,
    },
    "monthly-quality-trend": {
        "label": "Monthly quality trend",
        "description": "Track throughput, defects, and scrap cost over time.",
        "query": """
            SELECT
                substr(production_date, 1, 7) AS production_month,
                COUNT(*) AS batches,
                SUM(produced_units) AS total_units,
                ROUND(AVG(defect_flag) * 100, 2) AS defect_rate_pct,
                ROUND(SUM(estimated_scrap_cost_eur), 2) AS scrap_cost_eur
            FROM production
            GROUP BY substr(production_date, 1, 7)
            ORDER BY production_month
        """,
    },
    "energy-efficiency-by-plant": {
        "label": "Energy efficiency by plant",
        "description": "Benchmark energy intensity and emissions across manufacturing sites.",
        "query": """
            SELECT
                plant_id,
                ROUND(SUM(energy_consumption_kwh) * 1.0 / SUM(produced_units), 3) AS kwh_per_unit,
                ROUND(AVG(co2_emissions_kg), 2) AS avg_co2_kg_per_batch,
                ROUND(AVG(defect_flag) * 100, 2) AS defect_rate_pct
            FROM production
            GROUP BY plant_id
            ORDER BY kwh_per_unit ASC
        """,
    },
}

ANALYSIS_LABEL_TO_SLUG = {
    details["label"]: slug for slug, details in ANALYSIS_DEFINITIONS.items()
}


def _safe_percentage(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return round((numerator / denominator) * 100, 2)


def prepare_production_dataframe(dataframe: pd.DataFrame | None = None) -> pd.DataFrame:
    ensure_database()
    prepared = (
        dataframe.copy()
        if dataframe is not None
        else load_production_data(parse_dates=True)
    )

    if "production_date" in prepared.columns and not pd.api.types.is_datetime64_any_dtype(
        prepared["production_date"]
    ):
        prepared["production_date"] = pd.to_datetime(prepared["production_date"])

    produced_units = prepared["produced_units"].replace(0, pd.NA)
    prepared["yield_rate_pct"] = (
        (prepared["produced_units"] - prepared["scrap_units"]).clip(lower=0) / produced_units
    ).fillna(0) * 100
    prepared["on_time_flag"] = (prepared["shipment_status"] == "On-Time").astype(int)
    prepared["energy_intensity_kwh_per_unit"] = (
        prepared["energy_consumption_kwh"] / produced_units
    ).fillna(0)
    prepared["scrap_cost_per_unit_eur"] = (
        prepared["estimated_scrap_cost_eur"] / produced_units
    ).fillna(0)
    prepared["production_month"] = prepared["production_date"].dt.to_period("M").astype(str)

    return prepared


def list_analyses() -> list[dict[str, str]]:
    return [
        {
            "slug": slug,
            "label": details["label"],
            "description": details["description"],
        }
        for slug, details in ANALYSIS_DEFINITIONS.items()
    ]


def run_named_analysis(analysis_slug: str) -> pd.DataFrame:
    if analysis_slug not in ANALYSIS_DEFINITIONS:
        raise KeyError(f"Unknown analysis '{analysis_slug}'")

    ensure_database()
    return run_query(ANALYSIS_DEFINITIONS[analysis_slug]["query"])


def get_summary_metrics(dataframe: pd.DataFrame | None = None) -> dict[str, float]:
    prepared = prepare_production_dataframe(dataframe)

    total_units = float(prepared["produced_units"].sum())
    total_scrap_units = float(prepared["scrap_units"].sum())

    return {
        "total_batches": int(len(prepared)),
        "total_defects": int(prepared["defect_flag"].sum()),
        "average_produced_units": round(float(prepared["produced_units"].mean()), 2),
        "defect_rate_pct": round(float(prepared["defect_flag"].mean() * 100), 2),
        "on_time_delivery_rate_pct": round(float(prepared["on_time_flag"].mean() * 100), 2),
        "average_yield_rate_pct": round(float(prepared["yield_rate_pct"].mean()), 2),
        "total_scrap_cost_eur": round(float(prepared["estimated_scrap_cost_eur"].sum()), 2),
        "energy_intensity_kwh_per_unit": round(
            float(prepared["energy_consumption_kwh"].sum() / total_units),
            3,
        ),
        "scrap_share_pct": _safe_percentage(total_scrap_units, total_units),
    }


def get_monthly_kpi_frame(dataframe: pd.DataFrame | None = None) -> pd.DataFrame:
    prepared = prepare_production_dataframe(dataframe)
    result = (
        prepared.groupby("production_month", as_index=False)
        .agg(
            total_units=("produced_units", "sum"),
            defect_rate_pct=("defect_flag", lambda values: round(values.mean() * 100, 2)),
            on_time_rate_pct=("on_time_flag", lambda values: round(values.mean() * 100, 2)),
            scrap_cost_eur=("estimated_scrap_cost_eur", "sum"),
        )
        .sort_values("production_month")
    )
    result["scrap_cost_eur"] = result["scrap_cost_eur"].round(2)
    return result


def get_plant_benchmark_frame(dataframe: pd.DataFrame | None = None) -> pd.DataFrame:
    prepared = prepare_production_dataframe(dataframe)
    result = (
        prepared.groupby("plant_id", as_index=False)
        .agg(
            total_units=("produced_units", "sum"),
            defect_rate_pct=("defect_flag", lambda values: round(values.mean() * 100, 2)),
            yield_rate_pct=("yield_rate_pct", "mean"),
            on_time_rate_pct=("on_time_flag", lambda values: round(values.mean() * 100, 2)),
            total_scrap_cost_eur=("estimated_scrap_cost_eur", "sum"),
            energy_intensity_kwh_per_unit=("energy_intensity_kwh_per_unit", "mean"),
        )
        .sort_values(["yield_rate_pct", "total_units"], ascending=[False, False])
        .reset_index(drop=True)
    )
    result["yield_rate_pct"] = result["yield_rate_pct"].round(2)
    result["total_scrap_cost_eur"] = result["total_scrap_cost_eur"].round(2)
    result["energy_intensity_kwh_per_unit"] = result["energy_intensity_kwh_per_unit"].round(3)
    return result


def get_machine_risk_frame(dataframe: pd.DataFrame | None = None) -> pd.DataFrame:
    prepared = prepare_production_dataframe(dataframe)
    result = (
        prepared.groupby("machine_id", as_index=False)
        .agg(
            defect_rate_pct=("defect_flag", lambda values: round(values.mean() * 100, 2)),
            avg_downtime_min=("downtime_min", "mean"),
            avg_packing_delay_min=("packing_delay_min", "mean"),
            total_scrap_cost_eur=("estimated_scrap_cost_eur", "sum"),
        )
        .sort_values(["defect_rate_pct", "total_scrap_cost_eur"], ascending=[False, False])
        .reset_index(drop=True)
    )
    result["avg_downtime_min"] = result["avg_downtime_min"].round(2)
    result["avg_packing_delay_min"] = result["avg_packing_delay_min"].round(2)
    result["total_scrap_cost_eur"] = result["total_scrap_cost_eur"].round(2)
    return result


def get_shift_breakdown_frame(dataframe: pd.DataFrame | None = None) -> pd.DataFrame:
    prepared = prepare_production_dataframe(dataframe)
    result = (
        prepared.groupby("operator_shift", as_index=False)
        .agg(
            defect_rate_pct=("defect_flag", lambda values: round(values.mean() * 100, 2)),
            avg_output=("produced_units", "mean"),
            avg_packing_delay_min=("packing_delay_min", "mean"),
            on_time_rate_pct=("on_time_flag", lambda values: round(values.mean() * 100, 2)),
        )
        .sort_values("defect_rate_pct", ascending=False)
        .reset_index(drop=True)
    )
    result["avg_output"] = result["avg_output"].round(2)
    result["avg_packing_delay_min"] = result["avg_packing_delay_min"].round(2)
    return result


def get_glass_type_performance_frame(dataframe: pd.DataFrame | None = None) -> pd.DataFrame:
    prepared = prepare_production_dataframe(dataframe)
    result = (
        prepared.groupby("glass_type", as_index=False)
        .agg(
            defect_rate_pct=("defect_flag", lambda values: round(values.mean() * 100, 2)),
            avg_scrap_cost_eur=("estimated_scrap_cost_eur", "mean"),
            energy_intensity_kwh_per_unit=("energy_intensity_kwh_per_unit", "mean"),
            avg_yield_rate_pct=("yield_rate_pct", "mean"),
        )
        .sort_values(["avg_scrap_cost_eur", "defect_rate_pct"], ascending=[False, False])
        .reset_index(drop=True)
    )
    result["avg_scrap_cost_eur"] = result["avg_scrap_cost_eur"].round(2)
    result["energy_intensity_kwh_per_unit"] = result["energy_intensity_kwh_per_unit"].round(3)
    result["avg_yield_rate_pct"] = result["avg_yield_rate_pct"].round(2)
    return result


def get_top_delayed_batches_frame(
    dataframe: pd.DataFrame | None = None,
    *,
    limit: int = 10,
) -> pd.DataFrame:
    prepared = prepare_production_dataframe(dataframe)
    columns = [
        "batch_id",
        "production_date",
        "plant_id",
        "machine_id",
        "glass_type",
        "packing_delay_min",
        "downtime_min",
        "estimated_scrap_cost_eur",
        "shipment_status",
    ]
    result = prepared.sort_values(
        ["packing_delay_min", "estimated_scrap_cost_eur"],
        ascending=[False, False],
    ).head(limit)
    return result.loc[:, columns].reset_index(drop=True)


def get_executive_insights(dataframe: pd.DataFrame | None = None) -> list[str]:
    prepared = prepare_production_dataframe(dataframe)

    if prepared.empty:
        return ["No data available for the current selection."]

    plant_benchmark = get_plant_benchmark_frame(prepared)
    shift_breakdown = get_shift_breakdown_frame(prepared)
    glass_type_frame = get_glass_type_performance_frame(prepared)
    monthly_kpis = get_monthly_kpi_frame(prepared)

    best_plant = plant_benchmark.iloc[0]
    highest_risk_shift = shift_breakdown.iloc[0]
    highest_cost_glass = glass_type_frame.iloc[0]
    latest_month = monthly_kpis.iloc[-1]

    return [
        (
            f"{best_plant['plant_id']} leads the network with "
            f"{best_plant['yield_rate_pct']:.2f}% average yield."
        ),
        (
            f"{highest_risk_shift['operator_shift']} shift is the main quality hotspot "
            f"at {highest_risk_shift['defect_rate_pct']:.2f}% defect rate."
        ),
        (
            f"{highest_cost_glass['glass_type']} glass drives the highest scrap burden "
            f"at EUR {highest_cost_glass['avg_scrap_cost_eur']:.2f} average scrap cost per batch."
        ),
        (
            f"The latest visible month ({latest_month['production_month']}) delivered "
            f"{int(latest_month['total_units'])} units with a {latest_month['defect_rate_pct']:.2f}% defect rate."
        ),
    ]
