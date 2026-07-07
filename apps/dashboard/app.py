from __future__ import annotations

import sys
from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from glass_ai.analytics import (
    ANALYSIS_DEFINITIONS,
    ANALYSIS_LABEL_TO_SLUG,
    get_executive_insights,
    get_glass_type_performance_frame,
    get_machine_risk_frame,
    get_monthly_kpi_frame,
    get_plant_benchmark_frame,
    get_shift_breakdown_frame,
    get_summary_metrics,
    get_top_delayed_batches_frame,
    prepare_production_dataframe,
    run_named_analysis,
)
from glass_ai.chatbot import SUPPORTED_QUESTION_EXAMPLES, get_chatbot_response
from glass_ai.config import settings
from glass_ai.data import ensure_database, load_production_data
from glass_ai.ml import (
    build_feature_importance_frame,
    predict_defect_risk,
    train_defect_model,
)


SCENARIO_PRESETS: dict[str, dict[str, str | float]] = {
    "Stable line": {
        "plant_id": "Plant_A",
        "machine_id": "M2",
        "glass_type": "Tempered",
        "thickness_mm": 6.0,
        "furnace_temperature_c": 1522.0,
        "pressure_bar": 3.3,
        "line_speed_mps": 6.0,
        "cooling_time_sec": 74.0,
        "raw_material_quality": 0.90,
        "operator_shift": "Day",
        "operator_experience_years": 7.0,
        "ambient_humidity_pct": 54.0,
        "furnace_zone": "Zone_2",
    },
    "High-risk night shift": {
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
        "furnace_zone": "Zone_3",
    },
    "Humidity stress case": {
        "plant_id": "Plant_B",
        "machine_id": "M3",
        "glass_type": "Solar",
        "thickness_mm": 4.0,
        "furnace_temperature_c": 1541.0,
        "pressure_bar": 3.8,
        "line_speed_mps": 7.0,
        "cooling_time_sec": 61.0,
        "raw_material_quality": 0.83,
        "operator_shift": "Weekend",
        "operator_experience_years": 3.0,
        "ambient_humidity_pct": 78.0,
        "furnace_zone": "Zone_3",
    },
}


def apply_dashboard_theme() -> None:
    st.markdown(
        """
        <style>
            .stApp {
                background: #f7fafc;
                color: #102a43;
            }
            .block-container {
                max-width: 1280px;
                padding-top: 1.4rem;
                padding-bottom: 2rem;
            }
            [data-testid="stSidebar"] {
                background: #12304a;
            }
            [data-testid="stSidebar"] * {
                color: #f8fbff;
            }
            div[data-testid="stMetric"] {
                background: white;
                border: 1px solid #d9e2ec;
                border-radius: 14px;
                padding: 0.7rem 0.8rem;
            }
            div[data-testid="stMetricLabel"] {
                color: #486581 !important;
                font-weight: 600;
            }
            div[data-testid="stMetricValue"],
            div[data-testid="stMetricValue"] > div {
                color: #102a43 !important;
            }
            [data-testid="stSidebar"] div[data-testid="stMetric"] {
                background: #ffffff;
                border: 1px solid #d9e2ec;
                box-shadow: none;
            }
            [data-testid="stSidebar"] div[data-testid="stMetric"] * {
                color: #102a43 !important;
            }
            [data-testid="stSidebar"] div[data-testid="stMetricLabel"] {
                color: #486581 !important;
            }
            [data-testid="stSidebar"] div[data-testid="stMetricValue"],
            [data-testid="stSidebar"] div[data-testid="stMetricValue"] > div,
            [data-testid="stSidebar"] div[data-testid="stMetricValue"] p {
                color: #102a43 !important;
                font-weight: 700;
            }
            .helper-text {
                background: #eaf4fb;
                border: 1px solid #bfd7ea;
                border-radius: 12px;
                padding: 0.85rem 1rem;
                color: #102a43;
                line-height: 1.5;
            }
            .stButton > button,
            div[data-testid="stFormSubmitButton"] > button {
                border-radius: 12px;
                border: 1px solid #bfd7ea;
                background: #ffffff;
                color: #102a43 !important;
                font-weight: 600;
                min-height: 2.9rem;
                box-shadow: none;
            }
            .stButton > button:hover,
            div[data-testid="stFormSubmitButton"] > button:hover {
                border-color: #9fb3c8;
                background: #f7fafc;
                color: #102a43 !important;
            }
            .stButton > button[kind="primary"],
            div[data-testid="stFormSubmitButton"] > button[kind="primaryFormSubmit"],
            div[data-testid="stFormSubmitButton"] > button[kind="secondaryFormSubmit"] {
                background: linear-gradient(135deg, #1f77b4 0%, #0f609b 100%);
                border-color: #0f609b;
                color: #ffffff !important;
                box-shadow: 0 10px 20px rgba(31, 119, 180, 0.18);
            }
            .stButton > button[kind="primary"]:hover,
            div[data-testid="stFormSubmitButton"] > button[kind="primaryFormSubmit"]:hover,
            div[data-testid="stFormSubmitButton"] > button[kind="secondaryFormSubmit"]:hover {
                background: linear-gradient(135deg, #0f6ea8 0%, #0a4e7f 100%);
                border-color: #0a4e7f;
                color: #ffffff !important;
            }
            label,
            .stMarkdown,
            .stCaption,
            [data-testid="stWidgetLabel"],
            [data-testid="stWidgetLabel"] * {
                color: #102a43 !important;
            }
            div[data-baseweb="input"] > div,
            div[data-baseweb="select"] > div,
            textarea {
                background: #ffffff !important;
                border-color: #bfd7ea !important;
                color: #102a43 !important;
            }
            div[data-baseweb="input"] input,
            div[data-baseweb="select"] input,
            div[data-baseweb="select"] span,
            textarea {
                color: #102a43 !important;
            }
            div[data-baseweb="tag"] {
                background: #eaf4fb !important;
                color: #102a43 !important;
                border-radius: 999px !important;
            }
            div[data-baseweb="slider"] [role="slider"] {
                background: #1f77b4 !important;
                box-shadow: 0 0 0 2px #ffffff !important;
            }
            [data-testid="stAlert"] {
                border-radius: 12px;
            }
            [data-testid="stAlert"] * {
                color: #102a43 !important;
            }
            .table-shell {
                background: #ffffff;
                border: 1px solid #d9e2ec;
                border-radius: 12px;
                overflow: auto;
                max-width: 100%;
            }
            .light-table {
                width: 100%;
                border-collapse: separate;
                border-spacing: 0;
                min-width: 680px;
            }
            .light-table thead th {
                position: sticky;
                top: 0;
                z-index: 1;
                background: #f5f7fa;
                color: #243b53;
                font-size: 0.9rem;
                font-weight: 700;
                letter-spacing: 0.01em;
            }
            .light-table th,
            .light-table td {
                padding: 0.75rem 0.8rem;
                border-bottom: 1px solid #e5edf5;
                text-align: left;
                white-space: nowrap;
                color: #102a43;
            }
            .light-table tbody tr:nth-child(even) {
                background: #fbfdff;
            }
            .light-table tbody tr:hover {
                background: #eef6fb;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def format_display_frame(dataframe: pd.DataFrame) -> pd.DataFrame:
    formatted = dataframe.copy()
    numeric_columns = formatted.select_dtypes(include="number").columns
    formatted.loc[:, numeric_columns] = formatted.loc[:, numeric_columns].round(2)
    return formatted


def render_light_table(
    dataframe: pd.DataFrame,
    *,
    max_rows: int | None = None,
    height_px: int = 320,
) -> None:
    if dataframe.empty:
        st.info("No rows to display for the current selection.")
        return

    display_frame = dataframe.head(max_rows).copy() if max_rows is not None else dataframe.copy()
    display_frame = format_display_frame(display_frame)

    for column in display_frame.columns:
        if pd.api.types.is_datetime64_any_dtype(display_frame[column]):
            display_frame[column] = display_frame[column].dt.strftime("%Y-%m-%d %H:%M")

    display_frame = display_frame.where(pd.notna(display_frame), "")
    if max_rows is not None and len(dataframe) > len(display_frame):
        st.caption(f"Showing first {len(display_frame):,} of {len(dataframe):,} rows.")

    table_html = display_frame.to_html(index=False, classes="light-table", border=0, escape=True)
    st.markdown(
        f'<div class="table-shell" style="max-height: {height_px}px;">{table_html}</div>',
        unsafe_allow_html=True,
    )


@st.cache_data
def load_data_cached() -> pd.DataFrame:
    ensure_database()
    return prepare_production_dataframe(load_production_data(parse_dates=True))


@st.cache_data
def run_analysis_cached(analysis_slug: str) -> pd.DataFrame:
    return run_named_analysis(analysis_slug)


@st.cache_resource
def get_model_bundle():
    return train_defect_model(load_data_cached())


def build_sidebar(dataframe: pd.DataFrame) -> tuple[str, pd.DataFrame, dict[str, str]]:
    st.sidebar.title("Factory Dashboard")
    st.sidebar.caption("Simple navigation and high-contrast filters.")

    page = st.sidebar.radio(
        "Go to",
        ["Overview", "Prediction", "Explorer"],
    )

    plants = sorted(dataframe["plant_id"].unique().tolist())
    shifts = sorted(dataframe["operator_shift"].unique().tolist())
    glass_types = sorted(dataframe["glass_type"].unique().tolist())

    selected_plants = st.sidebar.multiselect("Plants", plants, default=plants)
    selected_shifts = st.sidebar.multiselect("Shifts", shifts, default=shifts)
    selected_glass_types = st.sidebar.multiselect("Glass types", glass_types, default=glass_types)

    min_date = dataframe["production_date"].min().date()
    max_date = dataframe["production_date"].max().date()
    date_range = st.sidebar.date_input(
        "Production window",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    if isinstance(date_range, tuple) and len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date = min_date
        end_date = max_date

    filtered_dataframe = dataframe[
        dataframe["plant_id"].isin(selected_plants)
        & dataframe["operator_shift"].isin(selected_shifts)
        & dataframe["glass_type"].isin(selected_glass_types)
        & dataframe["production_date"].dt.date.between(start_date, end_date)
    ].copy()

    st.sidebar.markdown("---")
    st.sidebar.metric("Visible batches", f"{len(filtered_dataframe):,}")
    st.sidebar.metric("Plants selected", len(selected_plants))
    st.sidebar.metric("Glass families", len(selected_glass_types))

    scope = {
        "date_range": f"{start_date.isoformat()} to {end_date.isoformat()}",
        "plants": f"{len(selected_plants)} plants",
        "shifts": f"{len(selected_shifts)} shifts",
        "glass_types": f"{len(selected_glass_types)} glass families",
    }
    return page, filtered_dataframe, scope


def render_header(summary_metrics: dict[str, float], scope: dict[str, str], row_count: int) -> None:
    st.title("Glass Manufacturing Dashboard")
    st.caption(
        "A clean view of factory performance, defect risk, and production analytics."
    )
    st.markdown(
        f"""
        <div class="helper-text">
            <strong>Current scope:</strong> {scope['date_range']} | {scope['plants']} |
            {scope['shifts']} | {scope['glass_types']} | {row_count:,} visible batches<br/>
            <strong>Headline:</strong> {summary_metrics['defect_rate_pct']:.2f}% defect rate,
            {summary_metrics['average_yield_rate_pct']:.2f}% yield,
            EUR {summary_metrics['total_scrap_cost_eur']:,.0f} estimated scrap cost.
        </div>
        """,
        unsafe_allow_html=True,
    )


def build_monthly_units_chart(monthly_kpis: pd.DataFrame) -> alt.Chart:
    month_order = monthly_kpis["production_month"].tolist()
    return (
        alt.Chart(monthly_kpis)
        .mark_line(point=True, strokeWidth=3, color="#1f77b4")
        .encode(
            x=alt.X("production_month:N", sort=month_order, title="Month"),
            y=alt.Y("total_units:Q", title="Total units"),
            tooltip=[
                alt.Tooltip("production_month:N", title="Month"),
                alt.Tooltip("total_units:Q", title="Units", format=","),
            ],
        )
        .properties(height=280)
        .configure_view(strokeOpacity=0)
    )


def build_monthly_defect_chart(monthly_kpis: pd.DataFrame) -> alt.Chart:
    month_order = monthly_kpis["production_month"].tolist()
    return (
        alt.Chart(monthly_kpis)
        .mark_bar(color="#d97706", cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
        .encode(
            x=alt.X("production_month:N", sort=month_order, title="Month"),
            y=alt.Y("defect_rate_pct:Q", title="Defect rate (%)"),
            tooltip=[
                alt.Tooltip("production_month:N", title="Month"),
                alt.Tooltip("defect_rate_pct:Q", title="Defect rate", format=".2f"),
                alt.Tooltip("scrap_cost_eur:Q", title="Scrap cost", format=",.0f"),
            ],
        )
        .properties(height=280)
        .configure_view(strokeOpacity=0)
    )


def build_analysis_chart(result: pd.DataFrame) -> alt.Chart | None:
    if result.empty or len(result.columns) < 2:
        return None

    first_column = result.columns[0]
    numeric_columns = [column for column in result.columns[1:] if pd.api.types.is_numeric_dtype(result[column])]
    if not numeric_columns:
        return None

    metric_name = numeric_columns[0]
    return (
        alt.Chart(result)
        .mark_bar(color="#1f77b4", cornerRadiusTopLeft=5, cornerRadiusTopRight=5)
        .encode(
            x=alt.X(f"{first_column}:N", title=None),
            y=alt.Y(f"{metric_name}:Q", title=metric_name.replace("_", " ").title()),
            tooltip=[first_column, alt.Tooltip(f"{metric_name}:Q", format=",.2f")],
        )
        .properties(height=300)
        .configure_view(strokeOpacity=0)
    )


def build_feature_chart(importance_frame: pd.DataFrame) -> alt.Chart:
    return (
        alt.Chart(importance_frame)
        .mark_bar(color="#1f77b4", cornerRadiusEnd=6)
        .encode(
            y=alt.Y("feature:N", sort="-x", title=None),
            x=alt.X("importance:Q", title="Importance"),
            tooltip=[
                alt.Tooltip("feature:N", title="Feature"),
                alt.Tooltip("importance:Q", title="Importance", format=".4f"),
            ],
        )
        .properties(height=280)
        .configure_view(strokeOpacity=0)
    )


def build_prediction_inputs(dataframe: pd.DataFrame, preset_name: str) -> dict[str, float | str]:
    preset = SCENARIO_PRESETS[preset_name]

    plant_options = sorted(dataframe["plant_id"].unique())
    machine_options = sorted(dataframe["machine_id"].unique())
    glass_options = sorted(dataframe["glass_type"].unique())
    shift_options = sorted(dataframe["operator_shift"].unique())
    zone_options = sorted(dataframe["furnace_zone"].unique())

    col1, col2 = st.columns(2, gap="medium")

    with col1:
        plant_id = st.selectbox("Plant", plant_options, index=plant_options.index(str(preset["plant_id"])))
        machine_id = st.selectbox("Machine", machine_options, index=machine_options.index(str(preset["machine_id"])))
        glass_type = st.selectbox("Glass type", glass_options, index=glass_options.index(str(preset["glass_type"])))
        thickness_mm = st.number_input("Thickness (mm)", min_value=1.0, value=float(preset["thickness_mm"]))
        furnace_temperature_c = st.number_input("Furnace temperature (C)", value=float(preset["furnace_temperature_c"]))
        pressure_bar = st.number_input("Pressure (bar)", value=float(preset["pressure_bar"]))
        raw_material_quality = st.slider(
            "Raw material quality",
            min_value=0.0,
            max_value=1.0,
            value=float(preset["raw_material_quality"]),
        )

    with col2:
        operator_shift = st.selectbox("Operator shift", shift_options, index=shift_options.index(str(preset["operator_shift"])))
        operator_experience_years = st.number_input(
            "Operator experience (years)",
            min_value=0.0,
            value=float(preset["operator_experience_years"]),
        )
        ambient_humidity_pct = st.number_input(
            "Ambient humidity (%)",
            min_value=0.0,
            max_value=100.0,
            value=float(preset["ambient_humidity_pct"]),
        )
        furnace_zone = st.selectbox("Furnace zone", zone_options, index=zone_options.index(str(preset["furnace_zone"])))
        line_speed_mps = st.number_input("Line speed (m/s)", value=float(preset["line_speed_mps"]))
        cooling_time_sec = st.number_input("Cooling time (sec)", value=float(preset["cooling_time_sec"]))

    return {
        "plant_id": plant_id,
        "machine_id": machine_id,
        "glass_type": glass_type,
        "thickness_mm": thickness_mm,
        "furnace_temperature_c": furnace_temperature_c,
        "pressure_bar": pressure_bar,
        "line_speed_mps": line_speed_mps,
        "cooling_time_sec": cooling_time_sec,
        "raw_material_quality": raw_material_quality,
        "operator_shift": operator_shift,
        "operator_experience_years": operator_experience_years,
        "ambient_humidity_pct": ambient_humidity_pct,
        "furnace_zone": furnace_zone,
    }


def render_overview_page(
    filtered_dataframe: pd.DataFrame,
    summary_metrics: dict[str, float],
    monthly_kpis: pd.DataFrame,
    plant_benchmark: pd.DataFrame,
    shift_breakdown: pd.DataFrame,
    glass_type_performance: pd.DataFrame,
    delayed_batches: pd.DataFrame,
    executive_insights: list[str],
) -> None:
    st.subheader("Overview")
    st.write("Start here for the quickest summary of factory performance.")

    metrics = st.columns(4)
    metrics[0].metric("Batches", f"{summary_metrics['total_batches']:,}")
    metrics[1].metric("Defect rate", f"{summary_metrics['defect_rate_pct']:.2f}%")
    metrics[2].metric("Yield", f"{summary_metrics['average_yield_rate_pct']:.2f}%")
    metrics[3].metric("On-time delivery", f"{summary_metrics['on_time_delivery_rate_pct']:.2f}%")

    metrics_2 = st.columns(3)
    metrics_2[0].metric("Average batch output", f"{summary_metrics['average_produced_units']:.0f} units")
    metrics_2[1].metric("Scrap cost", f"EUR {summary_metrics['total_scrap_cost_eur']:,.0f}")
    metrics_2[2].metric("Energy intensity", f"{summary_metrics['energy_intensity_kwh_per_unit']:.3f} kWh/unit")

    chart_left, chart_right = st.columns(2, gap="large")
    with chart_left:
        with st.container(border=True):
            st.markdown("**Monthly throughput**")
            st.altair_chart(build_monthly_units_chart(monthly_kpis), width="stretch")

    with chart_right:
        with st.container(border=True):
            st.markdown("**Monthly defect rate**")
            st.altair_chart(build_monthly_defect_chart(monthly_kpis), width="stretch")

    insight_left, insight_right = st.columns([1, 1], gap="large")
    with insight_left:
        with st.container(border=True):
            st.markdown("**Key insights**")
            for insight in executive_insights:
                st.markdown(f"- {insight}")

    with insight_right:
        with st.container(border=True):
            st.markdown("**Most delayed batches**")
            render_light_table(delayed_batches, max_rows=10, height_px=300)

    compare_left, compare_mid, compare_right = st.columns(3, gap="large")
    with compare_left:
        with st.container(border=True):
            st.markdown("**Plant benchmark**")
            render_light_table(plant_benchmark, height_px=260)
    with compare_mid:
        with st.container(border=True):
            st.markdown("**Shift performance**")
            render_light_table(shift_breakdown, height_px=260)
    with compare_right:
        with st.container(border=True):
            st.markdown("**Glass type cost risk**")
            render_light_table(glass_type_performance, height_px=260)

    st.markdown("**Filtered dataset preview**")
    if st.checkbox("Show filtered dataset preview", key="overview_preview_toggle"):
        render_light_table(filtered_dataframe, max_rows=25, height_px=420)


def render_prediction_page(dataframe: pd.DataFrame, model_bundle) -> None:
    st.subheader("Prediction")
    st.write("Use this page to simulate one batch and estimate defect risk.")

    metrics = st.columns(4)
    metrics[0].metric("Recall", f"{model_bundle.metrics['recall'] * 100:.2f}%")
    metrics[1].metric("Precision", f"{model_bundle.metrics['precision'] * 100:.2f}%")
    metrics[2].metric("ROC-AUC", f"{model_bundle.metrics['roc_auc'] * 100:.2f}%")
    metrics[3].metric("Threshold", f"{model_bundle.metrics['decision_threshold']:.2f}")

    st.markdown(
        """
        <div class="helper-text">
            Choose a preset, adjust the process settings, and run the prediction.
            The result explains the likely risk level and what the team should inspect next.
        </div>
        """,
        unsafe_allow_html=True,
    )

    with st.form("prediction_form"):
        preset_name = st.selectbox("Scenario preset", list(SCENARIO_PRESETS))
        features = build_prediction_inputs(dataframe, preset_name)
        submitted = st.form_submit_button("Estimate defect risk", type="primary", width="stretch")

    if submitted:
        st.session_state["prediction_result"] = predict_defect_risk(model_bundle, features)

    result = st.session_state.get("prediction_result")
    if result is None:
        st.info("Run a prediction to see the output here.")
    else:
        probability = float(result["probability"]) * 100
        risk_level = str(result["risk_level"])

        if risk_level == "high":
            st.error(f"High risk: {probability:.2f}% estimated defect probability.")
        elif risk_level == "medium":
            st.warning(f"Medium risk: {probability:.2f}% estimated defect probability.")
        else:
            st.success(f"Low risk: {probability:.2f}% estimated defect probability.")

        st.markdown("**Why the model is concerned**")
        alerts = result.get("alerts", [])
        if alerts:
            for alert in alerts:
                st.markdown(f"- {alert}")
        else:
            st.write("No critical alerts were generated.")

        st.markdown("**Recommended actions**")
        for recommendation in result.get("recommendations", []):
            st.markdown(f"- {recommendation}")

    st.markdown("**Model diagnostics**")
    if st.checkbox("Show feature importance", key="prediction_feature_importance_toggle"):
        importance_frame = build_feature_importance_frame(model_bundle)
        with st.container(border=True):
            st.altair_chart(build_feature_chart(importance_frame), width="stretch")
            render_light_table(importance_frame, height_px=320)

    if st.checkbox("Show confusion matrix", key="prediction_confusion_toggle"):
        confusion_frame = pd.DataFrame(
            model_bundle.confusion_matrix,
            index=["Actual normal", "Actual risk"],
            columns=["Predicted normal", "Predicted risk"],
        )
        render_light_table(confusion_frame.rename_axis("actual_label").reset_index(), height_px=200)


def render_explorer_page(filtered_dataframe: pd.DataFrame) -> None:
    st.subheader("Explorer")
    st.write("Use this page for one analysis at a time and for quick example questions.")

    analysis_labels = [details["label"] for details in ANALYSIS_DEFINITIONS.values()]
    selected_label = st.selectbox("Choose an analysis", analysis_labels, key="explorer_analysis")
    analysis_slug = ANALYSIS_LABEL_TO_SLUG[selected_label]
    st.caption(ANALYSIS_DEFINITIONS[analysis_slug]["description"])

    analysis_result = run_analysis_cached(analysis_slug)
    analysis_chart = build_analysis_chart(analysis_result)
    if analysis_chart is not None:
        st.altair_chart(analysis_chart, width="stretch")

    render_light_table(analysis_result, height_px=360)

    st.markdown("---")
    question_col, data_col = st.columns([0.9, 1.1], gap="large")

    with question_col:
        st.markdown("**Example questions**")
        example_question = st.selectbox("Examples", SUPPORTED_QUESTION_EXAMPLES, key="explorer_example")
        if st.button("Load example", width="stretch"):
            st.session_state["explorer_question"] = example_question

        question = st.text_input(
            "Question",
            key="explorer_question",
            placeholder="Ask about shifts, plants, defects, or delayed batches...",
        )

        if st.button("Run question", type="primary", width="stretch"):
            st.session_state["explorer_result"] = get_chatbot_response(question)

        result = st.session_state.get("explorer_result")
        if result is not None:
            render_light_table(result, height_px=320)

    with data_col:
        st.markdown("**Filtered dataset preview**")
        render_light_table(filtered_dataframe, max_rows=25, height_px=420)


def render_dashboard() -> None:
    st.set_page_config(
        page_title="Glass Manufacturing Dashboard",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    apply_dashboard_theme()

    dataframe = load_data_cached()
    page, filtered_dataframe, scope = build_sidebar(dataframe)

    if filtered_dataframe.empty:
        st.warning("No batches match the current filters. Please widen the selection in the sidebar.")
        return

    summary_metrics = get_summary_metrics(filtered_dataframe)
    render_header(summary_metrics, scope, len(filtered_dataframe))

    if page == "Overview":
        render_overview_page(
            filtered_dataframe=filtered_dataframe,
            summary_metrics=summary_metrics,
            monthly_kpis=get_monthly_kpi_frame(filtered_dataframe),
            plant_benchmark=get_plant_benchmark_frame(filtered_dataframe),
            shift_breakdown=get_shift_breakdown_frame(filtered_dataframe),
            glass_type_performance=get_glass_type_performance_frame(filtered_dataframe),
            delayed_batches=get_top_delayed_batches_frame(filtered_dataframe, limit=10),
            executive_insights=get_executive_insights(filtered_dataframe),
        )
        return

    model_bundle = get_model_bundle()

    if page == "Prediction":
        render_prediction_page(dataframe, model_bundle)
        return

    render_explorer_page(filtered_dataframe)


if __name__ == "__main__":
    render_dashboard()
