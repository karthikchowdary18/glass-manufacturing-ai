from glass_ai.analytics import (
    get_executive_insights,
    get_monthly_kpi_frame,
    get_summary_metrics,
    prepare_production_dataframe,
)
from glass_ai.data import load_production_data


def test_summary_metrics_have_business_kpis():
    dataframe = load_production_data()
    summary = get_summary_metrics(dataframe)

    assert summary["total_batches"] > 0
    assert summary["defect_rate_pct"] > 0
    assert summary["energy_intensity_kwh_per_unit"] > 0
    assert summary["total_scrap_cost_eur"] > 0


def test_monthly_kpi_frame_is_not_empty():
    dataframe = load_production_data()
    monthly_kpis = get_monthly_kpi_frame(dataframe)

    assert not monthly_kpis.empty
    assert "production_month" in monthly_kpis.columns
    assert "defect_rate_pct" in monthly_kpis.columns


def test_executive_insights_return_storylines():
    prepared = prepare_production_dataframe(load_production_data())
    insights = get_executive_insights(prepared)

    assert len(insights) >= 3
