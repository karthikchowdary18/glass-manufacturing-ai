from __future__ import annotations

import pandas as pd

from glass_ai.analytics import run_named_analysis
from glass_ai.data import run_query


SUPPORTED_QUESTION_EXAMPLES = [
    "Which machine is the biggest defect hotspot?",
    "Show shift quality performance",
    "Which plant has the best throughput and yield?",
    "Show monthly quality trend",
    "Which glass type drives the highest scrap cost?",
    "Show delayed batches",
]


def get_chatbot_response(question: str) -> pd.DataFrame | None:
    normalized_question = question.lower().strip()

    if "highest defects" in normalized_question or "defect hotspot" in normalized_question:
        return run_named_analysis("machine-defect-hotspots")

    if "shift quality" in normalized_question or "defect rate by shift" in normalized_question:
        return run_named_analysis("shift-quality-performance")

    if "throughput and yield" in normalized_question or "best plant" in normalized_question:
        return run_named_analysis("plant-throughput-and-yield")

    if "monthly quality" in normalized_question or "trend" in normalized_question:
        return run_named_analysis("monthly-quality-trend")

    if "glass type" in normalized_question and "scrap" in normalized_question:
        return run_named_analysis("glass-type-cost-risk")

    if "energy efficiency" in normalized_question or "energy by plant" in normalized_question:
        return run_named_analysis("energy-efficiency-by-plant")

    if "show delayed batches" in normalized_question or "delayed batches" in normalized_question:
        return run_query(
            """
            SELECT
                batch_id,
                plant_id,
                machine_id,
                glass_type,
                ROUND(packing_delay_min, 2) AS packing_delay_min,
                ROUND(downtime_min, 2) AS downtime_min,
                ROUND(estimated_scrap_cost_eur, 2) AS estimated_scrap_cost_eur
            FROM production
            WHERE shipment_status = 'Delayed'
            ORDER BY packing_delay_min DESC, estimated_scrap_cost_eur DESC
            LIMIT 10
            """
        )

    return None
