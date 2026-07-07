from __future__ import annotations

import math
import random
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from glass_ai.config import settings
from glass_ai.data import ensure_data_directories


PLANT_PROFILES = {
    "Plant_A": {
        "temp_offset": -8,
        "quality_bias": 0.02,
        "throughput_factor": 1.05,
        "energy_factor": 0.97,
        "delay_bias": -3,
        "defect_bias": -0.01,
    },
    "Plant_B": {
        "temp_offset": 6,
        "quality_bias": 0.0,
        "throughput_factor": 0.99,
        "energy_factor": 1.02,
        "delay_bias": 2,
        "defect_bias": 0.02,
    },
    "Plant_C": {
        "temp_offset": 12,
        "quality_bias": -0.01,
        "throughput_factor": 1.01,
        "energy_factor": 1.05,
        "delay_bias": 5,
        "defect_bias": 0.03,
    },
}

MACHINE_PROFILES = {
    "M1": {"throughput_factor": 1.08, "defect_bias": -0.02, "energy_bias": -25},
    "M2": {"throughput_factor": 1.02, "defect_bias": 0.0, "energy_bias": 0},
    "M3": {"throughput_factor": 0.98, "defect_bias": 0.02, "energy_bias": 20},
    "M4": {"throughput_factor": 0.95, "defect_bias": 0.04, "energy_bias": 30},
    "M5": {"throughput_factor": 1.03, "defect_bias": -0.01, "energy_bias": -10},
}

GLASS_PROFILES = {
    "Float": {
        "target_temp": 1485,
        "pressure_target": 2.6,
        "cooling_range": (42, 80),
        "speed_range": (6.0, 9.0),
        "thickness_range": (3.0, 8.0),
        "energy_per_unit": 0.9,
        "unit_cost": 16.0,
    },
    "Tempered": {
        "target_temp": 1525,
        "pressure_target": 3.4,
        "cooling_range": (55, 92),
        "speed_range": (4.8, 7.6),
        "thickness_range": (4.0, 12.0),
        "energy_per_unit": 1.15,
        "unit_cost": 24.0,
    },
    "Laminated": {
        "target_temp": 1505,
        "pressure_target": 3.1,
        "cooling_range": (60, 105),
        "speed_range": (4.0, 6.8),
        "thickness_range": (6.0, 14.0),
        "energy_per_unit": 1.25,
        "unit_cost": 29.0,
    },
    "Solar": {
        "target_temp": 1538,
        "pressure_target": 3.7,
        "cooling_range": (52, 88),
        "speed_range": (4.6, 7.1),
        "thickness_range": (3.0, 6.0),
        "energy_per_unit": 1.32,
        "unit_cost": 34.0,
    },
}

SHIFT_PROFILES = {
    "Day": {"defect_bias": -0.01, "delay_bias": -2},
    "Night": {"defect_bias": 0.03, "delay_bias": 3},
    "Weekend": {"defect_bias": 0.05, "delay_bias": 6},
}


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def seasonal_humidity(day_of_year: int) -> float:
    seasonal_component = 13 * math.sin((2 * math.pi * day_of_year) / 365)
    return 55 + seasonal_component


def generate_dataset(rows: int = 1800) -> pd.DataFrame:
    np.random.seed(42)
    random.seed(42)

    data: list[list[object]] = []
    start_date = datetime(2024, 1, 1)

    for index in range(rows):
        batch_id = index + 1
        production_date = start_date + timedelta(days=random.randint(0, 540))
        plant_id = random.choice(list(PLANT_PROFILES))
        machine_id = random.choice(list(MACHINE_PROFILES))
        glass_type = random.choice(list(GLASS_PROFILES))
        operator_shift = random.choices(
            population=list(SHIFT_PROFILES),
            weights=[0.52, 0.33, 0.15],
            k=1,
        )[0]

        plant_profile = PLANT_PROFILES[plant_id]
        machine_profile = MACHINE_PROFILES[machine_id]
        glass_profile = GLASS_PROFILES[glass_type]
        shift_profile = SHIFT_PROFILES[operator_shift]

        target_temp = glass_profile["target_temp"] + plant_profile["temp_offset"]
        thickness_mm = round(
            np.random.uniform(*glass_profile["thickness_range"]),
            2,
        )
        furnace_temp = round(np.random.normal(target_temp, 22), 2)
        cooling_time = round(np.random.uniform(*glass_profile["cooling_range"]), 2)
        pressure = round(np.random.normal(glass_profile["pressure_target"], 0.38), 2)
        line_speed = round(
            np.random.uniform(*glass_profile["speed_range"])
            * machine_profile["throughput_factor"],
            2,
        )

        raw_material_quality = round(
            clamp(
                np.random.normal(0.89 + plant_profile["quality_bias"], 0.05),
                0.7,
                0.99,
            ),
            2,
        )
        operator_experience_years = round(
            clamp(
                np.random.normal(
                    6.5 if operator_shift == "Day" else 4.8,
                    2.2,
                ),
                0.5,
                18.0,
            ),
            1,
        )
        ambient_humidity_pct = round(
            clamp(
                np.random.normal(
                    seasonal_humidity(production_date.timetuple().tm_yday),
                    6.0,
                ),
                30.0,
                85.0,
            ),
            2,
        )
        furnace_zone = random.choice(["Zone_1", "Zone_2", "Zone_3"])

        base_units = np.random.randint(420, 1100)
        produced_units = int(
            base_units
            * plant_profile["throughput_factor"]
            * machine_profile["throughput_factor"]
        )

        energy_intensity = (
            glass_profile["energy_per_unit"]
            * (1 + ((furnace_temp - glass_profile["target_temp"]) / 320))
            * plant_profile["energy_factor"]
        )
        energy = round(
            max(
                300.0,
                produced_units * energy_intensity + machine_profile["energy_bias"],
            ),
            2,
        )

        temperature_gap = abs(furnace_temp - target_temp)
        pressure_gap = abs(pressure - glass_profile["pressure_target"])
        speed_gap = max(0.0, line_speed - glass_profile["speed_range"][1])

        temperature_risk = temperature_gap / 12
        pressure_risk = pressure_gap / 0.18
        speed_risk = speed_gap / 0.2
        quality_risk = max(0.0, 0.91 - raw_material_quality) * 14
        humidity_risk = max(0.0, ambient_humidity_pct - 62) / 8

        defect_signal = -4.05
        defect_signal += 0.92 * temperature_risk
        defect_signal += 0.62 * pressure_risk
        defect_signal += 0.54 * speed_risk
        defect_signal += 1.05 * quality_risk
        defect_signal += 0.42 * humidity_risk if glass_type in {"Laminated", "Solar"} else 0.16 * humidity_risk
        defect_signal += 0.28 if operator_shift == "Night" else 0.52 if operator_shift == "Weekend" else 0.0
        defect_signal += machine_profile["defect_bias"] * 8
        defect_signal += plant_profile["defect_bias"] * 7
        defect_signal += 0.14 if furnace_zone == "Zone_3" else -0.05 if furnace_zone == "Zone_1" else 0.0
        defect_signal -= 0.14 * operator_experience_years
        defect_signal += np.random.normal(0, 0.12)

        defect_probability = clamp(1 / (1 + math.exp(-defect_signal)), 0.01, 0.72)

        defect_flag = 1 if random.random() < defect_probability else 0

        base_defect_rate = np.random.uniform(0.003, 0.012)
        defect_rate = base_defect_rate
        if defect_flag:
            defect_rate += np.random.uniform(0.015, 0.05)
            defect_rate += defect_probability * 0.08
            defect_rate += temperature_gap / 1100
        defect_rate = round(clamp(defect_rate, 0.004, 0.18), 4)

        scrap_units = int(round(produced_units * defect_rate * np.random.uniform(0.55, 0.9)))
        rework_units = int(
            round(
                produced_units
                * defect_rate
                * (np.random.uniform(0.12, 0.32) if defect_flag else np.random.uniform(0.01, 0.05))
            )
        )
        downtime_min = round(
            max(
                0.0,
                np.random.normal(8 + shift_profile["delay_bias"] + defect_flag * 14, 6),
            ),
            2,
        )
        packing_delay = round(
            max(
                0.0,
                np.random.normal(
                    10
                    + plant_profile["delay_bias"]
                    + shift_profile["delay_bias"]
                    + downtime_min * 0.35
                    + defect_flag * 8,
                    7,
                ),
            ),
            2,
        )

        estimated_scrap_cost = round(
            scrap_units * glass_profile["unit_cost"] * np.random.uniform(0.9, 1.12),
            2,
        )
        co2_emissions_kg = round(energy * 0.41, 2)
        shipment_status = (
            "Delayed"
            if packing_delay > 22 or downtime_min > 30 or defect_rate > 0.085
            else "On-Time"
        )

        data.append(
            [
                batch_id,
                production_date.date().isoformat(),
                plant_id,
                machine_id,
                glass_type,
                thickness_mm,
                furnace_temp,
                cooling_time,
                pressure,
                line_speed,
                raw_material_quality,
                operator_shift,
                operator_experience_years,
                ambient_humidity_pct,
                furnace_zone,
                energy,
                produced_units,
                scrap_units,
                rework_units,
                defect_rate,
                defect_flag,
                estimated_scrap_cost,
                packing_delay,
                downtime_min,
                shipment_status,
                co2_emissions_kg,
            ]
        )

    columns = [
        "batch_id",
        "production_date",
        "plant_id",
        "machine_id",
        "glass_type",
        "thickness_mm",
        "furnace_temperature_c",
        "cooling_time_sec",
        "pressure_bar",
        "line_speed_mps",
        "raw_material_quality",
        "operator_shift",
        "operator_experience_years",
        "ambient_humidity_pct",
        "furnace_zone",
        "energy_consumption_kwh",
        "produced_units",
        "scrap_units",
        "rework_units",
        "defect_rate",
        "defect_flag",
        "estimated_scrap_cost_eur",
        "packing_delay_min",
        "downtime_min",
        "shipment_status",
        "co2_emissions_kg",
    ]

    return pd.DataFrame(data, columns=columns)


def main() -> None:
    ensure_data_directories()
    dataset = generate_dataset()
    dataset.to_csv(settings.csv_path, index=False)
    print(f"Dataset created successfully at {settings.csv_path}")
    print(dataset.head())


if __name__ == "__main__":
    main()
