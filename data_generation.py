import pandas as pd
import numpy as np
import random
from datetime import datetime, timedelta

np.random.seed(42)

n = 1000

data = []

start_date = datetime(2024, 1, 1)

for i in range(n):
    batch_id = i + 1
    production_date = start_date + timedelta(days=random.randint(0, 365))
    plant_id = random.choice(['Plant_A', 'Plant_B'])
    machine_id = random.choice(['M1', 'M2', 'M3', 'M4'])
    glass_type = random.choice(['Tempered', 'Laminated', 'Float'])
    
    thickness_mm = round(np.random.uniform(2, 12), 2)
    furnace_temp = round(np.random.normal(1500, 50), 2)
    cooling_time = round(np.random.uniform(30, 120), 2)
    pressure = round(np.random.uniform(1, 5), 2)
    line_speed = round(np.random.uniform(1, 10), 2)
    
    raw_material_quality = round(np.random.uniform(0.7, 1.0), 2)
    operator_shift = random.choice(['Day', 'Night'])
    
    energy = round(np.random.uniform(500, 1500), 2)
    produced_units = random.randint(200, 1000)
    
    # defect logic (important for ML later)
    defect_probability = (
        0.3 * (furnace_temp > 1550) +
        0.2 * (pressure > 4) +
        0.2 * (raw_material_quality < 0.8) +
        0.1 * (operator_shift == 'Night')
    )
    
    defect_flag = 1 if random.random() < defect_probability else 0
    defect_rate = round(defect_flag * np.random.uniform(0.01, 0.1), 3)
    
    packing_delay = round(np.random.uniform(0, 60), 2)
    shipment_status = random.choice(['On-Time', 'Delayed'])
    
    data.append([
        batch_id, production_date, plant_id, machine_id, glass_type,
        thickness_mm, furnace_temp, cooling_time, pressure, line_speed,
        raw_material_quality, operator_shift, energy, produced_units,
        defect_rate, defect_flag, packing_delay, shipment_status
    ])

columns = [
    "batch_id", "production_date", "plant_id", "machine_id", "glass_type",
    "thickness_mm", "furnace_temperature_c", "cooling_time_sec", "pressure_bar",
    "line_speed_mps", "raw_material_quality", "operator_shift",
    "energy_consumption_kwh", "produced_units", "defect_rate", "defect_flag",
    "packing_delay_min", "shipment_status"
]

df = pd.DataFrame(data, columns=columns)

# Save dataset
df.to_csv("glass_production.csv", index=False)

print("Dataset created successfully!")
print(df.head())