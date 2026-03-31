import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Load data
df = pd.read_csv("glass_production.csv")

# Features (important inputs)
X = df[[
    "furnace_temperature_c",
    "pressure_bar",
    "raw_material_quality",
    "line_speed_mps",
    "cooling_time_sec"
]]

# Target
y = df["defect_flag"]

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Model
model = RandomForestClassifier(class_weight="balanced")
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print(classification_report(y_test, y_pred))