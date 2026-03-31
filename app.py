import os
import sqlite3
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier


# Page config

st.set_page_config(page_title="Glass Manufacturing AI Assistant", layout="wide")

st.sidebar.title("Glass Manufacturing AI Assistant")
st.sidebar.info("SQL analytics, defect prediction, and chatbot insights for production data.")


# File paths

CSV_PATH = "glass_production.csv"
DB_PATH = "glass_factory.db"


# Database setup

def ensure_database():
    """Create the SQLite database from CSV if it does not exist."""
    if not os.path.exists(CSV_PATH):
        st.error(f"Required data file not found: {CSV_PATH}")
        st.stop()

    if not os.path.exists(DB_PATH):
        df_temp = pd.read_csv(CSV_PATH)
        conn = sqlite3.connect(DB_PATH)
        df_temp.to_sql("production", conn, if_exists="replace", index=False)
        conn.close()

@st.cache_data
def load_data():
    """Load CSV data."""
    return pd.read_csv(CSV_PATH)

@st.cache_resource
def get_connection():
    """Create a reusable SQLite connection."""
    return sqlite3.connect(DB_PATH, check_same_thread=False)

def run_query(query: str) -> pd.DataFrame:
    """Run SQL query and return DataFrame."""
    conn = get_connection()
    return pd.read_sql(query, conn)


# Model training

@st.cache_resource
def train_model(dataframe: pd.DataFrame):
    feature_cols = [
        "furnace_temperature_c",
        "pressure_bar",
        "raw_material_quality",
        "line_speed_mps",
        "cooling_time_sec",
    ]

    X = dataframe[feature_cols]
    y = dataframe["defect_flag"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=5,
        random_state=42
    )
    model.fit(X_train, y_train)

    score = model.score(X_test, y_test)
    return model, feature_cols, score


# Helper functions

def format_small_number(x):
    if isinstance(x, float):
        return round(x, 4)
    return x

def chatbot_response(question: str):
    q = question.lower().strip()

    if "highest defects" in q or "most defects" in q:
        query = """
        SELECT machine_id, SUM(defect_flag) AS total_defects
        FROM production
        GROUP BY machine_id
        ORDER BY total_defects DESC
        LIMIT 1
        """
        return run_query(query)

    elif "average output by machine" in q:
        query = """
        SELECT machine_id, ROUND(AVG(produced_units), 2) AS avg_output
        FROM production
        GROUP BY machine_id
        ORDER BY avg_output DESC
        """
        return run_query(query)

    elif "defect rate by shift" in q:
        query = """
        SELECT operator_shift, ROUND(AVG(defect_flag), 4) AS defect_rate
        FROM production
        GROUP BY operator_shift
        """
        return run_query(query)

    elif "average temperature for defective batches" in q or "defective batches temperature" in q:
        query = """
        SELECT ROUND(AVG(furnace_temperature_c), 2) AS avg_temp
        FROM production
        WHERE defect_flag = 1
        """
        return run_query(query)

    elif "produced units by plant" in q or "plant performance" in q:
        query = """
        SELECT plant_id, SUM(produced_units) AS total_units
        FROM production
        GROUP BY plant_id
        ORDER BY total_units DESC
        """
        return run_query(query)

    elif "show defective batches" in q:
        query = """
        SELECT batch_id, machine_id, plant_id,
               ROUND(furnace_temperature_c, 2) AS furnace_temperature_c,
               ROUND(pressure_bar, 2) AS pressure_bar,
               ROUND(raw_material_quality, 2) AS raw_material_quality,
               produced_units
        FROM production
        WHERE defect_flag = 1
        LIMIT 10
        """
        return run_query(query)

    return None


# Initialize app data

ensure_database()
df = load_data()
model, feature_cols, model_score = train_model(df)


# UI

st.title("Glass Manufacturing Analytics & AI Assistant")
st.write("An end-to-end mini project with SQL analytics, defect prediction, and chatbot support.")

tab1, tab2, tab3, tab4 = st.tabs([
    "Dataset Preview",
    "SQL Analytics",
    "Defect Prediction",
    "Chatbot"
])


# Tab 1: Dataset Preview

with tab1:
    st.subheader("Production Dataset")
    st.dataframe(df.head(20), width="stretch")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Batches", len(df))
    col2.metric("Total Defects", int(df["defect_flag"].sum()))
    col3.metric("Average Produced Units", round(df["produced_units"].mean(), 2))

    st.subheader("Quick Summary")
    st.dataframe(df.describe(), width="stretch")


# Tab 2: SQL Analytics

with tab2:
    st.subheader("SQL-Based Manufacturing Insights")

    query_option = st.selectbox(
        "Choose an analysis",
        [
            "Average output by machine",
            "Total defects by machine",
            "Defect rate by shift",
            "Produced units by plant",
            "Average defective batch conditions",
        ],
        key="sql_analysis_select"
    )

    if query_option == "Average output by machine":
        query = """
        SELECT machine_id, ROUND(AVG(produced_units), 2) AS avg_output
        FROM production
        GROUP BY machine_id
        ORDER BY avg_output DESC
        """
        result = run_query(query)
        st.dataframe(result, width="stretch")
        st.bar_chart(result.set_index("machine_id"))

    elif query_option == "Total defects by machine":
        query = """
        SELECT machine_id, SUM(defect_flag) AS total_defects
        FROM production
        GROUP BY machine_id
        ORDER BY total_defects DESC
        """
        result = run_query(query)
        st.dataframe(result, width="stretch")
        st.bar_chart(result.set_index("machine_id"))

    elif query_option == "Defect rate by shift":
        query = """
        SELECT operator_shift, ROUND(AVG(defect_flag), 4) AS defect_rate
        FROM production
        GROUP BY operator_shift
        """
        result = run_query(query)
        st.dataframe(result, width="stretch")
        st.bar_chart(result.set_index("operator_shift"))

    elif query_option == "Produced units by plant":
        query = """
        SELECT plant_id, SUM(produced_units) AS total_units
        FROM production
        GROUP BY plant_id
        ORDER BY total_units DESC
        """
        result = run_query(query)
        st.dataframe(result, width="stretch")
        st.bar_chart(result.set_index("plant_id"))

    elif query_option == "Average defective batch conditions":
        query = """
        SELECT
            ROUND(AVG(furnace_temperature_c), 2) AS avg_temp,
            ROUND(AVG(pressure_bar), 2) AS avg_pressure,
            ROUND(AVG(raw_material_quality), 2) AS avg_quality,
            ROUND(AVG(line_speed_mps), 2) AS avg_line_speed
        FROM production
        WHERE defect_flag = 1
        """
        result = run_query(query)
        st.dataframe(result, width="stretch")


# Tab 3: Defect Prediction

with tab3:
    st.subheader("Predict Defect Risk")
    st.write(f"Model accuracy: **{round(model_score * 100, 2)}%**")

    col1, col2 = st.columns(2)

    with col1:
        furnace_temperature_c = st.number_input(
            "Furnace Temperature (°C)",
            value=1500.0,
            key="temp_input"
        )
        pressure_bar = st.number_input(
            "Pressure (bar)",
            value=3.0,
            key="pressure_input"
        )
        raw_material_quality = st.number_input(
            "Raw Material Quality",
            min_value=0.0,
            max_value=1.0,
            value=0.85,
            key="quality_input"
        )

    with col2:
        line_speed_mps = st.number_input(
            "Line Speed (m/s)",
            value=5.0,
            key="speed_input"
        )
        cooling_time_sec = st.number_input(
            "Cooling Time (sec)",
            value=60.0,
            key="cooling_input"
        )

    if st.button("Predict Defect", key="predict_button"):
        input_df = pd.DataFrame([{
            "furnace_temperature_c": furnace_temperature_c,
            "pressure_bar": pressure_bar,
            "raw_material_quality": raw_material_quality,
            "line_speed_mps": line_speed_mps,
            "cooling_time_sec": cooling_time_sec,
        }])

        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.error(f"High defect risk detected. Probability: {round(probability * 100, 2)}%")
        else:
            st.success(f"Low defect risk. Probability: {round(probability * 100, 2)}%")

    st.subheader("Feature Importance")
    importance_df = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    st.dataframe(importance_df, width="stretch")
    st.bar_chart(importance_df.set_index("feature"))


# Tab 4: Chatbot

with tab4:
    st.subheader("Manufacturing Chatbot")
    st.write("Ask questions about production data.")

    st.markdown("""
**Supported examples**
- Which machine has the highest defects?
- Show defect rate by shift
- Show average output by machine
- Average temperature for defective batches
- Produced units by plant
- Show defective batches
""")

    user_question = st.text_input("Enter your question", key="chatbot_question")

    if st.button("Ask Chatbot", key="ask_chatbot_button"):
        result = chatbot_response(user_question)

        if result is not None:
            st.dataframe(result.applymap(format_small_number), width="stretch")
        else:
            st.warning("I do not understand that question yet. Please try one of the supported examples.")