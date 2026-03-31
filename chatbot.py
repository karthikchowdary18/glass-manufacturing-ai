import sqlite3
import pandas as pd

pd.set_option("display.max_columns", None)
pd.set_option("display.width", 1000)
pd.set_option("display.max_colwidth", None)

conn = sqlite3.connect("glass_factory.db")

def run_query(query):
    return pd.read_sql(query, conn)

def format_result(df):
    if df.empty:
        return "No results found."
    return df.to_string(index=False)

def chatbot(question):
    q = question.lower().strip()

    if "highest defects" in q or "most defects" in q:
        query = """
        SELECT machine_id, SUM(defect_flag) AS total_defects
        FROM production
        GROUP BY machine_id
        ORDER BY total_defects DESC
        LIMIT 1
        """
        result = run_query(query)
        return "Machine with highest defects:\n" + format_result(result)

    elif "average output by machine" in q:
        query = """
        SELECT machine_id, ROUND(AVG(produced_units), 2) AS avg_output
        FROM production
        GROUP BY machine_id
        ORDER BY avg_output DESC
        """
        result = run_query(query)
        return "Average output by machine:\n" + format_result(result)

    elif "defect rate by shift" in q:
        query = """
        SELECT operator_shift, ROUND(AVG(defect_flag), 4) AS defect_rate
        FROM production
        GROUP BY operator_shift
        """
        result = run_query(query)
        return "Defect rate by shift:\n" + format_result(result)

    elif "average temperature for defective batches" in q or "defective batches temperature" in q:
        query = """
        SELECT ROUND(AVG(furnace_temperature_c), 2) AS avg_temp
        FROM production
        WHERE defect_flag = 1
        """
        result = run_query(query)
        return "Average furnace temperature for defective batches:\n" + format_result(result)

    elif "plant performance" in q or "produced units by plant" in q:
        query = """
        SELECT plant_id, SUM(produced_units) AS total_units
        FROM production
        GROUP BY plant_id
        ORDER BY total_units DESC
        """
        result = run_query(query)
        return "Produced units by plant:\n" + format_result(result)

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
        result = run_query(query)
        return "Sample defective batches:\n" + format_result(result)

    elif "help" in q:
        return (
            "You can ask:\n"
            "- Which machine has the highest defects?\n"
            "- Show defect rate by shift\n"
            "- Show average output by machine\n"
            "- Average temperature for defective batches\n"
            "- Produced units by plant\n"
            "- Show defective batches"
        )

    else:
        return "Sorry, I do not understand that yet. Type 'help' to see supported questions."

print("Glass Manufacturing Chatbot")
print("Type 'exit' to quit.")
print("Type 'help' to see supported questions.\n")

while True:
    user_input = input("You: ")
    if user_input.lower() == "exit":
        print("Chatbot: Goodbye!")
        break

    answer = chatbot(user_input)
    print("\nChatbot:", answer, "\n")