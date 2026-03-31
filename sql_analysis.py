import sqlite3
import pandas as pd

# Load CSV
df = pd.read_csv("glass_production.csv")

# Create SQLite DB
conn = sqlite3.connect("glass_factory.db")

# Save table
df.to_sql("production", conn, if_exists="replace", index=False)

print("Data loaded into SQLite!")