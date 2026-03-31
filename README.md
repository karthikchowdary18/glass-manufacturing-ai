# Glass Manufacturing Analytics, Defect Prediction & AI Chatbot

An end-to-end manufacturing analytics project built with Python, SQLite, scikit-learn, and Streamlit.

This project simulates a glass manufacturing environment and demonstrates how production data can be analyzed, used for defect prediction, and queried through a chatbot-style interface.

## Features

- Synthetic glass manufacturing dataset generation
- SQL-based analytics using SQLite
- Defect prediction using Random Forest
- Streamlit dashboard for interactive exploration
- Chatbot interface for natural-language production queries

## Tech Stack

- Python
- Pandas
- NumPy
- SQLite
- scikit-learn
- Streamlit
- Matplotlib

## Project Modules

### 1. Data Generation
Generates a realistic production dataset with:
- furnace temperature
- pressure
- raw material quality
- line speed
- cooling time
- produced units
- defect flag

### 2. SQL Analytics
Runs manufacturing-focused queries such as:
- average output by machine
- total defects by machine
- defect rate by shift
- produced units by plant
- defective batch conditions

### 3. Defect Prediction
Trains a Random Forest classifier to predict defect risk based on production parameters.

### 4. Chatbot
Supports questions like:
- Which machine has the highest defects?
- Show defect rate by shift
- Show average output by machine
- Average temperature for defective batches
- Produced units by plant
- Show defective batches

## Project Structure

```text
glass-manufacturing-ai/
├── app.py
├── chatbot.py
├── data_generation.py
├── defect_prediction_model.py
├── sql_analysis.py
├── glass_production.csv
├── glass_factory.db
├── requirements.txt
└── README.md
