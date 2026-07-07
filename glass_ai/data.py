from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd

from glass_ai.config import settings


def ensure_data_directories() -> None:
    settings.csv_path.parent.mkdir(parents=True, exist_ok=True)
    settings.db_path.parent.mkdir(parents=True, exist_ok=True)


def load_production_data(
    csv_path: Path | None = None,
    *,
    parse_dates: bool = False,
) -> pd.DataFrame:
    target_path = csv_path or settings.csv_path
    dataframe = pd.read_csv(target_path)

    if parse_dates and "production_date" in dataframe.columns:
        dataframe["production_date"] = pd.to_datetime(dataframe["production_date"])

    return dataframe


def build_sqlite_database(
    csv_path: Path | None = None,
    db_path: Path | None = None,
) -> Path:
    source_csv = csv_path or settings.csv_path
    target_db = db_path or settings.db_path

    ensure_data_directories()

    if not source_csv.exists():
        raise FileNotFoundError(f"Production dataset not found at {source_csv}")

    dataframe = pd.read_csv(source_csv)
    with sqlite3.connect(target_db) as connection:
        dataframe.to_sql(settings.sqlite_table, connection, if_exists="replace", index=False)

    return target_db


def ensure_database() -> Path:
    ensure_data_directories()

    if not settings.csv_path.exists():
        raise FileNotFoundError(
            f"Production dataset not found at {settings.csv_path}. "
            "Generate or move the CSV before starting the apps."
        )

    if not settings.db_path.exists():
        return build_sqlite_database()

    return settings.db_path


def run_query(query: str) -> pd.DataFrame:
    ensure_database()
    with sqlite3.connect(settings.db_path) as connection:
        return pd.read_sql_query(query, connection)
