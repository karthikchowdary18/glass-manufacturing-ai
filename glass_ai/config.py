from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _resolve_path(value: str | None, default: str) -> Path:
    raw_path = Path(value or default)
    return raw_path if raw_path.is_absolute() else PROJECT_ROOT / raw_path


@dataclass(frozen=True)
class Settings:
    project_root: Path = PROJECT_ROOT
    project_name: str = os.getenv("PROJECT_NAME", "Glass Manufacturing AI")
    app_env: str = os.getenv("APP_ENV", "development")
    csv_path: Path = _resolve_path(os.getenv("CSV_PATH"), "data/raw/glass_production.csv")
    db_path: Path = _resolve_path(os.getenv("DB_PATH"), "data/processed/glass_factory.db")
    sqlite_table: str = os.getenv("SQLITE_TABLE", "production")
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    dashboard_host: str = os.getenv("DASHBOARD_HOST", "0.0.0.0")
    dashboard_port: int = int(os.getenv("DASHBOARD_PORT", "8501"))


settings = Settings()

