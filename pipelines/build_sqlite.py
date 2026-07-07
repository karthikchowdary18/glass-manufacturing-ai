from __future__ import annotations

from glass_ai.data import build_sqlite_database


def main() -> None:
    database_path = build_sqlite_database()
    print(f"Data loaded into SQLite at {database_path}")


if __name__ == "__main__":
    main()

