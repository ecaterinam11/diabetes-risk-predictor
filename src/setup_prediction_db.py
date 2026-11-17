"""Create the user_predictions table in the hosted PostgreSQL database.

Usage (from the project root):
    python src/setup_prediction_db.py

The connection string is read from the DATABASE_URL environment variable or,
if that is not set, from .streamlit/secrets.toml. Neither is committed to Git.

The script runs sql/04_user_predictions.sql, then proves the table works by
inserting one test row inside a transaction that is rolled back, so nothing
is left behind. It is safe to run more than once.
"""

import os
import sys
import tomllib
from pathlib import Path

from sqlalchemy import func, insert, select, text

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from prediction_log import INPUT_COLUMNS, database_url, make_engine, user_predictions  # noqa: E402

SQL_PATH = ROOT / "sql" / "04_user_predictions.sql"
SECRETS_PATH = ROOT / ".streamlit" / "secrets.toml"


def find_url() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url and SECRETS_PATH.exists():
        url = tomllib.loads(SECRETS_PATH.read_text(encoding="utf-8")).get("DATABASE_URL")
    if not url:
        raise SystemExit(
            "No DATABASE_URL found.\n"
            "Copy .streamlit/secrets.toml.example to .streamlit/secrets.toml and paste in "
            "the connection string from your database provider."
        )
    return database_url(url)


def main() -> None:
    engine = make_engine(find_url())
    if engine.dialect.name != "postgresql":
        raise SystemExit("DATABASE_URL must point at a PostgreSQL database.")
    print(f"Connecting to {engine.url.host} ...")  # host only: never print the password

    with engine.begin() as conn:
        conn.exec_driver_sql(SQL_PATH.read_text(encoding="utf-8"))
    print("Created user_predictions and v_prediction_summary (or they already existed).")

    test_row = {name: 0 for name in INPUT_COLUMNS} | {
        "age_group_id": 5, "bmi": 24.5, "general_health": 2,
        "model_version": "setup-test", "predicted_probability": 0.0312, "risk_level": "Low",
    }
    with engine.connect() as conn:
        transaction = conn.begin()
        conn.execute(insert(user_predictions), test_row)
        conn.execute(text("SELECT * FROM v_prediction_summary")).fetchall()
        transaction.rollback()
        rows = conn.execute(select(func.count()).select_from(user_predictions)).scalar_one()
    print("Test insert worked and was rolled back.")
    print(f"The table currently holds {rows:,} prediction(s). Setup complete.")
    engine.dispose()


if __name__ == "__main__":
    main()
