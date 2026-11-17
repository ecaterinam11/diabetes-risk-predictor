"""Tests for prediction logging, run against a temporary SQLite database."""

import re
import sys
from pathlib import Path

import pytest
from sqlalchemy import select

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "app"))

from prediction_log import (  # noqa: E402
    INPUT_COLUMNS,
    database_url,
    log_prediction,
    make_engine,
    usage_summary,
    user_predictions,
)

ANSWERS = dict(
    is_male=1, age_group_id=6, bmi=25.9515, has_high_bp=1, has_high_chol=1, has_heart_disease=0,
    had_stroke=0, has_difficulty_walking=0, general_health=3, poor_physical_days=0,
    poor_mental_days=0, is_physically_active=0, eats_fruit_daily=0, eats_veg_daily=0, is_smoker=0,
    # extra keys the app also holds, which must never reach the database
    age=45, consent=True,
)


@pytest.fixture
def engine(tmp_path):
    engine = make_engine(f"sqlite:///{(tmp_path / 'log.db').as_posix()}")
    yield engine
    engine.dispose()


def test_row_is_stored_with_only_the_model_columns(engine):
    log_prediction(engine, ANSWERS, 0.15834, "Moderate", "2026-11-17")
    with engine.connect() as conn:
        rows = conn.execute(select(user_predictions)).mappings().all()
    assert len(rows) == 1
    row = rows[0]
    assert float(row["bmi"]) == 26.0                       # rounded to one decimal
    assert float(row["predicted_probability"]) == 0.1583   # rounded to four
    assert row["risk_level"] == "Moderate" and row["created_at"] is not None
    assert "age" not in row and "consent" not in row


def test_usage_summary_counts_by_level(engine):
    assert usage_summary(engine).empty
    for probability, level in [(0.03, "Low"), (0.05, "Low"), (0.40, "High")]:
        log_prediction(engine, ANSWERS, probability, level, "2026-11-17")
    summary = usage_summary(engine)
    assert list(summary["risk_level"]) == ["Low", "High"]
    assert list(summary["predictions"]) == [2, 1]
    assert list(summary["avg_predicted_pct"]) == [4.0, 40.0]


def test_database_url_handling(monkeypatch):
    monkeypatch.delenv("DATABASE_URL", raising=False)
    assert database_url().startswith("sqlite:///")
    assert database_url("postgres://u:p@host/db") == "postgresql://u:p@host/db"
    monkeypatch.setenv("DATABASE_URL", "postgresql://u:p@host/db")
    assert database_url() == "postgresql://u:p@host/db"


def test_sql_file_matches_the_python_table():
    """sql/04_user_predictions.sql and the SQLAlchemy table must list the same columns."""
    sql = (ROOT / "sql" / "04_user_predictions.sql").read_text(encoding="utf-8")
    body = sql.split("CREATE TABLE IF NOT EXISTS user_predictions (")[1].split("\n);")[0]
    sql_columns = re.findall(r"^\s{4}([a-z_]+)\s+[A-Z]", body, flags=re.MULTILINE)
    assert sql_columns == [c.name for c in user_predictions.columns]
    assert set(INPUT_COLUMNS) <= set(sql_columns)


def test_logged_columns_match_the_model():
    """Every input the model uses is logged, and nothing else."""
    import json
    info = json.loads((ROOT / "models" / "model_info.json").read_text(encoding="utf-8"))
    assert sorted(INPUT_COLUMNS) == sorted(info["features"])
