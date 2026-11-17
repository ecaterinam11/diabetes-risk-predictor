"""Anonymous logging of predictions to a SQL database.

In production the database is hosted PostgreSQL, found through the DATABASE_URL
secret, and its table is created by sql/04_user_predictions.sql.

With no DATABASE_URL set (for example when running the app on your own machine)
predictions go to a local SQLite file instead, so the app works with no setup.

Nothing in this file imports Streamlit, so it can be tested on its own.
"""

import os
from pathlib import Path

import pandas as pd
from sqlalchemy import (
    BigInteger,
    Column,
    DateTime,
    Identity,
    Integer,
    MetaData,
    Numeric,
    SmallInteger,
    String,
    Table,
    create_engine,
    func,
    insert,
    text,
)
from sqlalchemy.engine import Engine

LOCAL_DB_PATH = Path(__file__).resolve().parents[1] / "data" / "predictions.db"

# The model inputs that are stored, in table order
INPUT_COLUMNS = [
    "is_male", "age_group_id", "bmi", "has_high_bp", "has_high_chol", "has_heart_disease",
    "had_stroke", "has_difficulty_walking", "general_health", "poor_physical_days",
    "poor_mental_days", "is_physically_active", "eats_fruit_daily", "eats_veg_daily", "is_smoker",
]

metadata = MetaData()

# Mirrors sql/04_user_predictions.sql (a test checks the two stay in step)
user_predictions = Table(
    "user_predictions",
    metadata,
    Column("prediction_id", BigInteger().with_variant(Integer, "sqlite"), Identity(always=True), primary_key=True),
    Column("created_at", DateTime(timezone=True), nullable=False, server_default=func.now()),
    Column("model_version", String(20), nullable=False),
    *[
        Column(name, Numeric(4, 1) if name == "bmi" else SmallInteger, nullable=False)
        for name in INPUT_COLUMNS
    ],
    Column("predicted_probability", Numeric(5, 4), nullable=False),
    Column("risk_level", String(10), nullable=False),
)

SUMMARY_SQL = text("""
    SELECT
        risk_level,
        COUNT(*)                                     AS predictions,
        ROUND(100.0 * AVG(predicted_probability), 1) AS avg_predicted_pct
    FROM user_predictions
    GROUP BY risk_level
""")


def database_url(secret: str | None = None) -> str:
    """The hosted database if one is configured, otherwise a local SQLite file."""
    url = secret or os.environ.get("DATABASE_URL")
    if not url:
        LOCAL_DB_PATH.parent.mkdir(exist_ok=True)
        return f"sqlite:///{LOCAL_DB_PATH.as_posix()}"
    # Hosting providers hand out "postgres://", which SQLAlchemy no longer accepts
    if url.startswith("postgres://"):
        url = "postgresql://" + url[len("postgres://"):]
    return url


def make_engine(url: str) -> Engine:
    # pool_pre_ping reconnects if the free-tier database went to sleep
    engine = create_engine(url, pool_pre_ping=True)
    if engine.dialect.name == "sqlite":
        metadata.create_all(engine)  # the local file needs its table creating
    return engine


def log_prediction(engine: Engine, answers: dict, probability: float, level: str, model_version: str) -> None:
    """Insert one anonymous prediction. Only the model's inputs and outputs are stored."""
    row = {name: answers[name] for name in INPUT_COLUMNS}
    row["bmi"] = round(float(row["bmi"]), 1)
    row |= {
        "model_version": model_version,
        "predicted_probability": round(float(probability), 4),
        "risk_level": level,
    }
    with engine.begin() as conn:
        conn.execute(insert(user_predictions), row)


def usage_summary(engine: Engine) -> pd.DataFrame:
    """Aggregate counts for the app's Usage tab. Never returns individual rows."""
    with engine.connect() as conn:
        summary = pd.read_sql(SUMMARY_SQL, conn)
    order = {"Low": 0, "Moderate": 1, "High": 2}
    summary["avg_predicted_pct"] = summary["avg_predicted_pct"].astype(float)
    return summary.sort_values("risk_level", key=lambda s: s.map(order)).reset_index(drop=True)
