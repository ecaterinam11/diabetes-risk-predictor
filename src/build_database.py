"""Build the SQLite database from the raw CSV.

Usage (from the project root):
    python src/build_database.py

Steps:
    1. Create the tables from sql/01_schema.sql
    2. Split the flat CSV into patients, lifestyle_factors and clinical_metrics
    3. Validate: row counts, foreign keys, and that joining the three tables
       back together reproduces the original CSV exactly

The database is written to data/diabetes.db, which is not committed to Git.
It is rebuilt from scratch on every run.
"""

from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine, event, text

ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = ROOT / "data" / "raw" / "cdc_diabetes_health_indicators.csv"
DB_PATH = ROOT / "data" / "diabetes.db"
SCHEMA_PATH = ROOT / "sql" / "01_schema.sql"

# CSV column -> database column, grouped by destination table
PATIENTS = {
    "ID": "patient_id",
    "Sex": "sex",
    "Age": "age_group_id",
    "Education": "education_id",
    "Income": "income_id",
    "AnyHealthcare": "has_healthcare",
    "NoDocbcCost": "skipped_doctor_cost",
    "Diabetes_binary": "has_diabetes",
}
LIFESTYLE = {
    "ID": "patient_id",
    "Smoker": "is_smoker",
    "PhysActivity": "is_physically_active",
    "Fruits": "eats_fruit_daily",
    "Veggies": "eats_veg_daily",
    "HvyAlcoholConsump": "is_heavy_drinker",
}
CLINICAL = {
    "ID": "patient_id",
    "BMI": "bmi",
    "HighBP": "has_high_bp",
    "HighChol": "has_high_chol",
    "CholCheck": "had_chol_check",
    "Stroke": "had_stroke",
    "HeartDiseaseorAttack": "has_heart_disease",
    "DiffWalk": "has_difficulty_walking",
    "GenHlth": "general_health",
    "MentHlth": "poor_mental_days",
    "PhysHlth": "poor_physical_days",
}

# Joins the three tables back into the shape of the original CSV
RECONSTRUCT_SQL = """
SELECT
    p.patient_id                               AS ID,
    p.has_diabetes                             AS Diabetes_binary,
    c.has_high_bp                              AS HighBP,
    c.has_high_chol                            AS HighChol,
    c.had_chol_check                           AS CholCheck,
    c.bmi                                      AS BMI,
    l.is_smoker                                AS Smoker,
    c.had_stroke                               AS Stroke,
    c.has_heart_disease                        AS HeartDiseaseorAttack,
    l.is_physically_active                     AS PhysActivity,
    l.eats_fruit_daily                         AS Fruits,
    l.eats_veg_daily                           AS Veggies,
    l.is_heavy_drinker                         AS HvyAlcoholConsump,
    p.has_healthcare                           AS AnyHealthcare,
    p.skipped_doctor_cost                      AS NoDocbcCost,
    c.general_health                           AS GenHlth,
    c.poor_mental_days                         AS MentHlth,
    c.poor_physical_days                       AS PhysHlth,
    c.has_difficulty_walking                   AS DiffWalk,
    CASE p.sex WHEN 'M' THEN 1 ELSE 0 END      AS Sex,
    p.age_group_id                             AS Age,
    p.education_id                             AS Education,
    p.income_id                                AS Income
FROM patients AS p
JOIN lifestyle_factors AS l ON l.patient_id = p.patient_id
JOIN clinical_metrics  AS c ON c.patient_id = p.patient_id
ORDER BY p.patient_id
"""


def make_engine():
    engine = create_engine(f"sqlite:///{DB_PATH.as_posix()}")

    # SQLite only enforces foreign keys when asked to, on every connection
    @event.listens_for(engine, "connect")
    def _enable_foreign_keys(dbapi_connection, _):
        dbapi_connection.execute("PRAGMA foreign_keys = ON")

    return engine


def create_schema(engine) -> None:
    raw = engine.raw_connection()
    try:
        raw.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
        raw.commit()
    finally:
        raw.close()


def load_tables(engine, df: pd.DataFrame) -> None:
    patients = df[list(PATIENTS)].rename(columns=PATIENTS)
    patients["sex"] = patients["sex"].map({0: "F", 1: "M"})
    lifestyle = df[list(LIFESTYLE)].rename(columns=LIFESTYLE)
    clinical = df[list(CLINICAL)].rename(columns=CLINICAL)

    # patients first: the other two tables have foreign keys pointing at it
    with engine.begin() as conn:
        for name, table in [
            ("patients", patients),
            ("lifestyle_factors", lifestyle),
            ("clinical_metrics", clinical),
        ]:
            table.to_sql(name, conn, if_exists="append", index=False, chunksize=10_000)
            print(f"  loaded {len(table):>9,} rows into {name}")


def validate(engine, df: pd.DataFrame) -> None:
    with engine.connect() as conn:
        for name in ["patients", "lifestyle_factors", "clinical_metrics"]:
            count = conn.execute(text(f"SELECT COUNT(*) FROM {name}")).scalar_one()
            assert count == len(df), f"{name}: {count:,} rows, expected {len(df):,}"
            print(f"  row count OK      {name}: {count:,}")

        broken = conn.execute(text("PRAGMA foreign_key_check")).fetchall()
        assert not broken, f"{len(broken)} foreign key violations"
        print("  foreign keys OK   no violations")

        rebuilt = pd.read_sql(text(RECONSTRUCT_SQL), conn)

    original = df.sort_values("ID").reset_index(drop=True)
    pd.testing.assert_frame_equal(rebuilt[original.columns], original, check_dtype=False)
    print(f"  round trip OK     3-table join reproduces all {original.size:,} CSV values")


def main() -> None:
    if not CSV_PATH.exists():
        raise SystemExit("Raw CSV not found. Run: python src/download_data.py")

    DB_PATH.unlink(missing_ok=True)
    df = pd.read_csv(CSV_PATH)
    print(f"Read {len(df):,} rows from {CSV_PATH.name}")

    engine = make_engine()
    print("Creating schema ...")
    create_schema(engine)
    print("Loading tables ...")
    load_tables(engine, df)
    print("Validating ...")
    validate(engine, df)
    engine.dispose()
    print(f"Done: {DB_PATH.relative_to(ROOT)} ({DB_PATH.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
