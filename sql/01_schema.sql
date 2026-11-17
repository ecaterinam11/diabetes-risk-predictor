-- =============================================================================
-- 01_schema.sql
-- Normalised schema for the CDC Diabetes Health Indicators dataset (BRFSS 2015).
--
-- The raw data is one flat CSV with 23 columns. Here it is split into:
--   * three lookup tables that give the coded survey answers a readable label
--   * patients           - one row per respondent: demographics and the outcome
--   * lifestyle_factors  - one row per respondent: behaviours
--   * clinical_metrics   - one row per respondent: health measures
--
-- Written for SQLite. Run with foreign keys switched on:
--   PRAGMA foreign_keys = ON;
-- =============================================================================

DROP TABLE IF EXISTS clinical_metrics;
DROP TABLE IF EXISTS lifestyle_factors;
DROP TABLE IF EXISTS patients;
DROP TABLE IF EXISTS age_groups;
DROP TABLE IF EXISTS education_levels;
DROP TABLE IF EXISTS income_brackets;

-- -----------------------------------------------------------------------------
-- Lookup tables
-- -----------------------------------------------------------------------------

CREATE TABLE age_groups (
    age_group_id INTEGER PRIMARY KEY,
    label        TEXT    NOT NULL UNIQUE,
    min_age      INTEGER NOT NULL,
    max_age      INTEGER              -- NULL for the open-ended 80+ band
);

INSERT INTO age_groups (age_group_id, label, min_age, max_age) VALUES
    (1,  '18-24', 18, 24),
    (2,  '25-29', 25, 29),
    (3,  '30-34', 30, 34),
    (4,  '35-39', 35, 39),
    (5,  '40-44', 40, 44),
    (6,  '45-49', 45, 49),
    (7,  '50-54', 50, 54),
    (8,  '55-59', 55, 59),
    (9,  '60-64', 60, 64),
    (10, '65-69', 65, 69),
    (11, '70-74', 70, 74),
    (12, '75-79', 75, 79),
    (13, '80+',   80, NULL);

CREATE TABLE education_levels (
    education_id INTEGER PRIMARY KEY,
    label        TEXT NOT NULL UNIQUE
);

INSERT INTO education_levels (education_id, label) VALUES
    (1, 'Never attended school or kindergarten only'),
    (2, 'Grades 1-8'),
    (3, 'Grades 9-11'),
    (4, 'High school graduate or GED'),
    (5, 'Some college or technical school'),
    (6, 'College graduate');

CREATE TABLE income_brackets (
    income_id  INTEGER PRIMARY KEY,
    label      TEXT    NOT NULL UNIQUE,
    min_income INTEGER NOT NULL,     -- US dollars per year
    max_income INTEGER               -- NULL for the open-ended top band
);

INSERT INTO income_brackets (income_id, label, min_income, max_income) VALUES
    (1, 'Under $10,000',      0,     9999),
    (2, '$10,000 - $14,999',  10000, 14999),
    (3, '$15,000 - $19,999',  15000, 19999),
    (4, '$20,000 - $24,999',  20000, 24999),
    (5, '$25,000 - $34,999',  25000, 34999),
    (6, '$35,000 - $49,999',  35000, 49999),
    (7, '$50,000 - $74,999',  50000, 74999),
    (8, '$75,000 or more',    75000, NULL);

-- -----------------------------------------------------------------------------
-- patients: who the respondent is, their access to care, and the outcome
-- -----------------------------------------------------------------------------

CREATE TABLE patients (
    patient_id          INTEGER PRIMARY KEY,
    sex                 TEXT    NOT NULL CHECK (sex IN ('F', 'M')),
    age_group_id        INTEGER NOT NULL REFERENCES age_groups (age_group_id),
    education_id        INTEGER NOT NULL REFERENCES education_levels (education_id),
    income_id           INTEGER NOT NULL REFERENCES income_brackets (income_id),
    has_healthcare      INTEGER NOT NULL CHECK (has_healthcare IN (0, 1)),
    skipped_doctor_cost INTEGER NOT NULL CHECK (skipped_doctor_cost IN (0, 1)),
    has_diabetes        INTEGER NOT NULL CHECK (has_diabetes IN (0, 1))   -- target
);

-- -----------------------------------------------------------------------------
-- lifestyle_factors: behaviours (one row per patient)
-- -----------------------------------------------------------------------------

CREATE TABLE lifestyle_factors (
    patient_id          INTEGER PRIMARY KEY REFERENCES patients (patient_id),
    is_smoker           INTEGER NOT NULL CHECK (is_smoker IN (0, 1)),
    is_physically_active INTEGER NOT NULL CHECK (is_physically_active IN (0, 1)),
    eats_fruit_daily    INTEGER NOT NULL CHECK (eats_fruit_daily IN (0, 1)),
    eats_veg_daily      INTEGER NOT NULL CHECK (eats_veg_daily IN (0, 1)),
    is_heavy_drinker    INTEGER NOT NULL CHECK (is_heavy_drinker IN (0, 1))
);

-- -----------------------------------------------------------------------------
-- clinical_metrics: health measures (one row per patient)
-- -----------------------------------------------------------------------------

CREATE TABLE clinical_metrics (
    patient_id           INTEGER PRIMARY KEY REFERENCES patients (patient_id),
    bmi                  INTEGER NOT NULL CHECK (bmi BETWEEN 10 AND 100),
    has_high_bp          INTEGER NOT NULL CHECK (has_high_bp IN (0, 1)),
    has_high_chol        INTEGER NOT NULL CHECK (has_high_chol IN (0, 1)),
    had_chol_check       INTEGER NOT NULL CHECK (had_chol_check IN (0, 1)),
    had_stroke           INTEGER NOT NULL CHECK (had_stroke IN (0, 1)),
    has_heart_disease    INTEGER NOT NULL CHECK (has_heart_disease IN (0, 1)),
    has_difficulty_walking INTEGER NOT NULL CHECK (has_difficulty_walking IN (0, 1)),
    general_health       INTEGER NOT NULL CHECK (general_health BETWEEN 1 AND 5),      -- 1 excellent ... 5 poor
    poor_mental_days     INTEGER NOT NULL CHECK (poor_mental_days BETWEEN 0 AND 30),   -- in the last 30 days
    poor_physical_days   INTEGER NOT NULL CHECK (poor_physical_days BETWEEN 0 AND 30)  -- in the last 30 days
);

-- -----------------------------------------------------------------------------
-- Indexes on the columns the analysis queries group and filter by
-- -----------------------------------------------------------------------------

CREATE INDEX idx_patients_age_group ON patients (age_group_id);
CREATE INDEX idx_patients_income    ON patients (income_id);
CREATE INDEX idx_patients_diabetes  ON patients (has_diabetes);
