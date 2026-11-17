-- =============================================================================
-- 04_user_predictions.sql
-- Logging table for the web app. Written for PostgreSQL (Neon or Supabase).
--
-- Each row is one anonymous prediction made on the website. The table stores
-- only what the model used: no name, no IP address, no exact age, height or
-- weight. A row is written only when the visitor ticks the consent box.
--
-- Run it once against the hosted database:
--   python src/setup_prediction_db.py
-- or paste it into the database provider's SQL editor.
-- =============================================================================

CREATE TABLE IF NOT EXISTS user_predictions (
    prediction_id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    created_at             TIMESTAMPTZ  NOT NULL DEFAULT now(),
    model_version          VARCHAR(20)  NOT NULL,   -- the date the model was trained

    -- the 15 model inputs
    is_male                SMALLINT     NOT NULL CHECK (is_male IN (0, 1)),
    age_group_id           SMALLINT     NOT NULL CHECK (age_group_id BETWEEN 1 AND 13),
    bmi                    NUMERIC(4,1) NOT NULL CHECK (bmi BETWEEN 5 AND 200),
    has_high_bp            SMALLINT     NOT NULL CHECK (has_high_bp IN (0, 1)),
    has_high_chol          SMALLINT     NOT NULL CHECK (has_high_chol IN (0, 1)),
    has_heart_disease      SMALLINT     NOT NULL CHECK (has_heart_disease IN (0, 1)),
    had_stroke             SMALLINT     NOT NULL CHECK (had_stroke IN (0, 1)),
    has_difficulty_walking SMALLINT     NOT NULL CHECK (has_difficulty_walking IN (0, 1)),
    general_health         SMALLINT     NOT NULL CHECK (general_health BETWEEN 1 AND 5),
    poor_physical_days     SMALLINT     NOT NULL CHECK (poor_physical_days BETWEEN 0 AND 30),
    poor_mental_days       SMALLINT     NOT NULL CHECK (poor_mental_days BETWEEN 0 AND 30),
    is_physically_active   SMALLINT     NOT NULL CHECK (is_physically_active IN (0, 1)),
    eats_fruit_daily       SMALLINT     NOT NULL CHECK (eats_fruit_daily IN (0, 1)),
    eats_veg_daily         SMALLINT     NOT NULL CHECK (eats_veg_daily IN (0, 1)),
    is_smoker              SMALLINT     NOT NULL CHECK (is_smoker IN (0, 1)),

    -- the model's output
    predicted_probability  NUMERIC(5,4) NOT NULL CHECK (predicted_probability BETWEEN 0 AND 1),
    risk_level             VARCHAR(10)  NOT NULL CHECK (risk_level IN ('Low', 'Moderate', 'High'))
);

CREATE INDEX IF NOT EXISTS idx_user_predictions_created_at ON user_predictions (created_at);

-- -----------------------------------------------------------------------------
-- v_prediction_summary: what the app's "Usage" tab reads.
-- One row per risk level, with no individual predictions exposed.
-- -----------------------------------------------------------------------------

CREATE OR REPLACE VIEW v_prediction_summary AS
SELECT
    risk_level,
    COUNT(*)                                       AS predictions,
    ROUND(100.0 * AVG(predicted_probability), 1)   AS avg_predicted_pct,
    ROUND(AVG(bmi), 1)                             AS avg_bmi,
    ROUND(100.0 * AVG(has_high_bp), 1)             AS high_bp_pct
FROM user_predictions
GROUP BY risk_level;

-- -----------------------------------------------------------------------------
-- Example monitoring queries (not run by the app)
-- -----------------------------------------------------------------------------

-- Predictions per day, with a running total
-- SELECT
--     created_at::date                                     AS day,
--     COUNT(*)                                             AS predictions,
--     SUM(COUNT(*)) OVER (ORDER BY created_at::date)       AS running_total
-- FROM user_predictions
-- GROUP BY created_at::date
-- ORDER BY day;

-- Do website visitors look like the survey population the model was trained on?
-- (In the survey: 13.9% average risk, 42.9% with high blood pressure, average BMI 28.4)
-- SELECT
--     COUNT(*)                                     AS predictions,
--     ROUND(100.0 * AVG(predicted_probability), 1) AS avg_predicted_pct,
--     ROUND(100.0 * AVG(has_high_bp), 1)           AS high_bp_pct,
--     ROUND(AVG(bmi), 1)                           AS avg_bmi
-- FROM user_predictions;
