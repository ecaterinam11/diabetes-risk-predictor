-- =============================================================================
-- 02_views.sql
-- Feature engineering in SQL. Run after 01_schema.sql and the data load.
--
--   v_patient_profile      one readable row per patient, codes replaced by labels
--   v_engineered_features  new features built with CASE WHEN and arithmetic
--   v_summary_by_age_band  summary statistics for each life-stage age band
--   training_features      the single view the model reads its data from
-- =============================================================================

DROP VIEW IF EXISTS training_features;
DROP VIEW IF EXISTS v_summary_by_age_band;
DROP VIEW IF EXISTS v_engineered_features;
DROP VIEW IF EXISTS v_patient_profile;

-- -----------------------------------------------------------------------------
-- v_patient_profile
-- Joins the three patient tables to the lookup tables, so a person reading the
-- data sees '60-64' and 'College graduate' instead of 9 and 6.
-- -----------------------------------------------------------------------------

CREATE VIEW v_patient_profile AS
SELECT
    p.patient_id,
    p.has_diabetes,
    p.sex,
    a.label  AS age_group,
    e.label  AS education,
    i.label  AS income,
    p.has_healthcare,
    p.skipped_doctor_cost,
    l.is_smoker,
    l.is_physically_active,
    l.eats_fruit_daily,
    l.eats_veg_daily,
    l.is_heavy_drinker,
    c.bmi,
    c.has_high_bp,
    c.has_high_chol,
    c.had_chol_check,
    c.had_stroke,
    c.has_heart_disease,
    c.has_difficulty_walking,
    c.general_health,
    c.poor_mental_days,
    c.poor_physical_days
FROM patients AS p
JOIN age_groups        AS a ON a.age_group_id = p.age_group_id
JOIN education_levels  AS e ON e.education_id = p.education_id
JOIN income_brackets   AS i ON i.income_id    = p.income_id
JOIN lifestyle_factors AS l ON l.patient_id   = p.patient_id
JOIN clinical_metrics  AS c ON c.patient_id   = p.patient_id;

-- -----------------------------------------------------------------------------
-- v_engineered_features
-- New features derived from the raw columns.
-- -----------------------------------------------------------------------------

CREATE VIEW v_engineered_features AS
SELECT
    p.patient_id,

    -- WHO body mass index categories. BMI is stored as a whole number,
    -- so "under 18.5" means 18 or less and "25 to 29.9" means 25 to 29.
    CASE
        WHEN c.bmi <= 18 THEN 'Underweight'
        WHEN c.bmi <= 24 THEN 'Healthy weight'
        WHEN c.bmi <= 29 THEN 'Overweight'
        WHEN c.bmi <= 34 THEN 'Obese class I'
        WHEN c.bmi <= 39 THEN 'Obese class II'
        ELSE                  'Obese class III'
    END AS bmi_category,

    -- The same categories as an ordered number, for sorting and modelling
    CASE
        WHEN c.bmi <= 18 THEN 1
        WHEN c.bmi <= 24 THEN 2
        WHEN c.bmi <= 29 THEN 3
        WHEN c.bmi <= 34 THEN 4
        WHEN c.bmi <= 39 THEN 5
        ELSE                  6
    END AS bmi_category_rank,

    -- Thirteen five-year age groups collapsed into four life stages
    CASE
        WHEN a.min_age < 35 THEN '18-34'
        WHEN a.min_age < 50 THEN '35-49'
        WHEN a.min_age < 65 THEN '50-64'
        ELSE                     '65+'
    END AS age_band,

    -- Eight income brackets collapsed into three
    CASE
        WHEN i.min_income < 25000 THEN 'Low (under $25k)'
        WHEN i.min_income < 50000 THEN 'Middle ($25k-$50k)'
        ELSE                           'High ($50k+)'
    END AS income_band,

    CASE WHEN p.education_id = 6 THEN 1 ELSE 0 END AS is_college_graduate,

    CASE c.general_health
        WHEN 1 THEN 'Excellent'
        WHEN 2 THEN 'Very good'
        WHEN 3 THEN 'Good'
        WHEN 4 THEN 'Fair'
        ELSE        'Poor'
    END AS general_health_label,

    -- 14 or more unhealthy days in 30 is the CDC's definition of "frequent distress"
    CASE WHEN c.poor_physical_days >= 14 THEN 1 ELSE 0 END AS has_frequent_physical_distress,
    CASE WHEN c.poor_mental_days   >= 14 THEN 1 ELSE 0 END AS has_frequent_mental_distress,

    -- How many of four cardiovascular conditions the patient has (0 to 4)
    c.has_high_bp + c.has_high_chol + c.has_heart_disease + c.had_stroke
        AS cardio_condition_count,

    -- How many of five healthy habits the patient has (0 to 5)
    l.is_physically_active
        + l.eats_fruit_daily
        + l.eats_veg_daily
        + (1 - l.is_smoker)
        + (1 - l.is_heavy_drinker)
        AS healthy_habit_count
FROM patients AS p
JOIN age_groups        AS a ON a.age_group_id = p.age_group_id
JOIN income_brackets   AS i ON i.income_id    = p.income_id
JOIN lifestyle_factors AS l ON l.patient_id   = p.patient_id
JOIN clinical_metrics  AS c ON c.patient_id   = p.patient_id;

-- -----------------------------------------------------------------------------
-- v_summary_by_age_band
-- Summary statistics for each life-stage age band.
-- -----------------------------------------------------------------------------

CREATE VIEW v_summary_by_age_band AS
SELECT
    f.age_band,
    COUNT(*)                                        AS patients,
    ROUND(100.0 * AVG(p.has_diabetes), 1)           AS diabetes_pct,
    ROUND(AVG(c.bmi), 1)                            AS avg_bmi,
    ROUND(100.0 * AVG(f.bmi_category_rank >= 4), 1) AS obese_pct,
    ROUND(100.0 * AVG(c.has_high_bp), 1)            AS high_bp_pct,
    ROUND(100.0 * AVG(c.has_high_chol), 1)          AS high_chol_pct,
    ROUND(100.0 * AVG(l.is_physically_active), 1)   AS active_pct,
    ROUND(AVG(f.cardio_condition_count), 2)         AS avg_cardio_conditions,
    ROUND(AVG(f.healthy_habit_count), 2)            AS avg_healthy_habits
FROM patients AS p
JOIN v_engineered_features AS f ON f.patient_id = p.patient_id
JOIN lifestyle_factors     AS l ON l.patient_id = p.patient_id
JOIN clinical_metrics      AS c ON c.patient_id = p.patient_id
GROUP BY f.age_band;

-- -----------------------------------------------------------------------------
-- training_features
-- Everything the model needs, one row per patient, in a single SELECT.
-- Raw columns first, engineered columns after. The modelling step chooses which
-- columns to use: the two counts are sums of other columns here, so a linear
-- model should use either the counts or their components, not both.
-- -----------------------------------------------------------------------------

CREATE VIEW training_features AS
SELECT
    p.patient_id,
    p.has_diabetes,                                   -- target

    -- demographics and access to care
    CASE p.sex WHEN 'M' THEN 1 ELSE 0 END AS is_male,
    p.age_group_id,
    p.education_id,
    p.income_id,
    p.has_healthcare,
    p.skipped_doctor_cost,

    -- lifestyle
    l.is_smoker,
    l.is_physically_active,
    l.eats_fruit_daily,
    l.eats_veg_daily,
    l.is_heavy_drinker,

    -- clinical
    c.bmi,
    c.has_high_bp,
    c.has_high_chol,
    c.had_chol_check,
    c.had_stroke,
    c.has_heart_disease,
    c.has_difficulty_walking,
    c.general_health,
    c.poor_mental_days,
    c.poor_physical_days,

    -- engineered
    f.bmi_category,
    f.bmi_category_rank,
    f.age_band,
    f.income_band,
    f.is_college_graduate,
    f.has_frequent_physical_distress,
    f.has_frequent_mental_distress,
    f.cardio_condition_count,
    f.healthy_habit_count
FROM patients AS p
JOIN lifestyle_factors     AS l ON l.patient_id = p.patient_id
JOIN clinical_metrics      AS c ON c.patient_id = p.patient_id
JOIN v_engineered_features AS f ON f.patient_id = p.patient_id;
