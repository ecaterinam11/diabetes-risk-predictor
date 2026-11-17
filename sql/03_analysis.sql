-- =============================================================================
-- 03_analysis.sql
-- Analysis queries on the diabetes database. Run after 01_schema.sql, the data
-- load and 02_views.sql.
--
-- Each query starts with a line of the form "-- Qn: title". The script
-- src/run_analysis.py uses those lines to run every query and save the results
-- to docs/analysis_results.md.
-- =============================================================================

-- Q1: Overall picture
-- Technique: aggregate functions.
SELECT
    COUNT(*)                              AS patients,
    SUM(has_diabetes)                     AS with_diabetes,
    ROUND(100.0 * AVG(has_diabetes), 1)   AS diabetes_pct,
    ROUND(AVG(bmi), 1)                    AS avg_bmi,
    ROUND(100.0 * AVG(has_high_bp), 1)    AS high_bp_pct,
    ROUND(100.0 * AVG(has_high_chol), 1)  AS high_chol_pct
FROM training_features;

-- Q2: Diabetes rate by age band, split by sex
-- Technique: GROUP BY with conditional aggregation to pivot sex into columns.
SELECT
    age_band,
    COUNT(*)                                                           AS patients,
    ROUND(100.0 * AVG(CASE WHEN is_male = 0 THEN has_diabetes END), 1) AS women_pct,
    ROUND(100.0 * AVG(CASE WHEN is_male = 1 THEN has_diabetes END), 1) AS men_pct,
    ROUND(100.0 * AVG(CASE WHEN is_male = 1 THEN has_diabetes END)
        - 100.0 * AVG(CASE WHEN is_male = 0 THEN has_diabetes END), 1) AS gap_points
FROM training_features
GROUP BY age_band
ORDER BY age_band;

-- Q3: Risk by BMI category, relative to a healthy weight
-- Technique: CTEs, with one CTE used as a baseline for the others.
WITH by_bmi AS (
    SELECT
        bmi_category_rank,
        bmi_category,
        COUNT(*)          AS patients,
        AVG(has_diabetes) AS diabetes_rate
    FROM training_features
    GROUP BY bmi_category_rank, bmi_category
),
baseline AS (
    SELECT diabetes_rate AS healthy_rate
    FROM by_bmi
    WHERE bmi_category = 'Healthy weight'
)
SELECT
    b.bmi_category,
    b.patients,
    ROUND(100.0 * b.diabetes_rate, 1)             AS diabetes_pct,
    ROUND(b.diabetes_rate / base.healthy_rate, 2) AS risk_vs_healthy_weight
FROM by_bmi AS b
CROSS JOIN baseline AS base
ORDER BY b.bmi_category_rank;

-- Q4: How the rate changes from one age group to the next
-- Technique: window functions - LAG for the change, running SUM for the cumulative share.
WITH by_age AS (
    SELECT
        a.age_group_id,
        a.label             AS age_group,
        COUNT(*)            AS patients,
        SUM(p.has_diabetes) AS cases
    FROM patients AS p
    JOIN age_groups AS a ON a.age_group_id = p.age_group_id
    GROUP BY a.age_group_id, a.label
)
SELECT
    age_group,
    patients,
    ROUND(100.0 * cases / patients, 1) AS diabetes_pct,
    ROUND(100.0 * cases / patients
        - LAG(100.0 * cases / patients) OVER (ORDER BY age_group_id), 1)
                                       AS change_from_previous,
    ROUND(100.0 * SUM(cases) OVER (ORDER BY age_group_id)
        / SUM(cases) OVER (), 1)       AS cumulative_pct_of_all_cases
FROM by_age
ORDER BY age_group_id;

-- Q5: Which yes/no factors matter most
-- Technique: UNION ALL to stack factors into one long table, then an odds ratio.
WITH factors AS (
    SELECT 'High blood pressure'    AS factor, has_high_bp            AS present, has_diabetes FROM training_features
    UNION ALL SELECT 'High cholesterol',       has_high_chol,          has_diabetes FROM training_features
    UNION ALL SELECT 'Heart disease',          has_heart_disease,      has_diabetes FROM training_features
    UNION ALL SELECT 'Stroke',                 had_stroke,             has_diabetes FROM training_features
    UNION ALL SELECT 'Difficulty walking',     has_difficulty_walking, has_diabetes FROM training_features
    UNION ALL SELECT 'Smoker',                 is_smoker,              has_diabetes FROM training_features
    UNION ALL SELECT 'Physically active',      is_physically_active,   has_diabetes FROM training_features
    UNION ALL SELECT 'Heavy drinker',          is_heavy_drinker,       has_diabetes FROM training_features
    UNION ALL SELECT 'Eats fruit daily',       eats_fruit_daily,       has_diabetes FROM training_features
    UNION ALL SELECT 'Eats vegetables daily',  eats_veg_daily,         has_diabetes FROM training_features
),
counts AS (
    SELECT
        factor,
        SUM(present = 1 AND has_diabetes = 1) AS yes_cases,
        SUM(present = 1 AND has_diabetes = 0) AS yes_healthy,
        SUM(present = 0 AND has_diabetes = 1) AS no_cases,
        SUM(present = 0 AND has_diabetes = 0) AS no_healthy
    FROM factors
    GROUP BY factor
)
SELECT
    factor,
    ROUND(100.0 * (yes_cases + yes_healthy)
        / (yes_cases + yes_healthy + no_cases + no_healthy), 1)   AS pct_of_patients_with_factor,
    ROUND(100.0 * yes_cases / (yes_cases + yes_healthy), 1)       AS diabetes_pct_if_yes,
    ROUND(100.0 * no_cases / (no_cases + no_healthy), 1)          AS diabetes_pct_if_no,
    ROUND(1.0 * yes_cases * no_healthy / (yes_healthy * no_cases), 2) AS odds_ratio
FROM counts
ORDER BY odds_ratio DESC;

-- Q6: The income and education gradient
-- Technique: joins to lookup tables, HAVING to drop small groups, RANK to order them.
WITH groups AS (
    SELECT
        t.income_band,
        e.label                              AS education,
        COUNT(*)                             AS patients,
        ROUND(100.0 * AVG(t.has_diabetes), 1) AS diabetes_pct
    FROM training_features AS t
    JOIN education_levels  AS e ON e.education_id = t.education_id
    GROUP BY t.income_band, e.education_id, e.label
    HAVING COUNT(*) >= 1000
)
SELECT
    RANK() OVER (ORDER BY diabetes_pct DESC) AS risk_rank,
    income_band,
    education,
    patients,
    diabetes_pct
FROM groups
ORDER BY risk_rank;

-- Q7: The highest-risk BMI category within each age band
-- Technique: ROW_NUMBER with PARTITION BY to pick the top rows per group.
WITH cells AS (
    SELECT
        age_band,
        bmi_category,
        COUNT(*)                            AS patients,
        ROUND(100.0 * AVG(has_diabetes), 1) AS diabetes_pct
    FROM training_features
    GROUP BY age_band, bmi_category
    HAVING COUNT(*) >= 500
),
ranked AS (
    SELECT
        *,
        ROW_NUMBER() OVER (PARTITION BY age_band ORDER BY diabetes_pct DESC) AS rank_in_band,
        ROUND(diabetes_pct - AVG(diabetes_pct) OVER (PARTITION BY age_band), 1)
                                                                              AS points_above_band_average
    FROM cells
)
SELECT age_band, rank_in_band, bmi_category, patients, diabetes_pct, points_above_band_average
FROM ranked
WHERE rank_in_band <= 2
ORDER BY age_band, rank_in_band;

-- Q8: How concentrated diabetes is among people with cardiovascular conditions
-- Technique: running totals with window functions, counting from the highest-risk group down.
WITH by_count AS (
    SELECT
        cardio_condition_count,
        COUNT(*)          AS patients,
        SUM(has_diabetes) AS cases
    FROM training_features
    GROUP BY cardio_condition_count
)
SELECT
    cardio_condition_count,
    patients,
    ROUND(100.0 * cases / patients, 1) AS diabetes_pct,
    ROUND(100.0 * SUM(patients) OVER (ORDER BY cardio_condition_count DESC)
        / SUM(patients) OVER (), 1)    AS cumulative_pct_of_patients,
    ROUND(100.0 * SUM(cases) OVER (ORDER BY cardio_condition_count DESC)
        / SUM(cases) OVER (), 1)       AS cumulative_pct_of_cases
FROM by_count
ORDER BY cardio_condition_count DESC;
