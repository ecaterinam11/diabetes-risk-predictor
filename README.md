# Type 2 Diabetes Risk Predictor

A web app that estimates a person's risk of type 2 diabetes from 15 questions about
their health and lifestyle. The data lives in a SQL database, the model's features are
built with SQL views, and the app writes its predictions back to SQL.

<!-- LIVE_DEMO_LINK -->

> **Disclaimer:** This is an educational portfolio project. It is not a medical
> device and is not a substitute for professional medical advice.

## At a glance

| | |
|---|---|
| **Data** | CDC Diabetes Health Indicators (BRFSS 2015): 253,680 US adults, 21 survey questions |
| **SQL** | Normalised SQLite schema with constraints, 4 feature views, 8 analysis queries, a PostgreSQL logging table |
| **Model** | Logistic regression, ROC-AUC 0.818 on 50,736 held-out people, with calibrated probabilities |
| **App** | Streamlit: risk level, probability, and a chart explaining each person's result |
| **Tests** | 28 automated tests covering the prediction logic, the logging and the app itself |

Three decisions shaped the project:

1. **Calibration over class balancing.** Reweighting the classes would have made the app
   tell the average person their risk was 39% when the truth is 14%.
2. **A misleading feature was removed.** Heavy drinkers report less diabetes in the survey,
   so the model was lowering the risk it showed them. Dropping the feature cost 0.001 ROC-AUC.
3. **Interpretability over a small accuracy gain.** XGBoost scored 0.01 higher, but a
   logistic regression lets the app show each person which answers drive their result.

## Architecture

```mermaid
flowchart TD
    csv["CDC survey CSV"] --> load["build_database.py"]
    load --> db[("SQLite: 6 tables")]
    db --> views["SQL views: training_features"]
    db --> analysis["SQL analysis: 8 queries"]
    views --> train["train_model.py"]
    train --> model["pipeline.joblib"]
    model --> app["Streamlit app"]
    app -- "opt-in INSERT" --> pg[("PostgreSQL: user_predictions")]
    pg -- "GROUP BY totals" --> app
```

1. **Load.** The raw CSV is split into normalised tables and validated against the original.
2. **Engineer.** SQL views build the features, so the model reads one `SELECT`.
3. **Analyse.** Eight queries explore the data using CTEs and window functions.
4. **Model.** A logistic regression is trained, evaluated and saved.
5. **Serve.** The app loads the saved model, and logs predictions to PostgreSQL with consent.

## Database schema

The raw data is one flat CSV. [`sql/01_schema.sql`](sql/01_schema.sql) splits it into
three tables linked by `patient_id`, plus three lookup tables that turn the survey's
coded answers into readable labels. Yes/no and numeric columns have `CHECK` constraints on their allowed values.

```mermaid
erDiagram
    age_groups ||--o{ patients : "age_group_id"
    education_levels ||--o{ patients : "education_id"
    income_brackets ||--o{ patients : "income_id"
    patients ||--|| lifestyle_factors : "patient_id"
    patients ||--|| clinical_metrics : "patient_id"

    patients {
        int patient_id PK
        text sex
        int age_group_id FK
        int education_id FK
        int income_id FK
        int has_healthcare
        int skipped_doctor_cost
        int has_diabetes
    }
    lifestyle_factors {
        int patient_id PK
        int is_smoker
        int is_physically_active
        int eats_fruit_daily
        int eats_veg_daily
        int is_heavy_drinker
    }
    clinical_metrics {
        int patient_id PK
        int bmi
        int has_high_bp
        int has_high_chol
        int had_chol_check
        int had_stroke
        int has_heart_disease
        int has_difficulty_walking
        int general_health
        int poor_mental_days
        int poor_physical_days
    }
    age_groups {
        int age_group_id PK
        text label
        int min_age
        int max_age
    }
    education_levels {
        int education_id PK
        text label
    }
    income_brackets {
        int income_id PK
        text label
        int min_income
        int max_income
    }
```

[`src/build_database.py`](src/build_database.py) loads the data and then validates it:
each table holds 253,680 rows, there are no foreign key violations, and joining the
three tables back together reproduces every value in the original CSV.

## Feature engineering in SQL

[`sql/02_views.sql`](sql/02_views.sql) builds the model's features inside the database
rather than in Python:

| View | Purpose |
|---|---|
| `v_patient_profile` | One readable row per patient, with codes replaced by labels |
| `v_engineered_features` | New features built with `CASE WHEN` and arithmetic |
| `v_summary_by_age_band` | Summary statistics for four life-stage age bands |
| `training_features` | The single view the model reads from (`SELECT * FROM training_features`) |

Engineered features: WHO BMI category, life-stage age band, income band, college
graduate flag, frequent physical and mental distress (14 or more unhealthy days in 30),
a count of cardiovascular conditions (0-4) and a count of healthy habits (0-5).

The diabetes rate rises steadily with the number of cardiovascular conditions
(high blood pressure, high cholesterol, heart disease, stroke):

| Conditions | Patients | Diabetes rate |
|---|---|---|
| 0 | 97,968 | 3.7% |
| 1 | 81,002 | 12.2% |
| 2 | 57,010 | 26.5% |
| 3 | 15,217 | 36.4% |
| 4 | 2,483 | 47.4% |

## Analysis in SQL

[`sql/03_analysis.sql`](sql/03_analysis.sql) holds eight queries, and
[`docs/analysis_results.md`](docs/analysis_results.md) holds their full output.

| Query | Question | SQL technique |
|---|---|---|
| Q1 | What does the population look like overall? | Aggregate functions |
| Q2 | How does the diabetes rate differ by age and sex? | Conditional aggregation (pivot) |
| Q3 | How much does risk rise with BMI, relative to a healthy weight? | CTEs with a baseline |
| Q4 | How does the rate change from one age group to the next? | `LAG` and a running `SUM` |
| Q5 | Which yes/no factors matter most? | `UNION ALL` and odds ratios |
| Q6 | How do income and education combine? | Joins, `HAVING`, `RANK` |
| Q7 | Which BMI category is riskiest within each age band? | `ROW_NUMBER` with `PARTITION BY` |
| Q8 | How concentrated is diabetes among people with heart conditions? | Running totals |

Key findings:

- **Blood pressure is the strongest single factor.** People with high blood pressure have
  5 times the odds of diabetes (24.4% against 6.0%).
- **Risk climbs with every BMI category.** Compared with a healthy weight, the rate is 2 times
  higher when overweight and 5.9 times higher in obese class III.
- **Cases are concentrated.** The 29.5% of people with two or more cardiovascular conditions
  account for 61.7% of all diabetes cases.
- **Income and education compound.** The rate is 32.0% for low-income respondents who left
  school by grade 8, and 7.9% for high-income college graduates.
- **The gap between men and women widens with age,** from under 1 point below 50 to 4.9 points at 65 and over.

These are associations in survey data, not causes.

## Model

[`src/train_model.py`](src/train_model.py) reads `training_features` from the database,
holds out 20% of patients as a test set, and compares four models.

| Model | Inputs | ROC-AUC | PR-AUC | Brier score | Average predicted risk |
|---|---|---|---|---|---|
| **Logistic regression (used in the app)** | 15 | 0.818 | 0.406 | 0.099 | 14.0% |
| Logistic regression, all features | 21 | 0.822 | 0.411 | 0.099 | 14.0% |
| Logistic regression, balanced class weights | 15 | 0.818 | 0.405 | 0.178 | 38.7% |
| XGBoost | 21 | 0.828 | 0.424 | 0.097 | 13.9% |

The actual diabetes rate in the test set is 13.9%. Five-fold cross-validation on the
training set gives a ROC-AUC of 0.822 (standard deviation 0.001).

**Why logistic regression.** XGBoost is better by about 0.01 ROC-AUC. That is too small
a gain to give up coefficients that can be read as odds ratios and shown to the user.

**Why 15 inputs.** The app model leaves out six of the 21 survey columns. Adding them all
back improves ROC-AUC by under 0.005.

- *Income, education and three access-to-care questions* are left out because a health
  tool should not need to ask for someone's income.
- *Heavy drinking* is left out because it misleads. Heavy drinkers in the survey report
  less diabetes (5.8% against 14.4%), most likely because people who are diagnosed are told
  to cut down. A model that used it gave heavy drinkers an odds ratio of 0.44, which more
  than halved the risk it showed them.

**Why no class weighting.** Only 13.9% of patients have diabetes. Balancing the classes
leaves the ranking unchanged (ROC-AUC 0.818) but makes the model predict an average risk
of 38.7%, almost three times the truth. The app shows people a probability, so the
probability has to be honest.

![ROC and precision-recall curves](docs/figures/roc_pr_curves.png)

![Calibration plot](docs/figures/calibration.png)

### Risk levels

The app turns the predicted probability into three levels. On the test set:

| Level | Predicted risk | Share of people | Actual diabetes rate | Share of all cases |
|---|---|---|---|---|
| Low | Under 10% | 56.1% | 3.8% | 15.1% |
| Moderate | 10% to 25% | 25.1% | 17.8% | 32.1% |
| High | 25% or more | 18.8% | 39.2% | 52.8% |

### Odds ratios

Each value is the effect of one factor with the other fourteen held constant.

![Odds ratios](docs/figures/odds_ratios.png)

These are smaller than the single-factor odds ratios in the SQL analysis. High blood
pressure falls from 5.0 on its own to 2.1 in the model, because people with high blood
pressure also tend to be older and heavier, and the model credits those separately.

Smoking and diet have almost no effect once the other factors are accounted for. That
describes this survey and should not be read as advice.

## Web app

[`app/app.py`](app/app.py) is a Streamlit app. It asks 15 questions and returns:

- a risk level (low, moderate or high) and the estimated probability
- a chart of which answers raise or lower that person's estimate, compared with the
  average survey respondent
- an "About the model" tab with the accuracy figures, odds ratios and limitations

No answer is pre-selected for the yes/no questions, so the app never assumes one. The
prediction logic lives in [`app/risk_model.py`](app/risk_model.py), separate from the
interface, and is covered by tests.

### Prediction logging

The app writes back to SQL. When a visitor ticks the consent box, their answers and
result are inserted into a `user_predictions` table in PostgreSQL, defined in
[`sql/04_user_predictions.sql`](sql/04_user_predictions.sql). A "Usage" tab reads the
table back with a `GROUP BY` query and shows totals only.

Privacy choices:

- **Opt-in.** The consent box is never pre-ticked, and nothing is stored without it.
- **Minimal.** Only the 15 model inputs and the result are stored. There is no name,
  IP address or session ID, and age is kept as a five-year band, not an exact age.
  Height and weight are not stored, only the BMI worked out from them.
- **Fail-safe.** If the database is unreachable the visitor still gets their result.

With no database configured, the app logs to a local SQLite file, so it runs on any
machine with no setup. To use a hosted database, copy
[`.streamlit/secrets.toml.example`](.streamlit/secrets.toml.example) to
`.streamlit/secrets.toml`, paste in the connection string, and run
`python src/setup_prediction_db.py` to create the table.

## Limitations

- **The data is self-reported, from 2015, and covers US adults only.** Nobody's blood
  sugar was measured.
- **The target is broad.** The survey records prediabetes and diabetes together and does
  not separate type 1 from type 2. Most adult cases are type 2.
- **Associations, not causes.** The model describes who reports diabetes. It cannot say
  that changing a behaviour would change someone's risk.
- **Missing risk factors.** Family history, ethnicity and blood test results are not in the
  data, and doctors rely on all three.
- **Not clinically validated.** The risk levels are cut-offs chosen for this project, not
  a medical standard.

## Run it yourself

Built and tested with Python 3.14. The commands below are for Windows; on macOS or Linux,
activate the environment with `source .venv/bin/activate`.

```bash
py -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
```

The saved model is in the repository, so the app runs straight away:

```bash
streamlit run app/app.py
```

To rebuild everything from the raw data:

```bash
python src/download_data.py
python src/build_database.py
python src/run_analysis.py
python src/train_model.py
python -m pytest
```

## Deployment

The app is set up for [Streamlit Community Cloud](https://streamlit.io/cloud):

- **Main file:** `app/app.py`
- **Python version:** 3.14, set under "Advanced settings" when deploying.
- **Dependencies:** `requirements.txt` holds only what the app needs, with pinned versions,
  so the hosted app loads the model with the same scikit-learn version that saved it.
- **Database:** add the PostgreSQL connection string as a secret named `DATABASE_URL`.
  Without it the app still runs, and saves predictions to a temporary file.

## Repository layout

| Path | Contents |
|---|---|
| [`sql/`](sql/) | Schema, feature views, analysis queries and the logging table |
| [`src/`](src/) | Scripts that download the data, build the database, run the analysis and train the model |
| [`app/`](app/) | Streamlit app, prediction logic and prediction logging |
| [`models/`](models/) | Saved model pipeline and its metrics |
| [`notebooks/`](notebooks/) | Data exploration notebook |
| [`docs/`](docs/) | Data dictionary, analysis results and model figures |
| [`tests/`](tests/) | Automated tests |
| `data/` | Raw CSV and SQLite database (rebuilt by the scripts, not committed) |

## Data source

CDC Diabetes Health Indicators, from the
[UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators),
derived from the CDC's Behavioral Risk Factor Surveillance System 2015 survey.
Column definitions are in the [data dictionary](docs/data_dictionary.md).
