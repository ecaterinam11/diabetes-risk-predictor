# Data dictionary

**Dataset:** CDC Diabetes Health Indicators
([UCI Machine Learning Repository, dataset 891](https://archive.ics.uci.edu/dataset/891/cdc+diabetes+health+indicators))

**Origin:** the CDC's Behavioral Risk Factor Surveillance System (BRFSS) 2015, a
telephone survey of US adults. Every value is **self-reported**.

**Size:** 253,680 rows, 23 columns, no missing values. All columns are integers.

## Target

| Column | Values | Meaning |
|---|---|---|
| `Diabetes_binary` | 0 / 1 | 1 = respondent has been told they have prediabetes or diabetes. 13.9% of rows are 1. |

The survey does not distinguish type 1 from type 2 diabetes. About 90-95% of
adult diabetes cases are type 2, so the target is a reasonable proxy, but this
is a limitation.

## Identifier

| Column | Values | Meaning |
|---|---|---|
| `ID` | 0 - 253,679 | Row identifier, unique. Becomes `patient_id` in the database. |

## Clinical measures

| Column | Values | Meaning |
|---|---|---|
| `HighBP` | 0 / 1 | Told by a health professional they have high blood pressure |
| `HighChol` | 0 / 1 | Told they have high cholesterol |
| `CholCheck` | 0 / 1 | Had a cholesterol check in the last 5 years |
| `BMI` | 12 - 98 | Body mass index (whole number) |
| `Stroke` | 0 / 1 | Ever told they had a stroke |
| `HeartDiseaseorAttack` | 0 / 1 | Coronary heart disease or heart attack |
| `GenHlth` | 1 - 5 | General health: 1 excellent, 2 very good, 3 good, 4 fair, 5 poor |
| `MentHlth` | 0 - 30 | Days of poor mental health in the last 30 |
| `PhysHlth` | 0 - 30 | Days of poor physical health in the last 30 |
| `DiffWalk` | 0 / 1 | Serious difficulty walking or climbing stairs |

## Lifestyle

| Column | Values | Meaning |
|---|---|---|
| `Smoker` | 0 / 1 | Smoked at least 100 cigarettes in their lifetime |
| `PhysActivity` | 0 / 1 | Physical activity in the last 30 days, not counting their job |
| `Fruits` | 0 / 1 | Eats fruit at least once a day |
| `Veggies` | 0 / 1 | Eats vegetables at least once a day |
| `HvyAlcoholConsump` | 0 / 1 | Heavy drinker (men over 14 drinks a week, women over 7) |

## Demographics and access to care

| Column | Values | Meaning |
|---|---|---|
| `Sex` | 0 / 1 | 0 female, 1 male |
| `Age` | 1 - 13 | Five-year age band (see below) |
| `Education` | 1 - 6 | See below |
| `Income` | 1 - 8 | Annual household income band (see below) |
| `AnyHealthcare` | 0 / 1 | Has any kind of health care coverage |
| `NoDocbcCost` | 0 / 1 | Needed a doctor in the last year but could not afford it |

### Age bands

| Code | Age | Code | Age | Code | Age |
|---|---|---|---|---|---|
| 1 | 18-24 | 6 | 45-49 | 11 | 70-74 |
| 2 | 25-29 | 7 | 50-54 | 12 | 75-79 |
| 3 | 30-34 | 8 | 55-59 | 13 | 80+ |
| 4 | 35-39 | 9 | 60-64 | | |
| 5 | 40-44 | 10 | 65-69 | | |

### Education

| Code | Level |
|---|---|
| 1 | Never attended school, or kindergarten only |
| 2 | Grades 1-8 |
| 3 | Grades 9-11 |
| 4 | Grade 12 or GED (high school graduate) |
| 5 | Some college or technical school |
| 6 | College graduate |

### Income (US dollars per year)

| Code | Band | Code | Band |
|---|---|---|---|
| 1 | Under 10,000 | 5 | 25,000 - 34,999 |
| 2 | 10,000 - 14,999 | 6 | 35,000 - 49,999 |
| 3 | 15,000 - 19,999 | 7 | 50,000 - 74,999 |
| 4 | 20,000 - 24,999 | 8 | 75,000 or more |
