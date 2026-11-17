"""Prediction logic for the web app, kept separate from the user interface.

Nothing in this file imports Streamlit, so it can be tested on its own.
"""

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

MODEL_DIR = Path(__file__).resolve().parents[1] / "models"

GENERAL_HEALTH = {"Excellent": 1, "Very good": 2, "Good": 3, "Fair": 4, "Poor": 5}

# For each yes/no input: (wording when the answer is yes, wording when it is no)
BINARY_LABELS = {
    "is_male": ("Being male", "Being female"),
    "is_smoker": ("Having smoked", "Never having smoked"),
    "is_physically_active": ("Being physically active", "Not being physically active"),
    "eats_fruit_daily": ("Eating fruit daily", "Not eating fruit daily"),
    "eats_veg_daily": ("Eating vegetables daily", "Not eating vegetables daily"),
    "is_heavy_drinker": ("Heavy drinking", "Not drinking heavily"),
    "has_high_bp": ("High blood pressure", "No high blood pressure"),
    "has_high_chol": ("High cholesterol", "No high cholesterol"),
    "had_stroke": ("A previous stroke", "No previous stroke"),
    "has_heart_disease": ("Heart disease", "No heart disease"),
    "has_difficulty_walking": ("Difficulty walking", "No difficulty walking"),
}
NUMERIC_LABELS = {
    "general_health": "Your general health rating",
    "poor_mental_days": "Your days of poor mental health",
    "poor_physical_days": "Your days of poor physical health",
}


@dataclass
class Prediction:
    probability: float          # between 0 and 1
    level: str                  # "Low", "Moderate" or "High"
    contributions: pd.DataFrame  # columns: factor, log_odds, odds_multiplier


def load_model(model_dir: Path = MODEL_DIR):
    """Return the fitted pipeline and the details saved with it."""
    pipeline = joblib.load(model_dir / "pipeline.joblib")
    info = json.loads((model_dir / "model_info.json").read_text(encoding="utf-8"))
    return pipeline, info


def age_to_group(age: int) -> int:
    """Turn an age in years into the survey's age group code (1 = 18-24 ... 13 = 80+)."""
    if age < 18:
        raise ValueError("The model only covers adults aged 18 and over.")
    if age < 25:
        return 1
    return min(13, (age - 25) // 5 + 2)


def bmi_from(height_cm: float, weight_kg: float) -> float:
    return weight_kg / (height_cm / 100) ** 2


def bmi_category(bmi: float) -> str:
    """WHO categories, matching the CASE WHEN in sql/02_views.sql."""
    if bmi < 18.5:
        return "Underweight"
    if bmi < 25:
        return "Healthy weight"
    if bmi < 30:
        return "Overweight"
    if bmi < 35:
        return "Obese class I"
    if bmi < 40:
        return "Obese class II"
    return "Obese class III"


def risk_level(probability: float, bands: dict) -> str:
    """bands maps each level to the lowest probability that falls in it."""
    level = None
    for name, lower in sorted(bands.items(), key=lambda item: item[1]):
        if probability >= lower:
            level = name
    return level


def predict(pipeline, info: dict, answers: dict) -> Prediction:
    """Score one person.

    `answers` holds one value for each name in info["features"].
    """
    row = pd.DataFrame([answers])[info["features"]]
    probability = float(pipeline.predict_proba(row)[0, 1])

    # A logistic regression adds up one term per feature on the log-odds scale.
    # Each term here is measured against the average survey respondent.
    transformed = np.asarray(pipeline.named_steps["prep"].transform(row)).ravel()
    coefficients = pipeline.named_steps["model"].coef_[0]
    terms = coefficients * (transformed - np.asarray(info["feature_baseline"]))

    binary, numeric = info["binary_features"], info["numeric_features"]
    rows, i = [], 0
    for name in binary:
        yes, no = BINARY_LABELS[name]
        rows.append((yes if answers[name] else no, terms[i]))
        i += 1
    for name in numeric:
        rows.append((NUMERIC_LABELS[name], terms[i]))
        i += 1
    rows.append(("Your BMI", terms[i]))
    i += 1
    rows.append(("Your age", terms[i:].sum()))  # the remaining terms are the age groups

    contributions = pd.DataFrame(rows, columns=["factor", "log_odds"])
    contributions["odds_multiplier"] = np.exp(contributions["log_odds"])
    contributions = contributions.sort_values("log_odds", ascending=False).reset_index(drop=True)

    return Prediction(probability, risk_level(probability, info["risk_bands"]), contributions)
