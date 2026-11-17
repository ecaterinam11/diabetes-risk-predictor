"""Tests for the app's prediction logic.

Run from the project root:
    python -m pytest
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app"))

from risk_model import age_to_group, bmi_category, bmi_from, load_model, predict, risk_level  # noqa: E402

HEALTHY = dict(
    is_male=0, is_smoker=0, is_physically_active=1, eats_fruit_daily=1, eats_veg_daily=1,
    is_heavy_drinker=0, has_high_bp=0, has_high_chol=0, had_stroke=0, has_heart_disease=0,
    has_difficulty_walking=0, general_health=2, poor_mental_days=0, poor_physical_days=0,
    bmi=22.0, age_group_id=2,
)
HIGH_RISK = HEALTHY | dict(
    age_group_id=10, bmi=34.0, has_high_bp=1, has_high_chol=1, general_health=4, is_physically_active=0,
)


@pytest.fixture(scope="module")
def model():
    return load_model()


@pytest.mark.parametrize("age, group", [(18, 1), (24, 1), (25, 2), (29, 2), (30, 3), (64, 9), (65, 10), (79, 12), (80, 13), (101, 13)])
def test_age_to_group(age, group):
    assert age_to_group(age) == group


def test_age_under_18_is_rejected():
    with pytest.raises(ValueError):
        age_to_group(17)


def test_bmi():
    assert bmi_from(170, 75) == pytest.approx(25.95, abs=0.01)
    assert bmi_category(18.4) == "Underweight"
    assert bmi_category(24.9) == "Healthy weight"
    assert bmi_category(25.0) == "Overweight"
    assert bmi_category(41.0) == "Obese class III"


def test_risk_level_boundaries():
    bands = {"Low": 0.0, "Moderate": 0.10, "High": 0.25}
    assert risk_level(0.0999, bands) == "Low"
    assert risk_level(0.10, bands) == "Moderate"
    assert risk_level(0.2499, bands) == "Moderate"
    assert risk_level(0.25, bands) == "High"


def test_predictions_are_sensible(model):
    pipeline, info = model
    healthy = predict(pipeline, info, HEALTHY)
    high = predict(pipeline, info, HIGH_RISK)
    assert healthy.level == "Low" and healthy.probability < 0.02
    assert high.level == "High" and high.probability > 0.40


def test_contributions_add_up_to_the_prediction(model):
    """Baseline log-odds plus every contribution must equal the model's own log-odds."""
    pipeline, info = model
    clf = pipeline.named_steps["model"]
    baseline_log_odds = clf.intercept_[0] + clf.coef_[0] @ np.asarray(info["feature_baseline"])
    for person in (HEALTHY, HIGH_RISK):
        result = predict(pipeline, info, person)
        log_odds = baseline_log_odds + result.contributions["log_odds"].sum()
        assert 1 / (1 + np.exp(-log_odds)) == pytest.approx(result.probability, abs=1e-4)


def test_bmi_is_capped(model):
    pipeline, info = model
    at_cap = predict(pipeline, info, HIGH_RISK | {"bmi": float(info["bmi_cap"])})
    above = predict(pipeline, info, HIGH_RISK | {"bmi": 90.0})
    assert above.probability == pytest.approx(at_cap.probability)
