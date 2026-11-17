"""End-to-end tests that run the Streamlit app without a browser."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app" / "app.py")
YES_NO_KEYS = ["high_bp", "high_chol", "heart", "stroke", "walking", "active", "fruit", "veg", "smoker", "drinker"]


@pytest.fixture
def app():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    return at


def submit(at):
    return at.button[0].click().run()


def test_page_loads_with_disclaimer(app):
    assert app.title[0].value == "Type 2 Diabetes Risk Estimator"
    assert "not medical advice" in app.warning[0].value
    assert len(app.metric) == 3  # only the "About the model" metrics, no result yet


def test_unanswered_questions_block_the_result(app):
    submit(app)
    assert not app.exception
    assert "Please answer every question" in app.error[0].value
    assert "Risk level" not in [m.label for m in app.metric]


def test_high_risk_profile(app):
    app.radio(key="sex").set_value("Male")
    app.number_input(key="age").set_value(67)
    app.number_input(key="weight").set_value(98)
    app.select_slider(key="general").set_value("Fair")
    for key in YES_NO_KEYS:
        app.radio(key=key).set_value("Yes" if key in ("high_bp", "high_chol") else "No")
    submit(app)

    assert not app.exception and not app.error
    metrics = {m.label: m.value for m in app.metric}
    assert metrics["Risk level"] == "High"
    assert metrics["Survey average"] == "14%"


def test_low_risk_profile(app):
    app.radio(key="sex").set_value("Female")
    app.number_input(key="age").set_value(27)
    app.number_input(key="weight").set_value(62)
    app.select_slider(key="general").set_value("Very good")
    for key in YES_NO_KEYS:
        app.radio(key=key).set_value("Yes" if key in ("active", "fruit", "veg") else "No")
    submit(app)

    assert not app.exception and not app.error
    assert {m.label: m.value for m in app.metric}["Risk level"] == "Low"
