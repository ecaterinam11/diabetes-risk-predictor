"""End-to-end tests that run the Streamlit app without a browser."""

from pathlib import Path

import pytest
import streamlit as st
from sqlalchemy import create_engine, text
from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / "app" / "app.py")
YES_NO_KEYS = ["high_bp", "high_chol", "heart", "stroke", "walking", "active", "fruit", "veg", "smoker"]


@pytest.fixture
def log_url(tmp_path, monkeypatch):
    """Point the app's prediction log at a throwaway database for each test."""
    url = f"sqlite:///{(tmp_path / 'log.db').as_posix()}"
    monkeypatch.setenv("DATABASE_URL", url)
    st.cache_resource.clear()
    st.cache_data.clear()
    yield url
    st.cache_resource.clear()
    st.cache_data.clear()


@pytest.fixture
def app(log_url):
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    return at


def logged_rows(url: str) -> list:
    engine = create_engine(url)
    with engine.connect() as conn:
        rows = conn.execute(text("SELECT risk_level, age_group_id, bmi FROM user_predictions")).fetchall()
    engine.dispose()
    return rows


def fill(at, sex, age, weight, general, yes):
    at.radio(key="sex").set_value(sex)
    at.number_input(key="age").set_value(age)
    at.number_input(key="weight").set_value(weight)
    at.select_slider(key="general").set_value(general)
    for key in YES_NO_KEYS:
        at.radio(key=key).set_value("Yes" if key in yes else "No")


def submit(at):
    return at.button[0].click().run()


def test_page_loads_with_disclaimer(app):
    assert app.title[0].value == "Type 2 Diabetes Risk Estimator"
    assert "not medical advice" in app.warning[0].value
    assert [t.label for t in app.tabs] == ["Estimate my risk", "About the model", "Usage"]
    assert "Risk level" not in [m.label for m in app.metric]


def test_unanswered_questions_block_the_result(app, log_url):
    app.checkbox(key="consent").check()
    submit(app)
    assert not app.exception
    assert "Please answer every question" in app.error[0].value
    assert "Risk level" not in [m.label for m in app.metric]
    assert logged_rows(log_url) == []


def test_high_risk_profile(app):
    fill(app, "Male", 67, 98, "Fair", yes=("high_bp", "high_chol"))
    submit(app)
    assert not app.exception and not app.error
    metrics = {m.label: m.value for m in app.metric}
    assert metrics["Risk level"] == "High"
    assert metrics["Survey average"] == "14%"


def test_low_risk_profile(app):
    fill(app, "Female", 27, 62, "Very good", yes=("active", "fruit", "veg"))
    submit(app)
    assert not app.exception and not app.error
    assert {m.label: m.value for m in app.metric}["Risk level"] == "Low"


def test_nothing_is_saved_without_consent(app, log_url):
    assert app.checkbox(key="consent").value is False  # opt-in, never pre-ticked
    fill(app, "Male", 67, 98, "Fair", yes=("high_bp", "high_chol"))
    submit(app)
    assert not app.exception
    assert logged_rows(log_url) == []


def test_prediction_is_saved_with_consent(app, log_url):
    fill(app, "Male", 67, 98, "Fair", yes=("high_bp", "high_chol"))
    app.checkbox(key="consent").check()
    submit(app)
    assert not app.exception
    assert any("saved anonymously" in c.value for c in app.caption)

    rows = logged_rows(log_url)
    assert len(rows) == 1
    level, age_group, bmi = rows[0]
    assert level == "High" and age_group == 10 and float(bmi) == 33.9

    # the Usage tab reads the new row back
    assert {m.label: m.value for m in app.metric}["Predictions saved"] == "1"
