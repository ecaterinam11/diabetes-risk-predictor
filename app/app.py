"""Type 2 diabetes risk estimator - Streamlit web app.

Run from the project root:
    streamlit run app/app.py
"""

from pathlib import Path

import altair as alt
import pandas as pd
import streamlit as st

from prediction_log import database_url, log_prediction, make_engine, usage_summary
from risk_model import (
    GENERAL_HEALTH,
    age_to_group,
    bmi_category,
    bmi_from,
    load_model,
    predict,
)

ROOT = Path(__file__).resolve().parents[1]
REPO_URL = "https://github.com/ecaterinam11/diabetes-risk-predictor"
RAISES, LOWERS = "#eb6834", "#2a78d6"
LEVEL_NOTES = {
    "Low": "Your estimated risk is below 10%.",
    "Moderate": "Your estimated risk is between 10% and 25%.",
    "High": "Your estimated risk is 25% or more.",
}

st.set_page_config(page_title="Diabetes Risk Estimator", page_icon="🩺", layout="centered")


@st.cache_resource
def get_model():
    return load_model()


@st.cache_resource
def get_log_engine():
    """Connect to the prediction log: hosted PostgreSQL if configured, local SQLite if not."""
    try:
        secret = st.secrets.get("DATABASE_URL")
    except Exception:  # no secrets file at all
        secret = None
    return make_engine(database_url(secret))


def yes_no(label: str, key: str, help: str | None = None) -> int | None:
    """A yes/no question with nothing pre-selected, so no answer is assumed."""
    answer = st.radio(label, ["No", "Yes"], index=None, horizontal=True, key=key, help=help)
    return None if answer is None else int(answer == "Yes")


def input_form() -> dict | None:
    """Show the questionnaire. Returns the answers once the form is submitted."""
    with st.form("risk_form"):
        st.subheader("About you")
        col1, col2 = st.columns(2)
        sex = col1.radio("Sex", ["Female", "Male"], index=None, horizontal=True, key="sex")
        age = col2.number_input("Age", min_value=18, max_value=110, value=45, step=1, key="age")
        col1, col2 = st.columns(2)
        height = col1.number_input("Height (cm)", min_value=120, max_value=220, value=170, step=1, key="height")
        weight = col2.number_input("Weight (kg)", min_value=30, max_value=250, value=75, step=1, key="weight")

        st.subheader("Your health")
        col1, col2 = st.columns(2)
        with col1:
            high_bp = yes_no("High blood pressure?", "high_bp", "Have you been told by a health professional that you have it?")
            high_chol = yes_no("High cholesterol?", "high_chol", "Have you been told by a health professional that you have it?")
            heart = yes_no("Heart disease or a heart attack?", "heart")
        with col2:
            stroke = yes_no("Ever had a stroke?", "stroke")
            walking = yes_no("Serious difficulty walking or climbing stairs?", "walking")
        general = st.select_slider(
            "In general, how would you rate your health?",
            options=list(GENERAL_HEALTH), value="Good", key="general",
        )
        col1, col2 = st.columns(2)
        physical_days = col1.slider("Days of poor physical health in the last 30", 0, 30, 0, key="physical_days")
        mental_days = col2.slider("Days of poor mental health in the last 30", 0, 30, 0, key="mental_days")

        st.subheader("Your lifestyle")
        col1, col2 = st.columns(2)
        with col1:
            active = yes_no("Any exercise in the last 30 days?", "active", "Physical activity outside of your job.")
            fruit = yes_no("Fruit at least once a day?", "fruit")
            veg = yes_no("Vegetables at least once a day?", "veg")
        with col2:
            smoker = yes_no("Smoked 100+ cigarettes in your life?", "smoker", "That is about 5 packs in total.")

        consent = st.checkbox(
            "Save my answers anonymously to help improve this project (optional)",
            value=False, key="consent",
            help="Stores only the answers above and your result. No name, IP address, exact age, "
                 "height or weight is saved.",
        )
        submitted = st.form_submit_button("Estimate my risk", type="primary", width="stretch")

    if not submitted:
        return None
    yes_no_answers = [high_bp, high_chol, heart, stroke, walking, active, fruit, veg, smoker]
    missing = sum(a is None for a in yes_no_answers) + (sex is None)
    if missing:
        st.error(f"Please answer every question. {missing} still need{'s' if missing == 1 else ''} an answer.")
        return None
    return {
        "is_male": int(sex == "Male"),
        "age": int(age),
        "age_group_id": age_to_group(int(age)),
        "bmi": bmi_from(height, weight),
        "has_high_bp": high_bp,
        "has_high_chol": high_chol,
        "has_heart_disease": heart,
        "had_stroke": stroke,
        "has_difficulty_walking": walking,
        "general_health": GENERAL_HEALTH[general],
        "poor_physical_days": physical_days,
        "poor_mental_days": mental_days,
        "is_physically_active": active,
        "eats_fruit_daily": fruit,
        "eats_veg_daily": veg,
        "is_smoker": smoker,
        "consent": bool(consent),
    }


def contribution_chart(contributions: pd.DataFrame) -> alt.Chart:
    data = contributions[contributions["log_odds"].abs() >= 0.05].copy()
    data["effect"] = data["log_odds"].map(lambda v: "Raises your estimate" if v > 0 else "Lowers your estimate")
    data["label"] = data["odds_multiplier"].map(lambda v: f"×{v:.2f}")
    base = alt.Chart(data).encode(
        y=alt.Y("factor:N", sort=list(data["factor"]), title=None, axis=alt.Axis(labelLimit=260)),
        x=alt.X("log_odds:Q", title="Effect on your odds, compared with the average respondent",
                axis=alt.Axis(labels=False, ticks=False, grid=False)),
    )
    bars = base.mark_bar(cornerRadius=3, height=16).encode(
        color=alt.Color(
            "effect:N", title=None,
            scale=alt.Scale(domain=["Raises your estimate", "Lowers your estimate"], range=[RAISES, LOWERS]),
            legend=alt.Legend(orient="top"),
        ),
        tooltip=[
            alt.Tooltip("factor:N", title="Factor"),
            alt.Tooltip("label:N", title="Multiplies your odds by"),
        ],
    )
    # Each label sits on the empty side of the centre line, opposite its bar
    text = base.mark_text(
        dx=alt.expr("datum.log_odds > 0 ? -6 : 6"),
        align=alt.expr("datum.log_odds > 0 ? 'right' : 'left'"),
    ).encode(x=alt.datum(0), text="label:N")
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#8a8984").encode(x="x:Q")
    return (bars + text + zero).properties(height=alt.Step(30))


def show_result(answers: dict, pipeline, info: dict) -> None:
    result = predict(pipeline, info, answers)
    percent = result.probability * 100
    average = info["base_rate"] * 100

    st.divider()
    st.subheader("Your result")
    col1, col2, col3 = st.columns(3)
    col1.metric("Risk level", result.level)
    col2.metric("Estimated risk", f"{percent:.0f}%" if percent >= 1 else "Under 1%")
    col3.metric("Survey average", f"{average:.0f}%")
    st.progress(min(result.probability, 1.0))

    people = max(1, round(percent))
    st.write(
        f"{LEVEL_NOTES[result.level]} Out of 100 survey respondents who gave answers like yours, "
        f"about **{people}** reported having prediabetes or diabetes."
    )
    bmi = answers["bmi"]
    st.caption(f"Your BMI works out at {bmi:.1f} ({bmi_category(bmi)}).")

    st.subheader("What is driving your estimate")
    st.write(
        "Each bar shows how one answer moves your estimate, compared with the average person "
        "in the survey. Longer bars matter more."
    )
    st.altair_chart(contribution_chart(result.contributions), width="stretch")
    with st.expander("Show these numbers as a table"):
        table = result.contributions.assign(
            direction=lambda d: d["log_odds"].map(lambda v: "Raises" if v > 0 else "Lowers"),
            odds_multiplier=lambda d: d["odds_multiplier"].round(2),
        )[["factor", "direction", "odds_multiplier"]]
        st.dataframe(table, hide_index=True, width="stretch")

    if answers["age"] >= 80:
        st.caption("Everyone aged 80 and over is treated as one group, because that is how the survey recorded age.")
    if bmi > info["bmi_cap"]:
        st.caption(f"BMI values above {info['bmi_cap']} are treated as {info['bmi_cap']} by the model.")

    if answers["consent"]:
        # Logging must never break the result the visitor came for
        try:
            log_prediction(get_log_engine(), answers, result.probability, result.level, info["trained_on"])
            get_usage.clear()
            st.caption("✓ Your answers were saved anonymously. Thank you.")
        except Exception:
            st.caption("Your answers could not be saved this time. Your result is not affected.")

    st.info(
        "This estimate describes patterns in survey data. It cannot tell you whether you have "
        "diabetes. If you are concerned, speak to a doctor, who can arrange a blood test."
    )


@st.cache_data(ttl=60)
def get_usage() -> pd.DataFrame:
    return usage_summary(get_log_engine())


def usage_tab() -> None:
    st.subheader("How the tool is being used")
    st.write(
        "When a visitor ticks the consent box, their answers and result are written to a "
        "PostgreSQL table. This page reads that table back with a SQL `GROUP BY` query. "
        "Only totals are shown, never individual answers."
    )
    try:
        usage = get_usage()
    except Exception:
        st.caption("Usage figures are unavailable right now.")
        return
    if usage.empty:
        st.caption("No predictions have been saved yet.")
        return

    total = int(usage["predictions"].sum())
    average = (usage["predictions"] * usage["avg_predicted_pct"]).sum() / total
    col1, col2 = st.columns(2)
    col1.metric("Predictions saved", f"{total:,}")
    col2.metric("Average estimated risk", f"{average:.0f}%", help="Among saved predictions. The survey average is 14%.")
    st.dataframe(
        usage.rename(columns={
            "risk_level": "Risk level", "predictions": "Predictions",
            "avg_predicted_pct": "Average estimated risk (%)",
        }),
        hide_index=True, width="stretch",
    )
    st.caption("People who choose to use a risk tool are not a random sample, so these figures will differ from the survey.")


def about_tab(info: dict) -> None:
    chosen = info["test_metrics"]["Logistic regression (app model)"]
    st.subheader("How this works")
    st.write(
        "The estimate comes from a logistic regression trained on the CDC's 2015 Behavioral Risk "
        f"Factor Surveillance System survey: {info['training_rows'] + info['test_rows']:,} US adults "
        "who answered questions about their health by phone. The data is stored in a SQL database, "
        "and the model's features are built with SQL views."
    )
    col1, col2, col3 = st.columns(3)
    col1.metric("ROC-AUC", f"{chosen['roc_auc']:.3f}", help="How well the model ranks people by risk. 0.5 is a coin flip, 1.0 is perfect.")
    col2.metric("Tested on", f"{info['test_rows']:,} people", help="People the model never saw during training.")
    col3.metric("Questions asked", len(info["features"]))

    st.subheader("Do the risk levels hold up?")
    st.write("On people the model had never seen, the three levels matched reality closely:")
    bands = pd.DataFrame(info["risk_band_results"]).rename(columns={
        "band": "Level", "share_of_people_pct": "Share of people (%)",
        "avg_predicted_pct": "Average predicted risk (%)", "actual_diabetes_pct": "Actual diabetes rate (%)",
    })[["Level", "Share of people (%)", "Average predicted risk (%)", "Actual diabetes rate (%)"]]
    st.dataframe(bands, hide_index=True, width="stretch")

    st.subheader("Odds ratios")
    st.write("The effect of each factor with all the others held constant. Above 1 raises the odds, below 1 lowers them.")
    st.image(str(ROOT / "docs" / "figures" / "odds_ratios.png"))

    st.subheader("Limitations")
    st.markdown(
        "- The survey is **self-reported**, from **2015**, and covers **US adults only**.\n"
        "- It records prediabetes and diabetes together and does not separate type 1 from type 2. "
        "Most adult cases are type 2.\n"
        "- The model finds **associations, not causes**. Changing an answer changes the estimate, "
        "but that does not mean changing the behaviour would change your risk by the same amount.\n"
        "- Alcohol is deliberately left out. Heavy drinkers in the survey reported *less* diabetes, "
        "most likely because people who are diagnosed are told to cut down, so including it would "
        "have lowered the risk shown to heavy drinkers.\n"
        "- It knows nothing about family history, ethnicity, blood sugar or medication, "
        "which doctors use to assess risk."
    )
    st.write(f"The code, SQL and full evaluation are on [GitHub]({REPO_URL}).")


def main() -> None:
    pipeline, info = get_model()

    st.title("Type 2 Diabetes Risk Estimator")
    st.warning(
        "**Educational project, not medical advice.** This tool was built for a data science "
        "portfolio. It is not a diagnosis and is no substitute for seeing a health professional.",
        icon="⚠️",
    )
    st.write("Answer a few questions to see how your risk compares with 250,000 survey respondents.")

    estimate, about, usage = st.tabs(["Estimate my risk", "About the model", "Usage"])
    with estimate:
        answers = input_form()
        if answers:
            show_result(answers, pipeline, info)
    with about:
        about_tab(info)
    with usage:
        usage_tab()


main()
