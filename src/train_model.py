"""Train, evaluate and save the diabetes risk model.

Usage (from the project root, after building the database):
    python src/train_model.py

Reads the `training_features` view from the SQLite database, compares four
models on a held-out test set, and saves:

    models/pipeline.joblib      the chosen model with its preprocessing
    models/model_info.json      features, risk bands, metrics and odds ratios
    docs/figures/*.png          ROC, precision-recall, calibration, odds ratios
"""

import json
from datetime import date
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sqlalchemy import create_engine, text
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
DB_PATH = ROOT / "data" / "diabetes.db"
MODEL_DIR = ROOT / "models"
FIGURE_DIR = ROOT / "docs" / "figures"

SEED = 42
TARGET = "has_diabetes"
BMI_CAP = 50  # BMI above this is treated as 50: the few extreme values distort a linear model

# --- Feature sets -------------------------------------------------------------
# The web app asks only health and lifestyle questions (16 inputs).
APP_BINARY = [
    "is_male", "is_smoker", "is_physically_active", "eats_fruit_daily", "eats_veg_daily",
    "is_heavy_drinker", "has_high_bp", "has_high_chol", "had_stroke", "has_heart_disease",
    "has_difficulty_walking",
]
APP_NUMERIC = ["general_health", "poor_mental_days", "poor_physical_days"]
APP_FEATURES = APP_BINARY + APP_NUMERIC + ["bmi", "age_group_id"]

# The five columns left out of the app model: income, education and access to care.
EXTRA_BINARY = ["has_healthcare", "skipped_doctor_cost", "had_chol_check"]
EXTRA_NUMERIC = ["education_id", "income_id"]
ALL_FEATURES = APP_FEATURES + EXTRA_BINARY + EXTRA_NUMERIC

# Predicted probability cut-offs for the three risk levels shown in the app
RISK_BANDS = {"Low": 0.0, "Moderate": 0.10, "High": 0.25}

# Odds ratios are reported per this many units of each numeric feature
OR_UNITS = {"bmi": 5, "general_health": 1, "poor_mental_days": 10, "poor_physical_days": 10}

LABELS = {
    "is_male": "Male (vs female)",
    "is_smoker": "Smoker",
    "is_physically_active": "Physically active",
    "eats_fruit_daily": "Eats fruit daily",
    "eats_veg_daily": "Eats vegetables daily",
    "is_heavy_drinker": "Heavy drinker",
    "has_high_bp": "High blood pressure",
    "has_high_chol": "High cholesterol",
    "had_stroke": "Had a stroke",
    "has_heart_disease": "Heart disease",
    "has_difficulty_walking": "Difficulty walking",
    "bmi": "BMI, per 5 points",
    "general_health": "General health, per step worse",
    "poor_mental_days": "Poor mental health, per 10 days",
    "poor_physical_days": "Poor physical health, per 10 days",
}

BLUE, ORANGE, AQUA, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#1baf7a", "#0b0b0b", "#52514e", "#e4e3df"


def logistic_pipeline(binary, numeric, class_weight=None) -> Pipeline:
    """Logistic regression: yes/no columns as they are, numbers standardised,
    BMI capped then standardised, age group one-hot encoded (reference: 18-24)."""
    # np.clip is used directly (not a function defined here) so that the saved
    # model can be loaded by the web app without importing this script.
    cap_bmi = FunctionTransformer(np.clip, kw_args={"a_min": None, "a_max": BMI_CAP})
    prep = ColumnTransformer([
        ("binary", "passthrough", binary),
        ("numeric", StandardScaler(), numeric),
        ("bmi", Pipeline([("cap", cap_bmi), ("scale", StandardScaler())]), ["bmi"]),
        ("age", OneHotEncoder(drop="first", dtype=float), ["age_group_id"]),
    ])
    model = LogisticRegression(max_iter=2000, class_weight=class_weight)
    return Pipeline([("prep", prep), ("model", model)])


def load_data() -> pd.DataFrame:
    if not DB_PATH.exists():
        raise SystemExit("Database not found. Run: python src/build_database.py")
    engine = create_engine(f"sqlite:///{DB_PATH.as_posix()}")
    with engine.connect() as conn:
        df = pd.read_sql(text("SELECT * FROM training_features"), conn)
        age_labels = pd.read_sql(text("SELECT age_group_id, label FROM age_groups"), conn)
    engine.dispose()
    df.attrs["age_labels"] = dict(zip(age_labels["age_group_id"], age_labels["label"]))
    return df


def score(y_true, prob) -> dict:
    return {
        "roc_auc": round(float(roc_auc_score(y_true, prob)), 4),
        "pr_auc": round(float(average_precision_score(y_true, prob)), 4),
        "brier": round(float(brier_score_loss(y_true, prob)), 4),
        "mean_prediction": round(float(prob.mean()), 4),
    }


def odds_ratios(pipeline: Pipeline, age_labels: dict) -> pd.DataFrame:
    """Turn the model's coefficients into odds ratios in everyday units."""
    prep = pipeline.named_steps["prep"]
    coefs = pipeline.named_steps["model"].coef_[0]
    rows, i = [], 0

    for name in APP_BINARY:
        rows.append((LABELS[name], np.exp(coefs[i])))
        i += 1
    scales = prep.named_transformers_["numeric"].scale_
    for name, scale in zip(APP_NUMERIC, scales):
        rows.append((LABELS[name], np.exp(coefs[i] / scale * OR_UNITS[name])))
        i += 1
    bmi_scale = prep.named_transformers_["bmi"].named_steps["scale"].scale_[0]
    rows.append((LABELS["bmi"], np.exp(coefs[i] / bmi_scale * OR_UNITS["bmi"])))
    i += 1
    reference = age_labels[1]
    for age_id in prep.named_transformers_["age"].categories_[0][1:]:
        rows.append((f"Age {age_labels[int(age_id)]} (vs {reference})", np.exp(coefs[i])))
        i += 1

    assert i == len(coefs), "coefficient count does not match the feature list"
    return pd.DataFrame(rows, columns=["factor", "odds_ratio"]).round({"odds_ratio": 2})


def risk_band_table(y_true, prob) -> pd.DataFrame:
    cuts = list(RISK_BANDS.values()) + [1.0001]
    band = pd.cut(prob, cuts, right=False, labels=list(RISK_BANDS))
    table = pd.DataFrame({"band": band, "actual": np.asarray(y_true), "predicted": prob})
    out = table.groupby("band", observed=True).agg(
        people=("actual", "size"),
        actual_diabetes_pct=("actual", lambda s: round(100 * s.mean(), 1)),
        avg_predicted_pct=("predicted", lambda s: round(100 * s.mean(), 1)),
    )
    out.insert(1, "share_of_people_pct", (100 * out["people"] / out["people"].sum()).round(1))
    out["share_of_all_cases_pct"] = (
        100 * table.groupby("band", observed=True)["actual"].sum() / table["actual"].sum()
    ).round(1)
    return out.reset_index()


def style_axes(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(GRID)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0)
    ax.xaxis.label.set_color(MUTED)
    ax.yaxis.label.set_color(MUTED)


def plot_curves(y_test, probs: dict, base_rate: float) -> None:
    colors = {"Logistic regression (app model)": BLUE, "XGBoost": ORANGE}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), dpi=130)

    for name, color in colors.items():
        fpr, tpr, _ = roc_curve(y_test, probs[name])
        axes[0].plot(fpr, tpr, color=color, linewidth=2,
                     label=f"{name}: {roc_auc_score(y_test, probs[name]):.3f}")
        precision, recall, _ = precision_recall_curve(y_test, probs[name])
        keep = recall >= 0.01  # the first few points are noisy: they rest on a handful of people
        axes[1].plot(recall[keep], precision[keep], color=color, linewidth=2,
                     label=f"{name}: {average_precision_score(y_test, probs[name]):.3f}")

    axes[0].plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle=":", label="Random guess: 0.500")
    axes[0].set(xlabel="False positive rate", ylabel="True positive rate")
    axes[0].set_title("ROC curve (area under curve)", loc="left", fontweight="bold", color=INK)
    axes[1].axhline(base_rate, color=MUTED, linewidth=1, linestyle=":", label=f"Random guess: {base_rate:.3f}")
    axes[1].set(xlabel="Recall", ylabel="Precision", ylim=(0, 1))
    axes[1].set_title("Precision-recall curve (average precision)", loc="left", fontweight="bold", color=INK)
    for ax in axes:
        style_axes(ax)
        ax.legend(frameon=False, loc="lower right" if ax is axes[0] else "upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "roc_pr_curves.png", facecolor="white")
    plt.close(fig)


def plot_calibration(y_test, probs: dict) -> None:
    series = {
        "Logistic regression (app model)": (BLUE, "o"),
        "XGBoost": (ORANGE, "s"),
        "Logistic regression, balanced class weights": (AQUA, "^"),
    }
    fig, ax = plt.subplots(figsize=(6, 5), dpi=130)
    ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle=":", label="Perfectly calibrated")
    for name, (color, marker) in series.items():
        actual, predicted = calibration_curve(y_test, probs[name], n_bins=10, strategy="quantile")
        ax.plot(predicted, actual, color=color, marker=marker, markersize=6, linewidth=2, label=name)
    ax.set(xlabel="Predicted probability of diabetes", ylabel="Actual share with diabetes",
           xlim=(0, 0.9), ylim=(0, 0.9))
    ax.set_title("Calibration: do predicted risks match reality?", loc="left", fontweight="bold", color=INK)
    style_axes(ax)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "calibration.png", facecolor="white")
    plt.close(fig)


def plot_odds_ratios(ratios: pd.DataFrame) -> None:
    data = ratios.iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 7.5), dpi=130)
    ax.axvline(1, color=MUTED, linewidth=1)
    ax.hlines(data["factor"], 1, data["odds_ratio"], color=GRID, linewidth=2)
    ax.scatter(data["odds_ratio"], data["factor"], s=45, color=BLUE, zorder=3)
    for factor, value in zip(data["factor"], data["odds_ratio"]):
        ax.annotate(f"{value:.2f}", (max(value, 1), factor), xytext=(7, 0),
                    textcoords="offset points", va="center", ha="left", fontsize=8.5, color=MUTED)
    ax.set_xscale("log")
    ticks = [0.5, 1, 2, 5]
    ax.set_xticks(ticks, [str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlim(0.38, 8)
    ax.set_xlabel("Odds ratio, log scale (above 1 = higher odds of diabetes)")
    ax.set_title("Odds ratios from the logistic regression", loc="left", fontweight="bold", color=INK)
    style_axes(ax)
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(FIGURE_DIR / "odds_ratios.png", facecolor="white")
    plt.close(fig)


def main() -> None:
    MODEL_DIR.mkdir(exist_ok=True)
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    age_labels = df.attrs["age_labels"]
    X, y = df[ALL_FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=SEED
    )
    print(f"Train: {len(X_train):,} rows   Test: {len(X_test):,} rows   "
          f"Diabetes rate in test: {y_test.mean():.1%}")

    candidates = {
        "Logistic regression (app model)":
            (logistic_pipeline(APP_BINARY, APP_NUMERIC), APP_FEATURES),
        "Logistic regression, all 21 features":
            (logistic_pipeline(APP_BINARY + EXTRA_BINARY, APP_NUMERIC + EXTRA_NUMERIC), ALL_FEATURES),
        "Logistic regression, balanced class weights":
            (logistic_pipeline(APP_BINARY, APP_NUMERIC, class_weight="balanced"), APP_FEATURES),
        "XGBoost":
            (XGBClassifier(n_estimators=400, learning_rate=0.05, max_depth=4, subsample=0.8,
                           colsample_bytree=0.8, eval_metric="logloss", random_state=SEED),
             ALL_FEATURES),
    }

    metrics, probs, fitted = {}, {}, {}
    for name, (model, columns) in candidates.items():
        model.fit(X_train[columns], y_train)
        probs[name] = model.predict_proba(X_test[columns])[:, 1]
        metrics[name] = score(y_test, probs[name]) | {"features": len(columns)}
        fitted[name] = model

    table = pd.DataFrame(metrics).T[["features", "roc_auc", "pr_auc", "brier", "mean_prediction"]]
    print("\nTest set results")
    print(table.to_string())

    chosen = "Logistic regression (app model)"
    cv = cross_val_score(
        logistic_pipeline(APP_BINARY, APP_NUMERIC), X_train[APP_FEATURES], y_train,
        scoring="roc_auc", cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
    )
    print(f"\n5-fold cross-validated ROC-AUC (training set): {cv.mean():.4f} +/- {cv.std():.4f}")

    ratios = odds_ratios(fitted[chosen], age_labels)
    print("\nOdds ratios")
    print(ratios.to_string(index=False))

    bands = risk_band_table(y_test, probs[chosen])
    print("\nRisk bands on the test set")
    print(bands.to_string(index=False))

    plot_curves(y_test, probs, float(y_test.mean()))
    plot_calibration(y_test, probs)
    plot_odds_ratios(ratios)

    joblib.dump(fitted[chosen], MODEL_DIR / "pipeline.joblib")
    info = {
        "trained_on": date.today().isoformat(),
        "model": "Logistic regression",
        "scikit_learn_version": sklearn.__version__,
        "features": APP_FEATURES,
        "bmi_cap": BMI_CAP,
        "age_groups": {str(k): v for k, v in age_labels.items()},
        "risk_bands": RISK_BANDS,
        "training_rows": len(X_train),
        "test_rows": len(X_test),
        "base_rate": round(float(y.mean()), 4),
        "cv_roc_auc_mean": round(float(cv.mean()), 4),
        "cv_roc_auc_std": round(float(cv.std()), 4),
        "test_metrics": metrics,
        "risk_band_results": bands.to_dict(orient="records"),
        "odds_ratios": ratios.to_dict(orient="records"),
    }
    (MODEL_DIR / "model_info.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    print("\nSaved models/pipeline.joblib, models/model_info.json and 3 figures in docs/figures/")


if __name__ == "__main__":
    main()
