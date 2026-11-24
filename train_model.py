"""Train the predictive-maintenance model and save the bundle used by app.py.

Mirrors the steps in streamlit.ipynb: train/validation/test split, two
pipelines (logistic regression and random forest), model and threshold chosen
by maximum validation profit, and the full fitted pipeline saved with pickle.
"""
import pickle

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

RANDOM_STATE = 11
DATA_URL = (
    "https://archive.ics.uci.edu/static/public/601/"
    "ai4i+2020+predictive+maintenance+dataset.zip"
)
MODEL_PATH = "predictive_maintenance_model.pkl"

MAINTENANCE_COST = 500
FAILURE_COST = 10_000
MAINTENANCE_EFFECTIVENESS = 0.80

TARGET = "Machine failure"
CATEGORICAL_FEATURES = ["Type"]
NUMERIC_FEATURES = [
    "Air temperature [K]",
    "Process temperature [K]",
    "Rotational speed [rpm]",
    "Torque [Nm]",
    "Tool wear [min]",
]
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def make_ranking_table(
    y_true,
    y_score,
    maintenance_cost=MAINTENANCE_COST,
    failure_cost=FAILURE_COST,
    effectiveness=MAINTENANCE_EFFECTIVENESS,
):
    ranking = pd.DataFrame({"actual": np.asarray(y_true), "score": np.asarray(y_score)})
    ranking = ranking.sort_values("score", ascending=False).reset_index(drop=True)
    ranking["machines_targeted"] = np.arange(1, len(ranking) + 1)
    ranking["targeted_fraction"] = ranking["machines_targeted"] / len(ranking)
    total_failures = ranking["actual"].sum()
    ranking["failures_found"] = ranking["actual"].cumsum()
    ranking["cumulative_gain"] = ranking["failures_found"] / total_failures
    ranking["lift"] = ranking["cumulative_gain"] / ranking["targeted_fraction"]
    true_positive_value = effectiveness * failure_cost - maintenance_cost
    false_positive_value = -maintenance_cost
    ranking["incremental_profit"] = np.where(
        ranking["actual"].eq(1), true_positive_value, false_positive_value
    )
    ranking["cumulative_profit"] = ranking["incremental_profit"].cumsum()
    return ranking


def summarize_model(name, fitted_model, X_eval, y_eval):
    scores = fitted_model.predict_proba(X_eval)[:, 1]
    ranking = make_ranking_table(y_eval, scores)
    top_ten_count = max(1, int(np.ceil(0.10 * len(ranking))))
    top_ten = ranking.iloc[top_ten_count - 1]
    best = ranking.loc[ranking["cumulative_profit"].idxmax()]
    return {
        "model": name,
        "roc_auc": roc_auc_score(y_eval, scores),
        "average_precision": average_precision_score(y_eval, scores),
        "gain_at_10pct": top_ten["cumulative_gain"],
        "lift_at_10pct": top_ten["lift"],
        "max_validation_profit": best["cumulative_profit"],
        "profit_optimal_fraction": best["targeted_fraction"],
        "profit_optimal_threshold": best["score"],
    }


def build_models():
    preprocessor = make_column_transformer(
        (OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        (StandardScaler(), NUMERIC_FEATURES),
    )
    return {
        "Logistic regression": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]),
        "Random forest": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(
                n_estimators=300,
                min_samples_leaf=3,
                random_state=RANDOM_STATE,
                n_jobs=-1,
            )),
        ]),
    }


def train_bundle(df=None):
    if df is None:
        df = pd.read_csv(DATA_URL, compression="zip")
    X = df[FEATURE_COLUMNS].copy()
    y = df[TARGET].copy()

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, stratify=y, random_state=RANDOM_STATE
    )
    X_validation, X_test, y_validation, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
    )

    models = build_models()
    summaries = []
    for name, pipeline in models.items():
        pipeline.fit(X_train, y_train)
        summaries.append(summarize_model(name, pipeline, X_validation, y_validation))

    comparison = (
        pd.DataFrame(summaries)
        .set_index("model")
        .sort_values("max_validation_profit", ascending=False)
    )
    best_model_name = comparison.index[0]
    best_model = models[best_model_name]
    best_threshold = float(comparison.loc[best_model_name, "profit_optimal_threshold"])

    # Apply the unchanged policy once to the held-out test set.
    test_scores = best_model.predict_proba(X_test)[:, 1]
    maintain = test_scores >= best_threshold
    actual = np.asarray(y_test)
    test_profit = np.where(
        ~maintain,
        0,
        np.where(
            actual == 1,
            MAINTENANCE_EFFECTIVENESS * FAILURE_COST - MAINTENANCE_COST,
            -MAINTENANCE_COST,
        ),
    ).sum()

    bundle = {
        "model": best_model,
        "model_name": best_model_name,
        "threshold": best_threshold,
        "features": FEATURE_COLUMNS,
        "maintenance_cost": MAINTENANCE_COST,
        "failure_cost": FAILURE_COST,
        "maintenance_effectiveness": MAINTENANCE_EFFECTIVENESS,
        "sklearn_version": sklearn.__version__,
    }
    report = {
        "comparison": comparison,
        "test_roc_auc": roc_auc_score(y_test, test_scores),
        "test_machines_selected": int(maintain.sum()),
        "test_share_selected": float(maintain.mean()),
        "test_profit": float(test_profit),
    }
    return bundle, report


def main():
    bundle, report = train_bundle()
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(bundle, f)

    pd.set_option("display.width", 200)
    print(report["comparison"].round(3))
    print(f"\nSelected model: {bundle['model_name']}")
    print(f"Validation-selected threshold: {bundle['threshold']:.4f}")
    print(f"Test ROC AUC: {report['test_roc_auc']:.3f}")
    print(f"Machines selected on test set: {report['test_machines_selected']:,} "
          f"({report['test_share_selected']:.1%})")
    print(f"Estimated test-set incremental profit: ${report['test_profit']:,.0f}")
    print(f"Saved {MODEL_PATH} (scikit-learn {sklearn.__version__})")


if __name__ == "__main__":
    main()
