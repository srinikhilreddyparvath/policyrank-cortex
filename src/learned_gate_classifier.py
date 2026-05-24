"""
MVP 13.4: Learned Gate Classifier

Purpose:
--------
Train a small offline classifier from CORTEX evaluation logs to learn when
the baseline-aware gate should intervene.

This is safer than manually tuning thresholds.

Input:
------
outputs/scalable_eval_mvp13_3_1_resilient_250.csv

Core idea:
----------
For each query, we already have:

- baseline reward
- gated CORTEX reward
- full CORTEX reward
- gate confidence signals
- contract alignment signals
- label quality
- top-k retrieval confidence proxies

We create a supervised target:

    target_gate_helped = 1 if gated reward > full CORTEX reward
    target_gate_helped = 0 otherwise

Then we train a lightweight classifier to predict whether gate intervention
is likely to help.

Outputs:
--------
outputs/learned_gate_classifier_training_data.csv
outputs/learned_gate_classifier_feature_importance.csv
outputs/learned_gate_classifier_predictions.csv
outputs/learned_gate_classifier_summary.csv

This file does NOT modify the live ranking pipeline yet.
It is an offline analysis module.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


DEFAULT_INPUT_PATH = "outputs/scalable_eval_mvp13_3_1_resilient_250.csv"
OUTPUT_DIR = "outputs"

TRAINING_DATA_PATH = os.path.join(OUTPUT_DIR, "learned_gate_classifier_training_data.csv")
PREDICTIONS_PATH = os.path.join(OUTPUT_DIR, "learned_gate_classifier_predictions.csv")
FEATURE_IMPORTANCE_PATH = os.path.join(OUTPUT_DIR, "learned_gate_classifier_feature_importance.csv")
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "learned_gate_classifier_summary.csv")


FEATURE_COLUMNS = [
    "gate_baseline_confidence_score",
    "gate_contract_alignment_score",
    "gate_final_gate_score",
    "gate_exclusion_violation_rate",
    "gate_required_term_coverage",
    "gate_preferred_term_coverage",
    "gate_brand_preference_coverage",
    "gate_top1_score",
    "gate_top5_mean_score",
    "gate_label_quality_score",
]


OPTIONAL_FEATURE_COLUMNS = [
    "positive_contract_rows",
    "blocked_rows",
    "low_coverage",
]


DISPLAY_COLUMNS = [
    "query",
    "target_gate_helped",
    "predicted_gate_helped",
    "predicted_help_probability",
    "recommended_gate_action",
    "winner",
    "full_cortex_winner",
    "gate_decision",
    "baseline_reward_at_5",
    "cortex_reward_at_5",
    "full_cortex_reward_at_5",
    "absolute_lift",
    "full_cortex_absolute_lift",
    "gate_delta_vs_full_cortex",
    "gate_baseline_confidence_score",
    "gate_contract_alignment_score",
    "gate_final_gate_score",
    "gate_top1_score",
    "gate_top5_mean_score",
    "gate_label_quality_score",
    "baseline_top_product_title",
    "cortex_top_product_title",
    "full_cortex_top_product_title",
]


def ensure_output_dir() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def load_eval_data(input_path: str) -> pd.DataFrame:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Could not find input file: {input_path}")

    df = pd.read_csv(input_path)

    if len(df) == 0:
        raise ValueError(f"Input file is empty: {input_path}")

    if "error" in df.columns:
        df = df[df["error"].fillna("").astype(str).str.strip() == ""].copy()

    required_cols = [
        "query",
        "cortex_reward_at_5",
        "full_cortex_reward_at_5",
        "gate_delta_vs_full_cortex",
    ]

    missing_required = [col for col in required_cols if col not in df.columns]

    if missing_required:
        raise ValueError(f"Missing required columns: {missing_required}")

    numeric_cols = [
        "baseline_reward_at_5",
        "cortex_reward_at_5",
        "full_cortex_reward_at_5",
        "absolute_lift",
        "full_cortex_absolute_lift",
        "gate_delta_vs_full_cortex",
    ] + FEATURE_COLUMNS + OPTIONAL_FEATURE_COLUMNS

    for col in numeric_cols:
        if col in df.columns:
            df[col] = safe_numeric(df[col])

    return df.reset_index(drop=True)


def build_training_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    out = df.copy()

    # Main supervised target:
    # Did the gated slate beat the full CORTEX slate?
    out["target_gate_helped"] = (out["gate_delta_vs_full_cortex"] > 0.0001).astype(int)

    # Secondary target:
    # Did the gate hurt full CORTEX?
    out["target_gate_hurt"] = (out["gate_delta_vs_full_cortex"] < -0.0001).astype(int)

    usable_features = []

    for col in FEATURE_COLUMNS:
        if col in out.columns:
            usable_features.append(col)

    for col in OPTIONAL_FEATURE_COLUMNS:
        if col in out.columns:
            out[col] = safe_numeric(out[col])
            usable_features.append(col)

    if not usable_features:
        raise ValueError("No usable feature columns were found.")

    for col in usable_features:
        out[col] = safe_numeric(out[col])

    return out, usable_features


def train_random_forest(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=4,
        min_samples_leaf=8,
        random_state=42,
        class_weight="balanced",
    )

    model.fit(train_df[feature_cols], train_df[target_col])

    return model


def train_logistic_regression(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
) -> Pipeline:
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(train_df[feature_cols], train_df[target_col])

    return model


def evaluate_model(
    model,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    model_name: str,
) -> Dict:
    y_true = test_df[target_col].astype(int)

    y_pred = model.predict(test_df[feature_cols])

    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(test_df[feature_cols])[:, 1]
    else:
        y_prob = y_pred

    metrics = {
        "model": model_name,
        "test_rows": len(test_df),
        "positive_rate": float(y_true.mean()) if len(y_true) else 0.0,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "roc_auc": 0.0,
    }

    try:
        if len(set(y_true)) > 1:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
    except Exception:
        metrics["roc_auc"] = 0.0

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    metrics.update(
        {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        }
    )

    return metrics


def build_feature_importance(
    rf_model: RandomForestClassifier,
    logistic_model: Pipeline,
    feature_cols: List[str],
) -> pd.DataFrame:
    rf_importance = rf_model.feature_importances_

    logistic_classifier = logistic_model.named_steps["classifier"]
    logistic_coefficients = logistic_classifier.coef_[0]

    rows = []

    for feature, rf_value, coef_value in zip(
        feature_cols,
        rf_importance,
        logistic_coefficients,
    ):
        rows.append(
            {
                "feature": feature,
                "random_forest_importance": float(rf_value),
                "logistic_coefficient": float(coef_value),
                "abs_logistic_coefficient": float(abs(coef_value)),
            }
        )

    out = pd.DataFrame(rows)
    out = out.sort_values(
        by=["random_forest_importance", "abs_logistic_coefficient"],
        ascending=[False, False],
    )

    return out


def recommend_gate_action(probability: float) -> str:
    """
    Offline recommendation only.

    Conservative policy:
    - High probability of gate helping: light_rerank_candidate
    - Medium probability: full_cortex_default
    - Low probability: full_cortex_default

    We do not recommend preserve_baseline yet because preserve has not shown
    enough positive empirical evidence in the latest runs.
    """

    if probability >= 0.70:
        return "light_rerank_candidate"

    return "full_cortex_default"


def build_predictions(
    df: pd.DataFrame,
    model: RandomForestClassifier,
    feature_cols: List[str],
) -> pd.DataFrame:
    out = df.copy()

    out["predicted_help_probability"] = model.predict_proba(out[feature_cols])[:, 1]
    out["predicted_gate_helped"] = (out["predicted_help_probability"] >= 0.50).astype(int)
    out["recommended_gate_action"] = out["predicted_help_probability"].apply(
        recommend_gate_action
    )

    existing_display_cols = [
        col
        for col in DISPLAY_COLUMNS
        if col in out.columns
    ]

    return out[existing_display_cols].sort_values(
        by="predicted_help_probability",
        ascending=False,
    )


def build_policy_simulation(
    predictions_df: pd.DataFrame,
    threshold_values: List[float],
) -> pd.DataFrame:
    """
    Simulate a learned routing policy:

    If predicted_help_probability >= threshold:
        choose gated reward
    Else:
        choose full CORTEX reward

    This estimates whether a learned classifier could improve routing.
    """

    rows = []

    for threshold in threshold_values:
        tmp = predictions_df.copy()

        tmp["sim_use_gate"] = tmp["predicted_help_probability"] >= threshold

        tmp["sim_reward"] = tmp.apply(
            lambda row: row["cortex_reward_at_5"]
            if row["sim_use_gate"]
            else row["full_cortex_reward_at_5"],
            axis=1,
        )

        tmp["sim_lift_vs_baseline"] = tmp["sim_reward"] - tmp["baseline_reward_at_5"]

        full_cortex_avg_reward = float(tmp["full_cortex_reward_at_5"].mean())
        gated_avg_reward = float(tmp["cortex_reward_at_5"].mean())
        sim_avg_reward = float(tmp["sim_reward"].mean())

        rows.append(
            {
                "threshold": threshold,
                "queries_using_gate": int(tmp["sim_use_gate"].sum()),
                "queries_using_full_cortex": int((~tmp["sim_use_gate"]).sum()),
                "avg_sim_reward": sim_avg_reward,
                "avg_full_cortex_reward": full_cortex_avg_reward,
                "avg_gated_reward": gated_avg_reward,
                "avg_sim_delta_vs_full_cortex": sim_avg_reward - full_cortex_avg_reward,
                "avg_sim_delta_vs_current_gated": sim_avg_reward - gated_avg_reward,
                "avg_sim_lift_vs_baseline": float(tmp["sim_lift_vs_baseline"].mean()),
                "median_sim_lift_vs_baseline": float(tmp["sim_lift_vs_baseline"].median()),
            }
        )

    return pd.DataFrame(rows).sort_values(
        by="avg_sim_reward",
        ascending=False,
    )


def run_learned_gate_classifier(input_path: str) -> None:
    ensure_output_dir()

    raw_df = load_eval_data(input_path)
    training_df, feature_cols = build_training_frame(raw_df)

    if len(training_df) < 20:
        raise ValueError("Need at least 20 valid rows to train the classifier.")

    positive_count = int(training_df["target_gate_helped"].sum())
    negative_count = int((training_df["target_gate_helped"] == 0).sum())

    print()
    print("MVP 13.4 Learned Gate Classifier")
    print("=" * 80)
    print(f"Input: {input_path}")
    print(f"Valid rows: {len(training_df)}")
    print(f"Positive target rows: {positive_count}")
    print(f"Negative target rows: {negative_count}")
    print(f"Features: {feature_cols}")

    train_df, test_df = train_test_split(
        training_df,
        test_size=0.30,
        random_state=42,
        stratify=training_df["target_gate_helped"],
    )

    target_col = "target_gate_helped"

    rf_model = train_random_forest(
        train_df=train_df,
        feature_cols=feature_cols,
        target_col=target_col,
    )

    logistic_model = train_logistic_regression(
        train_df=train_df,
        feature_cols=feature_cols,
        target_col=target_col,
    )

    rf_metrics = evaluate_model(
        model=rf_model,
        test_df=test_df,
        feature_cols=feature_cols,
        target_col=target_col,
        model_name="random_forest",
    )

    logistic_metrics = evaluate_model(
        model=logistic_model,
        test_df=test_df,
        feature_cols=feature_cols,
        target_col=target_col,
        model_name="logistic_regression",
    )

    feature_importance_df = build_feature_importance(
        rf_model=rf_model,
        logistic_model=logistic_model,
        feature_cols=feature_cols,
    )

    predictions_df = build_predictions(
        df=training_df,
        model=rf_model,
        feature_cols=feature_cols,
    )

    threshold_values = [
        0.50,
        0.55,
        0.60,
        0.65,
        0.70,
        0.75,
        0.80,
        0.85,
        0.90,
    ]

    policy_simulation_df = build_policy_simulation(
        predictions_df=predictions_df,
        threshold_values=threshold_values,
    )

    summary_rows = [
        rf_metrics,
        logistic_metrics,
    ]

    for _, row in policy_simulation_df.iterrows():
        summary_rows.append(
            {
                "model": f"policy_sim_threshold_{row['threshold']}",
                "test_rows": len(training_df),
                "positive_rate": float(training_df["target_gate_helped"].mean()),
                "accuracy": None,
                "precision": None,
                "recall": None,
                "f1": None,
                "roc_auc": None,
                "true_negative": None,
                "false_positive": None,
                "false_negative": None,
                "true_positive": None,
                "queries_using_gate": int(row["queries_using_gate"]),
                "queries_using_full_cortex": int(row["queries_using_full_cortex"]),
                "avg_sim_reward": float(row["avg_sim_reward"]),
                "avg_full_cortex_reward": float(row["avg_full_cortex_reward"]),
                "avg_gated_reward": float(row["avg_gated_reward"]),
                "avg_sim_delta_vs_full_cortex": float(row["avg_sim_delta_vs_full_cortex"]),
                "avg_sim_delta_vs_current_gated": float(row["avg_sim_delta_vs_current_gated"]),
                "avg_sim_lift_vs_baseline": float(row["avg_sim_lift_vs_baseline"]),
                "median_sim_lift_vs_baseline": float(row["median_sim_lift_vs_baseline"]),
            }
        )

    summary_df = pd.DataFrame(summary_rows)

    training_df.to_csv(TRAINING_DATA_PATH, index=False)
    predictions_df.to_csv(PREDICTIONS_PATH, index=False)
    feature_importance_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
    summary_df.to_csv(SUMMARY_PATH, index=False)

    print()
    print("Model metrics")
    print("-" * 80)
    print(pd.DataFrame([rf_metrics, logistic_metrics]).to_string(index=False))

    print()
    print("Feature importance")
    print("-" * 80)
    print(feature_importance_df.to_string(index=False))

    print()
    print("Policy simulation")
    print("-" * 80)
    print(policy_simulation_df.to_string(index=False))

    print()
    print("Top gate-help candidates")
    print("-" * 80)
    print(predictions_df.head(15).to_string(index=False))

    print()
    print("Files written")
    print("-" * 80)
    print(f"- training data: {TRAINING_DATA_PATH}")
    print(f"- predictions: {PREDICTIONS_PATH}")
    print(f"- feature importance: {FEATURE_IMPORTANCE_PATH}")
    print(f"- summary: {SUMMARY_PATH}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT_PATH,
        help="Path to resilient scalable evaluation CSV.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_learned_gate_classifier(input_path=args.input)