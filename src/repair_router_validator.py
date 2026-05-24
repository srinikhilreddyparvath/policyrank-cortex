"""
MVP 13.9: Out-of-Sample Repair Router Validation

Purpose:
--------
Validate the Learned Repair Router using a clean train/test split.

MVP 13.8 trained a repair router and showed strong policy simulation on the
full dataset. MVP 13.9 makes the evaluation stricter:

    1. Load the critic-guided repair simulation data.
    2. Split queries into train/test.
    3. Train the repair router only on train.
    4. Predict the best policy only on held-out test.
    5. Compare reward on test:
        - baseline
        - current gated CORTEX
        - full CORTEX
        - learned repair router
        - oracle logged-policy upper bound

This gives a more honest out-of-sample validation.

Input:
------
outputs/critic_guided_repair_simulation.csv

Outputs:
--------
outputs/repair_router_oos_validation_summary.csv
outputs/repair_router_oos_validation_predictions.csv
outputs/repair_router_oos_validation_feature_importance.csv
outputs/repair_router_oos_validation_confusion_matrix.csv

Important:
----------
This remains offline. It does NOT modify the app or evaluator yet.
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
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


DEFAULT_INPUT_PATH = "outputs/critic_guided_repair_simulation.csv"
OUTPUT_DIR = "outputs"

SUMMARY_PATH = os.path.join(OUTPUT_DIR, "repair_router_oos_validation_summary.csv")
PREDICTIONS_PATH = os.path.join(OUTPUT_DIR, "repair_router_oos_validation_predictions.csv")
FEATURE_IMPORTANCE_PATH = os.path.join(OUTPUT_DIR, "repair_router_oos_validation_feature_importance.csv")
CONFUSION_MATRIX_PATH = os.path.join(OUTPUT_DIR, "repair_router_oos_validation_confusion_matrix.csv")


BASE_FEATURE_COLUMNS = [
    "critic_risk_score",
    "gate_baseline_confidence_score",
    "gate_contract_alignment_score",
    "gate_final_gate_score",
    "gate_top1_score",
    "gate_top5_mean_score",
    "gate_label_quality_score",
    "gate_exclusion_violation_rate",
    "query_token_count",
]

OPTIONAL_NUMERIC_FEATURE_COLUMNS = [
    "positive_contract_rows",
    "blocked_rows",
    "low_coverage",
]

OPTIONAL_BOOL_FEATURE_COLUMNS = [
    "needs_online_critic",
    "baseline_loss_rescued",
    "over_rerank_prevented",
]

OPTIONAL_CATEGORICAL_COLUMNS = [
    "critic_priority",
    "governance_route",
    "gate_decision",
    "selected_rl_action",
    "enforcement_status",
]

POLICY_LABELS = [
    "baseline",
    "gated_cortex",
    "full_cortex",
]

DISPLAY_COLUMNS = [
    "query",
    "split",
    "best_logged_policy",
    "predicted_policy",
    "prediction_matches_best_policy",
    "predicted_policy_confidence",
    "baseline_reward_at_5",
    "cortex_reward_at_5",
    "full_cortex_reward_at_5",
    "predicted_policy_reward_at_5",
    "best_logged_policy_reward_at_5",
    "predicted_delta_vs_current_gated",
    "predicted_delta_vs_full_cortex",
    "predicted_delta_vs_baseline",
    "oracle_delta_vs_current_gated",
    "critic_priority",
    "critic_risk_score",
    "critic_failure_modes",
    "critic_repair_actions",
    "primary_repair_action",
    "governance_route",
    "gate_decision",
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


def safe_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes"]).astype(int)


def load_data(input_path: str) -> pd.DataFrame:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Could not find input file: {input_path}")

    df = pd.read_csv(input_path)

    if len(df) == 0:
        raise ValueError(f"Input file is empty: {input_path}")

    required_cols = [
        "query",
        "baseline_reward_at_5",
        "cortex_reward_at_5",
        "full_cortex_reward_at_5",
    ]

    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    numeric_cols = (
        [
            "baseline_reward_at_5",
            "cortex_reward_at_5",
            "full_cortex_reward_at_5",
            "absolute_lift",
            "full_cortex_absolute_lift",
            "gate_delta_vs_full_cortex",
            "critic_risk_score",
            "gate_baseline_confidence_score",
            "gate_contract_alignment_score",
            "gate_final_gate_score",
            "gate_top1_score",
            "gate_top5_mean_score",
            "gate_label_quality_score",
            "gate_exclusion_violation_rate",
            "query_token_count",
            "conservative_repair_reward_at_5",
            "oracle_logged_policy_reward_at_5",
        ]
        + OPTIONAL_NUMERIC_FEATURE_COLUMNS
    )

    for col in numeric_cols:
        if col in df.columns:
            df[col] = safe_numeric(df[col])

    for col in OPTIONAL_BOOL_FEATURE_COLUMNS:
        if col in df.columns:
            df[col] = safe_bool(df[col])

    return df.reset_index(drop=True)


def get_policy_reward(row: pd.Series, policy: str) -> float:
    if policy == "baseline":
        return float(row.get("baseline_reward_at_5", 0.0))

    if policy == "gated_cortex":
        return float(row.get("cortex_reward_at_5", 0.0))

    if policy == "full_cortex":
        return float(row.get("full_cortex_reward_at_5", 0.0))

    return float(row.get("cortex_reward_at_5", 0.0))


def get_best_logged_policy(row: pd.Series) -> str:
    rewards = {
        "baseline": get_policy_reward(row, "baseline"),
        "gated_cortex": get_policy_reward(row, "gated_cortex"),
        "full_cortex": get_policy_reward(row, "full_cortex"),
    }

    return max(rewards, key=rewards.get)


def add_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["best_logged_policy"] = out.apply(get_best_logged_policy, axis=1)

    out["best_logged_policy_reward_at_5"] = out.apply(
        lambda row: get_policy_reward(row, row["best_logged_policy"]),
        axis=1,
    )

    out["oracle_delta_vs_current_gated"] = (
        out["best_logged_policy_reward_at_5"] - out["cortex_reward_at_5"]
    )

    out["oracle_delta_vs_full_cortex"] = (
        out["best_logged_policy_reward_at_5"] - out["full_cortex_reward_at_5"]
    )

    return out


def build_feature_frame(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    out = df.copy()

    feature_cols: List[str] = []

    for col in BASE_FEATURE_COLUMNS:
        if col in out.columns:
            out[col] = safe_numeric(out[col])
            feature_cols.append(col)

    for col in OPTIONAL_NUMERIC_FEATURE_COLUMNS:
        if col in out.columns:
            out[col] = safe_numeric(out[col])
            feature_cols.append(col)

    for col in OPTIONAL_BOOL_FEATURE_COLUMNS:
        if col in out.columns:
            out[col] = safe_bool(out[col])
            feature_cols.append(col)

    categorical_cols = [
        col
        for col in OPTIONAL_CATEGORICAL_COLUMNS
        if col in out.columns
    ]

    if categorical_cols:
        cat_df = pd.get_dummies(
            out[categorical_cols].fillna("unknown").astype(str),
            prefix=categorical_cols,
            dummy_na=False,
        )

        out = pd.concat(
            [
                out,
                cat_df,
            ],
            axis=1,
        )

        feature_cols.extend(cat_df.columns.tolist())

    if not feature_cols:
        raise ValueError("No usable feature columns found.")

    for col in feature_cols:
        out[col] = safe_numeric(out[col])

    return out, feature_cols


def align_test_columns(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """
    Ensures train/test have same feature columns after one-hot encoding.
    """

    train = train_df.copy()
    test = test_df.copy()

    for col in feature_cols:
        if col not in train.columns:
            train[col] = 0.0

        if col not in test.columns:
            test[col] = 0.0

    return train, test, feature_cols


def train_random_forest(
    train_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=8,
        min_samples_leaf=6,
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
                    max_iter=1500,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )

    model.fit(train_df[feature_cols], train_df[target_col])

    return model


def evaluate_classifier(
    model,
    test_df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    model_name: str,
) -> Dict:
    y_true = test_df[target_col].astype(str)
    y_pred = model.predict(test_df[feature_cols])

    metrics = {
        "model": model_name,
        "test_rows": len(test_df),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(
                y_true,
                y_pred,
                labels=POLICY_LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                y_true,
                y_pred,
                labels=POLICY_LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=POLICY_LABELS,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=POLICY_LABELS,
                average="weighted",
                zero_division=0,
            )
        ),
    }

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=POLICY_LABELS,
    )

    for i, true_label in enumerate(POLICY_LABELS):
        for j, pred_label in enumerate(POLICY_LABELS):
            metrics[f"cm_true_{true_label}_pred_{pred_label}"] = int(cm[i, j])

    return metrics


def build_feature_importance(
    rf_model: RandomForestClassifier,
    lr_model: Pipeline,
    feature_cols: List[str],
) -> pd.DataFrame:
    rows: List[Dict] = []

    rf_importance = rf_model.feature_importances_

    logistic_classifier = lr_model.named_steps["classifier"]
    logistic_classes = logistic_classifier.classes_
    logistic_coefs = logistic_classifier.coef_

    for idx, feature in enumerate(feature_cols):
        row = {
            "feature": feature,
            "random_forest_importance": float(rf_importance[idx]),
        }

        for class_index, class_name in enumerate(logistic_classes):
            row[f"logistic_coef_for_{class_name}"] = float(
                logistic_coefs[class_index][idx]
            )

        row["max_abs_logistic_coefficient"] = max(
            abs(float(logistic_coefs[class_index][idx]))
            for class_index in range(len(logistic_classes))
        )

        rows.append(row)

    out = pd.DataFrame(rows)

    out = out.sort_values(
        by=[
            "random_forest_importance",
            "max_abs_logistic_coefficient",
        ],
        ascending=[False, False],
    )

    return out


def predict_on_split(
    df: pd.DataFrame,
    model,
    feature_cols: List[str],
) -> pd.DataFrame:
    out = df.copy()

    predictions = model.predict(out[feature_cols])
    probabilities = model.predict_proba(out[feature_cols])
    classes = model.classes_

    out["predicted_policy"] = predictions
    out["predicted_policy_confidence"] = probabilities.max(axis=1)

    for class_index, class_name in enumerate(classes):
        out[f"predicted_prob_{class_name}"] = probabilities[:, class_index]

    out["predicted_policy_reward_at_5"] = out.apply(
        lambda row: get_policy_reward(row, row["predicted_policy"]),
        axis=1,
    )

    out["predicted_delta_vs_current_gated"] = (
        out["predicted_policy_reward_at_5"] - out["cortex_reward_at_5"]
    )

    out["predicted_delta_vs_full_cortex"] = (
        out["predicted_policy_reward_at_5"] - out["full_cortex_reward_at_5"]
    )

    out["predicted_delta_vs_baseline"] = (
        out["predicted_policy_reward_at_5"] - out["baseline_reward_at_5"]
    )

    out["prediction_matches_best_policy"] = (
        out["predicted_policy"].astype(str) == out["best_logged_policy"].astype(str)
    )

    return out


def build_reward_summary(
    df: pd.DataFrame,
    split_name: str,
) -> Dict:
    total = len(df)

    target_counts = df["best_logged_policy"].value_counts().to_dict()
    predicted_counts = df["predicted_policy"].value_counts().to_dict()

    baseline_avg = float(df["baseline_reward_at_5"].mean())
    gated_avg = float(df["cortex_reward_at_5"].mean())
    full_avg = float(df["full_cortex_reward_at_5"].mean())
    router_avg = float(df["predicted_policy_reward_at_5"].mean())
    oracle_avg = float(df["best_logged_policy_reward_at_5"].mean())

    return {
        "split": split_name,
        "rows": total,
        "baseline_avg_reward": baseline_avg,
        "current_gated_avg_reward": gated_avg,
        "full_cortex_avg_reward": full_avg,
        "learned_router_avg_reward": router_avg,
        "oracle_logged_policy_avg_reward": oracle_avg,
        "router_delta_vs_baseline": router_avg - baseline_avg,
        "router_delta_vs_current_gated": router_avg - gated_avg,
        "router_delta_vs_full_cortex": router_avg - full_avg,
        "oracle_delta_vs_current_gated": oracle_avg - gated_avg,
        "oracle_delta_vs_full_cortex": oracle_avg - full_avg,
        "target_baseline_count": int(target_counts.get("baseline", 0)),
        "target_gated_cortex_count": int(target_counts.get("gated_cortex", 0)),
        "target_full_cortex_count": int(target_counts.get("full_cortex", 0)),
        "predicted_baseline_count": int(predicted_counts.get("baseline", 0)),
        "predicted_gated_cortex_count": int(predicted_counts.get("gated_cortex", 0)),
        "predicted_full_cortex_count": int(predicted_counts.get("full_cortex", 0)),
        "prediction_match_rate": float(df["prediction_matches_best_policy"].mean()),
        "median_router_lift_vs_baseline": float(
            (df["predicted_policy_reward_at_5"] - df["baseline_reward_at_5"]).median()
        ),
        "median_current_gated_lift_vs_baseline": float(
            (df["cortex_reward_at_5"] - df["baseline_reward_at_5"]).median()
        ),
        "median_oracle_lift_vs_baseline": float(
            (df["best_logged_policy_reward_at_5"] - df["baseline_reward_at_5"]).median()
        ),
    }


def build_confusion_matrix_df(
    predictions_df: pd.DataFrame,
    split_name: str,
) -> pd.DataFrame:
    y_true = predictions_df["best_logged_policy"].astype(str)
    y_pred = predictions_df["predicted_policy"].astype(str)

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=POLICY_LABELS,
    )

    rows: List[Dict] = []

    for i, true_label in enumerate(POLICY_LABELS):
        for j, pred_label in enumerate(POLICY_LABELS):
            rows.append(
                {
                    "split": split_name,
                    "true_policy": true_label,
                    "predicted_policy": pred_label,
                    "count": int(cm[i, j]),
                }
            )

    return pd.DataFrame(rows)


def choose_prediction_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "prediction_matches_best_policy" in out.columns and "predicted_policy_confidence" in out.columns:
        out = out.sort_values(
            by=[
                "split",
                "prediction_matches_best_policy",
                "predicted_policy_confidence",
            ],
            ascending=[True, True, False],
        )

    existing = [
        col for col in DISPLAY_COLUMNS
        if col in out.columns
    ]

    prob_cols = [
        col for col in out.columns
        if col.startswith("predicted_prob_")
    ]

    return out[existing + prob_cols].copy()


def run_oos_validation(input_path: str, test_size: float) -> None:
    ensure_output_dir()

    raw_df = load_data(input_path)
    target_df = add_targets(raw_df)
    full_feature_df, feature_cols = build_feature_frame(target_df)

    if len(full_feature_df) < 30:
        raise ValueError("Need at least 30 rows for OOS validation.")

    train_df, test_df = train_test_split(
        full_feature_df,
        test_size=test_size,
        random_state=42,
        stratify=full_feature_df["best_logged_policy"],
    )

    train_df = train_df.copy()
    test_df = test_df.copy()

    train_df["split"] = "train"
    test_df["split"] = "test"

    train_df, test_df, feature_cols = align_test_columns(
        train_df=train_df,
        test_df=test_df,
        feature_cols=feature_cols,
    )

    target_col = "best_logged_policy"

    rf_model = train_random_forest(
        train_df=train_df,
        feature_cols=feature_cols,
        target_col=target_col,
    )

    lr_model = train_logistic_regression(
        train_df=train_df,
        feature_cols=feature_cols,
        target_col=target_col,
    )

    rf_metrics = evaluate_classifier(
        model=rf_model,
        test_df=test_df,
        feature_cols=feature_cols,
        target_col=target_col,
        model_name="random_forest_oos",
    )

    lr_metrics = evaluate_classifier(
        model=lr_model,
        test_df=test_df,
        feature_cols=feature_cols,
        target_col=target_col,
        model_name="logistic_regression_oos",
    )

    # Use RandomForest for router reward simulation, matching MVP 13.8.
    train_predictions = predict_on_split(
        df=train_df,
        model=rf_model,
        feature_cols=feature_cols,
    )

    test_predictions = predict_on_split(
        df=test_df,
        model=rf_model,
        feature_cols=feature_cols,
    )

    all_predictions = pd.concat(
        [
            train_predictions,
            test_predictions,
        ],
        ignore_index=True,
    )

    train_reward_summary = build_reward_summary(
        df=train_predictions,
        split_name="train",
    )

    test_reward_summary = build_reward_summary(
        df=test_predictions,
        split_name="test",
    )

    summary_rows = [
        {
            "section": "classifier_metrics",
            **rf_metrics,
        },
        {
            "section": "classifier_metrics",
            **lr_metrics,
        },
        {
            "section": "reward_simulation",
            "model": "random_forest_router_train",
            **train_reward_summary,
        },
        {
            "section": "reward_simulation",
            "model": "random_forest_router_test",
            **test_reward_summary,
        },
    ]

    summary_df = pd.DataFrame(summary_rows)

    feature_importance_df = build_feature_importance(
        rf_model=rf_model,
        lr_model=lr_model,
        feature_cols=feature_cols,
    )

    train_cm_df = build_confusion_matrix_df(
        predictions_df=train_predictions,
        split_name="train",
    )

    test_cm_df = build_confusion_matrix_df(
        predictions_df=test_predictions,
        split_name="test",
    )

    confusion_df = pd.concat(
        [
            train_cm_df,
            test_cm_df,
        ],
        ignore_index=True,
    )

    output_predictions_df = choose_prediction_columns(all_predictions)

    summary_df.to_csv(SUMMARY_PATH, index=False)
    output_predictions_df.to_csv(PREDICTIONS_PATH, index=False)
    feature_importance_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
    confusion_df.to_csv(CONFUSION_MATRIX_PATH, index=False)

    print()
    print("MVP 13.9 Out-of-Sample Repair Router Validation")
    print("=" * 110)
    print(f"Input: {input_path}")
    print(f"Total rows: {len(full_feature_df)}")
    print(f"Train rows: {len(train_df)}")
    print(f"Test rows: {len(test_df)}")
    print(f"Feature count: {len(feature_cols)}")

    print()
    print("Classifier metrics")
    print("-" * 110)
    print(pd.DataFrame([rf_metrics, lr_metrics]).to_string(index=False))

    print()
    print("Reward simulation summary")
    print("-" * 110)
    print(pd.DataFrame([train_reward_summary, test_reward_summary]).to_string(index=False))

    print()
    print("Feature importance")
    print("-" * 110)
    print(feature_importance_df.head(25).to_string(index=False))

    print()
    print("Files written")
    print("-" * 110)
    print(f"- summary: {SUMMARY_PATH}")
    print(f"- predictions: {PREDICTIONS_PATH}")
    print(f"- feature importance: {FEATURE_IMPORTANCE_PATH}")
    print(f"- confusion matrix: {CONFUSION_MATRIX_PATH}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT_PATH,
        help="Path to critic guided repair simulation CSV.",
    )

    parser.add_argument(
        "--test-size",
        type=float,
        default=0.30,
        help="Held-out test size fraction.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_oos_validation(
        input_path=args.input,
        test_size=args.test_size,
    )