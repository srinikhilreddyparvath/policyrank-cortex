"""
MVP 13.8: Learned Repair Router

Purpose:
--------
Train an offline repair-routing model that predicts which logged policy should
be selected for each query:

    1. baseline
    2. gated_cortex
    3. full_cortex

Input:
------
outputs/critic_guided_repair_simulation.csv

Outputs:
--------
outputs/learned_repair_router_training_data.csv
outputs/learned_repair_router_predictions.csv
outputs/learned_repair_router_feature_importance.csv
outputs/learned_repair_router_summary.csv
outputs/learned_repair_router_policy_simulation.csv

Fixes in this version:
----------------------
1. Removes LogisticRegression(multi_class="auto") for local sklearn compatibility.
2. Prevents product_type one-hot explosion.
3. Fixes KeyError: prediction_matches_best_policy by sorting before display-column filtering
   and including prediction_matches_best_policy in output columns.
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

TRAINING_DATA_PATH = os.path.join(OUTPUT_DIR, "learned_repair_router_training_data.csv")
PREDICTIONS_PATH = os.path.join(OUTPUT_DIR, "learned_repair_router_predictions.csv")
FEATURE_IMPORTANCE_PATH = os.path.join(OUTPUT_DIR, "learned_repair_router_feature_importance.csv")
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "learned_repair_router_summary.csv")
POLICY_SIMULATION_PATH = os.path.join(OUTPUT_DIR, "learned_repair_router_policy_simulation.csv")


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

DISPLAY_COLUMNS = [
    "query",
    "best_logged_policy",
    "predicted_policy",
    "prediction_matches_best_policy",
    "predicted_policy_confidence",
    "baseline_reward_at_5",
    "cortex_reward_at_5",
    "full_cortex_reward_at_5",
    "predicted_policy_reward_at_5",
    "best_logged_policy_reward_at_5",
    "current_gated_delta",
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


def load_simulation_data(input_path: str) -> pd.DataFrame:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Could not find repair simulation file: {input_path}")

    df = pd.read_csv(input_path)

    if len(df) == 0:
        raise ValueError(f"Repair simulation file is empty: {input_path}")

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
            "conservative_repair_reward_at_5",
            "oracle_logged_policy_reward_at_5",
            "conservative_delta_vs_current_gated",
            "oracle_delta_vs_current_gated",
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
        + OPTIONAL_NUMERIC_FEATURE_COLUMNS
    )

    for col in numeric_cols:
        if col in df.columns:
            df[col] = safe_numeric(df[col])

    for col in OPTIONAL_BOOL_FEATURE_COLUMNS:
        if col in df.columns:
            df[col] = safe_bool(df[col])

    return df.reset_index(drop=True)


def get_best_policy(row: pd.Series) -> str:
    rewards = {
        "baseline": float(row.get("baseline_reward_at_5", 0.0)),
        "gated_cortex": float(row.get("cortex_reward_at_5", 0.0)),
        "full_cortex": float(row.get("full_cortex_reward_at_5", 0.0)),
    }

    return max(rewards, key=rewards.get)


def get_policy_reward(row: pd.Series, policy: str) -> float:
    if policy == "baseline":
        return float(row.get("baseline_reward_at_5", 0.0))

    if policy == "gated_cortex":
        return float(row.get("cortex_reward_at_5", 0.0))

    if policy == "full_cortex":
        return float(row.get("full_cortex_reward_at_5", 0.0))

    return float(row.get("cortex_reward_at_5", 0.0))


def add_target_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["best_logged_policy"] = out.apply(get_best_policy, axis=1)

    out["best_logged_policy_reward_at_5"] = out.apply(
        lambda row: get_policy_reward(row, row["best_logged_policy"]),
        axis=1,
    )

    out["current_gated_delta"] = (
        out["cortex_reward_at_5"] - out["baseline_reward_at_5"]
    )

    out["full_cortex_delta"] = (
        out["full_cortex_reward_at_5"] - out["baseline_reward_at_5"]
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

    labels = ["baseline", "gated_cortex", "full_cortex"]

    metrics = {
        "model": model_name,
        "test_rows": len(test_df),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_precision": float(
            precision_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_recall": float(
            recall_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "macro_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=labels,
                average="macro",
                zero_division=0,
            )
        ),
        "weighted_f1": float(
            f1_score(
                y_true,
                y_pred,
                labels=labels,
                average="weighted",
                zero_division=0,
            )
        ),
    }

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=labels,
    )

    for i, true_label in enumerate(labels):
        for j, pred_label in enumerate(labels):
            metrics[f"cm_true_{true_label}_pred_{pred_label}"] = int(cm[i, j])

    return metrics


def build_feature_importance(
    rf_model: RandomForestClassifier,
    logistic_model: Pipeline,
    feature_cols: List[str],
) -> pd.DataFrame:
    rows: List[Dict] = []

    rf_importances = rf_model.feature_importances_

    logistic_classifier = logistic_model.named_steps["classifier"]
    logistic_classes = logistic_classifier.classes_
    logistic_coefs = logistic_classifier.coef_

    for idx, feature in enumerate(feature_cols):
        row = {
            "feature": feature,
            "random_forest_importance": float(rf_importances[idx]),
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


def predict_policy_routes(
    df: pd.DataFrame,
    model: RandomForestClassifier,
    feature_cols: List[str],
) -> pd.DataFrame:
    out = df.copy()

    predicted_policy = model.predict(out[feature_cols])
    probabilities = model.predict_proba(out[feature_cols])
    classes = model.classes_

    out["predicted_policy"] = predicted_policy
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


def build_policy_simulation(predictions_df: pd.DataFrame) -> pd.DataFrame:
    total = len(predictions_df)

    current_gated_avg = float(predictions_df["cortex_reward_at_5"].mean())
    full_cortex_avg = float(predictions_df["full_cortex_reward_at_5"].mean())
    baseline_avg = float(predictions_df["baseline_reward_at_5"].mean())
    predicted_router_avg = float(predictions_df["predicted_policy_reward_at_5"].mean())
    oracle_avg = float(predictions_df["best_logged_policy_reward_at_5"].mean())

    policy_counts = predictions_df["predicted_policy"].value_counts().to_dict()
    best_policy_counts = predictions_df["best_logged_policy"].value_counts().to_dict()

    rows = [
        {
            "simulation": "current_gated_cortex",
            "query_count": total,
            "avg_reward_at_5": current_gated_avg,
            "avg_delta_vs_baseline": current_gated_avg - baseline_avg,
            "avg_delta_vs_full_cortex": current_gated_avg - full_cortex_avg,
            "avg_delta_vs_current_gated": 0.0,
            "selected_baseline_count": 0,
            "selected_gated_cortex_count": total,
            "selected_full_cortex_count": 0,
        },
        {
            "simulation": "full_cortex",
            "query_count": total,
            "avg_reward_at_5": full_cortex_avg,
            "avg_delta_vs_baseline": full_cortex_avg - baseline_avg,
            "avg_delta_vs_full_cortex": 0.0,
            "avg_delta_vs_current_gated": full_cortex_avg - current_gated_avg,
            "selected_baseline_count": 0,
            "selected_gated_cortex_count": 0,
            "selected_full_cortex_count": total,
        },
        {
            "simulation": "learned_repair_router",
            "query_count": total,
            "avg_reward_at_5": predicted_router_avg,
            "avg_delta_vs_baseline": predicted_router_avg - baseline_avg,
            "avg_delta_vs_full_cortex": predicted_router_avg - full_cortex_avg,
            "avg_delta_vs_current_gated": predicted_router_avg - current_gated_avg,
            "selected_baseline_count": int(policy_counts.get("baseline", 0)),
            "selected_gated_cortex_count": int(policy_counts.get("gated_cortex", 0)),
            "selected_full_cortex_count": int(policy_counts.get("full_cortex", 0)),
        },
        {
            "simulation": "oracle_logged_policy_upper_bound",
            "query_count": total,
            "avg_reward_at_5": oracle_avg,
            "avg_delta_vs_baseline": oracle_avg - baseline_avg,
            "avg_delta_vs_full_cortex": oracle_avg - full_cortex_avg,
            "avg_delta_vs_current_gated": oracle_avg - current_gated_avg,
            "selected_baseline_count": int(best_policy_counts.get("baseline", 0)),
            "selected_gated_cortex_count": int(best_policy_counts.get("gated_cortex", 0)),
            "selected_full_cortex_count": int(best_policy_counts.get("full_cortex", 0)),
        },
    ]

    return pd.DataFrame(rows)


def choose_prediction_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    if "prediction_matches_best_policy" in out.columns and "predicted_policy_confidence" in out.columns:
        out = out.sort_values(
            by=[
                "prediction_matches_best_policy",
                "predicted_policy_confidence",
            ],
            ascending=[True, False],
        )
    elif "predicted_policy_confidence" in out.columns:
        out = out.sort_values(
            by="predicted_policy_confidence",
            ascending=False,
        )

    existing = [
        col
        for col in DISPLAY_COLUMNS
        if col in out.columns
    ]

    extra_prob_cols = [
        col
        for col in out.columns
        if col.startswith("predicted_prob_")
    ]

    return out[existing + extra_prob_cols].copy()


def build_summary_rows(
    rf_metrics: Dict,
    lr_metrics: Dict,
    policy_sim_df: pd.DataFrame,
    training_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict] = []

    target_counts = training_df["best_logged_policy"].value_counts().to_dict()

    base_context = {
        "total_rows": len(training_df),
        "target_baseline_count": int(target_counts.get("baseline", 0)),
        "target_gated_cortex_count": int(target_counts.get("gated_cortex", 0)),
        "target_full_cortex_count": int(target_counts.get("full_cortex", 0)),
    }

    for metrics in [rf_metrics, lr_metrics]:
        row = dict(base_context)
        row.update(metrics)
        rows.append(row)

    for _, sim_row in policy_sim_df.iterrows():
        row = dict(base_context)
        row.update(
            {
                "model": sim_row["simulation"],
                "test_rows": len(training_df),
                "accuracy": None,
                "balanced_accuracy": None,
                "macro_precision": None,
                "macro_recall": None,
                "macro_f1": None,
                "weighted_f1": None,
                "avg_reward_at_5": sim_row["avg_reward_at_5"],
                "avg_delta_vs_baseline": sim_row["avg_delta_vs_baseline"],
                "avg_delta_vs_full_cortex": sim_row["avg_delta_vs_full_cortex"],
                "avg_delta_vs_current_gated": sim_row["avg_delta_vs_current_gated"],
                "selected_baseline_count": sim_row["selected_baseline_count"],
                "selected_gated_cortex_count": sim_row["selected_gated_cortex_count"],
                "selected_full_cortex_count": sim_row["selected_full_cortex_count"],
            }
        )

        rows.append(row)

    return pd.DataFrame(rows)


def run_learned_repair_router(input_path: str) -> None:
    ensure_output_dir()

    raw_df = load_simulation_data(input_path)
    target_df = add_target_columns(raw_df)
    training_df, feature_cols = build_feature_frame(target_df)

    if len(training_df) < 30:
        raise ValueError("Need at least 30 rows for learned repair router.")

    target_col = "best_logged_policy"
    target_counts = training_df[target_col].value_counts()

    print()
    print("MVP 13.8 Learned Repair Router")
    print("=" * 100)
    print(f"Input: {input_path}")
    print(f"Rows: {len(training_df)}")
    print()
    print("Target distribution")
    print("-" * 100)
    print(target_counts.to_string())
    print()
    print("Feature count:", len(feature_cols))

    train_df, test_df = train_test_split(
        training_df,
        test_size=0.30,
        random_state=42,
        stratify=training_df[target_col],
    )

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
        model_name="random_forest_repair_router",
    )

    lr_metrics = evaluate_classifier(
        model=lr_model,
        test_df=test_df,
        feature_cols=feature_cols,
        target_col=target_col,
        model_name="logistic_regression_repair_router",
    )

    feature_importance_df = build_feature_importance(
        rf_model=rf_model,
        logistic_model=lr_model,
        feature_cols=feature_cols,
    )

    predictions_df = predict_policy_routes(
        df=training_df,
        model=rf_model,
        feature_cols=feature_cols,
    )

    policy_sim_df = build_policy_simulation(predictions_df)

    summary_df = build_summary_rows(
        rf_metrics=rf_metrics,
        lr_metrics=lr_metrics,
        policy_sim_df=policy_sim_df,
        training_df=training_df,
    )

    output_predictions_df = choose_prediction_columns(predictions_df)

    training_df.to_csv(TRAINING_DATA_PATH, index=False)
    output_predictions_df.to_csv(PREDICTIONS_PATH, index=False)
    feature_importance_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)
    summary_df.to_csv(SUMMARY_PATH, index=False)
    policy_sim_df.to_csv(POLICY_SIMULATION_PATH, index=False)

    print()
    print("Classifier metrics")
    print("-" * 100)
    print(pd.DataFrame([rf_metrics, lr_metrics]).to_string(index=False))

    print()
    print("Policy simulation")
    print("-" * 100)
    print(policy_sim_df.to_string(index=False))

    print()
    print("Top feature importance")
    print("-" * 100)
    print(feature_importance_df.head(25).to_string(index=False))

    print()
    print("Sample prediction errors / low confidence cases")
    print("-" * 100)
    print(output_predictions_df.head(20).to_string(index=False))

    print()
    print("Files written")
    print("-" * 100)
    print(f"- training data: {TRAINING_DATA_PATH}")
    print(f"- predictions: {PREDICTIONS_PATH}")
    print(f"- feature importance: {FEATURE_IMPORTANCE_PATH}")
    print(f"- summary: {SUMMARY_PATH}")
    print(f"- policy simulation: {POLICY_SIMULATION_PATH}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT_PATH,
        help="Path to critic guided repair simulation CSV.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_learned_repair_router(input_path=args.input)