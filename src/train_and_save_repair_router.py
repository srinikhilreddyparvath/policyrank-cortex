"""
MVP 14.1: Train and Save Learned Repair Router

Purpose:
--------
Train the Learned Repair Router on the full available repair simulation dataset
and persist it as a reusable model artifact.

This is the bridge from offline validation to future live integration.

Input:
------
outputs/critic_guided_repair_simulation.csv

Outputs:
--------
models/learned_repair_router.pkl
models/learned_repair_router_features.json
models/learned_repair_router_metadata.json
outputs/learned_repair_router_saved_model_summary.csv
outputs/learned_repair_router_saved_model_feature_importance.csv

Important:
----------
This does NOT modify Streamlit or the live evaluator yet.

It trains on all 1000 available rows after MVP 13.9 confirmed out-of-sample
generalization. The saved model can later be loaded by a router-integrated
evaluator or app component.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from datetime import datetime
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

MODELS_DIR = "models"
OUTPUT_DIR = "outputs"

MODEL_PATH = os.path.join(MODELS_DIR, "learned_repair_router.pkl")
FEATURES_PATH = os.path.join(MODELS_DIR, "learned_repair_router_features.json")
METADATA_PATH = os.path.join(MODELS_DIR, "learned_repair_router_metadata.json")

SUMMARY_PATH = os.path.join(OUTPUT_DIR, "learned_repair_router_saved_model_summary.csv")
FEATURE_IMPORTANCE_PATH = os.path.join(
    OUTPUT_DIR,
    "learned_repair_router_saved_model_feature_importance.csv",
)

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


def ensure_dirs() -> None:
    os.makedirs(MODELS_DIR, exist_ok=True)
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

    categorical_levels: Dict[str, List[str]] = {}

    for col in categorical_cols:
        categorical_levels[col] = sorted(
            out[col].fillna("unknown").astype(str).unique().tolist()
        )

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

    out.attrs["categorical_levels"] = categorical_levels

    return out, feature_cols


def train_random_forest(
    df: pd.DataFrame,
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

    model.fit(df[feature_cols], df[target_col])

    return model


def train_logistic_regression(
    df: pd.DataFrame,
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

    model.fit(df[feature_cols], df[target_col])

    return model


def evaluate_model(
    model,
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str,
    model_name: str,
) -> Dict:
    y_true = df[target_col].astype(str)
    y_pred = model.predict(df[feature_cols])

    metrics = {
        "model": model_name,
        "rows": len(df),
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

    cm = confusion_matrix(y_true, y_pred, labels=POLICY_LABELS)

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


def predict_rewards(
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

    out["prediction_matches_best_policy"] = (
        out["predicted_policy"].astype(str) == out["best_logged_policy"].astype(str)
    )

    return out


def build_reward_summary(
    df: pd.DataFrame,
    model_name: str,
) -> Dict:
    baseline_avg = float(df["baseline_reward_at_5"].mean())
    gated_avg = float(df["cortex_reward_at_5"].mean())
    full_avg = float(df["full_cortex_reward_at_5"].mean())
    router_avg = float(df["predicted_policy_reward_at_5"].mean())
    oracle_avg = float(df["best_logged_policy_reward_at_5"].mean())

    predicted_counts = df["predicted_policy"].value_counts().to_dict()
    oracle_counts = df["best_logged_policy"].value_counts().to_dict()

    return {
        "model": model_name,
        "rows": len(df),
        "avg_baseline_reward_at_5": baseline_avg,
        "avg_gated_cortex_reward_at_5": gated_avg,
        "avg_full_cortex_reward_at_5": full_avg,
        "avg_router_reward_at_5": router_avg,
        "avg_oracle_reward_at_5": oracle_avg,
        "router_delta_vs_baseline": router_avg - baseline_avg,
        "router_delta_vs_gated": router_avg - gated_avg,
        "router_delta_vs_full_cortex": router_avg - full_avg,
        "router_regret_vs_oracle": oracle_avg - router_avg,
        "prediction_match_rate": float(df["prediction_matches_best_policy"].mean()),
        "predicted_baseline_count": int(predicted_counts.get("baseline", 0)),
        "predicted_gated_cortex_count": int(predicted_counts.get("gated_cortex", 0)),
        "predicted_full_cortex_count": int(predicted_counts.get("full_cortex", 0)),
        "oracle_baseline_count": int(oracle_counts.get("baseline", 0)),
        "oracle_gated_cortex_count": int(oracle_counts.get("gated_cortex", 0)),
        "oracle_full_cortex_count": int(oracle_counts.get("full_cortex", 0)),
    }


def save_artifacts(
    model,
    feature_cols: List[str],
    categorical_levels: Dict[str, List[str]],
    metadata: Dict,
) -> None:
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    feature_payload = {
        "feature_columns": feature_cols,
        "base_feature_columns": BASE_FEATURE_COLUMNS,
        "optional_numeric_feature_columns": OPTIONAL_NUMERIC_FEATURE_COLUMNS,
        "optional_bool_feature_columns": OPTIONAL_BOOL_FEATURE_COLUMNS,
        "optional_categorical_columns": OPTIONAL_CATEGORICAL_COLUMNS,
        "categorical_levels": categorical_levels,
    }

    with open(FEATURES_PATH, "w", encoding="utf-8") as f:
        json.dump(feature_payload, f, indent=2)

    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)


def run_train_and_save(input_path: str) -> None:
    ensure_dirs()

    raw_df = load_data(input_path)
    target_df = add_targets(raw_df)
    feature_df, feature_cols = build_feature_frame(target_df)

    categorical_levels = feature_df.attrs.get("categorical_levels", {})

    target_col = "best_logged_policy"

    train_df, test_df = train_test_split(
        feature_df,
        test_size=0.30,
        random_state=42,
        stratify=feature_df[target_col],
    )

    rf_model_oos = train_random_forest(
        df=train_df,
        feature_cols=feature_cols,
        target_col=target_col,
    )

    lr_model_oos = train_logistic_regression(
        df=train_df,
        feature_cols=feature_cols,
        target_col=target_col,
    )

    rf_oos_metrics = evaluate_model(
        model=rf_model_oos,
        df=test_df,
        feature_cols=feature_cols,
        target_col=target_col,
        model_name="random_forest_oos_check",
    )

    lr_oos_metrics = evaluate_model(
        model=lr_model_oos,
        df=test_df,
        feature_cols=feature_cols,
        target_col=target_col,
        model_name="logistic_regression_oos_check",
    )

    # Final saved model trained on all available data after OOS validation was successful.
    final_rf_model = train_random_forest(
        df=feature_df,
        feature_cols=feature_cols,
        target_col=target_col,
    )

    final_lr_model = train_logistic_regression(
        df=feature_df,
        feature_cols=feature_cols,
        target_col=target_col,
    )

    final_predictions = predict_rewards(
        df=feature_df,
        model=final_rf_model,
        feature_cols=feature_cols,
    )

    final_reward_summary = build_reward_summary(
        df=final_predictions,
        model_name="saved_random_forest_router_full_data",
    )

    feature_importance_df = build_feature_importance(
        rf_model=final_rf_model,
        lr_model=final_lr_model,
        feature_cols=feature_cols,
    )

    target_counts = feature_df[target_col].value_counts().to_dict()

    metadata = {
        "model_name": "learned_repair_router",
        "model_type": "RandomForestClassifier",
        "created_at": datetime.utcnow().isoformat() + "Z",
        "input_path": input_path,
        "training_rows": int(len(feature_df)),
        "feature_count": int(len(feature_cols)),
        "target_column": target_col,
        "target_counts": {
            "baseline": int(target_counts.get("baseline", 0)),
            "gated_cortex": int(target_counts.get("gated_cortex", 0)),
            "full_cortex": int(target_counts.get("full_cortex", 0)),
        },
        "policy_labels": POLICY_LABELS,
        "oos_random_forest_metrics": rf_oos_metrics,
        "oos_logistic_regression_metrics": lr_oos_metrics,
        "final_full_data_reward_summary": final_reward_summary,
        "artifact_paths": {
            "model": MODEL_PATH,
            "features": FEATURES_PATH,
            "metadata": METADATA_PATH,
        },
        "notes": (
            "Saved model trained on all available data after MVP 13.9 "
            "validated out-of-sample reward improvement. Use only for dry-run "
            "integration until live evaluation is added."
        ),
    }

    save_artifacts(
        model=final_rf_model,
        feature_cols=feature_cols,
        categorical_levels=categorical_levels,
        metadata=metadata,
    )

    summary_rows = [
        rf_oos_metrics,
        lr_oos_metrics,
        final_reward_summary,
    ]

    summary_df = pd.DataFrame(summary_rows)

    summary_df.to_csv(SUMMARY_PATH, index=False)
    feature_importance_df.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    print()
    print("MVP 14.1 Train and Save Learned Repair Router")
    print("=" * 100)
    print(f"Input: {input_path}")
    print(f"Training rows: {len(feature_df)}")
    print(f"Feature count: {len(feature_cols)}")
    print()
    print("Target counts")
    print("-" * 100)
    print(pd.Series(target_counts).to_string())

    print()
    print("OOS check metrics")
    print("-" * 100)
    print(pd.DataFrame([rf_oos_metrics, lr_oos_metrics]).to_string(index=False))

    print()
    print("Final full-data reward summary")
    print("-" * 100)
    print(pd.DataFrame([final_reward_summary]).to_string(index=False))

    print()
    print("Top feature importance")
    print("-" * 100)
    print(feature_importance_df.head(25).to_string(index=False))

    print()
    print("Files written")
    print("-" * 100)
    print(f"- model: {MODEL_PATH}")
    print(f"- features: {FEATURES_PATH}")
    print(f"- metadata: {METADATA_PATH}")
    print(f"- summary: {SUMMARY_PATH}")
    print(f"- feature importance: {FEATURE_IMPORTANCE_PATH}")


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
    run_train_and_save(input_path=args.input)