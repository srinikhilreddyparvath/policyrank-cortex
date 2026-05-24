"""
MVP 14.2: Saved Repair Router Inference Smoke Test

Purpose:
--------
Load the persisted Learned Repair Router model and run inference on rows from
the critic-guided repair simulation file.

This verifies that the saved model artifacts are reusable:

    models/learned_repair_router.pkl
    models/learned_repair_router_features.json
    models/learned_repair_router_metadata.json

Input:
------
outputs/critic_guided_repair_simulation.csv

Outputs:
--------
outputs/repair_router_inference_smoke_test.csv
outputs/repair_router_inference_summary.csv

Important:
----------
This is still a dry-run smoke test.
It does NOT modify Streamlit or the scalable evaluator.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from typing import Dict, List, Tuple

import pandas as pd


DEFAULT_INPUT_PATH = "outputs/critic_guided_repair_simulation.csv"

MODEL_PATH = "models/learned_repair_router.pkl"
FEATURES_PATH = "models/learned_repair_router_features.json"
METADATA_PATH = "models/learned_repair_router_metadata.json"

OUTPUT_DIR = "outputs"
SMOKE_TEST_PATH = os.path.join(OUTPUT_DIR, "repair_router_inference_smoke_test.csv")
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "repair_router_inference_summary.csv")


POLICY_LABELS = [
    "baseline",
    "gated_cortex",
    "full_cortex",
]


DISPLAY_COLUMNS = [
    "query",
    "best_logged_policy",
    "router_predicted_policy",
    "router_prediction_matches_best_policy",
    "router_predicted_policy_confidence",
    "baseline_reward_at_5",
    "cortex_reward_at_5",
    "full_cortex_reward_at_5",
    "router_predicted_policy_reward_at_5",
    "best_logged_policy_reward_at_5",
    "router_delta_vs_baseline",
    "router_delta_vs_gated",
    "router_delta_vs_full_cortex",
    "router_regret_vs_oracle",
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


def load_json(path: str) -> Dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find JSON artifact: {path}")

    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_router_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Could not find model artifact: {model_path}")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return model


def load_input_data(input_path: str) -> pd.DataFrame:
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
        raise ValueError(f"Missing required input columns: {missing}")

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


def add_targets_if_missing(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    numeric_cols = [
        "baseline_reward_at_5",
        "cortex_reward_at_5",
        "full_cortex_reward_at_5",
        "critic_risk_score",
        "gate_baseline_confidence_score",
        "gate_contract_alignment_score",
        "gate_final_gate_score",
        "gate_top1_score",
        "gate_top5_mean_score",
        "gate_label_quality_score",
        "gate_exclusion_violation_rate",
        "query_token_count",
        "positive_contract_rows",
        "blocked_rows",
        "low_coverage",
    ]

    for col in numeric_cols:
        if col in out.columns:
            out[col] = safe_numeric(out[col])

    bool_cols = [
        "needs_online_critic",
        "baseline_loss_rescued",
        "over_rerank_prevented",
    ]

    for col in bool_cols:
        if col in out.columns:
            out[col] = safe_bool(out[col])

    if "best_logged_policy" not in out.columns:
        out["best_logged_policy"] = out.apply(get_best_logged_policy, axis=1)

    if "best_logged_policy_reward_at_5" not in out.columns:
        out["best_logged_policy_reward_at_5"] = out.apply(
            lambda row: get_policy_reward(row, row["best_logged_policy"]),
            axis=1,
        )

    return out


def build_feature_frame_from_artifact(
    df: pd.DataFrame,
    feature_payload: Dict,
) -> Tuple[pd.DataFrame, List[str]]:
    """
    Build the exact feature frame expected by the saved model.

    The saved feature artifact contains:
    - feature_columns
    - base_feature_columns
    - optional_numeric_feature_columns
    - optional_bool_feature_columns
    - optional_categorical_columns
    - categorical_levels
    """

    out = df.copy()

    expected_features = feature_payload.get("feature_columns", [])

    base_numeric = (
        feature_payload.get("base_feature_columns", [])
        + feature_payload.get("optional_numeric_feature_columns", [])
    )

    bool_cols = feature_payload.get("optional_bool_feature_columns", [])
    categorical_cols = feature_payload.get("optional_categorical_columns", [])

    for col in base_numeric:
        if col not in out.columns:
            out[col] = 0.0
        out[col] = safe_numeric(out[col])

    for col in bool_cols:
        if col not in out.columns:
            out[col] = 0
        out[col] = safe_bool(out[col])

    # Build categorical dummies.
    # Use current row values, then align exactly to expected features.
    existing_categorical_cols = [
        col for col in categorical_cols
        if col in out.columns
    ]

    if existing_categorical_cols:
        cat_df = pd.get_dummies(
            out[existing_categorical_cols].fillna("unknown").astype(str),
            prefix=existing_categorical_cols,
            dummy_na=False,
        )

        out = pd.concat(
            [
                out,
                cat_df,
            ],
            axis=1,
        )

    # Ensure every expected feature exists.
    for feature in expected_features:
        if feature not in out.columns:
            out[feature] = 0.0

        out[feature] = safe_numeric(out[feature])

    return out, expected_features


def run_router_inference(
    df: pd.DataFrame,
    model,
    feature_cols: List[str],
) -> pd.DataFrame:
    out = df.copy()

    predictions = model.predict(out[feature_cols])
    probabilities = model.predict_proba(out[feature_cols])
    classes = model.classes_

    out["router_predicted_policy"] = predictions
    out["router_predicted_policy_confidence"] = probabilities.max(axis=1)

    for class_index, class_name in enumerate(classes):
        out[f"router_predicted_prob_{class_name}"] = probabilities[:, class_index]

    out["router_predicted_policy_reward_at_5"] = out.apply(
        lambda row: get_policy_reward(row, row["router_predicted_policy"]),
        axis=1,
    )

    out["router_delta_vs_baseline"] = (
        out["router_predicted_policy_reward_at_5"] - out["baseline_reward_at_5"]
    )

    out["router_delta_vs_gated"] = (
        out["router_predicted_policy_reward_at_5"] - out["cortex_reward_at_5"]
    )

    out["router_delta_vs_full_cortex"] = (
        out["router_predicted_policy_reward_at_5"] - out["full_cortex_reward_at_5"]
    )

    out["router_regret_vs_oracle"] = (
        out["best_logged_policy_reward_at_5"] - out["router_predicted_policy_reward_at_5"]
    )

    out["router_prediction_matches_best_policy"] = (
        out["router_predicted_policy"].astype(str) == out["best_logged_policy"].astype(str)
    )

    out["router_near_oracle"] = out["router_regret_vs_oracle"] <= 0.01

    return out


def build_summary(df: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
    predicted_counts = df["router_predicted_policy"].value_counts().to_dict()
    oracle_counts = df["best_logged_policy"].value_counts().to_dict()

    baseline_avg = float(df["baseline_reward_at_5"].mean())
    gated_avg = float(df["cortex_reward_at_5"].mean())
    full_avg = float(df["full_cortex_reward_at_5"].mean())
    router_avg = float(df["router_predicted_policy_reward_at_5"].mean())
    oracle_avg = float(df["best_logged_policy_reward_at_5"].mean())

    summary = {
        "rows_scored": len(df),
        "model_name": metadata.get("model_name", "unknown"),
        "model_type": metadata.get("model_type", "unknown"),
        "training_rows": metadata.get("training_rows", None),
        "feature_count": metadata.get("feature_count", None),
        "avg_baseline_reward_at_5": baseline_avg,
        "avg_gated_cortex_reward_at_5": gated_avg,
        "avg_full_cortex_reward_at_5": full_avg,
        "avg_router_reward_at_5": router_avg,
        "avg_oracle_reward_at_5": oracle_avg,
        "router_delta_vs_baseline": router_avg - baseline_avg,
        "router_delta_vs_gated": router_avg - gated_avg,
        "router_delta_vs_full_cortex": router_avg - full_avg,
        "router_regret_vs_oracle": oracle_avg - router_avg,
        "prediction_match_rate": float(df["router_prediction_matches_best_policy"].mean()),
        "near_oracle_rate": float(df["router_near_oracle"].mean()),
        "predicted_baseline_count": int(predicted_counts.get("baseline", 0)),
        "predicted_gated_cortex_count": int(predicted_counts.get("gated_cortex", 0)),
        "predicted_full_cortex_count": int(predicted_counts.get("full_cortex", 0)),
        "oracle_baseline_count": int(oracle_counts.get("baseline", 0)),
        "oracle_gated_cortex_count": int(oracle_counts.get("gated_cortex", 0)),
        "oracle_full_cortex_count": int(oracle_counts.get("full_cortex", 0)),
    }

    return pd.DataFrame([summary])


def choose_display_columns(df: pd.DataFrame) -> pd.DataFrame:
    existing = [col for col in DISPLAY_COLUMNS if col in df.columns]

    prob_cols = [
        col for col in df.columns
        if col.startswith("router_predicted_prob_")
    ]

    out = df[existing + prob_cols].copy()

    out = out.sort_values(
        by=[
            "router_prediction_matches_best_policy",
            "router_predicted_policy_confidence",
        ],
        ascending=[True, False],
    )

    return out


def run_smoke_test(
    input_path: str,
    sample_size: int,
) -> None:
    ensure_output_dir()

    model = load_router_model(MODEL_PATH)
    feature_payload = load_json(FEATURES_PATH)
    metadata = load_json(METADATA_PATH)

    raw_df = load_input_data(input_path)
    raw_df = add_targets_if_missing(raw_df)

    if sample_size > 0 and sample_size < len(raw_df):
        df = raw_df.sample(
            n=sample_size,
            random_state=42,
        ).reset_index(drop=True)
    else:
        df = raw_df.copy()

    feature_df, feature_cols = build_feature_frame_from_artifact(
        df=df,
        feature_payload=feature_payload,
    )

    scored_df = run_router_inference(
        df=feature_df,
        model=model,
        feature_cols=feature_cols,
    )

    output_df = choose_display_columns(scored_df)
    summary_df = build_summary(scored_df, metadata=metadata)

    output_df.to_csv(SMOKE_TEST_PATH, index=False)
    summary_df.to_csv(SUMMARY_PATH, index=False)

    print()
    print("MVP 14.2 Saved Router Inference Smoke Test")
    print("=" * 100)
    print(f"Input: {input_path}")
    print(f"Rows scored: {len(scored_df)}")
    print(f"Model: {MODEL_PATH}")
    print(f"Features: {FEATURES_PATH}")
    print(f"Metadata: {METADATA_PATH}")

    print()
    print("Inference summary")
    print("-" * 100)
    print(summary_df.to_string(index=False))

    print()
    print("Sample predictions")
    print("-" * 100)
    print(output_df.head(20).to_string(index=False))

    print()
    print("Files written")
    print("-" * 100)
    print(f"- smoke test: {SMOKE_TEST_PATH}")
    print(f"- summary: {SUMMARY_PATH}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT_PATH,
        help="Path to critic guided repair simulation CSV.",
    )

    parser.add_argument(
        "--sample-size",
        type=int,
        default=0,
        help="Number of rows to score. Use 0 to score all rows.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_smoke_test(
        input_path=args.input,
        sample_size=args.sample_size,
    )