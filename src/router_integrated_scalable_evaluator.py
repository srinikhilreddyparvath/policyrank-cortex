"""
MVP 14.3: Router-Integrated Scalable Evaluator Dry Run

Purpose:
--------
Create a router-integrated scalable evaluation report using the saved Learned
Repair Router model.

This is a dry-run evaluator:
- It does NOT modify Streamlit.
- It does NOT modify the existing scalable_evaluator.py.
- It loads the saved router model from disk.
- It scores existing evaluation rows.
- It compares:
    1. Baseline semantic ranking
    2. Full CORTEX
    3. Gated CORTEX
    4. Router-Integrated CORTEX
    5. Oracle logged-policy upper bound

Inputs:
-------
models/learned_repair_router.pkl
models/learned_repair_router_features.json
models/learned_repair_router_metadata.json
outputs/critic_guided_repair_simulation.csv

Outputs:
--------
outputs/router_integrated_scalable_eval.csv
outputs/router_integrated_scalable_eval_summary.csv
outputs/router_integrated_scalable_eval_by_policy.csv
outputs/router_integrated_scalable_eval_by_route.csv
outputs/router_integrated_scalable_eval_by_split.csv
outputs/router_integrated_scalable_eval_high_impact.csv

Why this matters:
-----------------
MVP 14.2 proved the saved model can load and score rows.
MVP 14.3 turns that model into a scalable evaluator report that can be shown
as a formal CORTEX policy comparison artifact.
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

EVAL_PATH = os.path.join(OUTPUT_DIR, "router_integrated_scalable_eval.csv")
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "router_integrated_scalable_eval_summary.csv")
BY_POLICY_PATH = os.path.join(OUTPUT_DIR, "router_integrated_scalable_eval_by_policy.csv")
BY_ROUTE_PATH = os.path.join(OUTPUT_DIR, "router_integrated_scalable_eval_by_route.csv")
BY_SPLIT_PATH = os.path.join(OUTPUT_DIR, "router_integrated_scalable_eval_by_split.csv")
HIGH_IMPACT_PATH = os.path.join(OUTPUT_DIR, "router_integrated_scalable_eval_high_impact.csv")


POLICY_LABELS = [
    "baseline",
    "gated_cortex",
    "full_cortex",
]


NUMERIC_COLUMNS = [
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
    "positive_contract_rows",
    "blocked_rows",
    "low_coverage",
    "conservative_repair_reward_at_5",
    "oracle_logged_policy_reward_at_5",
]


BOOL_COLUMNS = [
    "needs_online_critic",
    "baseline_loss_rescued",
    "over_rerank_prevented",
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
    "baseline_lift_vs_baseline",
    "gated_lift_vs_baseline",
    "full_cortex_lift_vs_baseline",
    "router_lift_vs_baseline",
    "oracle_lift_vs_baseline",
    "router_delta_vs_gated",
    "router_delta_vs_full_cortex",
    "router_delta_vs_baseline",
    "router_regret_vs_oracle",
    "router_near_oracle",
    "router_improves_over_gated",
    "router_improves_over_full",
    "critic_priority",
    "critic_risk_score",
    "critic_failure_modes",
    "critic_repair_actions",
    "primary_repair_action",
    "governance_route",
    "gate_decision",
    "gate_reason",
    "gate_baseline_confidence_score",
    "gate_contract_alignment_score",
    "gate_final_gate_score",
    "gate_top1_score",
    "gate_top5_mean_score",
    "gate_label_quality_score",
    "gate_exclusion_violation_rate",
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


def load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find model artifact: {path}")

    with open(path, "rb") as f:
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

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = safe_numeric(df[col])

    for col in BOOL_COLUMNS:
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


def add_targets_if_missing(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

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

    for feature in expected_features:
        if feature not in out.columns:
            out[feature] = 0.0

        out[feature] = safe_numeric(out[feature])

    if not expected_features:
        raise ValueError("Feature artifact did not contain feature_columns.")

    return out, expected_features


def run_router_prediction(
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

    out["router_prediction_matches_best_policy"] = (
        out["router_predicted_policy"].astype(str) == out["best_logged_policy"].astype(str)
    )

    return out


def add_router_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["baseline_lift_vs_baseline"] = 0.0

    out["gated_lift_vs_baseline"] = (
        out["cortex_reward_at_5"] - out["baseline_reward_at_5"]
    )

    out["full_cortex_lift_vs_baseline"] = (
        out["full_cortex_reward_at_5"] - out["baseline_reward_at_5"]
    )

    out["router_lift_vs_baseline"] = (
        out["router_predicted_policy_reward_at_5"] - out["baseline_reward_at_5"]
    )

    out["oracle_lift_vs_baseline"] = (
        out["best_logged_policy_reward_at_5"] - out["baseline_reward_at_5"]
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

    out["router_near_oracle"] = out["router_regret_vs_oracle"] <= 0.01

    out["router_improves_over_gated"] = out["router_delta_vs_gated"] > 0.0001
    out["router_hurts_vs_gated"] = out["router_delta_vs_gated"] < -0.0001
    out["router_ties_gated"] = out["router_delta_vs_gated"].abs() <= 0.0001

    out["router_improves_over_full"] = out["router_delta_vs_full_cortex"] > 0.0001
    out["router_hurts_vs_full"] = out["router_delta_vs_full_cortex"] < -0.0001
    out["router_ties_full"] = out["router_delta_vs_full_cortex"].abs() <= 0.0001

    out["router_confidence_bucket"] = pd.cut(
        out["router_predicted_policy_confidence"],
        bins=[-0.001, 0.40, 0.55, 0.70, 0.85, 1.001],
        labels=[
            "very_low",
            "low",
            "medium",
            "high",
            "very_high",
        ],
    ).astype(str)

    return out


def summarize_slice(df: pd.DataFrame, label: str, value: str) -> Dict:
    total = len(df)

    predicted_counts = df["router_predicted_policy"].value_counts().to_dict()
    oracle_counts = df["best_logged_policy"].value_counts().to_dict()

    baseline_avg = float(df["baseline_reward_at_5"].mean())
    gated_avg = float(df["cortex_reward_at_5"].mean())
    full_avg = float(df["full_cortex_reward_at_5"].mean())
    router_avg = float(df["router_predicted_policy_reward_at_5"].mean())
    oracle_avg = float(df["best_logged_policy_reward_at_5"].mean())

    return {
        label: value,
        "query_count": total,
        "avg_baseline_reward_at_5": baseline_avg,
        "avg_gated_cortex_reward_at_5": gated_avg,
        "avg_full_cortex_reward_at_5": full_avg,
        "avg_router_reward_at_5": router_avg,
        "avg_oracle_reward_at_5": oracle_avg,
        "avg_gated_lift_vs_baseline": gated_avg - baseline_avg,
        "avg_full_cortex_lift_vs_baseline": full_avg - baseline_avg,
        "avg_router_lift_vs_baseline": router_avg - baseline_avg,
        "avg_oracle_lift_vs_baseline": oracle_avg - baseline_avg,
        "avg_router_delta_vs_gated": router_avg - gated_avg,
        "avg_router_delta_vs_full_cortex": router_avg - full_avg,
        "avg_router_regret_vs_oracle": oracle_avg - router_avg,
        "median_gated_lift_vs_baseline": float(df["gated_lift_vs_baseline"].median()),
        "median_full_cortex_lift_vs_baseline": float(df["full_cortex_lift_vs_baseline"].median()),
        "median_router_lift_vs_baseline": float(df["router_lift_vs_baseline"].median()),
        "median_oracle_lift_vs_baseline": float(df["oracle_lift_vs_baseline"].median()),
        "prediction_match_rate": float(df["router_prediction_matches_best_policy"].mean()),
        "near_oracle_rate": float(df["router_near_oracle"].mean()),
        "router_improves_over_gated_count": int(df["router_improves_over_gated"].sum()),
        "router_hurts_vs_gated_count": int(df["router_hurts_vs_gated"].sum()),
        "router_ties_gated_count": int(df["router_ties_gated"].sum()),
        "router_improves_over_full_count": int(df["router_improves_over_full"].sum()),
        "router_hurts_vs_full_count": int(df["router_hurts_vs_full"].sum()),
        "router_ties_full_count": int(df["router_ties_full"].sum()),
        "predicted_baseline_count": int(predicted_counts.get("baseline", 0)),
        "predicted_gated_cortex_count": int(predicted_counts.get("gated_cortex", 0)),
        "predicted_full_cortex_count": int(predicted_counts.get("full_cortex", 0)),
        "oracle_baseline_count": int(oracle_counts.get("baseline", 0)),
        "oracle_gated_cortex_count": int(oracle_counts.get("gated_cortex", 0)),
        "oracle_full_cortex_count": int(oracle_counts.get("full_cortex", 0)),
    }


def build_summary(df: pd.DataFrame, metadata: Dict) -> pd.DataFrame:
    summary = summarize_slice(
        df=df,
        label="scope",
        value="all",
    )

    summary["model_name"] = metadata.get("model_name", "unknown")
    summary["model_type"] = metadata.get("model_type", "unknown")
    summary["training_rows"] = metadata.get("training_rows", None)
    summary["feature_count"] = metadata.get("feature_count", None)

    return pd.DataFrame([summary])


def build_by_policy(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []

    for policy, group in df.groupby("router_predicted_policy"):
        rows.append(
            summarize_slice(
                df=group,
                label="router_predicted_policy",
                value=str(policy),
            )
        )

    out = pd.DataFrame(rows)

    if len(out) > 0:
        out = out.sort_values(
            by="query_count",
            ascending=False,
        )

    return out


def build_by_route(df: pd.DataFrame) -> pd.DataFrame:
    if "governance_route" not in df.columns:
        return pd.DataFrame()

    rows: List[Dict] = []

    for route, group in df.groupby("governance_route"):
        rows.append(
            summarize_slice(
                df=group,
                label="governance_route",
                value=str(route),
            )
        )

    out = pd.DataFrame(rows)

    if len(out) > 0:
        out = out.sort_values(
            by="query_count",
            ascending=False,
        )

    return out


def build_by_split(df: pd.DataFrame) -> pd.DataFrame:
    if "split" not in df.columns:
        return pd.DataFrame()

    rows: List[Dict] = []

    for split, group in df.groupby("split"):
        rows.append(
            summarize_slice(
                df=group,
                label="split",
                value=str(split),
            )
        )

    out = pd.DataFrame(rows)

    if len(out) > 0:
        out = out.sort_values(
            by="query_count",
            ascending=False,
        )

    return out


def build_high_impact(df: pd.DataFrame) -> pd.DataFrame:
    high_impact = df[
        (
            df["router_delta_vs_gated"] >= 0.05
        )
        | (
            df["router_delta_vs_full_cortex"] >= 0.05
        )
        | (
            df["router_regret_vs_oracle"] >= 0.05
        )
        | (
            df["router_prediction_matches_best_policy"] == False
        )
    ].copy()

    high_impact = high_impact.sort_values(
        by=[
            "router_delta_vs_gated",
            "router_delta_vs_full_cortex",
            "router_regret_vs_oracle",
        ],
        ascending=[False, False, False],
    )

    existing = [col for col in DISPLAY_COLUMNS if col in high_impact.columns]

    prob_cols = [
        col for col in high_impact.columns
        if col.startswith("router_predicted_prob_")
    ]

    return high_impact[existing + prob_cols].copy()


def choose_eval_columns(df: pd.DataFrame) -> pd.DataFrame:
    existing = [col for col in DISPLAY_COLUMNS if col in df.columns]

    extra_cols = [
        "router_confidence_bucket",
        "router_hurts_vs_gated",
        "router_ties_gated",
        "router_hurts_vs_full",
        "router_ties_full",
    ]

    existing_extra = [col for col in extra_cols if col in df.columns]

    prob_cols = [
        col for col in df.columns
        if col.startswith("router_predicted_prob_")
    ]

    out = df[existing + existing_extra + prob_cols].copy()

    out = out.sort_values(
        by=[
            "router_prediction_matches_best_policy",
            "router_predicted_policy_confidence",
        ],
        ascending=[True, False],
    )

    return out


def print_report(
    summary_df: pd.DataFrame,
    by_policy_df: pd.DataFrame,
    by_route_df: pd.DataFrame,
    by_split_df: pd.DataFrame,
) -> None:
    print()
    print("MVP 14.3 Router-Integrated Scalable Evaluator Dry Run")
    print("=" * 120)

    print()
    print("Summary")
    print("-" * 120)
    print(summary_df.to_string(index=False))

    print()
    print("By router predicted policy")
    print("-" * 120)
    print(by_policy_df.to_string(index=False))

    if len(by_route_df) > 0:
        print()
        print("By governance route")
        print("-" * 120)
        print(by_route_df.to_string(index=False))

    if len(by_split_df) > 0:
        print()
        print("By split")
        print("-" * 120)
        print(by_split_df.to_string(index=False))

    print()
    print("Files written")
    print("-" * 120)
    print(f"- eval: {EVAL_PATH}")
    print(f"- summary: {SUMMARY_PATH}")
    print(f"- by policy: {BY_POLICY_PATH}")
    print(f"- by route: {BY_ROUTE_PATH}")
    print(f"- by split: {BY_SPLIT_PATH}")
    print(f"- high impact: {HIGH_IMPACT_PATH}")


def run_router_integrated_scalable_eval(
    input_path: str,
    sample_size: int,
) -> None:
    ensure_output_dir()

    model = load_model(MODEL_PATH)
    feature_payload = load_json(FEATURES_PATH)
    metadata = load_json(METADATA_PATH)

    raw_df = load_input_data(input_path)
    raw_df = add_targets_if_missing(raw_df)

    if sample_size > 0 and sample_size < len(raw_df):
        raw_df = raw_df.sample(
            n=sample_size,
            random_state=42,
        ).reset_index(drop=True)

    feature_df, feature_cols = build_feature_frame_from_artifact(
        df=raw_df,
        feature_payload=feature_payload,
    )

    scored_df = run_router_prediction(
        df=feature_df,
        model=model,
        feature_cols=feature_cols,
    )

    scored_df = add_router_metrics(scored_df)

    eval_df = choose_eval_columns(scored_df)
    summary_df = build_summary(scored_df, metadata=metadata)
    by_policy_df = build_by_policy(scored_df)
    by_route_df = build_by_route(scored_df)
    by_split_df = build_by_split(scored_df)
    high_impact_df = build_high_impact(scored_df)

    eval_df.to_csv(EVAL_PATH, index=False)
    summary_df.to_csv(SUMMARY_PATH, index=False)
    by_policy_df.to_csv(BY_POLICY_PATH, index=False)
    by_route_df.to_csv(BY_ROUTE_PATH, index=False)
    by_split_df.to_csv(BY_SPLIT_PATH, index=False)
    high_impact_df.to_csv(HIGH_IMPACT_PATH, index=False)

    print_report(
        summary_df=summary_df,
        by_policy_df=by_policy_df,
        by_route_df=by_route_df,
        by_split_df=by_split_df,
    )


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

    run_router_integrated_scalable_eval(
        input_path=args.input,
        sample_size=args.sample_size,
    )