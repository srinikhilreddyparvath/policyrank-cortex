"""
MVP 14: Router-Integrated CORTEX Evaluator Dry Run

Purpose:
--------
Create a clean dry-run evaluator for the learned repair router.

This does NOT modify the live Streamlit app or the existing scalable evaluator.
It reads the out-of-sample router validation predictions and produces a final
integrated report comparing:

    1. Baseline semantic ranking
    2. Full CORTEX
    3. Gated CORTEX
    4. Learned Repair Router
    5. Oracle logged-policy upper bound

Why this matters:
-----------------
MVP 13.9 proved that the Learned Repair Router works out-of-sample.
MVP 14 packages that into an integrated evaluation artifact that can be shown
as the next major CORTEX milestone.

Inputs:
-------
outputs/repair_router_oos_validation_predictions.csv

Optional:
---------
outputs/repair_router_oos_validation_summary.csv
outputs/repair_router_oos_validation_feature_importance.csv

Outputs:
--------
outputs/router_integrated_eval_summary.csv
outputs/router_integrated_eval_by_split.csv
outputs/router_integrated_eval_by_policy.csv
outputs/router_integrated_eval_by_route.csv
outputs/router_integrated_eval_high_impact.csv
outputs/router_integrated_eval_decision_audit.csv

Important:
----------
This is a dry-run report. It uses router predictions from the validation file.
It does not retrain models and does not change live ranking behavior.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List

import pandas as pd


DEFAULT_INPUT_PATH = "outputs/repair_router_oos_validation_predictions.csv"
OUTPUT_DIR = "outputs"

SUMMARY_PATH = os.path.join(OUTPUT_DIR, "router_integrated_eval_summary.csv")
BY_SPLIT_PATH = os.path.join(OUTPUT_DIR, "router_integrated_eval_by_split.csv")
BY_POLICY_PATH = os.path.join(OUTPUT_DIR, "router_integrated_eval_by_policy.csv")
BY_ROUTE_PATH = os.path.join(OUTPUT_DIR, "router_integrated_eval_by_route.csv")
HIGH_IMPACT_PATH = os.path.join(OUTPUT_DIR, "router_integrated_eval_high_impact.csv")
DECISION_AUDIT_PATH = os.path.join(OUTPUT_DIR, "router_integrated_eval_decision_audit.csv")


NUMERIC_COLUMNS = [
    "baseline_reward_at_5",
    "cortex_reward_at_5",
    "full_cortex_reward_at_5",
    "predicted_policy_reward_at_5",
    "best_logged_policy_reward_at_5",
    "predicted_delta_vs_current_gated",
    "predicted_delta_vs_full_cortex",
    "predicted_delta_vs_baseline",
    "oracle_delta_vs_current_gated",
    "critic_risk_score",
    "gate_baseline_confidence_score",
    "gate_contract_alignment_score",
    "gate_final_gate_score",
    "gate_top1_score",
    "gate_top5_mean_score",
    "gate_label_quality_score",
    "predicted_policy_confidence",
    "predicted_prob_baseline",
    "predicted_prob_gated_cortex",
    "predicted_prob_full_cortex",
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
    "predicted_prob_baseline",
    "predicted_prob_gated_cortex",
    "predicted_prob_full_cortex",
]


def ensure_output_dir() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def safe_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def load_predictions(input_path: str) -> pd.DataFrame:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Could not find router prediction file: {input_path}")

    df = pd.read_csv(input_path)

    if len(df) == 0:
        raise ValueError(f"Router prediction file is empty: {input_path}")

    required_cols = [
        "query",
        "split",
        "best_logged_policy",
        "predicted_policy",
        "baseline_reward_at_5",
        "cortex_reward_at_5",
        "full_cortex_reward_at_5",
        "predicted_policy_reward_at_5",
        "best_logged_policy_reward_at_5",
    ]

    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = safe_numeric(df[col])

    if "prediction_matches_best_policy" in df.columns:
        df["prediction_matches_best_policy"] = safe_bool(df["prediction_matches_best_policy"])
    else:
        df["prediction_matches_best_policy"] = (
            df["predicted_policy"].astype(str) == df["best_logged_policy"].astype(str)
        )

    return df.reset_index(drop=True)


def add_integrated_metrics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["baseline_lift_vs_baseline"] = 0.0

    out["gated_lift_vs_baseline"] = (
        out["cortex_reward_at_5"] - out["baseline_reward_at_5"]
    )

    out["full_cortex_lift_vs_baseline"] = (
        out["full_cortex_reward_at_5"] - out["baseline_reward_at_5"]
    )

    out["router_lift_vs_baseline"] = (
        out["predicted_policy_reward_at_5"] - out["baseline_reward_at_5"]
    )

    out["oracle_lift_vs_baseline"] = (
        out["best_logged_policy_reward_at_5"] - out["baseline_reward_at_5"]
    )

    out["router_delta_vs_gated"] = (
        out["predicted_policy_reward_at_5"] - out["cortex_reward_at_5"]
    )

    out["router_delta_vs_full_cortex"] = (
        out["predicted_policy_reward_at_5"] - out["full_cortex_reward_at_5"]
    )

    out["router_delta_vs_oracle"] = (
        out["predicted_policy_reward_at_5"] - out["best_logged_policy_reward_at_5"]
    )

    out["router_improves_over_gated"] = out["router_delta_vs_gated"] > 0.0001
    out["router_hurts_vs_gated"] = out["router_delta_vs_gated"] < -0.0001
    out["router_ties_gated"] = out["router_delta_vs_gated"].abs() <= 0.0001

    out["router_improves_over_full"] = out["router_delta_vs_full_cortex"] > 0.0001
    out["router_hurts_vs_full"] = out["router_delta_vs_full_cortex"] < -0.0001
    out["router_ties_full"] = out["router_delta_vs_full_cortex"].abs() <= 0.0001

    out["router_regret_vs_oracle"] = (
        out["best_logged_policy_reward_at_5"] - out["predicted_policy_reward_at_5"]
    )

    out["router_near_oracle"] = out["router_regret_vs_oracle"] <= 0.01

    out["router_confidence_bucket"] = pd.cut(
        out["predicted_policy_confidence"],
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

    predicted_counts = df["predicted_policy"].value_counts().to_dict()
    oracle_counts = df["best_logged_policy"].value_counts().to_dict()

    return {
        label: value,
        "query_count": total,
        "avg_baseline_reward_at_5": float(df["baseline_reward_at_5"].mean()),
        "avg_gated_cortex_reward_at_5": float(df["cortex_reward_at_5"].mean()),
        "avg_full_cortex_reward_at_5": float(df["full_cortex_reward_at_5"].mean()),
        "avg_router_reward_at_5": float(df["predicted_policy_reward_at_5"].mean()),
        "avg_oracle_reward_at_5": float(df["best_logged_policy_reward_at_5"].mean()),
        "avg_gated_lift_vs_baseline": float(df["gated_lift_vs_baseline"].mean()),
        "avg_full_cortex_lift_vs_baseline": float(df["full_cortex_lift_vs_baseline"].mean()),
        "avg_router_lift_vs_baseline": float(df["router_lift_vs_baseline"].mean()),
        "avg_oracle_lift_vs_baseline": float(df["oracle_lift_vs_baseline"].mean()),
        "avg_router_delta_vs_gated": float(df["router_delta_vs_gated"].mean()),
        "avg_router_delta_vs_full_cortex": float(df["router_delta_vs_full_cortex"].mean()),
        "avg_router_regret_vs_oracle": float(df["router_regret_vs_oracle"].mean()),
        "median_router_lift_vs_baseline": float(df["router_lift_vs_baseline"].median()),
        "median_gated_lift_vs_baseline": float(df["gated_lift_vs_baseline"].median()),
        "median_full_cortex_lift_vs_baseline": float(df["full_cortex_lift_vs_baseline"].median()),
        "prediction_match_rate": float(df["prediction_matches_best_policy"].mean()),
        "router_near_oracle_rate": float(df["router_near_oracle"].mean()),
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


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []

    rows.append(
        summarize_slice(
            df=df,
            label="scope",
            value="all",
        )
    )

    test_df = df[df["split"].astype(str).str.lower() == "test"].copy()

    if len(test_df) > 0:
        rows.append(
            summarize_slice(
                df=test_df,
                label="scope",
                value="test_only",
            )
        )

    train_df = df[df["split"].astype(str).str.lower() == "train"].copy()

    if len(train_df) > 0:
        rows.append(
            summarize_slice(
                df=train_df,
                label="scope",
                value="train_only",
            )
        )

    return pd.DataFrame(rows)


def build_by_split(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []

    for split, group in df.groupby("split"):
        rows.append(
            summarize_slice(
                df=group,
                label="split",
                value=str(split),
            )
        )

    return pd.DataFrame(rows)


def build_by_policy(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []

    for policy, group in df.groupby("predicted_policy"):
        rows.append(
            summarize_slice(
                df=group,
                label="predicted_policy",
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
            df["prediction_matches_best_policy"] == False
        )
    ].copy()

    high_impact = high_impact.sort_values(
        by=[
            "split",
            "router_delta_vs_gated",
            "router_delta_vs_full_cortex",
            "router_regret_vs_oracle",
        ],
        ascending=[True, False, False, False],
    )

    existing = [col for col in DISPLAY_COLUMNS if col in high_impact.columns]

    extra_cols = [
        "router_lift_vs_baseline",
        "router_delta_vs_gated",
        "router_delta_vs_full_cortex",
        "router_regret_vs_oracle",
        "router_confidence_bucket",
    ]

    existing_extra = [col for col in extra_cols if col in high_impact.columns]

    return high_impact[existing + existing_extra].copy()


def build_decision_audit(df: pd.DataFrame) -> pd.DataFrame:
    audit = df.copy()

    existing = [col for col in DISPLAY_COLUMNS if col in audit.columns]

    extra_cols = [
        "router_lift_vs_baseline",
        "gated_lift_vs_baseline",
        "full_cortex_lift_vs_baseline",
        "oracle_lift_vs_baseline",
        "router_delta_vs_gated",
        "router_delta_vs_full_cortex",
        "router_regret_vs_oracle",
        "router_confidence_bucket",
        "router_improves_over_gated",
        "router_improves_over_full",
        "router_near_oracle",
    ]

    existing_extra = [col for col in extra_cols if col in audit.columns]

    audit = audit[existing + existing_extra].copy()

    audit = audit.sort_values(
        by=[
            "split",
            "prediction_matches_best_policy",
            "predicted_policy_confidence",
        ],
        ascending=[True, True, False],
    )

    return audit


def print_report(
    summary_df: pd.DataFrame,
    by_split_df: pd.DataFrame,
    by_policy_df: pd.DataFrame,
    by_route_df: pd.DataFrame,
) -> None:
    print()
    print("MVP 14 Router-Integrated CORTEX Evaluator Dry Run")
    print("=" * 120)

    print()
    print("Integrated summary")
    print("-" * 120)
    print(summary_df.to_string(index=False))

    print()
    print("By split")
    print("-" * 120)
    print(by_split_df.to_string(index=False))

    print()
    print("By predicted policy")
    print("-" * 120)
    print(by_policy_df.to_string(index=False))

    if len(by_route_df) > 0:
        print()
        print("By governance route")
        print("-" * 120)
        print(by_route_df.to_string(index=False))

    print()
    print("Files written")
    print("-" * 120)
    print(f"- summary: {SUMMARY_PATH}")
    print(f"- by split: {BY_SPLIT_PATH}")
    print(f"- by policy: {BY_POLICY_PATH}")
    print(f"- by route: {BY_ROUTE_PATH}")
    print(f"- high impact: {HIGH_IMPACT_PATH}")
    print(f"- decision audit: {DECISION_AUDIT_PATH}")


def run_router_integrated_eval(input_path: str) -> None:
    ensure_output_dir()

    raw_df = load_predictions(input_path)
    df = add_integrated_metrics(raw_df)

    summary_df = build_summary(df)
    by_split_df = build_by_split(df)
    by_policy_df = build_by_policy(df)
    by_route_df = build_by_route(df)
    high_impact_df = build_high_impact(df)
    decision_audit_df = build_decision_audit(df)

    summary_df.to_csv(SUMMARY_PATH, index=False)
    by_split_df.to_csv(BY_SPLIT_PATH, index=False)
    by_policy_df.to_csv(BY_POLICY_PATH, index=False)
    by_route_df.to_csv(BY_ROUTE_PATH, index=False)
    high_impact_df.to_csv(HIGH_IMPACT_PATH, index=False)
    decision_audit_df.to_csv(DECISION_AUDIT_PATH, index=False)

    print_report(
        summary_df=summary_df,
        by_split_df=by_split_df,
        by_policy_df=by_policy_df,
        by_route_df=by_route_df,
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT_PATH,
        help="Path to out-of-sample router validation predictions CSV.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_router_integrated_eval(input_path=args.input)