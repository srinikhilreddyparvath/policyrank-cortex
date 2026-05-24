"""
MVP 14.6: Router Dry-Run Analyzer

Purpose:
--------
Analyze live router dry-run decisions saved from the Streamlit app.

Input:
------
storage/router_dry_run_log.csv

Outputs:
--------
outputs/router_dry_run_log_summary.csv
outputs/router_dry_run_log_by_policy.csv
outputs/router_dry_run_log_by_route.csv
outputs/router_dry_run_log_by_critic_priority.csv
outputs/router_dry_run_log_high_confidence.csv
outputs/router_dry_run_log_low_confidence.csv
outputs/router_dry_run_log_policy_probability_audit.csv

Why this matters:
-----------------
MVP 14.5 added live router dry-run logging.
MVP 14.6 turns those saved dry-run decisions into a small evaluation loop.

This helps answer:
- What does the router choose on manual test queries?
- Is it mostly selecting baseline, gated CORTEX, or full CORTEX?
- Which governance routes are being triggered?
- Which queries have high confidence?
- Which queries have low confidence / ambiguous router decisions?
- Are router probabilities close between gated and full CORTEX?
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List

import pandas as pd


DEFAULT_INPUT_PATH = "storage/router_dry_run_log.csv"
OUTPUT_DIR = "outputs"

SUMMARY_PATH = os.path.join(OUTPUT_DIR, "router_dry_run_log_summary.csv")
BY_POLICY_PATH = os.path.join(OUTPUT_DIR, "router_dry_run_log_by_policy.csv")
BY_ROUTE_PATH = os.path.join(OUTPUT_DIR, "router_dry_run_log_by_route.csv")
BY_CRITIC_PRIORITY_PATH = os.path.join(OUTPUT_DIR, "router_dry_run_log_by_critic_priority.csv")
HIGH_CONFIDENCE_PATH = os.path.join(OUTPUT_DIR, "router_dry_run_log_high_confidence.csv")
LOW_CONFIDENCE_PATH = os.path.join(OUTPUT_DIR, "router_dry_run_log_low_confidence.csv")
PROBABILITY_AUDIT_PATH = os.path.join(OUTPUT_DIR, "router_dry_run_log_policy_probability_audit.csv")


NUMERIC_COLUMNS = [
    "router_confidence",
    "router_prob_baseline",
    "router_prob_gated_cortex",
    "router_prob_full_cortex",
    "critic_risk_score",
    "gate_final_gate_score",
    "gate_contract_alignment_score",
    "gate_baseline_confidence_score",
    "gate_label_quality_score",
    "gate_exclusion_violation_rate",
    "positive_contract_rows",
    "blocked_rows",
    "low_coverage",
    "slate_reward_at_5",
]


DISPLAY_COLUMNS = [
    "timestamp",
    "query",
    "retrieval_mode",
    "contract_mode",
    "policy_mode",
    "selected_policy",
    "router_predicted_policy",
    "router_confidence",
    "router_prob_baseline",
    "router_prob_gated_cortex",
    "router_prob_full_cortex",
    "probability_margin_top1_top2",
    "governance_route",
    "critic_priority",
    "critic_risk_score",
    "gate_final_gate_score",
    "gate_contract_alignment_score",
    "gate_baseline_confidence_score",
    "gate_label_quality_score",
    "gate_exclusion_violation_rate",
    "positive_contract_rows",
    "blocked_rows",
    "low_coverage",
    "top_baseline_product",
    "top_cortex_product",
    "slate_reward_at_5",
]


def ensure_output_dir() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def load_router_log(input_path: str) -> pd.DataFrame:
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Could not find router dry-run log: {input_path}. "
            "Run Streamlit, use Router-Integrated CORTEX Dry Run, and click "
            "'Save router dry-run decision' first."
        )

    df = pd.read_csv(input_path)

    if len(df) == 0:
        raise ValueError(f"Router dry-run log is empty: {input_path}")

    required_cols = [
        "query",
        "router_predicted_policy",
        "router_confidence",
        "governance_route",
        "critic_priority",
    ]

    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns in router log: {missing}")

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = safe_numeric(df[col])

    return df.reset_index(drop=True)


def add_probability_audit_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    probability_cols = [
        "router_prob_baseline",
        "router_prob_gated_cortex",
        "router_prob_full_cortex",
    ]

    for col in probability_cols:
        if col not in out.columns:
            out[col] = 0.0

        out[col] = safe_numeric(out[col])

    def get_top_policy_probability(row: pd.Series) -> float:
        return max(
            float(row.get("router_prob_baseline", 0.0)),
            float(row.get("router_prob_gated_cortex", 0.0)),
            float(row.get("router_prob_full_cortex", 0.0)),
        )

    def get_second_policy_probability(row: pd.Series) -> float:
        probs = [
            float(row.get("router_prob_baseline", 0.0)),
            float(row.get("router_prob_gated_cortex", 0.0)),
            float(row.get("router_prob_full_cortex", 0.0)),
        ]

        probs = sorted(probs, reverse=True)

        if len(probs) < 2:
            return 0.0

        return probs[1]

    out["top_policy_probability"] = out.apply(get_top_policy_probability, axis=1)
    out["second_policy_probability"] = out.apply(get_second_policy_probability, axis=1)
    out["probability_margin_top1_top2"] = (
        out["top_policy_probability"] - out["second_policy_probability"]
    )

    out["router_is_low_confidence"] = out["router_confidence"] < 0.50
    out["router_is_high_confidence"] = out["router_confidence"] >= 0.70
    out["router_is_ambiguous"] = out["probability_margin_top1_top2"] < 0.10

    return out


def summarize_slice(df: pd.DataFrame, label: str, value: str) -> Dict:
    total = len(df)

    if total == 0:
        return {
            label: value,
            "query_count": 0,
        }

    policy_counts = df["router_predicted_policy"].value_counts().to_dict()

    summary = {
        label: value,
        "query_count": total,
        "avg_router_confidence": float(df["router_confidence"].mean()),
        "median_router_confidence": float(df["router_confidence"].median()),
        "avg_slate_reward_at_5": float(df["slate_reward_at_5"].mean())
        if "slate_reward_at_5" in df.columns
        else 0.0,
        "avg_critic_risk_score": float(df["critic_risk_score"].mean())
        if "critic_risk_score" in df.columns
        else 0.0,
        "avg_gate_final_gate_score": float(df["gate_final_gate_score"].mean())
        if "gate_final_gate_score" in df.columns
        else 0.0,
        "avg_gate_contract_alignment_score": float(df["gate_contract_alignment_score"].mean())
        if "gate_contract_alignment_score" in df.columns
        else 0.0,
        "avg_gate_baseline_confidence_score": float(df["gate_baseline_confidence_score"].mean())
        if "gate_baseline_confidence_score" in df.columns
        else 0.0,
        "avg_probability_margin_top1_top2": float(df["probability_margin_top1_top2"].mean())
        if "probability_margin_top1_top2" in df.columns
        else 0.0,
        "low_confidence_count": int(df["router_is_low_confidence"].sum())
        if "router_is_low_confidence" in df.columns
        else 0,
        "high_confidence_count": int(df["router_is_high_confidence"].sum())
        if "router_is_high_confidence" in df.columns
        else 0,
        "ambiguous_decision_count": int(df["router_is_ambiguous"].sum())
        if "router_is_ambiguous" in df.columns
        else 0,
        "predicted_baseline_count": int(policy_counts.get("baseline", 0)),
        "predicted_gated_cortex_count": int(policy_counts.get("gated_cortex", 0)),
        "predicted_full_cortex_count": int(policy_counts.get("full_cortex", 0)),
    }

    return summary


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)

    policy_counts = df["router_predicted_policy"].value_counts().to_dict()
    route_counts = df["governance_route"].value_counts().to_dict()
    priority_counts = df["critic_priority"].value_counts().to_dict()

    latest_timestamp = ""
    earliest_timestamp = ""

    if "timestamp" in df.columns:
        timestamps = df["timestamp"].dropna().astype(str)
        if len(timestamps) > 0:
            earliest_timestamp = timestamps.min()
            latest_timestamp = timestamps.max()

    row = {
        "total_logged_queries": total,
        "unique_queries": int(df["query"].nunique()) if "query" in df.columns else total,
        "earliest_timestamp": earliest_timestamp,
        "latest_timestamp": latest_timestamp,
        "avg_router_confidence": float(df["router_confidence"].mean()),
        "median_router_confidence": float(df["router_confidence"].median()),
        "avg_probability_margin_top1_top2": float(df["probability_margin_top1_top2"].mean()),
        "low_confidence_count": int(df["router_is_low_confidence"].sum()),
        "high_confidence_count": int(df["router_is_high_confidence"].sum()),
        "ambiguous_decision_count": int(df["router_is_ambiguous"].sum()),
        "avg_slate_reward_at_5": float(df["slate_reward_at_5"].mean())
        if "slate_reward_at_5" in df.columns
        else 0.0,
        "avg_critic_risk_score": float(df["critic_risk_score"].mean())
        if "critic_risk_score" in df.columns
        else 0.0,
        "avg_gate_final_gate_score": float(df["gate_final_gate_score"].mean())
        if "gate_final_gate_score" in df.columns
        else 0.0,
        "predicted_baseline_count": int(policy_counts.get("baseline", 0)),
        "predicted_gated_cortex_count": int(policy_counts.get("gated_cortex", 0)),
        "predicted_full_cortex_count": int(policy_counts.get("full_cortex", 0)),
        "critic_needed_route_count": int(route_counts.get("critic_needed_route", 0)),
        "full_cortex_route_count": int(route_counts.get("full_cortex_route", 0)),
        "mission_candidate_route_count": int(route_counts.get("mission_candidate_route", 0)),
        "light_rerank_route_count": int(route_counts.get("light_rerank_route", 0)),
        "critical_priority_count": int(priority_counts.get("critical", 0)),
        "high_priority_count": int(priority_counts.get("high", 0)),
        "medium_priority_count": int(priority_counts.get("medium", 0)),
        "low_priority_count": int(priority_counts.get("low", 0)),
    }

    return pd.DataFrame([row])


def build_group_summary(df: pd.DataFrame, group_col: str, output_label: str) -> pd.DataFrame:
    if group_col not in df.columns:
        return pd.DataFrame()

    rows: List[Dict] = []

    for group_value, group in df.groupby(group_col):
        rows.append(
            summarize_slice(
                df=group,
                label=output_label,
                value=str(group_value),
            )
        )

    out = pd.DataFrame(rows)

    if len(out) > 0:
        out = out.sort_values(
            by="query_count",
            ascending=False,
        )

    return out


def choose_display_columns(df: pd.DataFrame) -> pd.DataFrame:
    existing = [col for col in DISPLAY_COLUMNS if col in df.columns]

    if not existing:
        return df

    return df[existing].copy()


def build_probability_audit(df: pd.DataFrame) -> pd.DataFrame:
    audit_df = df.copy()

    existing = [col for col in DISPLAY_COLUMNS if col in audit_df.columns]

    audit_df = audit_df.sort_values(
        by=[
            "router_is_ambiguous",
            "probability_margin_top1_top2",
            "router_confidence",
        ],
        ascending=[False, True, True],
    )

    return audit_df[existing].copy()


def run_analyzer(input_path: str) -> None:
    ensure_output_dir()

    df = load_router_log(input_path)
    df = add_probability_audit_columns(df)

    summary_df = build_summary(df)
    by_policy_df = build_group_summary(
        df=df,
        group_col="router_predicted_policy",
        output_label="router_predicted_policy",
    )
    by_route_df = build_group_summary(
        df=df,
        group_col="governance_route",
        output_label="governance_route",
    )
    by_priority_df = build_group_summary(
        df=df,
        group_col="critic_priority",
        output_label="critic_priority",
    )

    high_confidence_df = df[df["router_is_high_confidence"]].copy()
    low_confidence_df = df[df["router_is_low_confidence"]].copy()
    probability_audit_df = build_probability_audit(df)

    summary_df.to_csv(SUMMARY_PATH, index=False)
    by_policy_df.to_csv(BY_POLICY_PATH, index=False)
    by_route_df.to_csv(BY_ROUTE_PATH, index=False)
    by_priority_df.to_csv(BY_CRITIC_PRIORITY_PATH, index=False)
    choose_display_columns(high_confidence_df).to_csv(HIGH_CONFIDENCE_PATH, index=False)
    choose_display_columns(low_confidence_df).to_csv(LOW_CONFIDENCE_PATH, index=False)
    probability_audit_df.to_csv(PROBABILITY_AUDIT_PATH, index=False)

    print()
    print("MVP 14.6 Router Dry-Run Analyzer")
    print("=" * 100)
    print(f"Input: {input_path}")
    print(f"Rows analyzed: {len(df)}")

    print()
    print("Summary")
    print("-" * 100)
    print(summary_df.to_string(index=False))

    print()
    print("By router predicted policy")
    print("-" * 100)
    print(by_policy_df.to_string(index=False))

    print()
    print("By governance route")
    print("-" * 100)
    print(by_route_df.to_string(index=False))

    print()
    print("By critic priority")
    print("-" * 100)
    print(by_priority_df.to_string(index=False))

    print()
    print("Files written")
    print("-" * 100)
    print(f"- summary: {SUMMARY_PATH}")
    print(f"- by policy: {BY_POLICY_PATH}")
    print(f"- by route: {BY_ROUTE_PATH}")
    print(f"- by critic priority: {BY_CRITIC_PRIORITY_PATH}")
    print(f"- high confidence: {HIGH_CONFIDENCE_PATH}")
    print(f"- low confidence: {LOW_CONFIDENCE_PATH}")
    print(f"- probability audit: {PROBABILITY_AUDIT_PATH}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT_PATH,
        help="Path to router dry-run log CSV.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_analyzer(input_path=args.input)