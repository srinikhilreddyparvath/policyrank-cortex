"""
MVP 13.4 Step 1: Gate Calibration Analyzer

Purpose:
--------
Analyze MVP 13.3.1 gated evaluation results and identify how the baseline-aware
gate should be calibrated.

This analyzer reads the scalable evaluator output CSV and produces:
1. Overall gated vs full CORTEX comparison
2. Gate decision breakdown
3. Helped/hurt analysis
4. Baseline-loss rescue analysis
5. Recommended threshold hints for MVP 13.4 adaptive gate
6. CSV outputs for deeper inspection

Input:
------
Default:
outputs/scalable_eval_mvp13_3_1_resilient_250.csv

Outputs:
--------
outputs/gate_calibration_summary.csv
outputs/gate_calibration_by_decision.csv
outputs/gate_calibration_helped_queries.csv
outputs/gate_calibration_hurt_queries.csv
outputs/gate_calibration_rescued_queries.csv
outputs/gate_calibration_threshold_hints.csv
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import pandas as pd


DEFAULT_INPUT_PATH = "outputs/scalable_eval_mvp13_3_1_resilient_250.csv"
OUTPUT_DIR = "outputs"

SUMMARY_PATH = os.path.join(OUTPUT_DIR, "gate_calibration_summary.csv")
BY_DECISION_PATH = os.path.join(OUTPUT_DIR, "gate_calibration_by_decision.csv")
HELPED_QUERIES_PATH = os.path.join(OUTPUT_DIR, "gate_calibration_helped_queries.csv")
HURT_QUERIES_PATH = os.path.join(OUTPUT_DIR, "gate_calibration_hurt_queries.csv")
RESCUED_QUERIES_PATH = os.path.join(OUTPUT_DIR, "gate_calibration_rescued_queries.csv")
THRESHOLD_HINTS_PATH = os.path.join(OUTPUT_DIR, "gate_calibration_threshold_hints.csv")


NUMERIC_COLUMNS = [
    "baseline_reward_at_5",
    "cortex_reward_at_5",
    "full_cortex_reward_at_5",
    "absolute_lift",
    "percentage_lift",
    "full_cortex_absolute_lift",
    "full_cortex_percentage_lift",
    "gate_delta_vs_full_cortex",
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


DISPLAY_COLUMNS = [
    "query",
    "winner",
    "full_cortex_winner",
    "baseline_loss_rescued",
    "over_rerank_prevented",
    "gate_decision",
    "gate_reason",
    "fallback_used",
    "contract_source",
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
    "product_type",
    "selected_rl_action",
    "enforcement_status",
    "low_coverage",
    "error",
]


def ensure_output_dir() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def safe_rate(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0

    return numerator / denominator


def load_eval(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Could not find evaluation file: {path}")

    df = pd.read_csv(path)

    if len(df) == 0:
        raise ValueError(f"Evaluation file is empty: {path}")

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    for bool_col in ["baseline_loss_rescued", "over_rerank_prevented"]:
        if bool_col in df.columns:
            df[bool_col] = df[bool_col].astype(str).str.lower().isin(["true", "1", "yes"])

    if "error" in df.columns:
        valid_df = df[df["error"].fillna("").astype(str).str.strip() == ""].copy()
    else:
        valid_df = df.copy()

    return valid_df.reset_index(drop=True)


def add_analysis_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["gate_helped_vs_full_cortex"] = out["gate_delta_vs_full_cortex"] > 0.0001
    out["gate_hurt_vs_full_cortex"] = out["gate_delta_vs_full_cortex"] < -0.0001
    out["gate_neutral_vs_full_cortex"] = out["gate_delta_vs_full_cortex"].abs() <= 0.0001

    out["gated_beats_baseline"] = out["absolute_lift"] > 0.0001
    out["full_cortex_beats_baseline"] = out["full_cortex_absolute_lift"] > 0.0001

    out["gated_loses_to_baseline"] = out["absolute_lift"] < -0.0001
    out["full_cortex_loses_to_baseline"] = out["full_cortex_absolute_lift"] < -0.0001

    out["gate_changed_outcome_positive"] = (
        (out["full_cortex_loses_to_baseline"] | (out["full_cortex_winner"] == "Tie"))
        & out["gated_beats_baseline"]
    )

    out["gate_changed_outcome_negative"] = (
        out["full_cortex_beats_baseline"]
        & (out["gated_loses_to_baseline"] | (out["winner"] == "Tie"))
    )

    return out


def build_overall_summary(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)

    gated_wins = int((df["winner"].astype(str).str.contains("Gated", case=False, na=False)).sum())
    baseline_wins_vs_gated = int((df["winner"] == "Baseline").sum())
    ties_vs_gated = int((df["winner"] == "Tie").sum())

    full_cortex_wins = int((df["full_cortex_winner"].astype(str).str.contains("Full_CORTEX", case=False, na=False)).sum())
    baseline_wins_vs_full_cortex = int((df["full_cortex_winner"] == "Baseline").sum())
    ties_vs_full_cortex = int((df["full_cortex_winner"] == "Tie").sum())

    helped_count = int(df["gate_helped_vs_full_cortex"].sum())
    hurt_count = int(df["gate_hurt_vs_full_cortex"].sum())
    neutral_count = int(df["gate_neutral_vs_full_cortex"].sum())

    rescued_count = int(df["baseline_loss_rescued"].sum()) if "baseline_loss_rescued" in df.columns else 0
    over_rerank_prevented_count = int(df["over_rerank_prevented"].sum()) if "over_rerank_prevented" in df.columns else 0

    positive_outcome_change_count = int(df["gate_changed_outcome_positive"].sum())
    negative_outcome_change_count = int(df["gate_changed_outcome_negative"].sum())

    fallback_count = 0
    if "fallback_used" in df.columns:
        fallback_count = int(df["fallback_used"].astype(str).str.lower().isin(["true", "1", "yes"]).sum())

    summary = {
        "total_valid_queries": total,
        "gated_wins": gated_wins,
        "baseline_wins_vs_gated": baseline_wins_vs_gated,
        "ties_vs_gated": ties_vs_gated,
        "gated_win_rate": safe_rate(gated_wins, total),
        "full_cortex_wins": full_cortex_wins,
        "baseline_wins_vs_full_cortex": baseline_wins_vs_full_cortex,
        "ties_vs_full_cortex": ties_vs_full_cortex,
        "full_cortex_win_rate": safe_rate(full_cortex_wins, total),
        "gated_win_gain_vs_full_cortex": gated_wins - full_cortex_wins,
        "baseline_loss_reduction_vs_full_cortex": baseline_wins_vs_full_cortex - baseline_wins_vs_gated,
        "avg_baseline_reward_at_5": float(df["baseline_reward_at_5"].mean()),
        "avg_gated_reward_at_5": float(df["cortex_reward_at_5"].mean()),
        "avg_full_cortex_reward_at_5": float(df["full_cortex_reward_at_5"].mean()),
        "avg_gated_absolute_lift": float(df["absolute_lift"].mean()),
        "avg_full_cortex_absolute_lift": float(df["full_cortex_absolute_lift"].mean()),
        "median_gated_absolute_lift": float(df["absolute_lift"].median()),
        "median_full_cortex_absolute_lift": float(df["full_cortex_absolute_lift"].median()),
        "avg_gate_delta_vs_full_cortex": float(df["gate_delta_vs_full_cortex"].mean()),
        "median_gate_delta_vs_full_cortex": float(df["gate_delta_vs_full_cortex"].median()),
        "helped_queries_count": helped_count,
        "hurt_queries_count": hurt_count,
        "neutral_queries_count": neutral_count,
        "helped_rate": safe_rate(helped_count, total),
        "hurt_rate": safe_rate(hurt_count, total),
        "neutral_rate": safe_rate(neutral_count, total),
        "baseline_loss_rescued_count": rescued_count,
        "over_rerank_prevented_count": over_rerank_prevented_count,
        "positive_outcome_change_count": positive_outcome_change_count,
        "negative_outcome_change_count": negative_outcome_change_count,
        "fallback_contract_count": fallback_count,
    }

    return pd.DataFrame([summary])


def build_by_decision_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []

    for decision, group in df.groupby("gate_decision", dropna=False):
        total = len(group)

        row = {
            "gate_decision": decision,
            "query_count": total,
            "query_share": safe_rate(total, len(df)),
            "avg_baseline_reward_at_5": float(group["baseline_reward_at_5"].mean()),
            "avg_gated_reward_at_5": float(group["cortex_reward_at_5"].mean()),
            "avg_full_cortex_reward_at_5": float(group["full_cortex_reward_at_5"].mean()),
            "avg_gated_absolute_lift": float(group["absolute_lift"].mean()),
            "avg_full_cortex_absolute_lift": float(group["full_cortex_absolute_lift"].mean()),
            "avg_gate_delta_vs_full_cortex": float(group["gate_delta_vs_full_cortex"].mean()),
            "median_gate_delta_vs_full_cortex": float(group["gate_delta_vs_full_cortex"].median()),
            "helped_count": int(group["gate_helped_vs_full_cortex"].sum()),
            "hurt_count": int(group["gate_hurt_vs_full_cortex"].sum()),
            "neutral_count": int(group["gate_neutral_vs_full_cortex"].sum()),
            "baseline_loss_rescued_count": int(group["baseline_loss_rescued"].sum()) if "baseline_loss_rescued" in group.columns else 0,
            "avg_gate_baseline_confidence_score": float(group["gate_baseline_confidence_score"].mean()),
            "avg_gate_contract_alignment_score": float(group["gate_contract_alignment_score"].mean()),
            "avg_gate_final_gate_score": float(group["gate_final_gate_score"].mean()),
            "avg_gate_top1_score": float(group["gate_top1_score"].mean()),
            "avg_gate_top5_mean_score": float(group["gate_top5_mean_score"].mean()),
            "avg_gate_label_quality_score": float(group["gate_label_quality_score"].mean()),
        }

        rows.append(row)

    out = pd.DataFrame(rows)

    if len(out) > 0:
        out = out.sort_values(by="query_count", ascending=False)

    return out


def select_display_columns(df: pd.DataFrame) -> pd.DataFrame:
    existing = [col for col in DISPLAY_COLUMNS if col in df.columns]
    return df[existing].copy()


def build_helped_hurt_tables(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    helped = df[df["gate_helped_vs_full_cortex"]].copy()
    hurt = df[df["gate_hurt_vs_full_cortex"]].copy()

    if "baseline_loss_rescued" in df.columns:
        rescued = df[df["baseline_loss_rescued"]].copy()
    else:
        rescued = df.iloc[0:0].copy()

    helped = helped.sort_values(by="gate_delta_vs_full_cortex", ascending=False)
    hurt = hurt.sort_values(by="gate_delta_vs_full_cortex", ascending=True)
    rescued = rescued.sort_values(by="gate_delta_vs_full_cortex", ascending=False)

    return (
        select_display_columns(helped),
        select_display_columns(hurt),
        select_display_columns(rescued),
    )


def quantile_or_none(series: pd.Series, q: float) -> float:
    clean = series.dropna()

    if len(clean) == 0:
        return 0.0

    return float(clean.quantile(q))


def build_threshold_hints(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build simple descriptive threshold hints.

    We compare signal distributions for:
    - cases where the gate helped
    - cases where the gate hurt
    - cases where full CORTEX lost to baseline

    This is not a trained model yet. It is a calibration report for the next MVP.
    """

    helped = df[df["gate_helped_vs_full_cortex"]].copy()
    hurt = df[df["gate_hurt_vs_full_cortex"]].copy()
    full_lost = df[df["full_cortex_loses_to_baseline"]].copy()
    gated_lost = df[df["gated_loses_to_baseline"]].copy()

    signal_cols = [
        "gate_baseline_confidence_score",
        "gate_contract_alignment_score",
        "gate_final_gate_score",
        "gate_top1_score",
        "gate_top5_mean_score",
        "gate_label_quality_score",
        "gate_exclusion_violation_rate",
    ]

    rows: List[Dict] = []

    for col in signal_cols:
        if col not in df.columns:
            continue

        rows.append(
            {
                "signal": col,
                "overall_mean": float(df[col].mean()),
                "overall_p25": quantile_or_none(df[col], 0.25),
                "overall_p50": quantile_or_none(df[col], 0.50),
                "overall_p75": quantile_or_none(df[col], 0.75),
                "helped_mean": float(helped[col].mean()) if len(helped) else 0.0,
                "helped_p50": quantile_or_none(helped[col], 0.50),
                "hurt_mean": float(hurt[col].mean()) if len(hurt) else 0.0,
                "hurt_p50": quantile_or_none(hurt[col], 0.50),
                "full_cortex_lost_mean": float(full_lost[col].mean()) if len(full_lost) else 0.0,
                "full_cortex_lost_p50": quantile_or_none(full_lost[col], 0.50),
                "gated_lost_mean": float(gated_lost[col].mean()) if len(gated_lost) else 0.0,
                "gated_lost_p50": quantile_or_none(gated_lost[col], 0.50),
            }
        )

    hints = pd.DataFrame(rows)

    return hints


def print_console_report(
    summary_df: pd.DataFrame,
    by_decision_df: pd.DataFrame,
    helped_df: pd.DataFrame,
    hurt_df: pd.DataFrame,
    rescued_df: pd.DataFrame,
    threshold_hints_df: pd.DataFrame,
) -> None:
    print()
    print("MVP 13.4 Gate Calibration Analyzer")
    print("=" * 80)

    print()
    print("Overall summary")
    print("-" * 80)
    print(summary_df.to_string(index=False))

    print()
    print("Gate decision summary")
    print("-" * 80)
    if len(by_decision_df) > 0:
        print(by_decision_df.to_string(index=False))
    else:
        print("No gate decision summary available.")

    print()
    print("Top helped queries")
    print("-" * 80)
    if len(helped_df) > 0:
        print(helped_df.head(10).to_string(index=False))
    else:
        print("No helped queries found.")

    print()
    print("Top hurt queries")
    print("-" * 80)
    if len(hurt_df) > 0:
        print(hurt_df.head(10).to_string(index=False))
    else:
        print("No hurt queries found.")

    print()
    print("Rescued baseline-loss queries")
    print("-" * 80)
    if len(rescued_df) > 0:
        print(rescued_df.head(10).to_string(index=False))
    else:
        print("No rescued queries found.")

    print()
    print("Threshold hints")
    print("-" * 80)
    if len(threshold_hints_df) > 0:
        print(threshold_hints_df.to_string(index=False))
    else:
        print("No threshold hints available.")

    print()
    print("Files written")
    print("-" * 80)
    print(f"- summary: {SUMMARY_PATH}")
    print(f"- by decision: {BY_DECISION_PATH}")
    print(f"- helped queries: {HELPED_QUERIES_PATH}")
    print(f"- hurt queries: {HURT_QUERIES_PATH}")
    print(f"- rescued queries: {RESCUED_QUERIES_PATH}")
    print(f"- threshold hints: {THRESHOLD_HINTS_PATH}")


def run_calibration_analysis(input_path: str) -> None:
    ensure_output_dir()

    df = load_eval(input_path)
    df = add_analysis_columns(df)

    summary_df = build_overall_summary(df)
    by_decision_df = build_by_decision_summary(df)

    helped_df, hurt_df, rescued_df = build_helped_hurt_tables(df)
    threshold_hints_df = build_threshold_hints(df)

    summary_df.to_csv(SUMMARY_PATH, index=False)
    by_decision_df.to_csv(BY_DECISION_PATH, index=False)
    helped_df.to_csv(HELPED_QUERIES_PATH, index=False)
    hurt_df.to_csv(HURT_QUERIES_PATH, index=False)
    rescued_df.to_csv(RESCUED_QUERIES_PATH, index=False)
    threshold_hints_df.to_csv(THRESHOLD_HINTS_PATH, index=False)

    print_console_report(
        summary_df=summary_df,
        by_decision_df=by_decision_df,
        helped_df=helped_df,
        hurt_df=hurt_df,
        rescued_df=rescued_df,
        threshold_hints_df=threshold_hints_df,
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT_PATH,
        help="Path to scalable evaluation CSV.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_calibration_analysis(input_path=args.input)