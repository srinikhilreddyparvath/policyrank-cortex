"""
MVP 13.6: Critic / Verifier Agent

Purpose:
--------
The Critic / Verifier Agent inspects CORTEX evaluation logs and diagnoses
why CORTEX may have lost to the semantic baseline or underperformed full CORTEX.

This is an offline analysis agent first.
It does NOT modify the live ranking pipeline yet.

Input:
------
outputs/scalable_eval_mvp13_3_1_resilient_1000.csv

Optional governance input:
------
outputs/agent_governance_routes.csv

Outputs:
--------
outputs/critic_verifier_report.csv
outputs/critic_verifier_summary.csv
outputs/critic_verifier_failure_modes.csv
outputs/critic_verifier_repair_actions.csv
outputs/critic_verifier_high_risk_queries.csv

Core idea:
----------
CORTEX should not only rank products. It should be able to critique its own
failures and recommend safe repair strategies.

Failure modes detected:
-----------------------
1. baseline_stronger_than_cortex
2. full_cortex_over_reranked
3. gated_cortex_underperformed_full_cortex
4. low_contract_alignment
5. high_exclusion_violation
6. retrieval_uncertainty
7. weak_top5_baseline_signal
8. possible_over_diversification
9. possible_contract_over_filtering
10. possible_label_noise
11. mission_query_needs_decomposition
12. fallback_contract_risk
13. gate_helped
14. gate_failed_to_rescue

Repair recommendations:
-----------------------
- use_full_cortex
- use_light_rerank
- use_baseline_safe_route
- rerun_or_repair_contract
- reduce_diversification_pressure
- strengthen_product_type_match
- apply_critic_before_final_slate
- activate_mission_agent_later
- collect_more_feedback
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Dict, List, Tuple

import pandas as pd


DEFAULT_INPUT_PATH = "outputs/scalable_eval_mvp13_3_1_resilient_1000.csv"
DEFAULT_GOVERNANCE_PATH = "outputs/agent_governance_routes.csv"
OUTPUT_DIR = "outputs"

REPORT_PATH = os.path.join(OUTPUT_DIR, "critic_verifier_report.csv")
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "critic_verifier_summary.csv")
FAILURE_MODES_PATH = os.path.join(OUTPUT_DIR, "critic_verifier_failure_modes.csv")
REPAIR_ACTIONS_PATH = os.path.join(OUTPUT_DIR, "critic_verifier_repair_actions.csv")
HIGH_RISK_PATH = os.path.join(OUTPUT_DIR, "critic_verifier_high_risk_queries.csv")


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
    "positive_contract_rows",
    "blocked_rows",
    "low_coverage",
    "route_cost_proxy",
    "route_reward_per_cost",
    "route_lift_per_cost",
]


MISSION_TRIGGERS = [
    "party",
    "watch party",
    "birthday",
    "wedding",
    "baby shower",
    "graduation",
    "moving",
    "move",
    "apartment",
    "dorm",
    "college",
    "camping",
    "hiking",
    "vacation",
    "travel",
    "trip",
    "gift",
    "gifts",
    "date night",
    "bbq",
    "barbecue",
    "cookout",
    "hosting",
    "host",
    "organizing",
    "christmas",
    "thanksgiving",
    "halloween",
    "world cup",
    "super bowl",
    "soccer",
    "football party",
]


def ensure_output_dir() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def safe_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin(["true", "1", "yes"])


def normalize_query(query: str) -> str:
    if query is None:
        return ""

    query = str(query).strip().lower()
    query = re.sub(r"\s+", " ", query)

    return query


def tokenize(query: str) -> List[str]:
    query = normalize_query(query)
    return re.findall(r"[a-zA-Z0-9]+", query)


def contains_any_phrase(text: str, phrases: List[str]) -> bool:
    text = normalize_query(text)

    return any(
        phrase in text
        for phrase in phrases
    )


def load_eval_data(input_path: str) -> pd.DataFrame:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Could not find input file: {input_path}")

    df = pd.read_csv(input_path)

    if len(df) == 0:
        raise ValueError(f"Input file is empty: {input_path}")

    if "error" in df.columns:
        df = df[df["error"].fillna("").astype(str).str.strip() == ""].copy()

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = safe_numeric(df[col])

    for col in ["baseline_loss_rescued", "over_rerank_prevented", "fallback_used"]:
        if col in df.columns:
            df[col] = safe_bool(df[col])

    return df.reset_index(drop=True)


def load_governance_data(governance_path: str) -> pd.DataFrame:
    if not os.path.exists(governance_path):
        return pd.DataFrame()

    df = pd.read_csv(governance_path)

    if len(df) == 0:
        return pd.DataFrame()

    keep_cols = [
        "query",
        "governance_route",
        "route_reason",
        "agents_to_run",
        "agents_to_skip",
        "route_cost_proxy",
        "route_reward_per_cost",
        "route_lift_per_cost",
        "contract_risk",
        "retrieval_uncertainty",
        "rerank_risk",
        "query_is_mission_candidate",
        "query_is_short_exact",
        "query_is_broad",
    ]

    existing_cols = [col for col in keep_cols if col in df.columns]

    out = df[existing_cols].copy()

    return out


def merge_governance(eval_df: pd.DataFrame, governance_df: pd.DataFrame) -> pd.DataFrame:
    if governance_df is None or len(governance_df) == 0:
        out = eval_df.copy()

        out["governance_route"] = "unknown_route"
        out["route_reason"] = ""
        out["agents_to_run"] = ""
        out["agents_to_skip"] = ""
        out["route_cost_proxy"] = 0.0
        out["route_reward_per_cost"] = 0.0
        out["route_lift_per_cost"] = 0.0
        out["contract_risk"] = False
        out["retrieval_uncertainty"] = False
        out["rerank_risk"] = False
        out["query_is_mission_candidate"] = False
        out["query_is_short_exact"] = False
        out["query_is_broad"] = False

        return out

    out = eval_df.merge(
        governance_df,
        on="query",
        how="left",
        suffixes=("", "_gov"),
    )

    fill_defaults = {
        "governance_route": "unknown_route",
        "route_reason": "",
        "agents_to_run": "",
        "agents_to_skip": "",
        "route_cost_proxy": 0.0,
        "route_reward_per_cost": 0.0,
        "route_lift_per_cost": 0.0,
        "contract_risk": False,
        "retrieval_uncertainty": False,
        "rerank_risk": False,
        "query_is_mission_candidate": False,
        "query_is_short_exact": False,
        "query_is_broad": False,
    }

    for col, default in fill_defaults.items():
        if col not in out.columns:
            out[col] = default
        else:
            out[col] = out[col].fillna(default)

    for col in ["contract_risk", "retrieval_uncertainty", "rerank_risk", "query_is_mission_candidate"]:
        out[col] = safe_bool(out[col])

    return out


def add_query_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["query_normalized"] = out["query"].fillna("").astype(str).apply(normalize_query)
    out["query_tokens"] = out["query_normalized"].apply(tokenize)
    out["query_token_count"] = out["query_tokens"].apply(len)
    out["query_char_count"] = out["query_normalized"].str.len()

    if "query_is_mission_candidate" not in out.columns:
        out["query_is_mission_candidate"] = out["query_normalized"].apply(
            lambda query: contains_any_phrase(query, MISSION_TRIGGERS)
        )
    else:
        existing = safe_bool(out["query_is_mission_candidate"])
        trigger_based = out["query_normalized"].apply(
            lambda query: contains_any_phrase(query, MISSION_TRIGGERS)
        )
        out["query_is_mission_candidate"] = existing | trigger_based

    return out


def add_basic_outcome_flags(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["gated_beats_baseline"] = out["absolute_lift"] > 0.0001
    out["gated_loses_to_baseline"] = out["absolute_lift"] < -0.0001
    out["gated_ties_baseline"] = out["absolute_lift"].abs() <= 0.0001

    out["full_cortex_beats_baseline"] = out["full_cortex_absolute_lift"] > 0.0001
    out["full_cortex_loses_to_baseline"] = out["full_cortex_absolute_lift"] < -0.0001
    out["full_cortex_ties_baseline"] = out["full_cortex_absolute_lift"].abs() <= 0.0001

    out["gate_helped_vs_full_cortex"] = out["gate_delta_vs_full_cortex"] > 0.0001
    out["gate_hurt_vs_full_cortex"] = out["gate_delta_vs_full_cortex"] < -0.0001
    out["gate_neutral_vs_full_cortex"] = out["gate_delta_vs_full_cortex"].abs() <= 0.0001

    out["baseline_stronger_than_gated"] = out["gated_loses_to_baseline"]
    out["baseline_stronger_than_full_cortex"] = out["full_cortex_loses_to_baseline"]

    return out


def infer_failure_modes(row: pd.Series) -> List[str]:
    modes: List[str] = []

    if bool(row.get("gated_loses_to_baseline", False)):
        modes.append("baseline_stronger_than_cortex")

    if bool(row.get("full_cortex_loses_to_baseline", False)):
        modes.append("full_cortex_over_reranked")

    if bool(row.get("gate_hurt_vs_full_cortex", False)):
        modes.append("gated_cortex_underperformed_full_cortex")

    if row.get("gate_contract_alignment_score", 1.0) < 0.50:
        modes.append("low_contract_alignment")

    if row.get("gate_exclusion_violation_rate", 0.0) > 0.20:
        modes.append("high_exclusion_violation")

    if row.get("gate_top1_score", 1.0) < 0.55:
        modes.append("retrieval_uncertainty")

    if row.get("gate_top5_mean_score", 1.0) < 0.50:
        modes.append("weak_top5_baseline_signal")

    if row.get("blocked_rows", 0.0) > 5:
        modes.append("possible_contract_over_filtering")

    if (
        row.get("gate_delta_vs_full_cortex", 0.0) < -0.03
        and row.get("gate_contract_alignment_score", 0.0) >= 0.70
        and row.get("gate_label_quality_score", 0.0) >= 0.85
    ):
        modes.append("possible_over_diversification")

    if (
        row.get("gate_label_quality_score", 0.0) >= 0.90
        and row.get("gated_loses_to_baseline", False)
        and row.get("full_cortex_loses_to_baseline", False)
    ):
        modes.append("possible_label_noise")

    if bool(row.get("query_is_mission_candidate", False)):
        modes.append("mission_query_needs_decomposition")

    if bool(row.get("fallback_used", False)):
        modes.append("fallback_contract_risk")

    if bool(row.get("gate_helped_vs_full_cortex", False)):
        modes.append("gate_helped")

    if (
        bool(row.get("full_cortex_loses_to_baseline", False))
        and bool(row.get("gated_loses_to_baseline", False))
    ):
        modes.append("gate_failed_to_rescue")

    if not modes:
        modes.append("no_major_failure_detected")

    return list(dict.fromkeys(modes))


def recommend_repair_actions(row: pd.Series, failure_modes: List[str]) -> List[str]:
    actions: List[str] = []

    modes = set(failure_modes)

    if "baseline_stronger_than_cortex" in modes and row.get("gate_baseline_confidence_score", 0.0) >= 0.78:
        actions.append("use_light_rerank_or_baseline_safe_route")

    if "full_cortex_over_reranked" in modes:
        actions.append("apply_baseline_aware_gate_before_final_slate")

    if "gated_cortex_underperformed_full_cortex" in modes:
        actions.append("use_full_cortex_for_similar_queries")

    if "low_contract_alignment" in modes:
        actions.append("rerun_or_repair_contract")

    if "high_exclusion_violation" in modes:
        actions.append("strengthen_blocked_term_enforcement")

    if "retrieval_uncertainty" in modes or "weak_top5_baseline_signal" in modes:
        actions.append("increase_retrieval_depth_or_expand_query")

    if "possible_contract_over_filtering" in modes:
        actions.append("relax_contract_filter_or_review_blocked_terms")

    if "possible_over_diversification" in modes:
        actions.append("reduce_diversification_pressure")

    if "mission_query_needs_decomposition" in modes:
        actions.append("activate_mission_agent_later")

    if "fallback_contract_risk" in modes:
        actions.append("prioritize_llm_contract_regeneration")

    if "possible_label_noise" in modes:
        actions.append("collect_more_feedback_or_review_label_quality")

    if "gate_failed_to_rescue" in modes:
        actions.append("send_to_critic_before_final_ranking")

    if not actions:
        if row.get("gate_helped_vs_full_cortex", False):
            actions.append("keep_current_gated_route")
        else:
            actions.append("keep_full_cortex_default")

    return list(dict.fromkeys(actions))


def compute_critic_risk_score(row: pd.Series, failure_modes: List[str]) -> float:
    """
    Risk score is a heuristic 0-1 score.

    Higher means this query should be inspected by the future online critic.
    """

    score = 0.0

    if "baseline_stronger_than_cortex" in failure_modes:
        score += 0.25

    if "full_cortex_over_reranked" in failure_modes:
        score += 0.20

    if "gated_cortex_underperformed_full_cortex" in failure_modes:
        score += 0.15

    if "low_contract_alignment" in failure_modes:
        score += 0.15

    if "high_exclusion_violation" in failure_modes:
        score += 0.12

    if "retrieval_uncertainty" in failure_modes:
        score += 0.10

    if "weak_top5_baseline_signal" in failure_modes:
        score += 0.10

    if "possible_contract_over_filtering" in failure_modes:
        score += 0.10

    if "possible_over_diversification" in failure_modes:
        score += 0.10

    if "mission_query_needs_decomposition" in failure_modes:
        score += 0.08

    if "gate_failed_to_rescue" in failure_modes:
        score += 0.18

    # Magnitude-based adjustments.
    if row.get("absolute_lift", 0.0) < -0.05:
        score += 0.15

    if row.get("gate_delta_vs_full_cortex", 0.0) < -0.05:
        score += 0.12

    if row.get("gate_contract_alignment_score", 1.0) < 0.40:
        score += 0.10

    return max(0.0, min(1.0, score))


def classify_critic_priority(risk_score: float) -> str:
    if risk_score >= 0.70:
        return "critical"

    if risk_score >= 0.45:
        return "high"

    if risk_score >= 0.20:
        return "medium"

    return "low"


def add_critic_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    failure_modes_list: List[List[str]] = []
    repair_actions_list: List[List[str]] = []
    risk_scores: List[float] = []
    priorities: List[str] = []

    for _, row in out.iterrows():
        failure_modes = infer_failure_modes(row)
        repair_actions = recommend_repair_actions(row, failure_modes)
        risk_score = compute_critic_risk_score(row, failure_modes)
        priority = classify_critic_priority(risk_score)

        failure_modes_list.append(failure_modes)
        repair_actions_list.append(repair_actions)
        risk_scores.append(risk_score)
        priorities.append(priority)

    out["critic_failure_modes"] = [";".join(modes) for modes in failure_modes_list]
    out["critic_repair_actions"] = [";".join(actions) for actions in repair_actions_list]
    out["critic_risk_score"] = risk_scores
    out["critic_priority"] = priorities

    out["needs_online_critic"] = out["critic_priority"].isin(["critical", "high"])
    out["needs_repair_action"] = ~out["critic_failure_modes"].str.contains("no_major_failure_detected")

    return out


def choose_report_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "query",
        "critic_priority",
        "critic_risk_score",
        "needs_online_critic",
        "needs_repair_action",
        "critic_failure_modes",
        "critic_repair_actions",
        "governance_route",
        "route_reason",
        "winner",
        "full_cortex_winner",
        "gate_decision",
        "baseline_loss_rescued",
        "over_rerank_prevented",
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
        "gate_exclusion_violation_rate",
        "positive_contract_rows",
        "blocked_rows",
        "low_coverage",
        "query_token_count",
        "query_is_mission_candidate",
        "baseline_top_product_title",
        "cortex_top_product_title",
        "full_cortex_top_product_title",
        "product_type",
        "selected_rl_action",
        "enforcement_status",
        "fallback_used",
    ]

    existing = [col for col in cols if col in df.columns]

    return df[existing].copy()


def explode_counts(df: pd.DataFrame, source_col: str, output_key: str) -> pd.DataFrame:
    rows: List[Dict] = []

    for _, row in df.iterrows():
        raw_value = str(row.get(source_col, "")).strip()

        if not raw_value:
            continue

        parts = [part.strip() for part in raw_value.split(";") if part.strip()]

        for part in parts:
            rows.append(
                {
                    output_key: part,
                    "count": 1,
                    "avg_critic_risk_score": row.get("critic_risk_score", 0.0),
                    "avg_absolute_lift": row.get("absolute_lift", 0.0),
                    "avg_gate_delta_vs_full_cortex": row.get("gate_delta_vs_full_cortex", 0.0),
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                output_key,
                "count",
                "query_share",
                "avg_critic_risk_score",
                "avg_absolute_lift",
                "avg_gate_delta_vs_full_cortex",
            ]
        )

    exploded = pd.DataFrame(rows)

    grouped = (
        exploded
        .groupby(output_key, as_index=False)
        .agg(
            count=("count", "sum"),
            avg_critic_risk_score=("avg_critic_risk_score", "mean"),
            avg_absolute_lift=("avg_absolute_lift", "mean"),
            avg_gate_delta_vs_full_cortex=("avg_gate_delta_vs_full_cortex", "mean"),
        )
    )

    grouped["query_share"] = grouped["count"] / len(df)

    grouped = grouped.sort_values(by="count", ascending=False)

    return grouped


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)

    priority_counts = df["critic_priority"].value_counts().to_dict()

    summary = {
        "total_queries": total,
        "critical_priority_count": int(priority_counts.get("critical", 0)),
        "high_priority_count": int(priority_counts.get("high", 0)),
        "medium_priority_count": int(priority_counts.get("medium", 0)),
        "low_priority_count": int(priority_counts.get("low", 0)),
        "needs_online_critic_count": int(df["needs_online_critic"].sum()),
        "needs_repair_action_count": int(df["needs_repair_action"].sum()),
        "avg_critic_risk_score": float(df["critic_risk_score"].mean()),
        "median_critic_risk_score": float(df["critic_risk_score"].median()),
        "avg_baseline_reward_at_5": float(df["baseline_reward_at_5"].mean()),
        "avg_gated_reward_at_5": float(df["cortex_reward_at_5"].mean()),
        "avg_full_cortex_reward_at_5": float(df["full_cortex_reward_at_5"].mean()),
        "avg_absolute_lift": float(df["absolute_lift"].mean()),
        "avg_full_cortex_absolute_lift": float(df["full_cortex_absolute_lift"].mean()),
        "avg_gate_delta_vs_full_cortex": float(df["gate_delta_vs_full_cortex"].mean()),
        "gated_loses_to_baseline_count": int(df["gated_loses_to_baseline"].sum()),
        "full_cortex_loses_to_baseline_count": int(df["full_cortex_loses_to_baseline"].sum()),
        "gate_helped_count": int(df["gate_helped_vs_full_cortex"].sum()),
        "gate_hurt_count": int(df["gate_hurt_vs_full_cortex"].sum()),
        "mission_query_count": int(df["query_is_mission_candidate"].sum()),
    }

    return pd.DataFrame([summary])


def print_report(
    summary_df: pd.DataFrame,
    failure_modes_df: pd.DataFrame,
    repair_actions_df: pd.DataFrame,
    high_risk_df: pd.DataFrame,
) -> None:
    print()
    print("MVP 13.6 Critic / Verifier Agent")
    print("=" * 100)

    print()
    print("Critic summary")
    print("-" * 100)
    print(summary_df.to_string(index=False))

    print()
    print("Failure modes")
    print("-" * 100)
    print(failure_modes_df.to_string(index=False))

    print()
    print("Repair actions")
    print("-" * 100)
    print(repair_actions_df.to_string(index=False))

    print()
    print("High-risk query examples")
    print("-" * 100)
    if len(high_risk_df) > 0:
        print(high_risk_df.head(20).to_string(index=False))
    else:
        print("No high-risk queries found.")

    print()
    print("Files written")
    print("-" * 100)
    print(f"- report: {REPORT_PATH}")
    print(f"- summary: {SUMMARY_PATH}")
    print(f"- failure modes: {FAILURE_MODES_PATH}")
    print(f"- repair actions: {REPAIR_ACTIONS_PATH}")
    print(f"- high risk queries: {HIGH_RISK_PATH}")


def run_critic_verifier(
    input_path: str,
    governance_path: str,
) -> None:
    ensure_output_dir()

    eval_df = load_eval_data(input_path)
    governance_df = load_governance_data(governance_path)

    df = merge_governance(eval_df, governance_df)
    df = add_query_features(df)
    df = add_basic_outcome_flags(df)
    df = add_critic_diagnostics(df)

    report_df = choose_report_columns(df)
    summary_df = build_summary(df)
    failure_modes_df = explode_counts(
        df=report_df,
        source_col="critic_failure_modes",
        output_key="failure_mode",
    )
    repair_actions_df = explode_counts(
        df=report_df,
        source_col="critic_repair_actions",
        output_key="repair_action",
    )

    high_risk_df = report_df[
        report_df["critic_priority"].isin(["critical", "high"])
    ].copy()

    high_risk_df = high_risk_df.sort_values(
        by=["critic_risk_score", "absolute_lift"],
        ascending=[False, True],
    )

    report_df.to_csv(REPORT_PATH, index=False)
    summary_df.to_csv(SUMMARY_PATH, index=False)
    failure_modes_df.to_csv(FAILURE_MODES_PATH, index=False)
    repair_actions_df.to_csv(REPAIR_ACTIONS_PATH, index=False)
    high_risk_df.to_csv(HIGH_RISK_PATH, index=False)

    print_report(
        summary_df=summary_df,
        failure_modes_df=failure_modes_df,
        repair_actions_df=repair_actions_df,
        high_risk_df=high_risk_df,
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT_PATH,
        help="Path to resilient scalable evaluation CSV.",
    )

    parser.add_argument(
        "--governance",
        type=str,
        default=DEFAULT_GOVERNANCE_PATH,
        help="Path to agent governance routes CSV.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_critic_verifier(
        input_path=args.input,
        governance_path=args.governance,
    )