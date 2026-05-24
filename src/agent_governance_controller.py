"""
MVP 13.5: Agent Governance Controller

Purpose:
--------
Create an offline governance controller for the CORTEX Engine.

The controller analyzes completed CORTEX evaluation logs and assigns each query
to an agent route. This is the foundation for autonomous agent routing.

This module does NOT change the live ranking pipeline yet.

It reads:
    outputs/scalable_eval_mvp13_3_1_resilient_1000.csv

It writes:
    outputs/agent_governance_routes.csv
    outputs/agent_governance_summary.csv
    outputs/agent_governance_route_quality.csv

Core idea:
----------
CORTEX should not run every agent with equal weight for every query.

Instead, each query should receive a governed route:

1. full_cortex_route
   - Default route.
   - Use complete CORTEX reranking.

2. light_rerank_route
   - Baseline is strong enough to partially trust.
   - Use blended baseline + CORTEX route.

3. baseline_safe_route
   - Baseline appears very strong and contract-aligned.
   - Candidate for preserving baseline or very minimal reranking.
   - For now this is only diagnostic; live preserve remains rare.

4. critic_needed_route
   - Query shows risk signals.
   - Needs a future critic/verifier agent before final slate.

5. mission_candidate_route
   - Query looks like an event, goal, lifestyle, or planning query.
   - Future Mission-Based Shopping Agent should activate.

6. fallback_contract_route
   - LLM failed and local fallback contract was used.
   - Needs careful monitoring.

The controller also computes:
- route share
- average rewards by route
- gate deltas by route
- agent cost estimate
- reward-per-cost proxy
- risk indicators
"""

from __future__ import annotations

import argparse
import os
import re
from typing import Dict, List

import pandas as pd


DEFAULT_INPUT_PATH = "outputs/scalable_eval_mvp13_3_1_resilient_1000.csv"
OUTPUT_DIR = "outputs"

ROUTES_PATH = os.path.join(OUTPUT_DIR, "agent_governance_routes.csv")
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "agent_governance_summary.csv")
ROUTE_QUALITY_PATH = os.path.join(OUTPUT_DIR, "agent_governance_route_quality.csv")


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


BRAND_OR_EXACT_TRIGGERS = [
    "iphone",
    "ipad",
    "samsung",
    "nike",
    "adidas",
    "lego",
    "sony",
    "xbox",
    "playstation",
    "nintendo",
    "canon",
    "hp",
    "dell",
    "apple",
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


def add_query_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["query_normalized"] = out["query"].fillna("").astype(str).apply(normalize_query)
    out["query_tokens"] = out["query_normalized"].apply(tokenize)
    out["query_token_count"] = out["query_tokens"].apply(len)
    out["query_char_count"] = out["query_normalized"].str.len()

    out["query_has_number"] = out["query_normalized"].str.contains(r"\d", regex=True)
    out["query_has_brand_or_exact_trigger"] = out["query_normalized"].apply(
        lambda x: contains_any_phrase(x, BRAND_OR_EXACT_TRIGGERS)
    )
    out["query_is_mission_candidate"] = out["query_normalized"].apply(
        lambda x: contains_any_phrase(x, MISSION_TRIGGERS)
    )

    out["query_is_short_exact"] = (
        (out["query_token_count"] <= 3)
        & (
            out["query_has_number"]
            | out["query_has_brand_or_exact_trigger"]
        )
    )

    out["query_is_broad"] = (
        (out["query_token_count"] <= 2)
        & (~out["query_has_number"])
        & (~out["query_has_brand_or_exact_trigger"])
    )

    return out


def add_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["gate_helped_vs_full_cortex"] = out["gate_delta_vs_full_cortex"] > 0.0001
    out["gate_hurt_vs_full_cortex"] = out["gate_delta_vs_full_cortex"] < -0.0001
    out["gate_neutral_vs_full_cortex"] = out["gate_delta_vs_full_cortex"].abs() <= 0.0001

    out["full_cortex_lost_to_baseline"] = out["full_cortex_absolute_lift"] < -0.0001
    out["gated_lost_to_baseline"] = out["absolute_lift"] < -0.0001

    out["high_baseline_confidence"] = out["gate_baseline_confidence_score"] >= 0.78
    out["high_contract_alignment"] = out["gate_contract_alignment_score"] >= 0.75
    out["very_high_contract_alignment"] = out["gate_contract_alignment_score"] >= 0.85
    out["high_label_quality"] = out["gate_label_quality_score"] >= 0.90
    out["high_top5_strength"] = out["gate_top5_mean_score"] >= 0.70

    out["contract_risk"] = (
        (out["gate_exclusion_violation_rate"] > 0.20)
        | (out["gate_contract_alignment_score"] < 0.50)
    )

    out["retrieval_uncertainty"] = (
        (out["gate_top1_score"] < 0.55)
        | (out["gate_top5_mean_score"] < 0.50)
    )

    out["rerank_risk"] = (
        out["full_cortex_lost_to_baseline"]
        | out["contract_risk"]
        | out["retrieval_uncertainty"]
    )

    return out


def estimate_route_cost(route: str) -> float:
    """
    Relative cost proxy, not milliseconds.

    Lower is cheaper.
    Higher means more agents / heavier reasoning.

    This is intentionally simple for MVP 13.5.
    """

    cost_map = {
        "baseline_safe_route": 1.0,
        "light_rerank_route": 2.0,
        "full_cortex_route": 3.0,
        "critic_needed_route": 4.0,
        "mission_candidate_route": 5.0,
        "fallback_contract_route": 3.5,
    }

    return cost_map.get(route, 3.0)


def assign_governance_route(row: pd.Series) -> Dict:
    """
    Assign a route based on query/ranking/gate signals.

    Route priority:
    1. fallback_contract_route
    2. mission_candidate_route
    3. critic_needed_route
    4. baseline_safe_route
    5. light_rerank_route
    6. full_cortex_route
    """

    query = row.get("query_normalized", "")

    fallback_used = bool(row.get("fallback_used", False))
    mission_candidate = bool(row.get("query_is_mission_candidate", False))

    high_baseline_confidence = bool(row.get("high_baseline_confidence", False))
    high_contract_alignment = bool(row.get("high_contract_alignment", False))
    very_high_contract_alignment = bool(row.get("very_high_contract_alignment", False))
    high_label_quality = bool(row.get("high_label_quality", False))
    high_top5_strength = bool(row.get("high_top5_strength", False))

    contract_risk = bool(row.get("contract_risk", False))
    retrieval_uncertainty = bool(row.get("retrieval_uncertainty", False))
    rerank_risk = bool(row.get("rerank_risk", False))

    query_is_short_exact = bool(row.get("query_is_short_exact", False))
    query_is_broad = bool(row.get("query_is_broad", False))

    if fallback_used:
        return {
            "governance_route": "fallback_contract_route",
            "route_reason": "LLM contract generation used fallback; route should be monitored.",
            "agents_to_run": "semantic_retrieval,llm_or_fallback_contract,contract_filter,q_learning,slate_enforcer,diversifier,baseline_gate",
            "agents_to_skip": "mission_agent,critic_agent,multimodal_agent",
        }

    if mission_candidate and row.get("query_token_count", 0) >= 3:
        return {
            "governance_route": "mission_candidate_route",
            "route_reason": "Query appears to express an occasion, goal, or shopping mission.",
            "agents_to_run": "mission_understanding,query_decomposition,semantic_retrieval,llm_contract,coverage_agent,q_learning,diversifier,baseline_gate",
            "agents_to_skip": "multimodal_agent",
        }

    if contract_risk or retrieval_uncertainty:
        return {
            "governance_route": "critic_needed_route",
            "route_reason": "Contract alignment or retrieval confidence is weak; future critic/verifier should inspect slate.",
            "agents_to_run": "semantic_retrieval,llm_contract,contract_filter,q_learning,slate_enforcer,diversifier,critic_agent,baseline_gate",
            "agents_to_skip": "mission_agent,multimodal_agent",
        }

    if (
        high_baseline_confidence
        and very_high_contract_alignment
        and high_label_quality
        and high_top5_strength
        and query_is_short_exact
    ):
        return {
            "governance_route": "baseline_safe_route",
            "route_reason": "Baseline appears very strong for a short exact/product-like query.",
            "agents_to_run": "semantic_retrieval,light_contract_check,baseline_gate",
            "agents_to_skip": "critic_agent,mission_agent,multimodal_agent",
        }

    if high_baseline_confidence and high_contract_alignment and high_label_quality:
        return {
            "governance_route": "light_rerank_route",
            "route_reason": "Baseline has strong confidence and alignment; light rerank is sufficient.",
            "agents_to_run": "semantic_retrieval,llm_contract,contract_filter,light_rerank,baseline_gate",
            "agents_to_skip": "critic_agent,mission_agent,multimodal_agent",
        }

    return {
        "governance_route": "full_cortex_route",
        "route_reason": "Default path; use full CORTEX policy reranking.",
        "agents_to_run": "semantic_retrieval,llm_contract,contract_filter,q_learning,slate_enforcer,diversifier,baseline_gate",
        "agents_to_skip": "critic_agent,mission_agent,multimodal_agent",
    }


def add_governance_routes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    route_dicts = out.apply(assign_governance_route, axis=1)
    route_df = pd.DataFrame(route_dicts.tolist())

    out = pd.concat(
        [
            out.reset_index(drop=True),
            route_df.reset_index(drop=True),
        ],
        axis=1,
    )

    out["route_cost_proxy"] = out["governance_route"].apply(estimate_route_cost)

    out["route_reward_per_cost"] = out.apply(
        lambda row: row["cortex_reward_at_5"] / row["route_cost_proxy"]
        if row["route_cost_proxy"] > 0
        else 0.0,
        axis=1,
    )

    out["route_lift_per_cost"] = out.apply(
        lambda row: row["absolute_lift"] / row["route_cost_proxy"]
        if row["route_cost_proxy"] > 0
        else 0.0,
        axis=1,
    )

    return out


def build_governance_summary(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)

    summary = {
        "total_queries": total,
        "unique_routes": int(df["governance_route"].nunique()),
        "avg_baseline_reward_at_5": float(df["baseline_reward_at_5"].mean()),
        "avg_gated_reward_at_5": float(df["cortex_reward_at_5"].mean()),
        "avg_full_cortex_reward_at_5": float(df["full_cortex_reward_at_5"].mean()),
        "avg_absolute_lift": float(df["absolute_lift"].mean()),
        "avg_full_cortex_absolute_lift": float(df["full_cortex_absolute_lift"].mean()),
        "avg_gate_delta_vs_full_cortex": float(df["gate_delta_vs_full_cortex"].mean()),
        "avg_route_cost_proxy": float(df["route_cost_proxy"].mean()),
        "avg_route_reward_per_cost": float(df["route_reward_per_cost"].mean()),
        "avg_route_lift_per_cost": float(df["route_lift_per_cost"].mean()),
        "mission_candidate_count": int((df["governance_route"] == "mission_candidate_route").sum()),
        "critic_needed_count": int((df["governance_route"] == "critic_needed_route").sum()),
        "baseline_safe_count": int((df["governance_route"] == "baseline_safe_route").sum()),
        "light_rerank_route_count": int((df["governance_route"] == "light_rerank_route").sum()),
        "full_cortex_route_count": int((df["governance_route"] == "full_cortex_route").sum()),
        "fallback_contract_route_count": int((df["governance_route"] == "fallback_contract_route").sum()),
    }

    return pd.DataFrame([summary])


def build_route_quality_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []

    for route, group in df.groupby("governance_route"):
        total = len(group)

        row = {
            "governance_route": route,
            "query_count": total,
            "query_share": total / len(df) if len(df) else 0.0,
            "avg_baseline_reward_at_5": float(group["baseline_reward_at_5"].mean()),
            "avg_gated_reward_at_5": float(group["cortex_reward_at_5"].mean()),
            "avg_full_cortex_reward_at_5": float(group["full_cortex_reward_at_5"].mean()),
            "avg_absolute_lift": float(group["absolute_lift"].mean()),
            "avg_full_cortex_absolute_lift": float(group["full_cortex_absolute_lift"].mean()),
            "avg_gate_delta_vs_full_cortex": float(group["gate_delta_vs_full_cortex"].mean()),
            "median_gate_delta_vs_full_cortex": float(group["gate_delta_vs_full_cortex"].median()),
            "avg_route_cost_proxy": float(group["route_cost_proxy"].mean()),
            "avg_route_reward_per_cost": float(group["route_reward_per_cost"].mean()),
            "avg_route_lift_per_cost": float(group["route_lift_per_cost"].mean()),
            "gate_helped_count": int(group["gate_helped_vs_full_cortex"].sum()),
            "gate_hurt_count": int(group["gate_hurt_vs_full_cortex"].sum()),
            "full_cortex_lost_to_baseline_count": int(group["full_cortex_lost_to_baseline"].sum()),
            "gated_lost_to_baseline_count": int(group["gated_lost_to_baseline"].sum()),
            "avg_gate_baseline_confidence_score": float(group["gate_baseline_confidence_score"].mean()),
            "avg_gate_contract_alignment_score": float(group["gate_contract_alignment_score"].mean()),
            "avg_gate_final_gate_score": float(group["gate_final_gate_score"].mean()),
            "avg_gate_label_quality_score": float(group["gate_label_quality_score"].mean()),
            "avg_query_token_count": float(group["query_token_count"].mean()),
        }

        rows.append(row)

    out = pd.DataFrame(rows)

    if len(out) > 0:
        out = out.sort_values(by="query_count", ascending=False)

    return out


def choose_display_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "query",
        "governance_route",
        "route_reason",
        "agents_to_run",
        "agents_to_skip",
        "route_cost_proxy",
        "route_reward_per_cost",
        "route_lift_per_cost",
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
        "query_token_count",
        "query_is_mission_candidate",
        "query_is_short_exact",
        "query_is_broad",
        "contract_risk",
        "retrieval_uncertainty",
        "rerank_risk",
        "baseline_top_product_title",
        "cortex_top_product_title",
        "full_cortex_top_product_title",
        "product_type",
        "selected_rl_action",
        "enforcement_status",
        "low_coverage",
        "fallback_used",
    ]

    existing = [col for col in cols if col in df.columns]

    return df[existing].copy()


def print_report(
    summary_df: pd.DataFrame,
    route_quality_df: pd.DataFrame,
    routes_df: pd.DataFrame,
) -> None:
    print()
    print("MVP 13.5 Agent Governance Controller")
    print("=" * 100)

    print()
    print("Governance summary")
    print("-" * 100)
    print(summary_df.to_string(index=False))

    print()
    print("Route quality summary")
    print("-" * 100)
    print(route_quality_df.to_string(index=False))

    print()
    print("Sample governed routes")
    print("-" * 100)
    print(routes_df.head(20).to_string(index=False))

    print()
    print("Files written")
    print("-" * 100)
    print(f"- routes: {ROUTES_PATH}")
    print(f"- summary: {SUMMARY_PATH}")
    print(f"- route quality: {ROUTE_QUALITY_PATH}")


def run_governance_controller(input_path: str) -> None:
    ensure_output_dir()

    df = load_eval_data(input_path)
    df = add_query_features(df)
    df = add_risk_features(df)
    df = add_governance_routes(df)

    routes_df = choose_display_columns(df)
    summary_df = build_governance_summary(df)
    route_quality_df = build_route_quality_summary(df)

    routes_df.to_csv(ROUTES_PATH, index=False)
    summary_df.to_csv(SUMMARY_PATH, index=False)
    route_quality_df.to_csv(ROUTE_QUALITY_PATH, index=False)

    print_report(
        summary_df=summary_df,
        route_quality_df=route_quality_df,
        routes_df=routes_df,
    )


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
    run_governance_controller(input_path=args.input)