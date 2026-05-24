"""
MVP 13.7: Critic-Guided Repair Simulator

Purpose:
--------
Use the Critic / Verifier Agent outputs to simulate repair policies offline.

This is the first step toward a self-healing CORTEX Engine.

Input:
------
outputs/critic_verifier_report.csv

Outputs:
--------
outputs/critic_guided_repair_simulation.csv
outputs/critic_guided_repair_summary.csv
outputs/critic_guided_repair_by_action.csv
outputs/critic_guided_repair_high_impact_queries.csv

Core idea:
----------
The Critic / Verifier Agent diagnoses failure modes and recommends repair actions.
This simulator asks:

    "If we followed those repair actions, would reward improve?"

Because we only have logged baseline, full CORTEX, and gated CORTEX rewards,
this simulator can safely estimate a subset of repairs:

1. use_full_cortex_for_similar_queries
   - Simulated by choosing full_cortex_reward_at_5.

2. apply_baseline_aware_gate_before_final_slate
   - Simulated by choosing cortex_reward_at_5.

3. use_light_rerank_or_baseline_safe_route
   - If baseline is stronger than both CORTEX versions, simulate baseline reward.
   - Otherwise choose max of baseline/gated/full for oracle-style diagnostic.

4. keep_current_gated_route
   - Simulated by choosing cortex_reward_at_5.

5. keep_full_cortex_default
   - Simulated by choosing full_cortex_reward_at_5.

Other repair actions such as:
- relax_contract_filter_or_review_blocked_terms
- increase_retrieval_depth_or_expand_query
- rerun_or_repair_contract
- reduce_diversification_pressure
- activate_mission_agent_later

cannot be fully simulated from current logs because they require rerunning retrieval,
contract filtering, diversification, or mission decomposition. These are flagged
as "requires_rerun".

This module is intentionally honest:
- It distinguishes directly simulatable repairs from future rerun-required repairs.
- It computes conservative simulation and oracle diagnostic simulation.
- It does not pretend to prove online improvement.
"""

from __future__ import annotations

import argparse
import os
from typing import Dict, List, Tuple

import pandas as pd


DEFAULT_INPUT_PATH = "outputs/critic_verifier_report.csv"
OUTPUT_DIR = "outputs"

SIMULATION_PATH = os.path.join(OUTPUT_DIR, "critic_guided_repair_simulation.csv")
SUMMARY_PATH = os.path.join(OUTPUT_DIR, "critic_guided_repair_summary.csv")
BY_ACTION_PATH = os.path.join(OUTPUT_DIR, "critic_guided_repair_by_action.csv")
HIGH_IMPACT_PATH = os.path.join(OUTPUT_DIR, "critic_guided_repair_high_impact_queries.csv")


NUMERIC_COLUMNS = [
    "critic_risk_score",
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
]


DIRECTLY_SIMULATABLE_ACTIONS = {
    "use_full_cortex_for_similar_queries",
    "apply_baseline_aware_gate_before_final_slate",
    "use_light_rerank_or_baseline_safe_route",
    "keep_current_gated_route",
    "keep_full_cortex_default",
}

RERUN_REQUIRED_ACTIONS = {
    "relax_contract_filter_or_review_blocked_terms",
    "increase_retrieval_depth_or_expand_query",
    "rerun_or_repair_contract",
    "strengthen_blocked_term_enforcement",
    "send_to_critic_before_final_ranking",
    "collect_more_feedback_or_review_label_quality",
    "reduce_diversification_pressure",
    "activate_mission_agent_later",
    "prioritize_llm_contract_regeneration",
}


def ensure_output_dir() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def safe_numeric(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce").fillna(0.0)


def load_critic_report(input_path: str) -> pd.DataFrame:
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Could not find critic report file: {input_path}")

    df = pd.read_csv(input_path)

    if len(df) == 0:
        raise ValueError(f"Critic report is empty: {input_path}")

    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = safe_numeric(df[col])

    required_cols = [
        "query",
        "baseline_reward_at_5",
        "cortex_reward_at_5",
        "full_cortex_reward_at_5",
        "critic_repair_actions",
    ]

    missing = [col for col in required_cols if col not in df.columns]

    if missing:
        raise ValueError(f"Missing required columns in critic report: {missing}")

    return df.reset_index(drop=True)


def parse_actions(raw_actions: str) -> List[str]:
    if raw_actions is None:
        return []

    raw_actions = str(raw_actions).strip()

    if not raw_actions:
        return []

    actions = [
        action.strip()
        for action in raw_actions.split(";")
        if action.strip()
    ]

    return list(dict.fromkeys(actions))


def choose_primary_action(actions: List[str]) -> str:
    """
    Choose a single primary action from the critic recommendations.

    Priority is based on actions that can be simulated safely from current logs.
    """

    priority_order = [
        "use_full_cortex_for_similar_queries",
        "apply_baseline_aware_gate_before_final_slate",
        "use_light_rerank_or_baseline_safe_route",
        "keep_current_gated_route",
        "keep_full_cortex_default",
        "send_to_critic_before_final_ranking",
        "rerun_or_repair_contract",
        "relax_contract_filter_or_review_blocked_terms",
        "increase_retrieval_depth_or_expand_query",
        "reduce_diversification_pressure",
        "strengthen_blocked_term_enforcement",
        "activate_mission_agent_later",
        "collect_more_feedback_or_review_label_quality",
    ]

    for action in priority_order:
        if action in actions:
            return action

    if actions:
        return actions[0]

    return "no_action"


def get_best_logged_reward(row: pd.Series) -> Tuple[float, str]:
    candidates = {
        "baseline": float(row.get("baseline_reward_at_5", 0.0)),
        "gated_cortex": float(row.get("cortex_reward_at_5", 0.0)),
        "full_cortex": float(row.get("full_cortex_reward_at_5", 0.0)),
    }

    best_policy = max(candidates, key=candidates.get)

    return candidates[best_policy], best_policy


def simulate_conservative_repair(row: pd.Series, primary_action: str) -> Tuple[float, str, bool]:
    """
    Conservative simulation:
    - Only use policies that exist in the current log.
    - If action requires rerun, keep current gated reward.
    """

    baseline_reward = float(row.get("baseline_reward_at_5", 0.0))
    gated_reward = float(row.get("cortex_reward_at_5", 0.0))
    full_reward = float(row.get("full_cortex_reward_at_5", 0.0))

    if primary_action == "use_full_cortex_for_similar_queries":
        return full_reward, "full_cortex", True

    if primary_action == "apply_baseline_aware_gate_before_final_slate":
        return gated_reward, "gated_cortex", True

    if primary_action == "use_light_rerank_or_baseline_safe_route":
        # Conservative: use baseline only if baseline beats both known CORTEX outcomes.
        if baseline_reward >= gated_reward and baseline_reward >= full_reward:
            return baseline_reward, "baseline_safe", True

        return gated_reward, "gated_cortex", True

    if primary_action == "keep_current_gated_route":
        return gated_reward, "gated_cortex", True

    if primary_action == "keep_full_cortex_default":
        return full_reward, "full_cortex", True

    # Rerun-required action. Keep current gated result until rerun is implemented.
    return gated_reward, "requires_rerun_kept_gated", False


def simulate_oracle_logged_policy(row: pd.Series) -> Tuple[float, str]:
    """
    Oracle diagnostic simulation:
    Choose the best reward among baseline, gated CORTEX, and full CORTEX.

    This is NOT deployable as-is because it uses observed outcomes.
    It estimates the upper bound of a perfect router over currently logged policies.
    """

    return get_best_logged_reward(row)


def classify_repair_status(
    conservative_delta_vs_current: float,
    conservative_delta_vs_full: float,
    directly_simulated: bool,
) -> str:
    if not directly_simulated:
        return "requires_rerun"

    if conservative_delta_vs_current > 0.0001:
        return "repair_improves_current_gated"

    if conservative_delta_vs_current < -0.0001:
        return "repair_hurts_current_gated"

    if conservative_delta_vs_full > 0.0001:
        return "repair_matches_current_but_beats_full"

    return "repair_neutral"


def add_repair_simulation(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    all_actions: List[List[str]] = []
    primary_actions: List[str] = []

    conservative_rewards: List[float] = []
    conservative_policies: List[str] = []
    directly_simulated_flags: List[bool] = []

    oracle_rewards: List[float] = []
    oracle_policies: List[str] = []

    repair_statuses: List[str] = []

    for _, row in out.iterrows():
        actions = parse_actions(row.get("critic_repair_actions", ""))
        primary_action = choose_primary_action(actions)

        conservative_reward, conservative_policy, directly_simulated = simulate_conservative_repair(
            row=row,
            primary_action=primary_action,
        )

        oracle_reward, oracle_policy = simulate_oracle_logged_policy(row)

        current_gated_reward = float(row.get("cortex_reward_at_5", 0.0))
        full_reward = float(row.get("full_cortex_reward_at_5", 0.0))

        conservative_delta_vs_current = conservative_reward - current_gated_reward
        conservative_delta_vs_full = conservative_reward - full_reward

        repair_status = classify_repair_status(
            conservative_delta_vs_current=conservative_delta_vs_current,
            conservative_delta_vs_full=conservative_delta_vs_full,
            directly_simulated=directly_simulated,
        )

        all_actions.append(actions)
        primary_actions.append(primary_action)

        conservative_rewards.append(conservative_reward)
        conservative_policies.append(conservative_policy)
        directly_simulated_flags.append(directly_simulated)

        oracle_rewards.append(oracle_reward)
        oracle_policies.append(oracle_policy)

        repair_statuses.append(repair_status)

    out["all_repair_actions_list"] = [";".join(actions) for actions in all_actions]
    out["primary_repair_action"] = primary_actions

    out["conservative_repair_reward_at_5"] = conservative_rewards
    out["conservative_repair_policy"] = conservative_policies
    out["repair_directly_simulated"] = directly_simulated_flags

    out["oracle_logged_policy_reward_at_5"] = oracle_rewards
    out["oracle_logged_policy"] = oracle_policies

    out["conservative_delta_vs_current_gated"] = (
        out["conservative_repair_reward_at_5"] - out["cortex_reward_at_5"]
    )

    out["conservative_delta_vs_full_cortex"] = (
        out["conservative_repair_reward_at_5"] - out["full_cortex_reward_at_5"]
    )

    out["oracle_delta_vs_current_gated"] = (
        out["oracle_logged_policy_reward_at_5"] - out["cortex_reward_at_5"]
    )

    out["oracle_delta_vs_full_cortex"] = (
        out["oracle_logged_policy_reward_at_5"] - out["full_cortex_reward_at_5"]
    )

    out["conservative_repair_lift_vs_baseline"] = (
        out["conservative_repair_reward_at_5"] - out["baseline_reward_at_5"]
    )

    out["oracle_lift_vs_baseline"] = (
        out["oracle_logged_policy_reward_at_5"] - out["baseline_reward_at_5"]
    )

    out["repair_status"] = repair_statuses

    return out


def build_summary(df: pd.DataFrame) -> pd.DataFrame:
    total = len(df)

    directly_simulated_count = int(df["repair_directly_simulated"].sum())
    rerun_required_count = total - directly_simulated_count

    improves_current_count = int((df["repair_status"] == "repair_improves_current_gated").sum())
    hurts_current_count = int((df["repair_status"] == "repair_hurts_current_gated").sum())
    neutral_count = int((df["repair_status"] == "repair_neutral").sum())
    requires_rerun_count = int((df["repair_status"] == "requires_rerun").sum())

    summary = {
        "total_queries": total,
        "directly_simulated_repairs": directly_simulated_count,
        "rerun_required_repairs": rerun_required_count,
        "repair_improves_current_count": improves_current_count,
        "repair_hurts_current_count": hurts_current_count,
        "repair_neutral_count": neutral_count,
        "requires_rerun_count": requires_rerun_count,
        "avg_baseline_reward_at_5": float(df["baseline_reward_at_5"].mean()),
        "avg_current_gated_reward_at_5": float(df["cortex_reward_at_5"].mean()),
        "avg_full_cortex_reward_at_5": float(df["full_cortex_reward_at_5"].mean()),
        "avg_conservative_repair_reward_at_5": float(df["conservative_repair_reward_at_5"].mean()),
        "avg_oracle_logged_policy_reward_at_5": float(df["oracle_logged_policy_reward_at_5"].mean()),
        "avg_conservative_delta_vs_current_gated": float(df["conservative_delta_vs_current_gated"].mean()),
        "avg_conservative_delta_vs_full_cortex": float(df["conservative_delta_vs_full_cortex"].mean()),
        "avg_oracle_delta_vs_current_gated": float(df["oracle_delta_vs_current_gated"].mean()),
        "avg_oracle_delta_vs_full_cortex": float(df["oracle_delta_vs_full_cortex"].mean()),
        "avg_current_gated_lift_vs_baseline": float(df["absolute_lift"].mean()),
        "avg_full_cortex_lift_vs_baseline": float(df["full_cortex_absolute_lift"].mean()),
        "avg_conservative_repair_lift_vs_baseline": float(df["conservative_repair_lift_vs_baseline"].mean()),
        "avg_oracle_lift_vs_baseline": float(df["oracle_lift_vs_baseline"].mean()),
        "median_current_gated_lift_vs_baseline": float(df["absolute_lift"].median()),
        "median_conservative_repair_lift_vs_baseline": float(df["conservative_repair_lift_vs_baseline"].median()),
        "median_oracle_lift_vs_baseline": float(df["oracle_lift_vs_baseline"].median()),
    }

    return pd.DataFrame([summary])


def build_by_action_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict] = []

    for action, group in df.groupby("primary_repair_action"):
        total = len(group)

        row = {
            "primary_repair_action": action,
            "query_count": total,
            "query_share": total / len(df) if len(df) else 0.0,
            "directly_simulated_count": int(group["repair_directly_simulated"].sum()),
            "requires_rerun_count": int((~group["repair_directly_simulated"]).sum()),
            "avg_baseline_reward_at_5": float(group["baseline_reward_at_5"].mean()),
            "avg_current_gated_reward_at_5": float(group["cortex_reward_at_5"].mean()),
            "avg_full_cortex_reward_at_5": float(group["full_cortex_reward_at_5"].mean()),
            "avg_conservative_repair_reward_at_5": float(group["conservative_repair_reward_at_5"].mean()),
            "avg_oracle_logged_policy_reward_at_5": float(group["oracle_logged_policy_reward_at_5"].mean()),
            "avg_conservative_delta_vs_current_gated": float(group["conservative_delta_vs_current_gated"].mean()),
            "avg_conservative_delta_vs_full_cortex": float(group["conservative_delta_vs_full_cortex"].mean()),
            "avg_oracle_delta_vs_current_gated": float(group["oracle_delta_vs_current_gated"].mean()),
            "avg_oracle_delta_vs_full_cortex": float(group["oracle_delta_vs_full_cortex"].mean()),
            "repair_improves_current_count": int((group["repair_status"] == "repair_improves_current_gated").sum()),
            "repair_hurts_current_count": int((group["repair_status"] == "repair_hurts_current_gated").sum()),
            "repair_neutral_count": int((group["repair_status"] == "repair_neutral").sum()),
            "requires_rerun_status_count": int((group["repair_status"] == "requires_rerun").sum()),
            "avg_critic_risk_score": float(group["critic_risk_score"].mean()) if "critic_risk_score" in group.columns else 0.0,
        }

        rows.append(row)

    out = pd.DataFrame(rows)

    if len(out) > 0:
        out = out.sort_values(
            by=["query_count", "avg_conservative_delta_vs_current_gated"],
            ascending=[False, False],
        )

    return out


def choose_output_columns(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "query",
        "critic_priority",
        "critic_risk_score",
        "needs_online_critic",
        "critic_failure_modes",
        "critic_repair_actions",
        "primary_repair_action",
        "repair_status",
        "repair_directly_simulated",
        "conservative_repair_policy",
        "oracle_logged_policy",
        "baseline_reward_at_5",
        "cortex_reward_at_5",
        "full_cortex_reward_at_5",
        "conservative_repair_reward_at_5",
        "oracle_logged_policy_reward_at_5",
        "absolute_lift",
        "full_cortex_absolute_lift",
        "conservative_repair_lift_vs_baseline",
        "oracle_lift_vs_baseline",
        "gate_delta_vs_full_cortex",
        "conservative_delta_vs_current_gated",
        "conservative_delta_vs_full_cortex",
        "oracle_delta_vs_current_gated",
        "oracle_delta_vs_full_cortex",
        "governance_route",
        "gate_decision",
        "winner",
        "full_cortex_winner",
        "baseline_loss_rescued",
        "over_rerank_prevented",
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
        "product_type",
        "selected_rl_action",
        "enforcement_status",
    ]

    existing = [col for col in cols if col in df.columns]

    return df[existing].copy()


def build_high_impact_queries(df: pd.DataFrame) -> pd.DataFrame:
    high_impact = df[
        (
            df["conservative_delta_vs_current_gated"] > 0.05
        )
        | (
            df["oracle_delta_vs_current_gated"] > 0.10
        )
        | (
            df["critic_priority"].isin(["critical", "high"])
            & (df["absolute_lift"] < 0)
        )
    ].copy()

    high_impact = high_impact.sort_values(
        by=[
            "conservative_delta_vs_current_gated",
            "oracle_delta_vs_current_gated",
            "critic_risk_score",
        ],
        ascending=[False, False, False],
    )

    return choose_output_columns(high_impact)


def print_report(
    summary_df: pd.DataFrame,
    by_action_df: pd.DataFrame,
    high_impact_df: pd.DataFrame,
) -> None:
    print()
    print("MVP 13.7 Critic-Guided Repair Simulator")
    print("=" * 110)

    print()
    print("Repair simulation summary")
    print("-" * 110)
    print(summary_df.to_string(index=False))

    print()
    print("Repair simulation by primary action")
    print("-" * 110)
    print(by_action_df.to_string(index=False))

    print()
    print("High-impact query examples")
    print("-" * 110)
    if len(high_impact_df) > 0:
        print(high_impact_df.head(20).to_string(index=False))
    else:
        print("No high-impact queries found.")

    print()
    print("Files written")
    print("-" * 110)
    print(f"- simulation: {SIMULATION_PATH}")
    print(f"- summary: {SUMMARY_PATH}")
    print(f"- by action: {BY_ACTION_PATH}")
    print(f"- high impact queries: {HIGH_IMPACT_PATH}")


def run_repair_simulator(input_path: str) -> None:
    ensure_output_dir()

    critic_df = load_critic_report(input_path)
    simulated_df = add_repair_simulation(critic_df)

    output_df = choose_output_columns(simulated_df)
    summary_df = build_summary(simulated_df)
    by_action_df = build_by_action_summary(simulated_df)
    high_impact_df = build_high_impact_queries(simulated_df)

    output_df.to_csv(SIMULATION_PATH, index=False)
    summary_df.to_csv(SUMMARY_PATH, index=False)
    by_action_df.to_csv(BY_ACTION_PATH, index=False)
    high_impact_df.to_csv(HIGH_IMPACT_PATH, index=False)

    print_report(
        summary_df=summary_df,
        by_action_df=by_action_df,
        high_impact_df=high_impact_df,
    )


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--input",
        type=str,
        default=DEFAULT_INPUT_PATH,
        help="Path to critic verifier report CSV.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_repair_simulator(input_path=args.input)