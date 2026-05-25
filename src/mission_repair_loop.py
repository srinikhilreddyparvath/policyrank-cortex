"""
MVP 15.7: Mission Repair Loop

Purpose:
--------
Take the Mission Critic Agent's missing-needs repair actions and actually
attempt to retrieve replacement products for those missing sub-intents.

This module builds on:
- MVP 15.1: Mission Agent
- MVP 15.2: Mission Slate Builder
- MVP 15.3: Mission Slate Guardrails
- MVP 15.5: Mission Coverage Analyzer
- MVP 15.6: Mission Critic Agent

What it does:
-------------
1. Critiques a mission query.
2. Reads the critic's repair actions.
3. Runs targeted repair retrieval for missing sub-intents.
4. Applies lightweight repair guardrails.
5. Produces a repaired mission slate and summary.

Outputs:
--------
outputs/mission_repair_loop_repaired_slate.csv
outputs/mission_repair_loop_summary.csv
outputs/mission_repair_loop_candidates.csv
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import List, Tuple

import pandas as pd

from src.mission_critic_agent import critique_mission_query
from src.mission_slate_builder import (
    load_product_data,
    prepare_product_dataframe,
    retrieve_for_sub_intent,
)
from src.mission_slate_guardrails import (
    evaluate_guardrail,
    build_guarded_mission_slate,
)


OUTPUT_DIR = "outputs"

REPAIRED_SLATE_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "mission_repair_loop_repaired_slate.csv",
)
REPAIR_SUMMARY_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "mission_repair_loop_summary.csv",
)
REPAIR_CANDIDATES_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "mission_repair_loop_candidates.csv",
)


@dataclass
class MissionRepairCandidate:
    query: str
    repair_query: str
    sub_intent: str
    role: str
    priority: float
    priority_bucket: str
    critic_action: str
    product_id: str
    product_title: str
    product_brand: str
    product_text_score: float
    product_role_score: float
    candidate_score: float
    accepted_by_guardrail: bool
    guardrail_reason: str
    title_intent_score: float
    title_role_score: float


@dataclass
class MissionRepairSummary:
    query: str
    critic_decision: str
    critic_priority: str
    critic_risk_score: float
    raw_guarded_slate_size: int
    repair_action_count: int
    attempted_repair_count: int
    accepted_repair_count: int
    final_repaired_slate_size: int
    repaired_sub_intents: str
    unrepaired_sub_intents: str
    repair_success_rate: float
    final_decision_after_repair: str
    explanation: str


def ensure_output_dir() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def build_repair_candidates_for_action(
    products_df: pd.DataFrame,
    query: str,
    repair_row: pd.Series,
    top_n: int = 10,
) -> pd.DataFrame:
    sub_intent = str(repair_row.get("sub_intent", ""))
    role = str(repair_row.get("role", ""))
    repair_query = str(repair_row.get("repair_query", f"{sub_intent} for {query}"))

    candidate_df = retrieve_for_sub_intent(
        products_df=products_df,
        sub_intent_name=repair_query,
        sub_intent_role=role,
        top_n=top_n,
    )

    if candidate_df.empty:
        return pd.DataFrame()

    candidate_df = candidate_df.copy()
    candidate_df["query"] = query
    candidate_df["repair_query"] = repair_query
    candidate_df["sub_intent"] = sub_intent
    candidate_df["role"] = role
    candidate_df["priority"] = float(repair_row.get("priority", 0.0))
    candidate_df["priority_bucket"] = str(repair_row.get("priority_bucket", "optional"))
    candidate_df["critic_action"] = str(repair_row.get("critic_action", ""))

    return candidate_df


def evaluate_repair_candidate(row: pd.Series) -> MissionRepairCandidate:
    guardrail_like_row = pd.Series(
        {
            "sub_intent": row.get("sub_intent", ""),
            "sub_intent_role": row.get("role", ""),
            "product_title": row.get("product_title", ""),
            "product_text_score": row.get("product_text_score", 0.0),
            "product_role_score": row.get("product_role_score", 0.0),
            "final_mission_score": row.get("candidate_score", 0.0),
        }
    )

    (
        accepted,
        guardrail_reason,
        title_intent_score,
        title_role_score,
        raw_final_score,
    ) = evaluate_guardrail(guardrail_like_row)

    return MissionRepairCandidate(
        query=str(row.get("query", "")),
        repair_query=str(row.get("repair_query", "")),
        sub_intent=str(row.get("sub_intent", "")),
        role=str(row.get("role", "")),
        priority=float(row.get("priority", 0.0)),
        priority_bucket=str(row.get("priority_bucket", "")),
        critic_action=str(row.get("critic_action", "")),
        product_id=str(row.get("product_id", "")),
        product_title=str(row.get("product_title", "")),
        product_brand=str(row.get("product_brand", "")),
        product_text_score=float(row.get("product_text_score", 0.0)),
        product_role_score=float(row.get("product_role_score", 0.0)),
        candidate_score=float(row.get("candidate_score", 0.0)),
        accepted_by_guardrail=bool(accepted),
        guardrail_reason=guardrail_reason,
        title_intent_score=float(title_intent_score),
        title_role_score=float(title_role_score),
    )


def choose_best_repairs(
    repair_candidates_df: pd.DataFrame,
    used_product_ids: set,
    max_repairs: int = 5,
) -> pd.DataFrame:
    if repair_candidates_df.empty:
        return pd.DataFrame()

    accepted = repair_candidates_df[
        repair_candidates_df["accepted_by_guardrail"] == True  # noqa: E712
    ].copy()

    if accepted.empty:
        return pd.DataFrame()

    priority_order = {
        "critical": 3,
        "important": 2,
        "optional": 1,
    }

    accepted["priority_rank"] = accepted["priority_bucket"].map(priority_order).fillna(0)

    accepted = accepted.sort_values(
        by=[
            "priority_rank",
            "priority",
            "candidate_score",
            "title_intent_score",
            "title_role_score",
        ],
        ascending=[False, False, False, False, False],
    )

    selected_rows = []
    repaired_sub_intents = set()

    for _, row in accepted.iterrows():
        if len(selected_rows) >= max_repairs:
            break

        product_id = str(row.get("product_id", ""))
        sub_intent = str(row.get("sub_intent", ""))

        if product_id in used_product_ids:
            continue

        if sub_intent in repaired_sub_intents:
            continue

        selected_rows.append(row.to_dict())
        used_product_ids.add(product_id)
        repaired_sub_intents.add(sub_intent)

    if not selected_rows:
        return pd.DataFrame()

    selected_df = pd.DataFrame(selected_rows)
    selected_df["repair_rank"] = range(1, len(selected_df) + 1)

    return selected_df


def build_repaired_mission_slate(
    query: str,
    slate_size: int = 10,
    max_repairs: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensure_output_dir()

    critic_report_df, critic_summary_df, repair_actions_df = critique_mission_query(
        query=query,
        slate_size=slate_size,
    )

    critic_row = critic_report_df.iloc[0]

    guarded_df, guarded_summary_df, rejected_df = build_guarded_mission_slate(
        query=query,
        slate_size=slate_size,
    )

    if not bool(critic_row.get("is_mission", False)):
        summary = MissionRepairSummary(
            query=query,
            critic_decision=str(critic_row.get("critic_decision", "")),
            critic_priority=str(critic_row.get("critic_priority", "")),
            critic_risk_score=float(critic_row.get("critic_risk_score", 0.0)),
            raw_guarded_slate_size=0,
            repair_action_count=0,
            attempted_repair_count=0,
            accepted_repair_count=0,
            final_repaired_slate_size=0,
            repaired_sub_intents="[]",
            unrepaired_sub_intents="[]",
            repair_success_rate=0.0,
            final_decision_after_repair="use_standard_cortex_pipeline",
            explanation="Single-product query; mission repair loop is not needed.",
        )

        return pd.DataFrame(), pd.DataFrame([asdict(summary)]), pd.DataFrame()

    if repair_actions_df.empty:
        guarded_final = guarded_df.copy()
        guarded_final["slate_source"] = "original_guarded_slate"
        guarded_final["final_rank"] = range(1, len(guarded_final) + 1)

        summary = MissionRepairSummary(
            query=query,
            critic_decision=str(critic_row.get("critic_decision", "")),
            critic_priority=str(critic_row.get("critic_priority", "")),
            critic_risk_score=float(critic_row.get("critic_risk_score", 0.0)),
            raw_guarded_slate_size=len(guarded_df),
            repair_action_count=0,
            attempted_repair_count=0,
            accepted_repair_count=0,
            final_repaired_slate_size=len(guarded_final),
            repaired_sub_intents="[]",
            unrepaired_sub_intents="[]",
            repair_success_rate=1.0,
            final_decision_after_repair="accept_slate",
            explanation="No repair actions were needed. The guarded slate is accepted.",
        )

        return guarded_final, pd.DataFrame([asdict(summary)]), pd.DataFrame()

    raw_products, data_path = load_product_data()
    products = prepare_product_dataframe(raw_products)

    used_product_ids = set()
    if not guarded_df.empty:
        used_product_ids = set(guarded_df["product_id"].astype(str).tolist())

    all_candidate_rows = []

    for _, repair_action in repair_actions_df.iterrows():
        candidate_df = build_repair_candidates_for_action(
            products_df=products,
            query=query,
            repair_row=repair_action,
            top_n=15,
        )

        if candidate_df.empty:
            continue

        for _, candidate in candidate_df.iterrows():
            evaluated = evaluate_repair_candidate(candidate)
            all_candidate_rows.append(asdict(evaluated))

    repair_candidates_df = (
        pd.DataFrame(all_candidate_rows)
        if all_candidate_rows
        else pd.DataFrame()
    )

    selected_repairs_df = choose_best_repairs(
        repair_candidates_df=repair_candidates_df,
        used_product_ids=used_product_ids,
        max_repairs=max_repairs,
    )

    guarded_final = guarded_df.copy()

    if not guarded_final.empty:
        guarded_final["slate_source"] = "original_guarded_slate"

    if not selected_repairs_df.empty:
        repair_slate_rows = []

        for _, row in selected_repairs_df.iterrows():
            repair_slate_rows.append(
                {
                    "query": query,
                    "mission_type": str(critic_row.get("mission_type", "")),
                    "mission_name": str(critic_row.get("mission_name", "")),
                    "mission_confidence": float(critic_row.get("mission_confidence", 0.0)),
                    "sub_intent": str(row.get("sub_intent", "")),
                    "sub_intent_priority": float(row.get("priority", 0.0)),
                    "sub_intent_role": str(row.get("role", "")),
                    "selected_rank": None,
                    "product_id": str(row.get("product_id", "")),
                    "product_title": str(row.get("product_title", "")),
                    "product_brand": str(row.get("product_brand", "")),
                    "product_text_score": float(row.get("product_text_score", 0.0)),
                    "product_role_score": float(row.get("product_role_score", 0.0)),
                    "final_mission_score": float(row.get("candidate_score", 0.0)),
                    "retrieval_reason": (
                        f"Repair retrieval selected this product for missing sub-intent "
                        f"'{row.get('sub_intent', '')}' using repair query "
                        f"'{row.get('repair_query', '')}'."
                    ),
                    "guardrail_accepted": True,
                    "guardrail_reason": str(row.get("guardrail_reason", "")),
                    "title_intent_score": float(row.get("title_intent_score", 0.0)),
                    "title_role_score": float(row.get("title_role_score", 0.0)),
                    "raw_final_mission_score": float(row.get("candidate_score", 0.0)),
                    "guarded_rank": None,
                    "slate_source": "repair_loop",
                }
            )

        repair_slate_df = pd.DataFrame(repair_slate_rows)
        final_slate_df = pd.concat([guarded_final, repair_slate_df], ignore_index=True)
    else:
        final_slate_df = guarded_final.copy()

    if final_slate_df.empty:
        final_slate_df = pd.DataFrame()

    if not final_slate_df.empty:
        final_slate_df = final_slate_df.sort_values(
            by=["slate_source", "sub_intent_priority", "final_mission_score"],
            ascending=[True, False, False],
        ).reset_index(drop=True)

        final_slate_df["final_rank"] = range(1, len(final_slate_df) + 1)

    repaired_sub_intents = (
        selected_repairs_df["sub_intent"].astype(str).tolist()
        if not selected_repairs_df.empty
        else []
    )

    all_missing_sub_intents = repair_actions_df["sub_intent"].astype(str).tolist()

    unrepaired_sub_intents = [
        item
        for item in all_missing_sub_intents
        if item not in repaired_sub_intents
    ]

    attempted_count = len(all_missing_sub_intents)
    accepted_count = len(repaired_sub_intents)

    repair_success_rate = (
        round(accepted_count / attempted_count, 4)
        if attempted_count
        else 1.0
    )

    if attempted_count == 0:
        final_decision = "accept_slate"
    elif accepted_count == attempted_count:
        final_decision = "accept_repaired_slate"
    elif accepted_count > 0:
        final_decision = "accept_with_remaining_gaps"
    else:
        final_decision = "repair_failed_retry_or_expand"

    explanation = (
        f"CORTEX attempted {attempted_count} repair actions for '{query}' and accepted "
        f"{accepted_count}. Repaired sub-intents: "
        f"{', '.join(repaired_sub_intents) if repaired_sub_intents else 'none'}. "
        f"Remaining gaps: {', '.join(unrepaired_sub_intents) if unrepaired_sub_intents else 'none'}."
    )

    summary = MissionRepairSummary(
        query=query,
        critic_decision=str(critic_row.get("critic_decision", "")),
        critic_priority=str(critic_row.get("critic_priority", "")),
        critic_risk_score=float(critic_row.get("critic_risk_score", 0.0)),
        raw_guarded_slate_size=len(guarded_df),
        repair_action_count=len(repair_actions_df),
        attempted_repair_count=attempted_count,
        accepted_repair_count=accepted_count,
        final_repaired_slate_size=len(final_slate_df),
        repaired_sub_intents=json.dumps(repaired_sub_intents),
        unrepaired_sub_intents=json.dumps(unrepaired_sub_intents),
        repair_success_rate=repair_success_rate,
        final_decision_after_repair=final_decision,
        explanation=explanation,
    )

    return final_slate_df, pd.DataFrame([asdict(summary)]), repair_candidates_df


def run_demo() -> None:
    ensure_output_dir()

    demo_queries = [
        "world cup watch party",
        "camping trip essentials",
        "new apartment kitchen setup",
        "beach vacation packing list",
    ]

    all_slates = []
    all_summaries = []
    all_candidates = []

    for query in demo_queries:
        slate_df, summary_df, candidates_df = build_repaired_mission_slate(
            query=query,
            slate_size=10,
            max_repairs=5,
        )

        if not slate_df.empty:
            all_slates.append(slate_df)

        all_summaries.append(summary_df)

        if not candidates_df.empty:
            all_candidates.append(candidates_df)

    final_slate_df = (
        pd.concat(all_slates, ignore_index=True)
        if all_slates
        else pd.DataFrame()
    )

    final_summary_df = pd.concat(all_summaries, ignore_index=True)

    final_candidates_df = (
        pd.concat(all_candidates, ignore_index=True)
        if all_candidates
        else pd.DataFrame()
    )

    final_slate_df.to_csv(REPAIRED_SLATE_OUTPUT, index=False)
    final_summary_df.to_csv(REPAIR_SUMMARY_OUTPUT, index=False)
    final_candidates_df.to_csv(REPAIR_CANDIDATES_OUTPUT, index=False)

    print()
    print("MVP 15.7 Mission Repair Loop Demo")
    print("=" * 100)

    print()
    print("Repair Summary")
    print("-" * 100)
    print(final_summary_df.to_string(index=False))

    if not final_slate_df.empty:
        print()
        print("Repaired Mission Slate Sample")
        print("-" * 100)
        cols = [
            "query",
            "final_rank",
            "slate_source",
            "sub_intent",
            "sub_intent_role",
            "product_title",
            "final_mission_score",
        ]
        print(final_slate_df[cols].head(40).to_string(index=False))

    if not final_candidates_df.empty:
        print()
        print("Repair Candidate Sample")
        print("-" * 100)
        cols = [
            "query",
            "repair_query",
            "sub_intent",
            "role",
            "product_title",
            "candidate_score",
            "accepted_by_guardrail",
            "guardrail_reason",
        ]
        print(final_candidates_df[cols].head(30).to_string(index=False))

    print()
    print("Files written")
    print("-" * 100)
    print(f"- repaired slate: {REPAIRED_SLATE_OUTPUT}")
    print(f"- repair summary: {REPAIR_SUMMARY_OUTPUT}")
    print(f"- repair candidates: {REPAIR_CANDIDATES_OUTPUT}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Mission query to repair.",
    )

    parser.add_argument(
        "--slate-size",
        type=int,
        default=10,
        help="Maximum raw mission slate size.",
    )

    parser.add_argument(
        "--max-repairs",
        type=int,
        default=5,
        help="Maximum accepted repair products to append.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.query:
        slate, summary, candidates = build_repaired_mission_slate(
            query=args.query,
            slate_size=args.slate_size,
            max_repairs=args.max_repairs,
        )

        print()
        print("Repair Summary")
        print("-" * 100)
        print(summary.to_string(index=False))

        if slate.empty:
            print()
            print("No repaired slate generated.")
        else:
            print()
            print("Repaired Mission Slate")
            print("-" * 100)
            print(
                slate[
                    [
                        "final_rank",
                        "slate_source",
                        "sub_intent",
                        "sub_intent_role",
                        "product_title",
                        "final_mission_score",
                    ]
                ].to_string(index=False)
            )

        if not candidates.empty:
            print()
            print("Repair Candidates")
            print("-" * 100)
            print(
                candidates[
                    [
                        "repair_query",
                        "sub_intent",
                        "role",
                        "product_title",
                        "candidate_score",
                        "accepted_by_guardrail",
                        "guardrail_reason",
                    ]
                ].head(30).to_string(index=False)
            )
    else:
        run_demo()