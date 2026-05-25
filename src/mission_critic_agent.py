"""
MVP 15.6: Mission Critic Agent

Purpose:
--------
Critique a mission-aware shopping slate after coverage analysis.

This module builds on:
- MVP 15.1: Mission Agent
- MVP 15.2: Mission Slate Builder
- MVP 15.3: Mission Slate Guardrails
- MVP 15.5: Mission Coverage Analyzer

What it answers:
----------------
1. Should CORTEX accept the mission slate?
2. Should CORTEX retry missing needs?
3. Are missing needs critical, important, or optional?
4. Is the slate too thin or risky?
5. What is the recommended repair action?

Outputs:
--------
outputs/mission_critic_report.csv
outputs/mission_critic_summary.csv
outputs/mission_critic_repair_actions.csv
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import List, Tuple

import pandas as pd

from src.mission_coverage_analyzer import analyze_mission_coverage


OUTPUT_DIR = "outputs"

MISSION_CRITIC_REPORT_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "mission_critic_report.csv",
)
MISSION_CRITIC_SUMMARY_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "mission_critic_summary.csv",
)
MISSION_CRITIC_REPAIR_ACTIONS_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "mission_critic_repair_actions.csv",
)


@dataclass
class MissionCriticReport:
    query: str
    is_mission: bool
    mission_type: str
    mission_name: str
    mission_confidence: float
    mission_readiness_label: str
    critic_decision: str
    critic_priority: str
    critic_risk_score: float
    recommended_next_action: str
    main_failure_mode: str
    covered_sub_intents: str
    missing_sub_intents: str
    critical_missing_sub_intents: str
    critique: str


@dataclass
class MissionRepairAction:
    query: str
    sub_intent: str
    role: str
    priority: float
    priority_bucket: str
    coverage_status: str
    recommended_action: str
    critic_action: str
    repair_query: str
    repair_reason: str


def ensure_output_dir() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def safe_json_loads(value: object) -> List[str]:
    if value is None or pd.isna(value):
        return []

    try:
        parsed = json.loads(str(value))
        if isinstance(parsed, list):
            return [str(item) for item in parsed]
    except Exception:
        return []

    return []


def calculate_critic_risk_score(summary_row: pd.Series) -> float:
    if not bool(summary_row.get("is_mission", False)):
        return 0.0

    coverage = float(summary_row.get("mission_coverage_score", 0.0))
    weighted = float(summary_row.get("weighted_coverage_score", 0.0))
    critical_missing = int(summary_row.get("critical_missing_count", 0))
    important_missing = int(summary_row.get("important_missing_count", 0))
    optional_missing = int(summary_row.get("optional_missing_count", 0))

    risk = 0.0

    risk += max(0.0, 1.0 - weighted) * 0.45
    risk += max(0.0, 1.0 - coverage) * 0.25
    risk += min(critical_missing * 0.18, 0.45)
    risk += min(important_missing * 0.08, 0.24)
    risk += min(optional_missing * 0.03, 0.09)

    return round(min(risk, 1.0), 4)


def critic_priority_from_risk(
    risk_score: float,
    critical_missing_count: int,
    readiness_label: str,
) -> str:
    if readiness_label == "single_product_query":
        return "none"

    if critical_missing_count >= 2 or risk_score >= 0.70:
        return "critical"

    if critical_missing_count == 1 or risk_score >= 0.45:
        return "high"

    if risk_score >= 0.25:
        return "medium"

    return "low"


def critic_decision_from_summary(summary_row: pd.Series, risk_score: float) -> str:
    readiness = str(summary_row.get("mission_readiness_label", "unknown"))
    critical_missing = int(summary_row.get("critical_missing_count", 0))
    important_missing = int(summary_row.get("important_missing_count", 0))

    if readiness == "single_product_query":
        return "use_standard_cortex_pipeline"

    if readiness == "mission_ready":
        return "accept_slate"

    if readiness == "mostly_ready":
        if important_missing > 0 or risk_score >= 0.30:
            return "accept_with_optional_retry"
        return "accept_slate"

    if readiness == "partially_ready":
        if critical_missing > 0:
            return "retry_critical_missing_needs"
        return "retry_important_missing_needs"

    return "reject_and_expand_mission"


def main_failure_mode(summary_row: pd.Series) -> str:
    readiness = str(summary_row.get("mission_readiness_label", "unknown"))
    critical_missing = int(summary_row.get("critical_missing_count", 0))
    important_missing = int(summary_row.get("important_missing_count", 0))
    optional_missing = int(summary_row.get("optional_missing_count", 0))

    if readiness == "single_product_query":
        return "not_a_mission_query"

    if critical_missing > 0:
        return "critical_mission_needs_missing"

    if important_missing > 0:
        return "important_mission_needs_missing"

    if optional_missing > 0:
        return "optional_mission_needs_missing"

    return "no_major_failure"


def build_critique(summary_row: pd.Series, risk_score: float, decision: str) -> str:
    query = str(summary_row.get("query", ""))
    covered = safe_json_loads(summary_row.get("covered_sub_intents", "[]"))
    missing = safe_json_loads(summary_row.get("missing_sub_intents", "[]"))
    critical_missing = safe_json_loads(summary_row.get("critical_missing_sub_intents", "[]"))
    readiness = str(summary_row.get("mission_readiness_label", "unknown"))
    weighted = float(summary_row.get("weighted_coverage_score", 0.0))

    covered_text = ", ".join(covered) if covered else "none"
    missing_text = ", ".join(missing) if missing else "none"
    critical_text = ", ".join(critical_missing) if critical_missing else "none"

    if readiness == "single_product_query":
        return (
            f"'{query}' appears to be a single-product query. "
            "Mission critique is not needed; route through the standard CORTEX pipeline."
        )

    if decision == "accept_slate":
        return (
            f"The mission slate for '{query}' is strong enough to accept. "
            f"It covers {covered_text}. Weighted coverage is {weighted:.2f}, "
            f"and critic risk is {risk_score:.2f}."
        )

    if decision == "accept_with_optional_retry":
        return (
            f"The mission slate for '{query}' is usable but incomplete. "
            f"It covers {covered_text}, but misses {missing_text}. "
            f"No critical needs are missing. CORTEX can accept the slate, but should optionally retry "
            f"the missing needs if latency or budget allows."
        )

    if decision == "retry_important_missing_needs":
        return (
            f"The mission slate for '{query}' is only partially ready. "
            f"It covers {covered_text}, but misses important needs: {missing_text}. "
            f"CORTEX should retry retrieval for these missing important needs before accepting."
        )

    if decision == "retry_critical_missing_needs":
        return (
            f"The mission slate for '{query}' is risky because critical needs are missing: {critical_text}. "
            f"CORTEX should not accept the slate until it retries those critical sub-intents."
        )

    return (
        f"The mission slate for '{query}' is not ready. "
        f"It covers {covered_text}, but misses {missing_text}. "
        "CORTEX should reject this slate and expand or retry the mission plan."
    )


def build_repair_actions(query: str, coverage_df: pd.DataFrame) -> pd.DataFrame:
    if coverage_df.empty:
        return pd.DataFrame()

    repair_rows: List[MissionRepairAction] = []

    missing_df = coverage_df[coverage_df["is_covered"] == False].copy()  # noqa: E712

    for _, row in missing_df.iterrows():
        sub_intent = str(row.get("sub_intent", ""))
        role = str(row.get("role", ""))
        priority = float(row.get("priority", 0.0))
        bucket = str(row.get("priority_bucket", "optional"))
        coverage_status = str(row.get("coverage_status", "missing"))
        recommended_action = str(row.get("recommended_action", ""))

        if bucket == "critical":
            critic_action = "must_retry"
        elif bucket == "important":
            critic_action = "retry_if_budget_allows"
        else:
            critic_action = "optional_retry"

        repair_query = f"{sub_intent} for {query}"

        repair_reason = (
            f"Sub-intent '{sub_intent}' is missing after guardrails. "
            f"It is classified as {bucket}, so the critic action is {critic_action}."
        )

        repair_rows.append(
            MissionRepairAction(
                query=query,
                sub_intent=sub_intent,
                role=role,
                priority=priority,
                priority_bucket=bucket,
                coverage_status=coverage_status,
                recommended_action=recommended_action,
                critic_action=critic_action,
                repair_query=repair_query,
                repair_reason=repair_reason,
            )
        )

    return pd.DataFrame([asdict(row) for row in repair_rows])


def critique_mission_query(
    query: str,
    slate_size: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensure_output_dir()

    coverage_df, coverage_summary_df = analyze_mission_coverage(
        query=query,
        slate_size=slate_size,
    )

    summary_row = coverage_summary_df.iloc[0]

    risk_score = calculate_critic_risk_score(summary_row)

    critical_missing_count = int(summary_row.get("critical_missing_count", 0))
    readiness = str(summary_row.get("mission_readiness_label", "unknown"))

    priority = critic_priority_from_risk(
        risk_score=risk_score,
        critical_missing_count=critical_missing_count,
        readiness_label=readiness,
    )

    decision = critic_decision_from_summary(
        summary_row=summary_row,
        risk_score=risk_score,
    )

    failure_mode = main_failure_mode(summary_row)
    critique = build_critique(summary_row, risk_score, decision)

    report = MissionCriticReport(
        query=query,
        is_mission=bool(summary_row.get("is_mission", False)),
        mission_type=str(summary_row.get("mission_type", "")),
        mission_name=str(summary_row.get("mission_name", "")),
        mission_confidence=float(summary_row.get("mission_confidence", 0.0)),
        mission_readiness_label=readiness,
        critic_decision=decision,
        critic_priority=priority,
        critic_risk_score=risk_score,
        recommended_next_action=str(summary_row.get("recommended_next_action", "")),
        main_failure_mode=failure_mode,
        covered_sub_intents=str(summary_row.get("covered_sub_intents", "[]")),
        missing_sub_intents=str(summary_row.get("missing_sub_intents", "[]")),
        critical_missing_sub_intents=str(
            summary_row.get("critical_missing_sub_intents", "[]")
        ),
        critique=critique,
    )

    report_df = pd.DataFrame([asdict(report)])
    repair_df = build_repair_actions(query=query, coverage_df=coverage_df)

    summary_df = report_df[
        [
            "query",
            "is_mission",
            "mission_type",
            "mission_readiness_label",
            "critic_decision",
            "critic_priority",
            "critic_risk_score",
            "main_failure_mode",
            "critique",
        ]
    ].copy()

    return report_df, summary_df, repair_df


def run_demo() -> None:
    ensure_output_dir()

    demo_queries = [
        "world cup watch party",
        "camping trip essentials",
        "new apartment kitchen setup",
        "beach vacation packing list",
        "adidas soccer cleats",
    ]

    all_reports = []
    all_summaries = []
    all_repairs = []

    for query in demo_queries:
        report_df, summary_df, repair_df = critique_mission_query(query=query)

        all_reports.append(report_df)
        all_summaries.append(summary_df)

        if not repair_df.empty:
            all_repairs.append(repair_df)

    final_report_df = pd.concat(all_reports, ignore_index=True)
    final_summary_df = pd.concat(all_summaries, ignore_index=True)
    final_repair_df = (
        pd.concat(all_repairs, ignore_index=True)
        if all_repairs
        else pd.DataFrame()
    )

    final_report_df.to_csv(MISSION_CRITIC_REPORT_OUTPUT, index=False)
    final_summary_df.to_csv(MISSION_CRITIC_SUMMARY_OUTPUT, index=False)
    final_repair_df.to_csv(MISSION_CRITIC_REPAIR_ACTIONS_OUTPUT, index=False)

    print()
    print("MVP 15.6 Mission Critic Agent Demo")
    print("=" * 100)

    print()
    print("Mission Critic Summary")
    print("-" * 100)
    print(final_summary_df.to_string(index=False))

    if not final_repair_df.empty:
        print()
        print("Mission Repair Actions")
        print("-" * 100)
        print(final_repair_df.to_string(index=False))

    print()
    print("Files written")
    print("-" * 100)
    print(f"- critic report: {MISSION_CRITIC_REPORT_OUTPUT}")
    print(f"- critic summary: {MISSION_CRITIC_SUMMARY_OUTPUT}")
    print(f"- repair actions: {MISSION_CRITIC_REPAIR_ACTIONS_OUTPUT}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Mission query to critique.",
    )

    parser.add_argument(
        "--slate-size",
        type=int,
        default=10,
        help="Maximum raw mission slate size.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.query:
        report, summary, repair = critique_mission_query(
            query=args.query,
            slate_size=args.slate_size,
        )

        print()
        print("Mission Critic Report")
        print("-" * 100)
        print(report.to_string(index=False))

        if not repair.empty:
            print()
            print("Mission Repair Actions")
            print("-" * 100)
            print(repair.to_string(index=False))
    else:
        run_demo()