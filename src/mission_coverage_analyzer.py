"""
MVP 15.5: Mission Coverage + Missing Needs Analyzer

Purpose:
--------
Analyze whether a mission-aware guarded slate sufficiently covers the user's shopping mission.

This module builds on:
- MVP 15.1: Mission Agent
- MVP 15.2: Mission Slate Builder
- MVP 15.3: Mission Slate Guardrails

What it answers:
----------------
1. Which sub-intents are covered?
2. Which sub-intents are missing after guardrails?
3. Are the missing needs critical or optional?
4. Is the mission slate complete enough?
5. What should CORTEX do next?

Outputs:
--------
outputs/mission_coverage_analysis.csv
outputs/mission_coverage_summary.csv
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import pandas as pd

from src.mission_agent import analyze_mission_query
from src.mission_slate_guardrails import build_guarded_mission_slate


OUTPUT_DIR = "outputs"
MISSION_COVERAGE_ANALYSIS_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "mission_coverage_analysis.csv",
)
MISSION_COVERAGE_SUMMARY_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "mission_coverage_summary.csv",
)


CRITICAL_PRIORITY_THRESHOLD = 0.80
IMPORTANT_PRIORITY_THRESHOLD = 0.60


@dataclass
class MissionCoverageRow:
    query: str
    mission_type: str
    mission_name: str
    sub_intent: str
    role: str
    priority: float
    priority_bucket: str
    is_covered: bool
    matched_product_count: int
    top_product_title: str
    avg_final_mission_score: float
    coverage_status: str
    recommended_action: str


@dataclass
class MissionCoverageSummary:
    query: str
    is_mission: bool
    mission_type: str
    mission_name: str
    mission_confidence: float
    requested_sub_intent_count: int
    covered_sub_intent_count: int
    missing_sub_intent_count: int
    critical_missing_count: int
    important_missing_count: int
    optional_missing_count: int
    mission_coverage_score: float
    weighted_coverage_score: float
    mission_readiness_label: str
    recommended_next_action: str
    covered_sub_intents: str
    missing_sub_intents: str
    critical_missing_sub_intents: str
    explanation: str


def ensure_output_dir() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def priority_bucket(priority: float) -> str:
    if priority >= CRITICAL_PRIORITY_THRESHOLD:
        return "critical"
    if priority >= IMPORTANT_PRIORITY_THRESHOLD:
        return "important"
    return "optional"


def coverage_status(is_covered: bool, bucket: str) -> str:
    if is_covered:
        return "covered"
    return f"missing_{bucket}"


def recommended_action_for_row(is_covered: bool, bucket: str) -> str:
    if is_covered:
        return "keep_in_slate"

    if bucket == "critical":
        return "retry_retrieval_or_expand_query"
    if bucket == "important":
        return "retry_if_budget_allows"
    return "optional_gap_acceptable"


def readiness_label(
    weighted_coverage_score: float,
    critical_missing_count: int,
    important_missing_count: int,
) -> str:
    if critical_missing_count == 0 and weighted_coverage_score >= 0.85:
        return "mission_ready"

    if critical_missing_count <= 1 and weighted_coverage_score >= 0.70:
        return "mostly_ready"

    if critical_missing_count <= 2 and weighted_coverage_score >= 0.50:
        return "partially_ready"

    return "not_ready"


def recommended_next_action(
    label: str,
    critical_missing_count: int,
    important_missing_count: int,
) -> str:
    if label == "mission_ready":
        return "accept_guarded_slate"

    if label == "mostly_ready":
        return "accept_with_optional_retry"

    if label == "partially_ready":
        if critical_missing_count > 0:
            return "retry_missing_critical_sub_intents"
        return "retry_missing_important_sub_intents"

    return "do_not_accept_retry_or_expand_mission"


def build_explanation(
    query: str,
    label: str,
    covered: List[str],
    missing: List[str],
    critical_missing: List[str],
) -> str:
    covered_text = ", ".join(covered) if covered else "none"
    missing_text = ", ".join(missing) if missing else "none"

    if label == "mission_ready":
        return (
            f"The mission slate for '{query}' is ready. "
            f"It covers the important shopping needs: {covered_text}."
        )

    if label == "mostly_ready":
        return (
            f"The mission slate for '{query}' is mostly ready. "
            f"It covers {covered_text}, but still misses {missing_text}. "
            "CORTEX can accept the slate or optionally retry missing parts."
        )

    if label == "partially_ready":
        if critical_missing:
            critical_text = ", ".join(critical_missing)
            return (
                f"The mission slate for '{query}' is only partially ready. "
                f"It covers {covered_text}, but misses critical needs: {critical_text}. "
                "CORTEX should retry retrieval for missing critical sub-intents."
            )

        return (
            f"The mission slate for '{query}' is partially ready. "
            f"It covers {covered_text}, but misses {missing_text}. "
            "CORTEX should retry important missing needs if budget allows."
        )

    return (
        f"The mission slate for '{query}' is not ready. "
        f"It covers {covered_text}, but misses {missing_text}. "
        "CORTEX should not accept this slate without retrying or expanding retrieval."
    )


def analyze_mission_coverage(
    query: str,
    slate_size: int = 10,
    max_per_sub_intent: int = 2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ensure_output_dir()

    mission = analyze_mission_query(query)

    guarded_df, guardrail_summary_df, rejected_df = build_guarded_mission_slate(
        query=query,
        slate_size=slate_size,
        max_per_sub_intent=max_per_sub_intent,
    )

    if not mission.is_mission:
        summary = MissionCoverageSummary(
            query=query,
            is_mission=False,
            mission_type=mission.mission_type,
            mission_name=mission.mission_name,
            mission_confidence=mission.confidence,
            requested_sub_intent_count=0,
            covered_sub_intent_count=0,
            missing_sub_intent_count=0,
            critical_missing_count=0,
            important_missing_count=0,
            optional_missing_count=0,
            mission_coverage_score=0.0,
            weighted_coverage_score=0.0,
            mission_readiness_label="single_product_query",
            recommended_next_action="use_standard_cortex_pipeline",
            covered_sub_intents="[]",
            missing_sub_intents="[]",
            critical_missing_sub_intents="[]",
            explanation=mission.reasoning,
        )

        return pd.DataFrame(), pd.DataFrame([asdict(summary)])

    coverage_rows: List[MissionCoverageRow] = []

    total_weight = 0.0
    covered_weight = 0.0

    covered_sub_intents: List[str] = []
    missing_sub_intents: List[str] = []
    critical_missing_sub_intents: List[str] = []

    critical_missing_count = 0
    important_missing_count = 0
    optional_missing_count = 0

    for sub_intent in mission.sub_intents:
        name = str(sub_intent.get("name", ""))
        role = str(sub_intent.get("role", ""))
        priority = float(sub_intent.get("priority", 0.0))
        bucket = priority_bucket(priority)

        total_weight += priority

        if guarded_df.empty:
            matches = pd.DataFrame()
        else:
            matches = guarded_df[guarded_df["sub_intent"] == name]

        is_covered = not matches.empty

        if is_covered:
            covered_weight += priority
            covered_sub_intents.append(name)
            matched_product_count = len(matches)
            top_product_title = str(matches.iloc[0].get("product_title", ""))
            avg_final_score = round(float(matches["final_mission_score"].mean()), 4)
        else:
            missing_sub_intents.append(name)
            matched_product_count = 0
            top_product_title = ""
            avg_final_score = 0.0

            if bucket == "critical":
                critical_missing_count += 1
                critical_missing_sub_intents.append(name)
            elif bucket == "important":
                important_missing_count += 1
            else:
                optional_missing_count += 1

        coverage_rows.append(
            MissionCoverageRow(
                query=query,
                mission_type=mission.mission_type,
                mission_name=mission.mission_name,
                sub_intent=name,
                role=role,
                priority=priority,
                priority_bucket=bucket,
                is_covered=is_covered,
                matched_product_count=matched_product_count,
                top_product_title=top_product_title,
                avg_final_mission_score=avg_final_score,
                coverage_status=coverage_status(is_covered, bucket),
                recommended_action=recommended_action_for_row(is_covered, bucket),
            )
        )

    requested_count = len(mission.sub_intents)
    covered_count = len(covered_sub_intents)
    missing_count = len(missing_sub_intents)

    mission_coverage_score = (
        round(covered_count / requested_count, 4)
        if requested_count
        else 0.0
    )

    weighted_coverage_score = (
        round(covered_weight / total_weight, 4)
        if total_weight
        else 0.0
    )

    label = readiness_label(
        weighted_coverage_score=weighted_coverage_score,
        critical_missing_count=critical_missing_count,
        important_missing_count=important_missing_count,
    )

    next_action = recommended_next_action(
        label=label,
        critical_missing_count=critical_missing_count,
        important_missing_count=important_missing_count,
    )

    explanation = build_explanation(
        query=query,
        label=label,
        covered=covered_sub_intents,
        missing=missing_sub_intents,
        critical_missing=critical_missing_sub_intents,
    )

    coverage_df = pd.DataFrame([asdict(row) for row in coverage_rows])

    summary = MissionCoverageSummary(
        query=query,
        is_mission=True,
        mission_type=mission.mission_type,
        mission_name=mission.mission_name,
        mission_confidence=mission.confidence,
        requested_sub_intent_count=requested_count,
        covered_sub_intent_count=covered_count,
        missing_sub_intent_count=missing_count,
        critical_missing_count=critical_missing_count,
        important_missing_count=important_missing_count,
        optional_missing_count=optional_missing_count,
        mission_coverage_score=mission_coverage_score,
        weighted_coverage_score=weighted_coverage_score,
        mission_readiness_label=label,
        recommended_next_action=next_action,
        covered_sub_intents=json.dumps(covered_sub_intents),
        missing_sub_intents=json.dumps(missing_sub_intents),
        critical_missing_sub_intents=json.dumps(critical_missing_sub_intents),
        explanation=explanation,
    )

    summary_df = pd.DataFrame([asdict(summary)])

    return coverage_df, summary_df


def run_demo() -> None:
    ensure_output_dir()

    demo_queries = [
        "world cup watch party",
        "camping trip essentials",
        "new apartment kitchen setup",
        "beach vacation packing list",
        "adidas soccer cleats",
    ]

    all_coverage_rows = []
    all_summary_rows = []

    for query in demo_queries:
        coverage_df, summary_df = analyze_mission_coverage(
            query=query,
            slate_size=10,
            max_per_sub_intent=2,
        )

        if not coverage_df.empty:
            all_coverage_rows.append(coverage_df)

        all_summary_rows.append(summary_df)

    final_coverage_df = (
        pd.concat(all_coverage_rows, ignore_index=True)
        if all_coverage_rows
        else pd.DataFrame()
    )

    final_summary_df = pd.concat(all_summary_rows, ignore_index=True)

    final_coverage_df.to_csv(MISSION_COVERAGE_ANALYSIS_OUTPUT, index=False)
    final_summary_df.to_csv(MISSION_COVERAGE_SUMMARY_OUTPUT, index=False)

    print()
    print("MVP 15.5 Mission Coverage Analyzer Demo")
    print("=" * 100)

    print()
    print("Mission Coverage Summary")
    print("-" * 100)
    print(final_summary_df.to_string(index=False))

    if not final_coverage_df.empty:
        print()
        print("Mission Coverage Detail Sample")
        print("-" * 100)
        print(
            final_coverage_df[
                [
                    "query",
                    "sub_intent",
                    "role",
                    "priority",
                    "priority_bucket",
                    "is_covered",
                    "coverage_status",
                    "recommended_action",
                    "top_product_title",
                ]
            ]
            .head(40)
            .to_string(index=False)
        )

    print()
    print("Files written")
    print("-" * 100)
    print(f"- coverage analysis: {MISSION_COVERAGE_ANALYSIS_OUTPUT}")
    print(f"- coverage summary: {MISSION_COVERAGE_SUMMARY_OUTPUT}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Mission query to analyze.",
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
        coverage, summary = analyze_mission_coverage(
            query=args.query,
            slate_size=args.slate_size,
        )

        print()
        print("Mission Coverage Summary")
        print("-" * 100)
        print(summary.to_string(index=False))

        if coverage.empty:
            print()
            print("No mission coverage details generated.")
        else:
            print()
            print("Mission Coverage Detail")
            print("-" * 100)
            print(
                coverage[
                    [
                        "sub_intent",
                        "role",
                        "priority",
                        "priority_bucket",
                        "is_covered",
                        "coverage_status",
                        "recommended_action",
                        "top_product_title",
                    ]
                ].to_string(index=False)
            )
    else:
        run_demo()