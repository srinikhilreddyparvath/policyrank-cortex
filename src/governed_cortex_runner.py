"""
MVP 18: End-to-End Governed CORTEX Runner

This module turns the CORTEX Governance Agent from a decision layer into an execution layer.

What MVP 17 did:
- Governance Agent recommended a route.

What MVP 18 does:
- Runs the Governance Agent.
- Reads the governance route.
- Executes the selected governed path.
- Produces one final governed CORTEX slate.
- Writes a summary and trace explaining what happened.

This runner does not fabricate CTR, ATC, or purchase engagement.
It uses existing CORTEX outputs:
- Governance decisions
- Strict repaired mission slate
- Behavior-aware CORTEX slate
- Governance signals and policy reasons

Routes handled:
- REJECT_REPAIR_NARROW_QUERY
- BASELINE_ONLY
- MISSION_REPAIR
- STRICT_REPAIR
- BEHAVIOR_AWARE_RERANK
- CRITIC_REVIEW

For now:
- Baseline-style routes create a governed fallback row explaining why CORTEX did not run aggressive repair.
- Mission/strict/behavior routes use the best available existing CORTEX slate.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Tuple


OUTPUT_DIR = Path("outputs")

GOVERNANCE_DECISIONS_PATH = OUTPUT_DIR / "cortex_governance_decisions.csv"
GOVERNANCE_SUMMARY_PATH = OUTPUT_DIR / "cortex_governance_summary.csv"
GOVERNANCE_TRACE_PATH = OUTPUT_DIR / "cortex_governance_trace.csv"

STRICT_SLATE_PATH = OUTPUT_DIR / "mission_strict_repair_slate.csv"
STRICT_SUMMARY_PATH = OUTPUT_DIR / "mission_strict_repair_summary.csv"
STRICT_REJECTIONS_PATH = OUTPUT_DIR / "mission_strict_repair_rejections.csv"

BEHAVIOR_SLATE_PATH = OUTPUT_DIR / "behavior_aware_cortex_slate.csv"
BEHAVIOR_SUMMARY_PATH = OUTPUT_DIR / "behavior_aware_cortex_summary.csv"
BEHAVIOR_REASONS_PATH = OUTPUT_DIR / "behavior_aware_cortex_policy_reasons.csv"

GOVERNED_FINAL_SLATE_PATH = OUTPUT_DIR / "governed_cortex_final_slate.csv"
GOVERNED_SUMMARY_PATH = OUTPUT_DIR / "governed_cortex_summary.csv"
GOVERNED_TRACE_PATH = OUTPUT_DIR / "governed_cortex_trace.csv"


ROUTE_BASELINE_ONLY = "BASELINE_ONLY"
ROUTE_MISSION_BUILD = "MISSION_BUILD"
ROUTE_MISSION_REPAIR = "MISSION_REPAIR"
ROUTE_STRICT_REPAIR = "STRICT_REPAIR"
ROUTE_BEHAVIOR_AWARE_RERANK = "BEHAVIOR_AWARE_RERANK"
ROUTE_CRITIC_REVIEW = "CRITIC_REVIEW"
ROUTE_REJECT_REPAIR_NARROW_QUERY = "REJECT_REPAIR_NARROW_QUERY"


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(value: object) -> str:
    return str(value or "").strip()


def lower_text(value: object) -> str:
    return clean_text(value).lower()


def safe_int(value: object, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []

    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    ensure_output_dir()

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def filter_rows_by_query(rows: List[Dict[str, str]], query: str) -> List[Dict[str, str]]:
    if not rows:
        return []

    if "query" not in rows[0]:
        return rows

    query_lower = query.lower()
    return [row for row in rows if lower_text(row.get("query")) == query_lower]


def latest_query_row(path: Path, query: str) -> Dict[str, str]:
    rows = filter_rows_by_query(read_csv(path), query)
    if not rows:
        return {}
    return rows[-1]


def append_or_replace_by_query(
    path: Path,
    new_rows: List[Dict[str, object]],
    fieldnames: List[str],
    query: str,
) -> None:
    existing = read_csv(path)
    existing = [row for row in existing if lower_text(row.get("query")) != query.lower()]
    write_csv(path, existing + new_rows, fieldnames)


def run_governance_agent(query: str, skip_refresh: bool = False) -> Tuple[bool, str]:
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "src.cortex_governance_agent",
        "--query",
        query,
    ]

    if skip_refresh:
        cmd.append("--skip-refresh")

    result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
    )

    output = ""

    if result.stdout:
        output += result.stdout

    if result.stderr:
        output += "\n\nSTDERR:\n" + result.stderr

    return result.returncode == 0, output


def get_governance_decision(query: str) -> Dict[str, str]:
    decision = latest_query_row(GOVERNANCE_DECISIONS_PATH, query)

    if not decision:
        raise RuntimeError(
            f"No governance decision found for query '{query}'. "
            f"Run src.cortex_governance_agent first."
        )

    return decision


def get_governance_summary(query: str) -> Dict[str, str]:
    return latest_query_row(GOVERNANCE_SUMMARY_PATH, query)


def normalize_behavior_row(
    query: str,
    row: Dict[str, str],
    governed_rank: int,
    route: str,
    decision: str,
) -> Dict[str, object]:
    return {
        "query": query,
        "governed_rank": governed_rank,
        "governed_source": "behavior_aware_cortex_slate",
        "governance_route": route,
        "governance_decision": decision,
        "product_title": clean_text(row.get("product_title")),
        "sub_intent": clean_text(row.get("sub_intent")),
        "sub_intent_role": clean_text(row.get("sub_intent_role")),
        "mission_stage": clean_text(row.get("mission_stage")),
        "behavior_rank": safe_int(row.get("behavior_rank")),
        "strict_rank": safe_int(row.get("original_strict_rank")),
        "behavior_score": safe_float(row.get("behavior_score")),
        "behavior_confidence": clean_text(row.get("behavior_confidence")),
        "coverage_contribution": safe_float(row.get("coverage_contribution")),
        "cold_start_proxy": clean_text(row.get("cold_start_proxy")),
        "exploration_flag": clean_text(row.get("exploration_flag")),
        "final_policy_score": safe_float(row.get("final_policy_score")),
        "policy_reason": clean_text(row.get("policy_reason")),
        "governed_action": "USE_BEHAVIOR_AWARE_FINAL_SLATE",
        "governed_reason": (
            "Governance selected a mission/behavior route. "
            "Final slate uses Behavior-Aware CORTEX ranking with policy reasons."
        ),
    }


def normalize_strict_row(
    query: str,
    row: Dict[str, str],
    governed_rank: int,
    route: str,
    decision: str,
) -> Dict[str, object]:
    return {
        "query": query,
        "governed_rank": governed_rank,
        "governed_source": "mission_strict_repair_slate",
        "governance_route": route,
        "governance_decision": decision,
        "product_title": clean_text(row.get("product_title")),
        "sub_intent": clean_text(row.get("sub_intent")),
        "sub_intent_role": clean_text(row.get("sub_intent_role")),
        "mission_stage": "",
        "behavior_rank": "",
        "strict_rank": safe_int(row.get("final_strict_rank")),
        "behavior_score": "",
        "behavior_confidence": "",
        "coverage_contribution": "",
        "cold_start_proxy": "",
        "exploration_flag": "",
        "final_policy_score": "",
        "policy_reason": "",
        "governed_action": "USE_STRICT_REPAIRED_SLATE",
        "governed_reason": (
            "Governance selected strict repair. Behavior-aware output was unavailable, "
            "so the runner used the strict repaired slate."
        ),
    }


def build_baseline_fallback_slate(
    query: str,
    route: str,
    decision: str,
    governance_decision: Dict[str, str],
) -> List[Dict[str, object]]:
    query_type = clean_text(governance_decision.get("query_type"))
    reason = clean_text(governance_decision.get("plain_english_reason"))

    return [
        {
            "query": query,
            "governed_rank": 1,
            "governed_source": "governance_baseline_fallback",
            "governance_route": route,
            "governance_decision": decision,
            "product_title": "Baseline route selected by governance agent",
            "sub_intent": "baseline_preservation",
            "sub_intent_role": "baseline",
            "mission_stage": "",
            "behavior_rank": "",
            "strict_rank": "",
            "behavior_score": "",
            "behavior_confidence": "",
            "coverage_contribution": "",
            "cold_start_proxy": "",
            "exploration_flag": "",
            "final_policy_score": "",
            "policy_reason": "BASELINE_PRESERVED_BY_GOVERNANCE",
            "governed_action": "PRESERVE_BASELINE_ROUTE",
            "governed_reason": (
                f"Governance classified this as {query_type}. "
                f"Aggressive mission repair/rerank was blocked. {reason}"
            ),
        }
    ]


def select_final_slate(
    query: str,
    governance_decision: Dict[str, str],
) -> Tuple[List[Dict[str, object]], str]:
    route = clean_text(governance_decision.get("recommended_route"))
    decision = clean_text(governance_decision.get("governance_decision"))

    if route in {ROUTE_REJECT_REPAIR_NARROW_QUERY, ROUTE_BASELINE_ONLY}:
        return (
            build_baseline_fallback_slate(
                query=query,
                route=route,
                decision=decision,
                governance_decision=governance_decision,
            ),
            "baseline_fallback",
        )

    behavior_rows = filter_rows_by_query(read_csv(BEHAVIOR_SLATE_PATH), query)

    if behavior_rows:
        behavior_rows = sorted(
            behavior_rows,
            key=lambda row: safe_int(row.get("behavior_rank"), default=999),
        )

        final_rows = [
            normalize_behavior_row(
                query=query,
                row=row,
                governed_rank=index,
                route=route,
                decision=decision,
            )
            for index, row in enumerate(behavior_rows, start=1)
        ]

        return final_rows, "behavior_aware"

    strict_rows = filter_rows_by_query(read_csv(STRICT_SLATE_PATH), query)

    if strict_rows:
        strict_rows = sorted(
            strict_rows,
            key=lambda row: safe_int(row.get("final_strict_rank"), default=999),
        )

        final_rows = [
            normalize_strict_row(
                query=query,
                row=row,
                governed_rank=index,
                route=route,
                decision=decision,
            )
            for index, row in enumerate(strict_rows, start=1)
        ]

        return final_rows, "strict_repair"

    return (
        build_baseline_fallback_slate(
            query=query,
            route=route,
            decision=decision,
            governance_decision=governance_decision,
        ),
        "fallback_no_slate_available",
    )


def summarize_final_slate(
    query: str,
    final_rows: List[Dict[str, object]],
    final_source: str,
    governance_decision: Dict[str, str],
    governance_summary: Dict[str, str],
) -> Dict[str, object]:
    route = clean_text(governance_decision.get("recommended_route"))
    decision = clean_text(governance_decision.get("governance_decision"))

    unique_sub_intents = {
        lower_text(row.get("sub_intent"))
        for row in final_rows
        if lower_text(row.get("sub_intent"))
    }

    policy_reasons = {
        clean_text(row.get("policy_reason"))
        for row in final_rows
        if clean_text(row.get("policy_reason"))
    }

    cold_start_count = sum(
        1 for row in final_rows
        if lower_text(row.get("cold_start_proxy")) == "true"
    )

    exploration_count = sum(
        1 for row in final_rows
        if lower_text(row.get("exploration_flag")) == "true"
    )

    behavior_ranked_count = sum(
        1 for row in final_rows
        if clean_text(row.get("governed_source")) == "behavior_aware_cortex_slate"
    )

    baseline_preserved = int(final_source.startswith("baseline") or "fallback" in final_source)

    return {
        "query": query,
        "governance_route": route,
        "governance_decision": decision,
        "query_type": clean_text(governance_decision.get("query_type")),
        "final_execution_source": final_source,
        "final_slate_size": len(final_rows),
        "unique_sub_intents": len(unique_sub_intents),
        "unique_policy_reasons": len(policy_reasons),
        "behavior_ranked_items": behavior_ranked_count,
        "cold_start_proxy_items": cold_start_count,
        "exploration_items": exploration_count,
        "baseline_preserved": baseline_preserved,
        "mission_likelihood_score": safe_float(governance_decision.get("mission_likelihood_score")),
        "brand_specificity_score": safe_float(governance_decision.get("brand_specificity_score")),
        "compound_intent_score": safe_float(governance_decision.get("compound_intent_score")),
        "narrow_query_score": safe_float(governance_decision.get("narrow_query_score")),
        "repair_risk_score": safe_float(governance_decision.get("repair_risk_score")),
        "coverage_gap_score": safe_float(governance_decision.get("coverage_gap_score")),
        "behavior_rescue_signal": safe_float(governance_decision.get("behavior_rescue_signal")),
        "critic_need_score": safe_float(governance_decision.get("critic_need_score")),
        "plain_english_reason": build_runner_explanation(
            query=query,
            final_source=final_source,
            governance_decision=governance_decision,
            final_rows=final_rows,
        ),
    }


def build_runner_explanation(
    query: str,
    final_source: str,
    governance_decision: Dict[str, str],
    final_rows: List[Dict[str, object]],
) -> str:
    route = clean_text(governance_decision.get("recommended_route"))
    decision = clean_text(governance_decision.get("governance_decision"))
    query_type = clean_text(governance_decision.get("query_type"))

    if final_source == "baseline_fallback":
        return (
            f"For '{query}', governance classified the query as {query_type} and selected {route}. "
            f"The governed runner preserved baseline-style handling and blocked aggressive mission repair."
        )

    if final_source == "behavior_aware":
        rescued = sum(
            1 for row in final_rows
            if lower_text(row.get("cold_start_proxy")) == "true"
        )
        return (
            f"For '{query}', governance selected {route}. The governed runner executed the behavior-aware "
            f"CORTEX slate with {len(final_rows)} items and {rescued} cold-start proxy/rescue items."
        )

    if final_source == "strict_repair":
        return (
            f"For '{query}', governance selected {route}. The governed runner used the strict repaired slate "
            f"because behavior-aware output was unavailable."
        )

    return (
        f"For '{query}', governance selected {route} with decision {decision}. "
        f"The runner produced the best available governed output from existing CORTEX artifacts."
    )


def build_trace_row(
    query: str,
    governance_ok: bool,
    governance_output: str,
    governance_decision: Dict[str, str],
    final_source: str,
    final_rows: List[Dict[str, object]],
    skip_refresh: bool,
) -> Dict[str, object]:
    return {
        "query": query,
        "governance_ok": governance_ok,
        "skip_refresh": skip_refresh,
        "recommended_route": clean_text(governance_decision.get("recommended_route")),
        "governance_decision": clean_text(governance_decision.get("governance_decision")),
        "query_type": clean_text(governance_decision.get("query_type")),
        "final_execution_source": final_source,
        "final_slate_size": len(final_rows),
        "governance_output_preview": governance_output[:1500],
    }


def run_governed_cortex(
    query: str,
    skip_refresh: bool = False,
) -> Tuple[List[Dict[str, object]], Dict[str, object], Dict[str, object]]:
    ensure_output_dir()

    governance_ok, governance_output = run_governance_agent(
        query=query,
        skip_refresh=skip_refresh,
    )

    if not governance_ok:
        raise RuntimeError(
            f"Governance Agent failed for query '{query}'.\n{governance_output}"
        )

    governance_decision = get_governance_decision(query)
    governance_summary = get_governance_summary(query)

    final_rows, final_source = select_final_slate(
        query=query,
        governance_decision=governance_decision,
    )

    summary = summarize_final_slate(
        query=query,
        final_rows=final_rows,
        final_source=final_source,
        governance_decision=governance_decision,
        governance_summary=governance_summary,
    )

    trace = build_trace_row(
        query=query,
        governance_ok=governance_ok,
        governance_output=governance_output,
        governance_decision=governance_decision,
        final_source=final_source,
        final_rows=final_rows,
        skip_refresh=skip_refresh,
    )

    final_slate_fields = [
        "query",
        "governed_rank",
        "governed_source",
        "governance_route",
        "governance_decision",
        "product_title",
        "sub_intent",
        "sub_intent_role",
        "mission_stage",
        "behavior_rank",
        "strict_rank",
        "behavior_score",
        "behavior_confidence",
        "coverage_contribution",
        "cold_start_proxy",
        "exploration_flag",
        "final_policy_score",
        "policy_reason",
        "governed_action",
        "governed_reason",
    ]

    summary_fields = [
        "query",
        "governance_route",
        "governance_decision",
        "query_type",
        "final_execution_source",
        "final_slate_size",
        "unique_sub_intents",
        "unique_policy_reasons",
        "behavior_ranked_items",
        "cold_start_proxy_items",
        "exploration_items",
        "baseline_preserved",
        "mission_likelihood_score",
        "brand_specificity_score",
        "compound_intent_score",
        "narrow_query_score",
        "repair_risk_score",
        "coverage_gap_score",
        "behavior_rescue_signal",
        "critic_need_score",
        "plain_english_reason",
    ]

    trace_fields = [
        "query",
        "governance_ok",
        "skip_refresh",
        "recommended_route",
        "governance_decision",
        "query_type",
        "final_execution_source",
        "final_slate_size",
        "governance_output_preview",
    ]

    append_or_replace_by_query(
        GOVERNED_FINAL_SLATE_PATH,
        final_rows,
        final_slate_fields,
        query=query,
    )

    append_or_replace_by_query(
        GOVERNED_SUMMARY_PATH,
        [summary],
        summary_fields,
        query=query,
    )

    append_or_replace_by_query(
        GOVERNED_TRACE_PATH,
        [trace],
        trace_fields,
        query=query,
    )

    return final_rows, summary, trace


def print_runner_result(
    query: str,
    final_rows: List[Dict[str, object]],
    summary: Dict[str, object],
    trace: Dict[str, object],
) -> None:
    print("\nGoverned CORTEX Runner Summary")
    print("-" * 100)
    print(f"Query: {query}")
    print(f"Route: {summary.get('governance_route')}")
    print(f"Decision: {summary.get('governance_decision')}")
    print(f"Execution source: {summary.get('final_execution_source')}")
    print(f"Final slate size: {summary.get('final_slate_size')}")
    print(f"Baseline preserved: {summary.get('baseline_preserved')}")
    print(f"Unique sub-intents: {summary.get('unique_sub_intents')}")
    print(f"Cold-start proxy items: {summary.get('cold_start_proxy_items')}")
    print(f"Exploration items: {summary.get('exploration_items')}")

    print("\nExplanation")
    print("-" * 100)
    print(summary.get("plain_english_reason", ""))

    print("\nTop Governed Slate Rows")
    print("-" * 100)

    for row in final_rows[:12]:
        print(
            f"{row.get('governed_rank')}. "
            f"[{row.get('governed_source')}] "
            f"{row.get('sub_intent')} | "
            f"{row.get('policy_reason')} | "
            f"{row.get('product_title')}"
        )

    print("\nFiles written:")
    print(f"- {GOVERNED_FINAL_SLATE_PATH}")
    print(f"- {GOVERNED_SUMMARY_PATH}")
    print(f"- {GOVERNED_TRACE_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="MVP 18 End-to-End Governed CORTEX Runner")
    parser.add_argument("--query", required=True, help="Shopping query to run through governed CORTEX.")
    parser.add_argument(
        "--skip-refresh",
        action="store_true",
        help="Use existing governance / behavior / strict outputs without refreshing dependencies.",
    )

    args = parser.parse_args()
    query = args.query.strip()

    final_rows, summary, trace = run_governed_cortex(
        query=query,
        skip_refresh=args.skip_refresh,
    )

    print_runner_result(query, final_rows, summary, trace)


if __name__ == "__main__":
    main()