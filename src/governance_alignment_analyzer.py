"""
MVP 23A: Governance Alignment Analyzer

Compares standalone Query Understanding Agent recommendations with current
CORTEX governance routes without changing governance behavior.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from src.query_understanding_agent import (
    BIAS_ORDER,
    QUERY_TYPE_ORDER,
    console_text,
    load_all_queries,
    select_queries,
    understand_query,
)


DEFAULT_OUTPUT_DIR = Path("outputs/governance_alignment")

RESULT_FIELDS = [
    "query",
    "query_type",
    "intent_bucket",
    "recommended_governance_bias",
    "confidence_score",
    "risk_score",
    "actual_governance_route",
    "actual_governance_decision",
    "final_execution_source",
    "baseline_preserved",
    "alignment_status",
    "alignment_score",
    "over_conservative_flag",
    "over_aggressive_flag",
    "missed_mission_opportunity_flag",
    "missed_behavior_aware_opportunity_flag",
    "strict_guardrail_needed_but_not_used_flag",
    "plain_english_alignment_reason",
]

ROUTE_BASELINE_ONLY = "BASELINE_ONLY"
ROUTE_MISSION_REPAIR = "MISSION_REPAIR"
ROUTE_STRICT_REPAIR = "STRICT_REPAIR"
ROUTE_BEHAVIOR_AWARE_RERANK = "BEHAVIOR_AWARE_RERANK"
ROUTE_CRITIC_REVIEW = "CRITIC_REVIEW"
ROUTE_REJECT_REPAIR_NARROW_QUERY = "REJECT_REPAIR_NARROW_QUERY"

AGGRESSIVE_ROUTES = {
    ROUTE_MISSION_REPAIR,
    ROUTE_STRICT_REPAIR,
    ROUTE_BEHAVIOR_AWARE_RERANK,
}


def clean_text(value: object) -> str:
    return str(value or "").strip()


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


def ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "runtime").mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def configure_governance_paths(output_dir: Path) -> Tuple[object, object]:
    import src.cortex_governance_agent as cga
    import src.governed_cortex_runner as gcr

    runtime_dir = output_dir / "runtime"
    decisions = runtime_dir / "cortex_governance_decisions.csv"
    summary = runtime_dir / "cortex_governance_summary.csv"
    trace = runtime_dir / "cortex_governance_trace.csv"

    cga.GOVERNANCE_DECISIONS_PATH = decisions
    cga.GOVERNANCE_SUMMARY_PATH = summary
    cga.GOVERNANCE_TRACE_PATH = trace

    gcr.GOVERNANCE_DECISIONS_PATH = decisions
    gcr.GOVERNANCE_SUMMARY_PATH = summary
    gcr.GOVERNANCE_TRACE_PATH = trace

    return cga, gcr


def run_governance_imported(query: str, output_dir: Path) -> Dict[str, object]:
    cga, gcr = configure_governance_paths(output_dir)
    _signals, decision = cga.run_governance(query=query, skip_refresh=True)
    decision_row = gcr.get_governance_decision(query)
    governance_summary = gcr.get_governance_summary(query)

    final_rows, final_source = gcr.select_final_slate(
        query=query,
        governance_decision=decision_row,
    )
    runner_summary = gcr.summarize_final_slate(
        query=query,
        final_rows=final_rows,
        final_source=final_source,
        governance_decision=decision_row,
        governance_summary=governance_summary,
    )

    return {
        "actual_governance_route": clean_text(decision.recommended_route),
        "actual_governance_decision": clean_text(decision.governance_decision),
        "final_execution_source": clean_text(runner_summary.get("final_execution_source")),
        "baseline_preserved": safe_int(runner_summary.get("baseline_preserved")),
    }


def run_governance_subprocess(query: str) -> Dict[str, object]:
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "src.cortex_governance_agent",
        "--query",
        query,
        "--skip-refresh",
    ]
    result = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if result.returncode != 0:
        output = (result.stdout or "") + "\n" + (result.stderr or "")
        raise RuntimeError(output[:1000])

    output = result.stdout or ""
    route_match = re.search(r"Recommended route:\s*(.+)", output)
    decision_match = re.search(r"Governance decision:\s*(.+)", output)

    return {
        "actual_governance_route": clean_text(route_match.group(1)) if route_match else "",
        "actual_governance_decision": clean_text(decision_match.group(1)) if decision_match else "",
        "final_execution_source": "",
        "baseline_preserved": 0,
    }


def get_actual_governance(query: str, output_dir: Path) -> Dict[str, object]:
    imported_error = ""
    try:
        return run_governance_imported(query=query, output_dir=output_dir)
    except Exception as exc:
        imported_error = str(exc)

    try:
        actual = run_governance_subprocess(query=query)
        if clean_text(actual.get("actual_governance_route")):
            return actual
    except Exception as exc:
        raise RuntimeError(f"Imported governance failed: {imported_error}; subprocess failed: {exc}") from exc

    raise RuntimeError(f"Imported governance failed: {imported_error}; subprocess did not return a route.")


def preferred_routes_for_bias(bias: str, risk_score: float) -> set[str]:
    if bias == "preserve_baseline":
        return {ROUTE_BASELINE_ONLY, ROUTE_REJECT_REPAIR_NARROW_QUERY}
    if bias == "reject_aggressive_repair":
        return {ROUTE_REJECT_REPAIR_NARROW_QUERY, ROUTE_BASELINE_ONLY}
    if bias == "allow_mission_repair":
        return {ROUTE_MISSION_REPAIR, ROUTE_STRICT_REPAIR}
    if bias == "allow_behavior_aware":
        return {ROUTE_BEHAVIOR_AWARE_RERANK, ROUTE_MISSION_REPAIR, ROUTE_STRICT_REPAIR}
    if bias == "strict_guardrails":
        return {ROUTE_STRICT_REPAIR, ROUTE_REJECT_REPAIR_NARROW_QUERY}
    if bias == "send_to_critic":
        routes = {ROUTE_CRITIC_REVIEW, ROUTE_STRICT_REPAIR}
        if risk_score >= 0.45:
            routes.add(ROUTE_BASELINE_ONLY)
        return routes
    return set()


def analyze_alignment(
    query: str,
    bias: str,
    query_type: str,
    risk_score: float,
    actual_route: str,
) -> Dict[str, object]:
    preferred = preferred_routes_for_bias(bias, risk_score)

    over_conservative = bias in {"allow_mission_repair", "allow_behavior_aware"} and actual_route == ROUTE_BASELINE_ONLY
    over_aggressive = bias in {"preserve_baseline", "reject_aggressive_repair"} and actual_route in AGGRESSIVE_ROUTES
    missed_mission = bias == "allow_mission_repair" and actual_route == ROUTE_BASELINE_ONLY
    missed_behavior = bias == "allow_behavior_aware" and actual_route == ROUTE_BASELINE_ONLY
    strict_needed_not_used = bias == "strict_guardrails" and actual_route not in {
        ROUTE_STRICT_REPAIR,
        ROUTE_REJECT_REPAIR_NARROW_QUERY,
    }

    if actual_route in preferred:
        status = "aligned"
        score = 1.0
    elif over_conservative:
        status = "over_conservative"
        score = 0.25
    elif over_aggressive:
        status = "over_aggressive"
        score = 0.20
    elif strict_needed_not_used:
        status = "strict_guardrail_gap"
        score = 0.35
    elif bias == "send_to_critic" and actual_route == ROUTE_BASELINE_ONLY:
        status = "partial_alignment"
        score = 0.60
    else:
        status = "misaligned"
        score = 0.0

    reason = (
        f"Query understanding classified '{query}' as {query_type} with bias {bias}. "
        f"Current governance selected {actual_route or 'unknown'}. Alignment status: {status}."
    )

    return {
        "alignment_status": status,
        "alignment_score": score,
        "over_conservative_flag": int(over_conservative),
        "over_aggressive_flag": int(over_aggressive),
        "missed_mission_opportunity_flag": int(missed_mission),
        "missed_behavior_aware_opportunity_flag": int(missed_behavior),
        "strict_guardrail_needed_but_not_used_flag": int(strict_needed_not_used),
        "plain_english_alignment_reason": reason,
    }


def analyze_query(query: str, output_dir: Path) -> Dict[str, object]:
    understanding = understand_query(query)
    try:
        actual = get_actual_governance(query=query, output_dir=output_dir)
    except Exception as exc:
        return {
            "query": understanding.query,
            "query_type": understanding.query_type,
            "intent_bucket": understanding.intent_bucket,
            "recommended_governance_bias": understanding.recommended_governance_bias,
            "confidence_score": understanding.confidence_score,
            "risk_score": understanding.risk_score,
            "actual_governance_route": "",
            "actual_governance_decision": "",
            "final_execution_source": "",
            "baseline_preserved": 0,
            "alignment_status": "governance_failed",
            "alignment_score": 0.0,
            "over_conservative_flag": 0,
            "over_aggressive_flag": 0,
            "missed_mission_opportunity_flag": 0,
            "missed_behavior_aware_opportunity_flag": 0,
            "strict_guardrail_needed_but_not_used_flag": 0,
            "plain_english_alignment_reason": f"Governance evaluation failed for '{query}': {exc}",
        }

    actual_route = clean_text(actual.get("actual_governance_route"))

    alignment = analyze_alignment(
        query=query,
        bias=understanding.recommended_governance_bias,
        query_type=understanding.query_type,
        risk_score=float(understanding.risk_score),
        actual_route=actual_route,
    )

    return {
        "query": understanding.query,
        "query_type": understanding.query_type,
        "intent_bucket": understanding.intent_bucket,
        "recommended_governance_bias": understanding.recommended_governance_bias,
        "confidence_score": understanding.confidence_score,
        "risk_score": understanding.risk_score,
        "actual_governance_route": actual_route,
        "actual_governance_decision": clean_text(actual.get("actual_governance_decision")),
        "final_execution_source": clean_text(actual.get("final_execution_source")),
        "baseline_preserved": safe_int(actual.get("baseline_preserved")),
        **alignment,
    }


def summarize(rows: List[Dict[str, object]]) -> Dict[str, object]:
    total = len(rows)
    success_rows = [row for row in rows if clean_text(row.get("actual_governance_route"))]
    failure_count = total - len(success_rows)
    aligned_count = sum(1 for row in success_rows if clean_text(row.get("alignment_status")) == "aligned")
    avg_alignment = sum(safe_float(row.get("alignment_score")) for row in success_rows) / max(len(success_rows), 1)
    route_counts = Counter(clean_text(row.get("actual_governance_route")) for row in success_rows)
    bias_counts = Counter(clean_text(row.get("recommended_governance_bias")) for row in rows)

    return {
        "total_queries": total,
        "success_count": len(success_rows),
        "failure_count": failure_count,
        "alignment_rate": round(aligned_count / max(len(success_rows), 1), 6),
        "avg_alignment_score": round(avg_alignment, 6),
        "over_conservative_count": sum(safe_int(row.get("over_conservative_flag")) for row in rows),
        "over_aggressive_count": sum(safe_int(row.get("over_aggressive_flag")) for row in rows),
        "missed_mission_opportunity_count": sum(safe_int(row.get("missed_mission_opportunity_flag")) for row in rows),
        "missed_behavior_aware_opportunity_count": sum(safe_int(row.get("missed_behavior_aware_opportunity_flag")) for row in rows),
        "strict_guardrail_needed_but_not_used_count": sum(safe_int(row.get("strict_guardrail_needed_but_not_used_flag")) for row in rows),
        "top_actual_governance_route": route_counts.most_common(1)[0][0] if route_counts else "",
        "top_recommended_governance_bias": bias_counts.most_common(1)[0][0] if bias_counts else "",
    }


def group_by_field(rows: List[Dict[str, object]], field: str, ordered_values: List[str]) -> List[Dict[str, object]]:
    groups: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[clean_text(row.get(field)) or "unknown"].append(row)

    output = []
    for value in ordered_values + sorted(set(groups) - set(ordered_values)):
        group = groups.get(value, [])
        if not group:
            continue
        output.append(
            {
                field: value,
                "query_count": len(group),
                "alignment_rate": round(
                    sum(1 for row in group if clean_text(row.get("alignment_status")) == "aligned") / max(len(group), 1),
                    6,
                ),
                "avg_alignment_score": round(
                    sum(safe_float(row.get("alignment_score")) for row in group) / max(len(group), 1),
                    6,
                ),
                "over_conservative_count": sum(safe_int(row.get("over_conservative_flag")) for row in group),
                "over_aggressive_count": sum(safe_int(row.get("over_aggressive_flag")) for row in group),
            }
        )
    return output


def flag_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    flags = [
        "over_conservative_flag",
        "over_aggressive_flag",
        "missed_mission_opportunity_flag",
        "missed_behavior_aware_opportunity_flag",
        "strict_guardrail_needed_but_not_used_flag",
    ]
    output = []
    for flag in flags:
        flagged = [row for row in rows if safe_int(row.get(flag)) == 1]
        output.append(
            {
                "flag": flag,
                "query_count": len(flagged),
                "query_share": round(len(flagged) / max(len(rows), 1), 6),
                "example_queries": " | ".join(clean_text(row.get("query")) for row in flagged[:5]),
            }
        )
    return output


def write_outputs(rows: List[Dict[str, object]], output_dir: Path) -> Dict[str, object]:
    ensure_output_dir(output_dir)
    summary = summarize(rows)
    by_query_type = group_by_field(rows, "query_type", QUERY_TYPE_ORDER)
    by_bias = group_by_field(rows, "recommended_governance_bias", BIAS_ORDER)
    flags = flag_rows(rows)

    write_csv(output_dir / "governance_alignment_results.csv", rows, RESULT_FIELDS)
    write_csv(output_dir / "governance_alignment_summary.csv", [summary], list(summary.keys()))
    write_csv(
        output_dir / "governance_alignment_by_query_type.csv",
        by_query_type,
        ["query_type", "query_count", "alignment_rate", "avg_alignment_score", "over_conservative_count", "over_aggressive_count"],
    )
    write_csv(
        output_dir / "governance_alignment_by_bias.csv",
        by_bias,
        [
            "recommended_governance_bias",
            "query_count",
            "alignment_rate",
            "avg_alignment_score",
            "over_conservative_count",
            "over_aggressive_count",
        ],
    )
    write_csv(output_dir / "governance_alignment_flags.csv", flags, ["flag", "query_count", "query_share", "example_queries"])
    return summary


def run_single_query(query: str, output_dir: Path) -> None:
    print("\nMVP 23A Governance Alignment Analyzer")
    print("-" * 100)
    print(f"query: {console_text(query)}")

    row = analyze_query(query=query, output_dir=output_dir)
    summary = write_outputs([row], output_dir=output_dir)

    print("\nAlignment result")
    print("-" * 100)
    for key in RESULT_FIELDS:
        print(f"{key}: {console_text(row.get(key))}")

    print("\nSummary")
    print("-" * 100)
    for key, value in summary.items():
        print(f"{key}: {value}")


def run_batch(sample_size: int, query_mode: str, start_index: int, output_dir: Path) -> None:
    all_queries, source = load_all_queries(query_mode)
    selected = select_queries(
        queries=all_queries,
        query_mode=query_mode,
        sample_size=sample_size,
        start_index=start_index,
    )

    print("\nMVP 23A Governance Alignment Analyzer")
    print("-" * 100)
    print(f"query_mode: {query_mode}")
    print(f"source: {source}")
    print(f"total_available_unique_queries: {len(all_queries)}")
    print(f"start_index: {start_index}")
    print(f"sample_size: {sample_size}")
    print(f"selected_query_count: {len(selected)}")
    print(f"output_dir: {output_dir}")

    rows = []
    start = time.perf_counter()
    for index, (_query_index, query) in enumerate(selected, start=1):
        row = analyze_query(query=query, output_dir=output_dir)
        rows.append(row)
        if index <= 10 or index % 100 == 0 or index == len(selected):
            print(
                f"[{index}/{len(selected)}] {console_text(query)} -> "
                f"{row['recommended_governance_bias']} vs {row['actual_governance_route']} "
                f"({row['alignment_status']})"
            )

    summary = write_outputs(rows, output_dir=output_dir)
    elapsed = time.perf_counter() - start

    print("\nSummary")
    print("-" * 100)
    for key, value in summary.items():
        print(f"{key}: {value}")
    print(f"runtime_seconds: {elapsed:.4f}")

    print("\nFiles written under", output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MVP 23A Governance Alignment Analyzer")
    parser.add_argument("--query", default="", help="Single query to analyze.")
    parser.add_argument("--sample-size", type=int, default=100, help="Batch sample size.")
    parser.add_argument(
        "--query-mode",
        choices=["smoke", "esci", "stratified_esci"],
        default="smoke",
        help="Batch query mode.",
    )
    parser.add_argument("--start-index", type=int, default=0, help="Start offset for query selection.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)

    if args.query:
        run_single_query(query=args.query, output_dir=output_dir)
        return

    if args.sample_size <= 0:
        raise ValueError("--sample-size must be positive.")
    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative.")

    run_batch(
        sample_size=args.sample_size,
        query_mode=args.query_mode,
        start_index=args.start_index,
        output_dir=output_dir,
    )


if __name__ == "__main__":
    main()
