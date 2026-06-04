"""
MVP 23C: Calibrated Governed CORTEX Runner

Executes the calibrated governance route as a separate experimental path while
preserving the existing governed_cortex_runner.py and governance behavior.
"""

from __future__ import annotations

import argparse
import csv
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import src.governed_cortex_runner as gcr
from src.calibrated_route_execution_adapter import execute_calibrated_route
from src.governance_calibration_dry_run import (
    calibrate_query,
)
from src.governance_alignment_analyzer import (
    ROUTE_BASELINE_ONLY,
    ROUTE_BEHAVIOR_AWARE_RERANK,
    ROUTE_CRITIC_REVIEW,
    ROUTE_MISSION_REPAIR,
    ROUTE_REJECT_REPAIR_NARROW_QUERY,
    ROUTE_STRICT_REPAIR,
    clean_text,
    safe_float,
    safe_int,
)
from src.query_understanding_agent import (
    QUERY_TYPE_ORDER,
    console_text,
    load_all_queries,
    select_queries,
)


DEFAULT_OUTPUT_DIR = Path("outputs/calibrated_governed_cortex")

RESULT_FIELDS = [
    "query",
    "query_type",
    "recommended_governance_bias",
    "current_governance_route",
    "current_governance_decision",
    "calibrated_governance_route",
    "calibrated_governance_decision",
    "calibration_action",
    "route_changed_flag",
    "current_final_execution_source",
    "calibrated_final_execution_source",
    "calibrated_execution_source",
    "current_final_slate_size",
    "calibrated_final_slate_size",
    "current_unique_sub_intents",
    "calibrated_unique_sub_intents",
    "current_cold_start_proxy_items",
    "calibrated_cold_start_proxy_items",
    "calibrated_success",
    "calibrated_fallback_used",
    "calibrated_fallback_reason",
    "calibrated_adapter_trace",
    "error_message",
    "plain_english_comparison",
]

ROUTE_ORDER = [
    ROUTE_BASELINE_ONLY,
    ROUTE_REJECT_REPAIR_NARROW_QUERY,
    ROUTE_MISSION_REPAIR,
    ROUTE_STRICT_REPAIR,
    ROUTE_BEHAVIOR_AWARE_RERANK,
    ROUTE_CRITIC_REVIEW,
]


def ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "runtime").mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def governance_decision_dict(
    query: str,
    route: str,
    decision: str,
    query_type: str,
    bias: str,
) -> Dict[str, str]:
    return {
        "query": query,
        "recommended_route": route,
        "governance_decision": decision,
        "query_type": query_type,
        "plain_english_reason": (
            f"Calibrated experimental runner route for {query_type} query with "
            f"query-understanding bias {bias}."
        ),
    }


def fallback_slate(
    query: str,
    route: str,
    decision: str,
    query_type: str,
    bias: str,
    source: str,
    action: str,
    reason: str,
) -> Tuple[List[Dict[str, object]], str]:
    if route in {ROUTE_BASELINE_ONLY, ROUTE_REJECT_REPAIR_NARROW_QUERY}:
        rows = gcr.build_baseline_fallback_slate(
            query=query,
            route=route,
            decision=decision,
            governance_decision=governance_decision_dict(query, route, decision, query_type, bias),
        )
        return rows, source

    return (
        [
            {
                "query": query,
                "governed_rank": 1,
                "governed_source": source,
                "governance_route": route,
                "governance_decision": decision,
                "product_title": reason,
                "sub_intent": action,
                "sub_intent_role": "fallback",
                "mission_stage": "",
                "behavior_rank": "",
                "strict_rank": "",
                "behavior_score": "",
                "behavior_confidence": "",
                "coverage_contribution": "",
                "cold_start_proxy": "",
                "exploration_flag": "",
                "final_policy_score": "",
                "policy_reason": action.upper(),
                "governed_action": action.upper(),
                "governed_reason": reason,
            }
        ],
        source,
    )


def strict_slate(query: str, route: str, decision: str) -> Tuple[List[Dict[str, object]], str]:
    strict_rows = gcr.filter_rows_by_query(gcr.read_csv(gcr.STRICT_SLATE_PATH), query)
    if not strict_rows:
        return [], ""

    strict_rows = sorted(
        strict_rows,
        key=lambda row: gcr.safe_int(row.get("final_strict_rank"), default=999),
    )
    return (
        [
            gcr.normalize_strict_row(
                query=query,
                row=row,
                governed_rank=index,
                route=route,
                decision=decision,
            )
            for index, row in enumerate(strict_rows, start=1)
        ],
        "strict_repair",
    )


def behavior_slate(query: str, route: str, decision: str) -> Tuple[List[Dict[str, object]], str]:
    behavior_rows = gcr.filter_rows_by_query(gcr.read_csv(gcr.BEHAVIOR_SLATE_PATH), query)
    if not behavior_rows:
        return [], ""

    behavior_rows = sorted(
        behavior_rows,
        key=lambda row: gcr.safe_int(row.get("behavior_rank"), default=999),
    )
    return (
        [
            gcr.normalize_behavior_row(
                query=query,
                row=row,
                governed_rank=index,
                route=route,
                decision=decision,
            )
            for index, row in enumerate(behavior_rows, start=1)
        ],
        "behavior_aware",
    )


def select_experimental_slate(
    query: str,
    route: str,
    decision: str,
    query_type: str,
    bias: str,
) -> Tuple[List[Dict[str, object]], str, bool]:
    if route in {ROUTE_BASELINE_ONLY, ROUTE_REJECT_REPAIR_NARROW_QUERY}:
        rows, source = fallback_slate(
            query=query,
            route=route,
            decision=decision,
            query_type=query_type,
            bias=bias,
            source="baseline_fallback",
            action="preserve_baseline_route",
            reason="Baseline-style route selected by calibrated governance.",
        )
        return rows, source, True

    if route == ROUTE_CRITIC_REVIEW:
        rows, source = fallback_slate(
            query=query,
            route=route,
            decision=decision,
            query_type=query_type,
            bias=bias,
            source="critic_review_fallback",
            action="send_to_critic_review",
            reason="Calibrated route requests critic review; risky reranking is not executed in this runner.",
        )
        return rows, source, True

    if route == ROUTE_BEHAVIOR_AWARE_RERANK:
        rows, source = behavior_slate(query=query, route=route, decision=decision)
        if rows:
            return rows, source, False
        rows, source = strict_slate(query=query, route=route, decision=decision)
        if rows:
            return rows, "strict_repair_fallback_for_behavior_aware", True

    if route in {ROUTE_MISSION_REPAIR, ROUTE_STRICT_REPAIR}:
        rows, source = strict_slate(query=query, route=route, decision=decision)
        if rows:
            return rows, source, False
        rows, source = behavior_slate(query=query, route=route, decision=decision)
        if rows:
            return rows, "behavior_aware_available_for_mission_route", False

    rows, source = fallback_slate(
        query=query,
        route=route,
        decision=decision,
        query_type=query_type,
        bias=bias,
        source="fallback_no_slate_available",
        action="fallback_no_slate_available",
        reason="No existing strict or behavior-aware slate artifact was available for the calibrated route.",
    )
    return rows, source, True


def slate_metrics(rows: List[Dict[str, object]], source: str) -> Dict[str, object]:
    unique_sub_intents = {
        gcr.lower_text(row.get("sub_intent"))
        for row in rows
        if gcr.lower_text(row.get("sub_intent"))
    }
    cold_start_items = sum(
        1 for row in rows
        if gcr.lower_text(row.get("cold_start_proxy")) == "true"
    )
    return {
        "final_execution_source": source,
        "final_slate_size": len(rows),
        "unique_sub_intents": len(unique_sub_intents),
        "cold_start_proxy_items": cold_start_items,
    }


def execute_query(query: str, output_dir: Path) -> Dict[str, object]:
    try:
        calibration = calibrate_query(query=query, output_dir=output_dir)
        query_type = clean_text(calibration.get("query_type"))
        bias = clean_text(calibration.get("recommended_governance_bias"))
        current_route = clean_text(calibration.get("current_governance_route"))
        current_decision = clean_text(calibration.get("current_governance_decision"))
        calibrated_route = clean_text(calibration.get("calibrated_governance_route"))
        calibrated_decision = clean_text(calibration.get("calibrated_governance_decision"))

        current_rows, current_source, current_fallback = select_experimental_slate(
            query=query,
            route=current_route,
            decision=current_decision,
            query_type=query_type,
            bias=bias,
        )
        adapter_result = execute_calibrated_route(
            query=query,
            calibrated_route=calibrated_route,
            query_understanding={
                "query_type": query_type,
                "recommended_governance_bias": bias,
                "confidence_score": calibration.get("query_understanding_confidence"),
                "risk_score": calibration.get("query_understanding_risk"),
            },
            max_items=12,
        )
        current_metrics = slate_metrics(current_rows, current_source)
        calibrated_rows = adapter_result.get("final_slate", [])
        calibrated_source = clean_text(adapter_result.get("execution_source"))
        calibrated_metrics = {
            "final_execution_source": calibrated_source,
            "final_slate_size": safe_int(adapter_result.get("final_slate_size")),
            "unique_sub_intents": safe_int(adapter_result.get("unique_sub_intents")),
            "cold_start_proxy_items": sum(
                1 for row in calibrated_rows
                if gcr.lower_text(row.get("cold_start_proxy")) == "true"
            ),
        }

        success = bool(calibrated_rows)
        comparison = (
            f"Current route {current_route or 'unknown'} produced "
            f"{current_metrics['final_slate_size']} rows from {current_source or 'unknown'}; "
            f"calibrated route {calibrated_route or 'unknown'} produced "
            f"{calibrated_metrics['final_slate_size']} rows from {calibrated_source or 'unknown'}."
        )

        return {
            "query": clean_text(query),
            "query_type": query_type,
            "recommended_governance_bias": bias,
            "current_governance_route": current_route,
            "current_governance_decision": current_decision,
            "calibrated_governance_route": calibrated_route,
            "calibrated_governance_decision": calibrated_decision,
            "calibration_action": clean_text(calibration.get("calibration_action")),
            "route_changed_flag": safe_int(calibration.get("route_changed_flag")),
            "current_final_execution_source": current_metrics["final_execution_source"],
            "calibrated_final_execution_source": calibrated_metrics["final_execution_source"],
            "calibrated_execution_source": calibrated_source,
            "current_final_slate_size": current_metrics["final_slate_size"],
            "calibrated_final_slate_size": calibrated_metrics["final_slate_size"],
            "current_unique_sub_intents": current_metrics["unique_sub_intents"],
            "calibrated_unique_sub_intents": calibrated_metrics["unique_sub_intents"],
            "current_cold_start_proxy_items": current_metrics["cold_start_proxy_items"],
            "calibrated_cold_start_proxy_items": calibrated_metrics["cold_start_proxy_items"],
            "calibrated_success": int(success),
            "calibrated_fallback_used": int(bool(adapter_result.get("fallback_used"))),
            "calibrated_fallback_reason": clean_text(adapter_result.get("fallback_reason")),
            "calibrated_adapter_trace": clean_text(adapter_result.get("adapter_trace")),
            "error_message": "",
            "plain_english_comparison": comparison,
        }
    except Exception as exc:
        return {
            "query": clean_text(query),
            "query_type": "",
            "recommended_governance_bias": "",
            "current_governance_route": "",
            "current_governance_decision": "",
            "calibrated_governance_route": "",
            "calibrated_governance_decision": "",
            "calibration_action": "execution_failed",
            "route_changed_flag": 0,
            "current_final_execution_source": "",
            "calibrated_final_execution_source": "",
            "calibrated_execution_source": "",
            "current_final_slate_size": 0,
            "calibrated_final_slate_size": 0,
            "current_unique_sub_intents": 0,
            "calibrated_unique_sub_intents": 0,
            "current_cold_start_proxy_items": 0,
            "calibrated_cold_start_proxy_items": 0,
            "calibrated_success": 0,
            "calibrated_fallback_used": 1,
            "calibrated_fallback_reason": "Calibrated execution failed.",
            "calibrated_adapter_trace": "",
            "error_message": str(exc)[:1000],
            "plain_english_comparison": f"Calibrated execution failed for '{query}': {exc}",
        }


def avg(rows: List[Dict[str, object]], field: str) -> float:
    return round(sum(safe_float(row.get(field)) for row in rows) / max(len(rows), 1), 6)


def summarize(rows: List[Dict[str, object]]) -> Dict[str, object]:
    total = len(rows)
    success_rows = [row for row in rows if safe_int(row.get("calibrated_success")) == 1]
    route_changed = sum(safe_int(row.get("route_changed_flag")) for row in rows)
    current_route_counts = Counter(clean_text(row.get("current_governance_route")) for row in success_rows)
    calibrated_route_counts = Counter(clean_text(row.get("calibrated_governance_route")) for row in success_rows)
    current_source_counts = Counter(clean_text(row.get("current_final_execution_source")) for row in success_rows)
    calibrated_source_counts = Counter(clean_text(row.get("calibrated_execution_source")) for row in success_rows)
    adapter_fallback_count = sum(safe_int(row.get("calibrated_fallback_used")) for row in rows)

    return {
        "total_queries": total,
        "success_count": len(success_rows),
        "failure_count": total - len(success_rows),
        "route_changed_count": route_changed,
        "route_changed_rate": round(route_changed / max(total, 1), 6),
        "avg_current_final_slate_size": avg(success_rows, "current_final_slate_size"),
        "avg_calibrated_final_slate_size": avg(success_rows, "calibrated_final_slate_size"),
        "avg_current_unique_sub_intents": avg(success_rows, "current_unique_sub_intents"),
        "avg_calibrated_unique_sub_intents": avg(success_rows, "calibrated_unique_sub_intents"),
        "current_top_route": current_route_counts.most_common(1)[0][0] if current_route_counts else "",
        "calibrated_top_route": calibrated_route_counts.most_common(1)[0][0] if calibrated_route_counts else "",
        "current_top_execution_source": current_source_counts.most_common(1)[0][0] if current_source_counts else "",
        "calibrated_top_execution_source": calibrated_source_counts.most_common(1)[0][0] if calibrated_source_counts else "",
        "fallback_count": adapter_fallback_count,
        "calibrated_adapter_fallback_count": adapter_fallback_count,
        "calibrated_adapter_fallback_rate": round(adapter_fallback_count / max(total, 1), 6),
    }


def group_by_route(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    groups: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[clean_text(row.get("calibrated_governance_route")) or "unknown"].append(row)

    output = []
    for route in ROUTE_ORDER + sorted(set(groups) - set(ROUTE_ORDER)):
        group = groups.get(route, [])
        if not group:
            continue
        output.append(
            {
                "calibrated_governance_route": route,
                "query_count": len(group),
                "query_share": round(len(group) / max(len(rows), 1), 6),
                "avg_calibrated_final_slate_size": avg(group, "calibrated_final_slate_size"),
                "avg_calibrated_unique_sub_intents": avg(group, "calibrated_unique_sub_intents"),
                "fallback_count": sum(safe_int(row.get("calibrated_fallback_used")) for row in group),
                "adapter_fallback_rate": round(
                    sum(safe_int(row.get("calibrated_fallback_used")) for row in group) / max(len(group), 1),
                    6,
                ),
            }
        )
    return output


def route_shift(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    counts = Counter(
        (
            clean_text(row.get("current_governance_route")) or "unknown",
            clean_text(row.get("calibrated_governance_route")) or "unknown",
        )
        for row in rows
    )
    total = len(rows)
    return [
        {
            "current_governance_route": current_route,
            "calibrated_governance_route": calibrated_route,
            "query_count": count,
            "query_share": round(count / max(total, 1), 6),
        }
        for (current_route, calibrated_route), count in sorted(
            counts.items(),
            key=lambda item: (-item[1], item[0][0], item[0][1]),
        )
    ]


def failures(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    return [
        {
            "query": row.get("query"),
            "current_governance_route": row.get("current_governance_route"),
            "calibrated_governance_route": row.get("calibrated_governance_route"),
            "calibrated_final_execution_source": row.get("calibrated_final_execution_source"),
            "calibrated_fallback_used": row.get("calibrated_fallback_used"),
            "calibrated_fallback_reason": row.get("calibrated_fallback_reason"),
            "error_message": row.get("error_message"),
        }
        for row in rows
        if safe_int(row.get("calibrated_success")) == 0 or clean_text(row.get("error_message"))
    ]


def write_outputs(rows: List[Dict[str, object]], output_dir: Path) -> Dict[str, object]:
    ensure_output_dir(output_dir)
    summary = summarize(rows)
    by_route = group_by_route(rows)
    shifts = route_shift(rows)
    failure_rows = failures(rows)

    write_csv(output_dir / "calibrated_governed_results.csv", rows, RESULT_FIELDS)
    write_csv(output_dir / "calibrated_governed_summary.csv", [summary], list(summary.keys()))
    write_csv(
        output_dir / "calibrated_governed_by_route.csv",
        by_route,
        [
            "calibrated_governance_route",
            "query_count",
            "query_share",
            "avg_calibrated_final_slate_size",
            "avg_calibrated_unique_sub_intents",
            "fallback_count",
            "adapter_fallback_rate",
        ],
    )
    write_csv(
        output_dir / "calibrated_governed_route_shift.csv",
        shifts,
        ["current_governance_route", "calibrated_governance_route", "query_count", "query_share"],
    )
    write_csv(
        output_dir / "calibrated_governed_failures.csv",
        failure_rows,
        [
            "query",
            "current_governance_route",
            "calibrated_governance_route",
            "calibrated_final_execution_source",
            "calibrated_fallback_used",
            "calibrated_fallback_reason",
            "error_message",
        ],
    )
    return summary


def print_summary(summary: Dict[str, object]) -> None:
    print("\nSummary")
    print("-" * 100)
    for key, value in summary.items():
        print(f"{key}: {value}")


def run_single_query(query: str, output_dir: Path) -> None:
    print("\nMVP 23C Calibrated Governed CORTEX Runner")
    print("-" * 100)
    print(f"query: {console_text(query)}")

    row = execute_query(query=query, output_dir=output_dir)
    summary = write_outputs([row], output_dir=output_dir)

    print("\nCalibrated governed result")
    print("-" * 100)
    for key in RESULT_FIELDS:
        print(f"{key}: {console_text(row.get(key))}")
    print_summary(summary)
    print("\nFiles written under", output_dir)


def run_batch(sample_size: int, query_mode: str, start_index: int, output_dir: Path) -> None:
    all_queries, source = load_all_queries(query_mode)
    selected = select_queries(
        queries=all_queries,
        query_mode=query_mode,
        sample_size=sample_size,
        start_index=start_index,
    )

    print("\nMVP 23C Calibrated Governed CORTEX Runner")
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
        row = execute_query(query=query, output_dir=output_dir)
        rows.append(row)
        if index <= 10 or index % 100 == 0 or index == len(selected):
            print(
                f"[{index}/{len(selected)}] {console_text(query)} -> "
                f"{row['current_governance_route']} => {row['calibrated_governance_route']} "
                f"({row['current_final_execution_source']} => {row['calibrated_execution_source']})"
            )

    summary = write_outputs(rows, output_dir=output_dir)
    elapsed = time.perf_counter() - start

    print_summary(summary)
    print(f"runtime_seconds: {elapsed:.4f}")
    print("\nFiles written under", output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MVP 23C Calibrated Governed CORTEX Runner")
    parser.add_argument("--query", default="", help="Single query to run.")
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
