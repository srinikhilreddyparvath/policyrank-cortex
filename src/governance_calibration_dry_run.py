"""
MVP 23B: Governance Calibration Dry Run

Proposes calibrated governance routes using Query Understanding signals and
current CORTEX governance outcomes without changing live governance behavior.
"""

from __future__ import annotations

import argparse
import csv
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

from src.governance_alignment_analyzer import (
    ROUTE_BASELINE_ONLY,
    ROUTE_BEHAVIOR_AWARE_RERANK,
    ROUTE_CRITIC_REVIEW,
    ROUTE_MISSION_REPAIR,
    ROUTE_REJECT_REPAIR_NARROW_QUERY,
    ROUTE_STRICT_REPAIR,
    analyze_query as analyze_alignment_query,
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


DEFAULT_OUTPUT_DIR = Path("outputs/governance_calibration")

RESULT_FIELDS = [
    "query",
    "query_type",
    "recommended_governance_bias",
    "query_understanding_confidence",
    "query_understanding_risk",
    "current_governance_route",
    "current_governance_decision",
    "calibrated_governance_route",
    "calibrated_governance_decision",
    "calibration_action",
    "calibration_strength",
    "calibration_reason",
    "route_changed_flag",
    "conservative_relaxation_flag",
    "strict_guardrail_upgrade_flag",
    "behavior_aware_upgrade_flag",
    "mission_repair_upgrade_flag",
    "reject_repair_confirmation_flag",
    "preserve_baseline_confirmation_flag",
]

ACTION_ORDER = [
    "confirm_preserve_baseline",
    "confirm_reject_repair",
    "upgrade_to_mission_repair",
    "upgrade_to_behavior_aware",
    "upgrade_to_strict_guardrails",
    "send_to_critic",
    "keep_current_route",
    "governance_failed",
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


def calibration_strength(confidence: float, risk: float) -> str:
    if confidence >= 0.80 and risk <= 0.25:
        return "high"
    if confidence >= 0.65 and risk <= 0.40:
        return "medium"
    return "low"


def decision_for_route(route: str, current_decision: str) -> str:
    if route == ROUTE_BASELINE_ONLY:
        return "PRESERVE_BASELINE"
    if route == ROUTE_BEHAVIOR_AWARE_RERANK:
        return "RUN_MISSION_REPAIR_AND_BEHAVIOR_AWARE"
    if route == ROUTE_MISSION_REPAIR:
        return "RUN_MISSION_REPAIR"
    if route == ROUTE_STRICT_REPAIR:
        return "RUN_STRICT_REPAIR"
    if route == ROUTE_REJECT_REPAIR_NARROW_QUERY:
        return "REJECT_REPAIR_NARROW_QUERY"
    if route == ROUTE_CRITIC_REVIEW:
        return "SEND_TO_CRITIC_REVIEW"
    return current_decision


def apply_calibration(
    query: str,
    query_type: str,
    bias: str,
    confidence: float,
    risk: float,
    current_route: str,
    current_decision: str,
) -> Dict[str, object]:
    calibrated_route = current_route
    action = "keep_current_route"

    if (
        current_route == ROUTE_BASELINE_ONLY
        and bias == "allow_behavior_aware"
        and confidence >= 0.70
        and risk <= 0.35
    ):
        calibrated_route = ROUTE_BEHAVIOR_AWARE_RERANK
        action = "upgrade_to_behavior_aware"
    elif (
        current_route == ROUTE_BASELINE_ONLY
        and bias == "allow_mission_repair"
        and confidence >= 0.65
        and risk <= 0.40
    ):
        calibrated_route = ROUTE_MISSION_REPAIR
        action = "upgrade_to_mission_repair"
    elif bias == "strict_guardrails":
        if current_route != ROUTE_REJECT_REPAIR_NARROW_QUERY:
            calibrated_route = ROUTE_STRICT_REPAIR
        action = "upgrade_to_strict_guardrails"
    elif bias == "reject_aggressive_repair":
        calibrated_route = ROUTE_REJECT_REPAIR_NARROW_QUERY
        action = "confirm_reject_repair"
    elif bias == "preserve_baseline":
        calibrated_route = ROUTE_BASELINE_ONLY
        action = "confirm_preserve_baseline"
    elif bias == "send_to_critic" and risk >= 0.50:
        calibrated_route = ROUTE_CRITIC_REVIEW
        action = "send_to_critic"

    calibrated_decision = decision_for_route(calibrated_route, current_decision)
    strength = calibration_strength(confidence, risk)
    route_changed = int(clean_text(current_route) != clean_text(calibrated_route))

    reason = (
        f"Query '{query}' is {query_type} with bias {bias}, confidence {confidence:.2f}, "
        f"and risk {risk:.2f}. Current route {current_route or 'unknown'} would dry-run "
        f"as {calibrated_route or 'unknown'} via {action}."
    )

    return {
        "calibrated_governance_route": calibrated_route,
        "calibrated_governance_decision": calibrated_decision,
        "calibration_action": action,
        "calibration_strength": strength,
        "calibration_reason": reason,
        "route_changed_flag": route_changed,
        "conservative_relaxation_flag": int(
            action in {"upgrade_to_behavior_aware", "upgrade_to_mission_repair"}
            and current_route == ROUTE_BASELINE_ONLY
        ),
        "strict_guardrail_upgrade_flag": int(action == "upgrade_to_strict_guardrails"),
        "behavior_aware_upgrade_flag": int(action == "upgrade_to_behavior_aware"),
        "mission_repair_upgrade_flag": int(action == "upgrade_to_mission_repair"),
        "reject_repair_confirmation_flag": int(action == "confirm_reject_repair"),
        "preserve_baseline_confirmation_flag": int(action == "confirm_preserve_baseline"),
    }


def calibrate_query(query: str, output_dir: Path) -> Dict[str, object]:
    alignment = analyze_alignment_query(query=query, output_dir=output_dir)
    current_route = clean_text(alignment.get("actual_governance_route"))
    current_decision = clean_text(alignment.get("actual_governance_decision"))
    bias = clean_text(alignment.get("recommended_governance_bias"))
    confidence = safe_float(alignment.get("confidence_score"))
    risk = safe_float(alignment.get("risk_score"))

    if not current_route:
        calibration = {
            "calibrated_governance_route": "",
            "calibrated_governance_decision": "",
            "calibration_action": "governance_failed",
            "calibration_strength": calibration_strength(confidence, risk),
            "calibration_reason": clean_text(alignment.get("plain_english_alignment_reason")),
            "route_changed_flag": 0,
            "conservative_relaxation_flag": 0,
            "strict_guardrail_upgrade_flag": 0,
            "behavior_aware_upgrade_flag": 0,
            "mission_repair_upgrade_flag": 0,
            "reject_repair_confirmation_flag": 0,
            "preserve_baseline_confirmation_flag": 0,
        }
    else:
        calibration = apply_calibration(
            query=clean_text(alignment.get("query")),
            query_type=clean_text(alignment.get("query_type")),
            bias=bias,
            confidence=confidence,
            risk=risk,
            current_route=current_route,
            current_decision=current_decision,
        )

    return {
        "query": clean_text(alignment.get("query")),
        "query_type": clean_text(alignment.get("query_type")),
        "recommended_governance_bias": bias,
        "query_understanding_confidence": confidence,
        "query_understanding_risk": risk,
        "current_governance_route": current_route,
        "current_governance_decision": current_decision,
        **calibration,
    }


def summarize(rows: List[Dict[str, object]]) -> Dict[str, object]:
    total = len(rows)
    success_rows = [row for row in rows if clean_text(row.get("current_governance_route"))]
    route_changed_count = sum(safe_int(row.get("route_changed_flag")) for row in success_rows)
    current_counts = Counter(clean_text(row.get("current_governance_route")) for row in success_rows)
    calibrated_counts = Counter(clean_text(row.get("calibrated_governance_route")) for row in success_rows)
    action_counts = Counter(clean_text(row.get("calibration_action")) for row in rows)

    return {
        "total_queries": total,
        "success_count": len(success_rows),
        "failure_count": total - len(success_rows),
        "route_changed_count": route_changed_count,
        "route_changed_rate": round(route_changed_count / max(len(success_rows), 1), 6),
        "conservative_relaxation_count": sum(safe_int(row.get("conservative_relaxation_flag")) for row in rows),
        "behavior_aware_upgrade_count": sum(safe_int(row.get("behavior_aware_upgrade_flag")) for row in rows),
        "mission_repair_upgrade_count": sum(safe_int(row.get("mission_repair_upgrade_flag")) for row in rows),
        "strict_guardrail_upgrade_count": sum(safe_int(row.get("strict_guardrail_upgrade_flag")) for row in rows),
        "reject_repair_confirmation_count": sum(safe_int(row.get("reject_repair_confirmation_flag")) for row in rows),
        "preserve_baseline_confirmation_count": sum(safe_int(row.get("preserve_baseline_confirmation_flag")) for row in rows),
        "top_current_route": current_counts.most_common(1)[0][0] if current_counts else "",
        "top_calibrated_route": calibrated_counts.most_common(1)[0][0] if calibrated_counts else "",
        "top_calibration_action": action_counts.most_common(1)[0][0] if action_counts else "",
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
        changed = sum(safe_int(row.get("route_changed_flag")) for row in group)
        output.append(
            {
                field: value,
                "query_count": len(group),
                "query_share": round(len(group) / max(len(rows), 1), 6),
                "route_changed_count": changed,
                "route_changed_rate": round(changed / max(len(group), 1), 6),
                "conservative_relaxation_count": sum(safe_int(row.get("conservative_relaxation_flag")) for row in group),
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


def write_outputs(rows: List[Dict[str, object]], output_dir: Path) -> Dict[str, object]:
    ensure_output_dir(output_dir)
    summary = summarize(rows)
    by_action = group_by_field(rows, "calibration_action", ACTION_ORDER)
    by_query_type = group_by_field(rows, "query_type", QUERY_TYPE_ORDER)
    shifts = route_shift(rows)

    write_csv(output_dir / "governance_calibration_results.csv", rows, RESULT_FIELDS)
    write_csv(output_dir / "governance_calibration_summary.csv", [summary], list(summary.keys()))
    write_csv(
        output_dir / "governance_calibration_by_action.csv",
        by_action,
        [
            "calibration_action",
            "query_count",
            "query_share",
            "route_changed_count",
            "route_changed_rate",
            "conservative_relaxation_count",
        ],
    )
    write_csv(
        output_dir / "governance_calibration_by_query_type.csv",
        by_query_type,
        [
            "query_type",
            "query_count",
            "query_share",
            "route_changed_count",
            "route_changed_rate",
            "conservative_relaxation_count",
        ],
    )
    write_csv(
        output_dir / "governance_calibration_route_shift.csv",
        shifts,
        ["current_governance_route", "calibrated_governance_route", "query_count", "query_share"],
    )
    return summary


def print_summary(summary: Dict[str, object]) -> None:
    print("\nSummary")
    print("-" * 100)
    for key, value in summary.items():
        print(f"{key}: {value}")


def run_single_query(query: str, output_dir: Path) -> None:
    print("\nMVP 23B Governance Calibration Dry Run")
    print("-" * 100)
    print(f"query: {console_text(query)}")

    row = calibrate_query(query=query, output_dir=output_dir)
    summary = write_outputs([row], output_dir=output_dir)

    print("\nCalibration result")
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

    print("\nMVP 23B Governance Calibration Dry Run")
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
        row = calibrate_query(query=query, output_dir=output_dir)
        rows.append(row)
        if index <= 10 or index % 100 == 0 or index == len(selected):
            print(
                f"[{index}/{len(selected)}] {console_text(query)} -> "
                f"{row['current_governance_route']} => {row['calibrated_governance_route']} "
                f"({row['calibration_action']})"
            )

    summary = write_outputs(rows, output_dir=output_dir)
    elapsed = time.perf_counter() - start

    print_summary(summary)
    print(f"runtime_seconds: {elapsed:.4f}")
    print("\nFiles written under", output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MVP 23B Governance Calibration Dry Run")
    parser.add_argument("--query", default="", help="Single query to dry-run.")
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
