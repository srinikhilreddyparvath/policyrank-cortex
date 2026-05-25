"""
MVP 19.3: Cost vs Value Governance Analyzer

This module estimates whether CORTEX's governed AI routing is worth the infrastructure cost
at production scale.

It does NOT claim real revenue lift.
It uses scenario-based configurable assumptions, based on scalable governed evaluation outputs.

Inputs:
- outputs/scalable_governed_eval.csv
- outputs/scalable_governed_eval_summary.csv
- outputs/scalable_governed_eval_by_route.csv
- outputs/scalable_governed_eval_by_execution_source.csv

Outputs:
- outputs/cost_value_governance_summary.csv
- outputs/cost_value_governance_by_route.csv
- outputs/cost_value_governance_scenarios.csv

Core idea:
Governance creates cost control because not every query needs expensive agentic processing.

Example:
- Narrow/product-specific query -> baseline fallback -> low cost
- Broad mission query -> behavior-aware CORTEX -> higher cost but more value opportunity
- Risky query -> critic/strict route -> higher cost, only used selectively

This analyzer estimates:
- daily/monthly AI infrastructure cost
- route-level cost contribution
- estimated value opportunity
- break-even value per query
- net value under conservative/base/optimistic scenarios
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


OUTPUT_DIR = Path("outputs")

SCALABLE_EVAL_PATH = OUTPUT_DIR / "scalable_governed_eval.csv"
SCALABLE_SUMMARY_PATH = OUTPUT_DIR / "scalable_governed_eval_summary.csv"
SCALABLE_BY_ROUTE_PATH = OUTPUT_DIR / "scalable_governed_eval_by_route.csv"
SCALABLE_BY_EXECUTION_SOURCE_PATH = OUTPUT_DIR / "scalable_governed_eval_by_execution_source.csv"

COST_VALUE_SUMMARY_PATH = OUTPUT_DIR / "cost_value_governance_summary.csv"
COST_VALUE_BY_ROUTE_PATH = OUTPUT_DIR / "cost_value_governance_by_route.csv"
COST_VALUE_SCENARIOS_PATH = OUTPUT_DIR / "cost_value_governance_scenarios.csv"


# Cost assumptions are per 1,000 queries routed through each execution path.
# These are scenario placeholders, not real vendor billing claims.
DEFAULT_COST_PER_1K_BY_EXECUTION_SOURCE = {
    "behavior_aware": 0.85,
    "strict_repair": 0.55,
    "baseline_fallback": 0.08,
    "fallback_no_slate_available": 0.05,
    "unknown": 0.10,
    "": 0.10,
}


# Route-level value assumptions are incremental value opportunity per successfully governed query.
# These are configurable scenario assumptions, not measured production revenue.
BASE_VALUE_PER_QUERY_BY_ROUTE = {
    "BEHAVIOR_AWARE_RERANK": 0.0060,
    "STRICT_REPAIR": 0.0045,
    "MISSION_REPAIR": 0.0040,
    "MISSION_BUILD": 0.0035,
    "CRITIC_REVIEW": 0.0030,
    "REJECT_REPAIR_NARROW_QUERY": 0.0010,
    "BASELINE_ONLY": 0.0008,
    "unknown": 0.0005,
    "": 0.0005,
}


SCENARIO_MULTIPLIERS = {
    "conservative": 0.50,
    "base": 1.00,
    "optimistic": 2.00,
}


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


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


def mean(values: List[float]) -> float:
    if not values:
        return 0.0

    return sum(values) / len(values)


def load_scalable_rows() -> List[Dict[str, str]]:
    rows = read_csv(SCALABLE_EVAL_PATH)

    if not rows:
        raise FileNotFoundError(
            f"{SCALABLE_EVAL_PATH} not found or empty. "
            "Run MVP 19 first: python -m src.scalable_governed_evaluator"
        )

    return rows


def successful_rows(rows: List[Dict[str, str]]) -> List[Dict[str, str]]:
    return [row for row in rows if safe_int(row.get("success")) == 1]


def estimate_route_mix(rows: List[Dict[str, str]]) -> Dict[str, float]:
    success_rows = successful_rows(rows)
    total = max(len(success_rows), 1)

    route_counts = Counter(clean_text(row.get("governance_route")) or "unknown" for row in success_rows)

    return {
        route: count / total
        for route, count in route_counts.items()
    }


def estimate_execution_mix(rows: List[Dict[str, str]]) -> Dict[str, float]:
    success_rows = successful_rows(rows)
    total = max(len(success_rows), 1)

    source_counts = Counter(clean_text(row.get("final_execution_source")) or "unknown" for row in success_rows)

    return {
        source: count / total
        for source, count in source_counts.items()
    }


def estimate_avg_cost_per_query(rows: List[Dict[str, str]]) -> float:
    success_rows = successful_rows(rows)

    if not success_rows:
        return 0.0

    costs = []

    for row in success_rows:
        source = clean_text(row.get("final_execution_source")) or "unknown"
        cost_per_1k = DEFAULT_COST_PER_1K_BY_EXECUTION_SOURCE.get(
            source,
            DEFAULT_COST_PER_1K_BY_EXECUTION_SOURCE["unknown"],
        )
        costs.append(cost_per_1k / 1000.0)

    return mean(costs)


def estimate_value_per_query(
    row: Dict[str, str],
    scenario: str,
) -> float:
    route = clean_text(row.get("governance_route")) or "unknown"
    base_value = BASE_VALUE_PER_QUERY_BY_ROUTE.get(
        route,
        BASE_VALUE_PER_QUERY_BY_ROUTE["unknown"],
    )

    multiplier = SCENARIO_MULTIPLIERS.get(scenario, 1.0)

    coverage_status = clean_text(row.get("candidate_coverage_status"))
    final_slate_size = safe_int(row.get("final_slate_size"))
    unique_sub_intents = safe_int(row.get("unique_sub_intents"))
    baseline_preserved = safe_int(row.get("baseline_preserved"))

    coverage_multiplier = 1.0

    if coverage_status == "SUPPORTED":
        coverage_multiplier += 0.20

    if unique_sub_intents >= 4:
        coverage_multiplier += 0.15

    if final_slate_size >= 8:
        coverage_multiplier += 0.10

    if coverage_status in {"LOW_COVERAGE", "LOW_SUB_INTENT_COVERAGE"}:
        coverage_multiplier -= 0.35

    if coverage_status in {"NO_FINAL_SLATE", "ERROR"}:
        coverage_multiplier = 0.0

    # Baseline preservation still has value because governance avoided unnecessary cost/risk,
    # but it should not receive the same lift as a full mission intervention.
    if baseline_preserved == 1:
        coverage_multiplier *= 0.60

    return max(0.0, base_value * multiplier * coverage_multiplier)


def estimate_cost_for_query(row: Dict[str, str]) -> float:
    source = clean_text(row.get("final_execution_source")) or "unknown"
    cost_per_1k = DEFAULT_COST_PER_1K_BY_EXECUTION_SOURCE.get(
        source,
        DEFAULT_COST_PER_1K_BY_EXECUTION_SOURCE["unknown"],
    )

    return cost_per_1k / 1000.0


def build_by_route_rows(
    rows: List[Dict[str, str]],
    daily_query_volume: int,
    scenario: str,
) -> List[Dict[str, object]]:
    success_rows = successful_rows(rows)
    total_success = max(len(success_rows), 1)

    grouped: Dict[str, List[Dict[str, str]]] = defaultdict(list)

    for row in success_rows:
        route = clean_text(row.get("governance_route")) or "unknown"
        grouped[route].append(row)

    output_rows = []

    for route, group_rows in sorted(grouped.items(), key=lambda item: item[0]):
        route_share = len(group_rows) / total_success
        estimated_daily_queries = daily_query_volume * route_share

        avg_cost_per_query = mean([estimate_cost_for_query(row) for row in group_rows])
        avg_value_per_query = mean([estimate_value_per_query(row, scenario) for row in group_rows])

        daily_cost = estimated_daily_queries * avg_cost_per_query
        monthly_cost = daily_cost * 30

        daily_value = estimated_daily_queries * avg_value_per_query
        monthly_value = daily_value * 30

        net_daily_value = daily_value - daily_cost
        net_monthly_value = monthly_value - monthly_cost

        avg_final_slate_size = mean([safe_float(row.get("final_slate_size")) for row in group_rows])
        avg_unique_sub_intents = mean([safe_float(row.get("unique_sub_intents")) for row in group_rows])
        avg_repair_risk = mean([safe_float(row.get("repair_risk_score")) for row in group_rows])
        avg_critic_need = mean([safe_float(row.get("critic_need_score")) for row in group_rows])

        output_rows.append(
            {
                "scenario": scenario,
                "governance_route": route,
                "observed_query_count": len(group_rows),
                "route_share": round(route_share, 6),
                "estimated_daily_queries": round(estimated_daily_queries, 2),
                "avg_cost_per_query": round(avg_cost_per_query, 8),
                "avg_value_per_query": round(avg_value_per_query, 8),
                "daily_cost": round(daily_cost, 4),
                "monthly_cost": round(monthly_cost, 4),
                "daily_value": round(daily_value, 4),
                "monthly_value": round(monthly_value, 4),
                "net_daily_value": round(net_daily_value, 4),
                "net_monthly_value": round(net_monthly_value, 4),
                "break_even_value_per_query": round(avg_cost_per_query, 8),
                "avg_final_slate_size": round(avg_final_slate_size, 4),
                "avg_unique_sub_intents": round(avg_unique_sub_intents, 4),
                "avg_repair_risk_score": round(avg_repair_risk, 4),
                "avg_critic_need_score": round(avg_critic_need, 4),
            }
        )

    return output_rows


def build_scenario_rows(
    rows: List[Dict[str, str]],
    daily_query_volume: int,
) -> List[Dict[str, object]]:
    scenario_rows = []

    success_rows = successful_rows(rows)

    for scenario in SCENARIO_MULTIPLIERS:
        if not success_rows:
            avg_cost_per_query = 0.0
            avg_value_per_query = 0.0
        else:
            avg_cost_per_query = mean([estimate_cost_for_query(row) for row in success_rows])
            avg_value_per_query = mean(
                [estimate_value_per_query(row, scenario) for row in success_rows]
            )

        daily_cost = daily_query_volume * avg_cost_per_query
        monthly_cost = daily_cost * 30

        daily_value = daily_query_volume * avg_value_per_query
        monthly_value = daily_value * 30

        net_daily_value = daily_value - daily_cost
        net_monthly_value = monthly_value - monthly_cost

        break_even_value_per_query = avg_cost_per_query
        break_even_monthly_value = monthly_cost

        scenario_rows.append(
            {
                "scenario": scenario,
                "daily_query_volume": daily_query_volume,
                "monthly_query_volume": daily_query_volume * 30,
                "avg_cost_per_query": round(avg_cost_per_query, 8),
                "avg_value_per_query": round(avg_value_per_query, 8),
                "daily_cost": round(daily_cost, 4),
                "monthly_cost": round(monthly_cost, 4),
                "daily_value": round(daily_value, 4),
                "monthly_value": round(monthly_value, 4),
                "net_daily_value": round(net_daily_value, 4),
                "net_monthly_value": round(net_monthly_value, 4),
                "break_even_value_per_query": round(break_even_value_per_query, 8),
                "break_even_monthly_value": round(break_even_monthly_value, 4),
                "value_to_cost_ratio": round(daily_value / daily_cost, 4) if daily_cost > 0 else 0.0,
            }
        )

    return scenario_rows


def build_summary_row(
    rows: List[Dict[str, str]],
    daily_query_volume: int,
    scenario: str,
) -> Dict[str, object]:
    success_rows = successful_rows(rows)
    total_rows = len(rows)
    total_success = len(success_rows)

    route_mix = estimate_route_mix(rows)
    execution_mix = estimate_execution_mix(rows)

    avg_cost_per_query = estimate_avg_cost_per_query(rows)
    avg_value_per_query = mean(
        [estimate_value_per_query(row, scenario) for row in success_rows]
    )

    daily_cost = daily_query_volume * avg_cost_per_query
    monthly_cost = daily_cost * 30

    daily_value = daily_query_volume * avg_value_per_query
    monthly_value = daily_value * 30

    net_daily_value = daily_value - daily_cost
    net_monthly_value = monthly_value - monthly_cost

    baseline_share = route_mix.get("BASELINE_ONLY", 0.0) + route_mix.get("REJECT_REPAIR_NARROW_QUERY", 0.0)
    behavior_share = execution_mix.get("behavior_aware", 0.0)
    fallback_share = execution_mix.get("baseline_fallback", 0.0) + execution_mix.get("fallback_no_slate_available", 0.0)

    supported_count = sum(
        1 for row in success_rows
        if clean_text(row.get("candidate_coverage_status")) == "SUPPORTED"
    )

    low_coverage_count = sum(
        1 for row in success_rows
        if clean_text(row.get("candidate_coverage_status")) in {
            "LOW_COVERAGE",
            "LOW_SUB_INTENT_COVERAGE",
            "NO_FINAL_SLATE",
        }
    )

    value_to_cost_ratio = daily_value / daily_cost if daily_cost > 0 else 0.0

    if net_monthly_value > 0:
        recommendation = "COST_JUSTIFIED_UNDER_ASSUMPTIONS"
    elif value_to_cost_ratio >= 0.80:
        recommendation = "NEAR_BREAK_EVEN_NEEDS_VALIDATION"
    else:
        recommendation = "NOT_YET_COST_JUSTIFIED_UNDER_ASSUMPTIONS"

    return {
        "scenario": scenario,
        "total_eval_queries": total_rows,
        "successful_eval_queries": total_success,
        "success_rate": round(total_success / max(total_rows, 1), 4),
        "daily_query_volume": daily_query_volume,
        "monthly_query_volume": daily_query_volume * 30,
        "avg_cost_per_query": round(avg_cost_per_query, 8),
        "avg_value_per_query": round(avg_value_per_query, 8),
        "daily_cost": round(daily_cost, 4),
        "monthly_cost": round(monthly_cost, 4),
        "daily_value": round(daily_value, 4),
        "monthly_value": round(monthly_value, 4),
        "net_daily_value": round(net_daily_value, 4),
        "net_monthly_value": round(net_monthly_value, 4),
        "break_even_value_per_query": round(avg_cost_per_query, 8),
        "value_to_cost_ratio": round(value_to_cost_ratio, 4),
        "baseline_or_reject_route_share": round(baseline_share, 4),
        "behavior_aware_execution_share": round(behavior_share, 4),
        "fallback_execution_share": round(fallback_share, 4),
        "supported_query_count": supported_count,
        "low_coverage_query_count": low_coverage_count,
        "recommendation": recommendation,
        "plain_english_summary": build_plain_english_summary(
            scenario=scenario,
            daily_query_volume=daily_query_volume,
            monthly_cost=monthly_cost,
            monthly_value=monthly_value,
            net_monthly_value=net_monthly_value,
            value_to_cost_ratio=value_to_cost_ratio,
            recommendation=recommendation,
        ),
    }


def build_plain_english_summary(
    scenario: str,
    daily_query_volume: int,
    monthly_cost: float,
    monthly_value: float,
    net_monthly_value: float,
    value_to_cost_ratio: float,
    recommendation: str,
) -> str:
    return (
        f"Under the {scenario} scenario with {daily_query_volume:,} queries/day, "
        f"CORTEX is estimated to cost ${monthly_cost:,.2f}/month and generate "
        f"${monthly_value:,.2f}/month in scenario value. Estimated net value is "
        f"${net_monthly_value:,.2f}/month with a value-to-cost ratio of {value_to_cost_ratio:.2f}. "
        f"Recommendation: {recommendation}. These are scenario assumptions, not measured production revenue."
    )


def run_cost_value_analysis(
    daily_query_volume: int,
    scenario: str,
) -> Tuple[Dict[str, object], List[Dict[str, object]], List[Dict[str, object]]]:
    ensure_output_dir()

    rows = load_scalable_rows()

    summary_row = build_summary_row(
        rows=rows,
        daily_query_volume=daily_query_volume,
        scenario=scenario,
    )

    by_route_rows = build_by_route_rows(
        rows=rows,
        daily_query_volume=daily_query_volume,
        scenario=scenario,
    )

    scenario_rows = build_scenario_rows(
        rows=rows,
        daily_query_volume=daily_query_volume,
    )

    summary_fields = [
        "scenario",
        "total_eval_queries",
        "successful_eval_queries",
        "success_rate",
        "daily_query_volume",
        "monthly_query_volume",
        "avg_cost_per_query",
        "avg_value_per_query",
        "daily_cost",
        "monthly_cost",
        "daily_value",
        "monthly_value",
        "net_daily_value",
        "net_monthly_value",
        "break_even_value_per_query",
        "value_to_cost_ratio",
        "baseline_or_reject_route_share",
        "behavior_aware_execution_share",
        "fallback_execution_share",
        "supported_query_count",
        "low_coverage_query_count",
        "recommendation",
        "plain_english_summary",
    ]

    by_route_fields = [
        "scenario",
        "governance_route",
        "observed_query_count",
        "route_share",
        "estimated_daily_queries",
        "avg_cost_per_query",
        "avg_value_per_query",
        "daily_cost",
        "monthly_cost",
        "daily_value",
        "monthly_value",
        "net_daily_value",
        "net_monthly_value",
        "break_even_value_per_query",
        "avg_final_slate_size",
        "avg_unique_sub_intents",
        "avg_repair_risk_score",
        "avg_critic_need_score",
    ]

    scenario_fields = [
        "scenario",
        "daily_query_volume",
        "monthly_query_volume",
        "avg_cost_per_query",
        "avg_value_per_query",
        "daily_cost",
        "monthly_cost",
        "daily_value",
        "monthly_value",
        "net_daily_value",
        "net_monthly_value",
        "break_even_value_per_query",
        "break_even_monthly_value",
        "value_to_cost_ratio",
    ]

    write_csv(COST_VALUE_SUMMARY_PATH, [summary_row], summary_fields)
    write_csv(COST_VALUE_BY_ROUTE_PATH, by_route_rows, by_route_fields)
    write_csv(COST_VALUE_SCENARIOS_PATH, scenario_rows, scenario_fields)

    return summary_row, by_route_rows, scenario_rows


def print_results(
    summary_row: Dict[str, object],
    by_route_rows: List[Dict[str, object]],
    scenario_rows: List[Dict[str, object]],
) -> None:
    print("\nCORTEX Cost vs Value Governance Analyzer")
    print("-" * 100)

    for key, value in summary_row.items():
        print(f"{key}: {value}")

    print("\nRoute-Level Cost / Value")
    print("-" * 100)

    for row in by_route_rows:
        print(
            f"{row.get('governance_route')}: "
            f"monthly_cost=${row.get('monthly_cost')}, "
            f"monthly_value=${row.get('monthly_value')}, "
            f"net=${row.get('net_monthly_value')}"
        )

    print("\nScenario Comparison")
    print("-" * 100)

    for row in scenario_rows:
        print(
            f"{row.get('scenario')}: "
            f"monthly_cost=${row.get('monthly_cost')}, "
            f"monthly_value=${row.get('monthly_value')}, "
            f"net=${row.get('net_monthly_value')}, "
            f"ratio={row.get('value_to_cost_ratio')}"
        )

    print("\nFiles written:")
    print(f"- {COST_VALUE_SUMMARY_PATH}")
    print(f"- {COST_VALUE_BY_ROUTE_PATH}")
    print(f"- {COST_VALUE_SCENARIOS_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser(description="MVP 19.3 CORTEX Cost vs Value Governance Analyzer")

    parser.add_argument(
        "--daily-query-volume",
        type=int,
        default=1_000_000,
        help="Estimated production query volume per day.",
    )

    parser.add_argument(
        "--scenario",
        choices=["conservative", "base", "optimistic"],
        default="base",
        help="Value assumption scenario.",
    )

    args = parser.parse_args()

    summary_row, by_route_rows, scenario_rows = run_cost_value_analysis(
        daily_query_volume=args.daily_query_volume,
        scenario=args.scenario,
    )

    print_results(
        summary_row=summary_row,
        by_route_rows=by_route_rows,
        scenario_rows=scenario_rows,
    )


if __name__ == "__main__":
    main()