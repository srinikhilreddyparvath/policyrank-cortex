"""
MVP 19: Scalable Governed CORTEX Evaluator

This evaluator runs the MVP 18 Governed CORTEX Runner across many queries.

Purpose:
- Move beyond a handful of smoke-test examples.
- Prove the governed system can route many queries.
- Measure route distribution, final slate success, baseline preservation, low/no coverage,
  behavior-aware execution, and governance decisions at scale.

This does NOT fabricate CTR, ATC, or purchase engagement.
It evaluates governed execution using:
- governance route decisions
- final governed slate size
- unique sub-intents
- behavior-aware execution source
- baseline preservation
- cold-start proxy/rescue counts
- exploration counts
- error/failure status

Default behavior:
- Uses fast mode / --skip-refresh for speed.
- This means it relies on existing generated artifacts when available.
- For unknown queries, the Governed Runner still calls Governance Agent with skip-refresh.
- Some unknown queries may become baseline/fallback due to no existing slate artifacts.

Outputs:
- outputs/scalable_governed_eval.csv
- outputs/scalable_governed_eval_summary.csv
- outputs/scalable_governed_eval_by_route.csv
- outputs/scalable_governed_eval_by_execution_source.csv
- outputs/scalable_governed_eval_errors.csv
"""

from __future__ import annotations

import argparse
import csv
import random
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


DATA_PATH = Path("data/esci_balanced_sample.csv")
OUTPUT_DIR = Path("outputs")

GOVERNED_SUMMARY_PATH = OUTPUT_DIR / "governed_cortex_summary.csv"
GOVERNED_FINAL_SLATE_PATH = OUTPUT_DIR / "governed_cortex_final_slate.csv"

SCALABLE_EVAL_PATH = OUTPUT_DIR / "scalable_governed_eval.csv"
SCALABLE_SUMMARY_PATH = OUTPUT_DIR / "scalable_governed_eval_summary.csv"
SCALABLE_BY_ROUTE_PATH = OUTPUT_DIR / "scalable_governed_eval_by_route.csv"
SCALABLE_BY_EXECUTION_SOURCE_PATH = OUTPUT_DIR / "scalable_governed_eval_by_execution_source.csv"
SCALABLE_ERRORS_PATH = OUTPUT_DIR / "scalable_governed_eval_errors.csv"


SMOKE_TEST_QUERIES = [
    "new apartment kitchen setup",
    "beach vacation packing list",
    "adidas soccer cleats",
    "world cup watch party",
    "camping trip essentials",
]


MISSION_LIKE_SEED_QUERIES = [
    "new apartment kitchen setup",
    "beach vacation packing list",
    "world cup watch party",
    "camping trip essentials",
    "college dorm essentials",
    "baby shower decorations",
    "office desk setup",
    "diwali party supplies",
    "birthday party supplies",
    "hiking trip snacks",
    "home gym setup",
    "moving day essentials",
    "road trip snacks",
    "bbq party supplies",
    "school lunch packing",
]


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


def load_esci_queries(max_queries: int | None = None, seed: int = 42) -> List[str]:
    """
    Loads unique queries from data/esci_balanced_sample.csv when available.

    The evaluator is still useful without this file, because it falls back to seed queries.
    """
    if not DATA_PATH.exists():
        return []

    queries = []

    with DATA_PATH.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)

        if "query" not in reader.fieldnames:
            return []

        seen = set()

        for row in reader:
            query = clean_text(row.get("query"))
            if not query:
                continue

            key = query.lower()

            if key in seen:
                continue

            seen.add(key)
            queries.append(query)

    random.seed(seed)
    random.shuffle(queries)

    if max_queries is not None:
        queries = queries[:max_queries]

    return queries


def build_query_list(
    sample_size: int,
    query_mode: str,
    seed: int,
    custom_queries: List[str] | None = None,
) -> List[str]:
    if custom_queries:
        deduped = []
        seen = set()

        for query in custom_queries:
            query = clean_text(query)
            if not query:
                continue
            key = query.lower()
            if key in seen:
                continue
            seen.add(key)
            deduped.append(query)

        return deduped[:sample_size]

    if query_mode == "smoke":
        return SMOKE_TEST_QUERIES[:sample_size]

    if query_mode == "mission_seed":
        return MISSION_LIKE_SEED_QUERIES[:sample_size]

    esci_queries = load_esci_queries(max_queries=sample_size, seed=seed)

    if esci_queries:
        # Keep a few core smoke tests at the top when sample size allows.
        combined = []
        seen = set()

        for query in SMOKE_TEST_QUERIES + esci_queries:
            key = query.lower()
            if key not in seen:
                combined.append(query)
                seen.add(key)

        return combined[:sample_size]

    # Fallback if data file is missing.
    return (SMOKE_TEST_QUERIES + MISSION_LIKE_SEED_QUERIES)[:sample_size]


def run_governed_runner(query: str, skip_refresh: bool = True) -> Tuple[bool, str]:
    cmd = [
        sys.executable,
        "-u",
        "-m",
        "src.governed_cortex_runner",
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

    combined_output = ""

    if result.stdout:
        combined_output += result.stdout

    if result.stderr:
        combined_output += "\n\nSTDERR:\n" + result.stderr

    return result.returncode == 0, combined_output


def infer_candidate_coverage_status(summary_row: Dict[str, str]) -> str:
    final_slate_size = safe_int(summary_row.get("final_slate_size"))
    final_source = clean_text(summary_row.get("final_execution_source"))
    baseline_preserved = safe_int(summary_row.get("baseline_preserved"))
    unique_sub_intents = safe_int(summary_row.get("unique_sub_intents"))

    if final_slate_size <= 0:
        return "NO_FINAL_SLATE"

    if baseline_preserved == 1 or "fallback" in final_source:
        return "GOVERNANCE_ONLY_OR_BASELINE_FALLBACK"

    if final_slate_size < 3:
        return "LOW_COVERAGE"

    if unique_sub_intents < 2 and final_slate_size >= 3:
        return "LOW_SUB_INTENT_COVERAGE"

    return "SUPPORTED"


def build_eval_row(
    query: str,
    success: bool,
    runtime_output: str,
) -> Dict[str, object]:
    summary = latest_query_row(GOVERNED_SUMMARY_PATH, query)

    if not success:
        return {
            "query": query,
            "success": 0,
            "error": runtime_output[:1000],
            "governance_route": "",
            "governance_decision": "",
            "query_type": "",
            "final_execution_source": "",
            "final_slate_size": 0,
            "unique_sub_intents": 0,
            "unique_policy_reasons": 0,
            "behavior_ranked_items": 0,
            "cold_start_proxy_items": 0,
            "exploration_items": 0,
            "baseline_preserved": 0,
            "candidate_coverage_status": "ERROR",
            "mission_likelihood_score": 0.0,
            "brand_specificity_score": 0.0,
            "compound_intent_score": 0.0,
            "narrow_query_score": 0.0,
            "repair_risk_score": 0.0,
            "coverage_gap_score": 0.0,
            "behavior_rescue_signal": 0.0,
            "critic_need_score": 0.0,
            "plain_english_reason": "",
        }

    coverage_status = infer_candidate_coverage_status(summary)

    return {
        "query": query,
        "success": 1,
        "error": "",
        "governance_route": clean_text(summary.get("governance_route")),
        "governance_decision": clean_text(summary.get("governance_decision")),
        "query_type": clean_text(summary.get("query_type")),
        "final_execution_source": clean_text(summary.get("final_execution_source")),
        "final_slate_size": safe_int(summary.get("final_slate_size")),
        "unique_sub_intents": safe_int(summary.get("unique_sub_intents")),
        "unique_policy_reasons": safe_int(summary.get("unique_policy_reasons")),
        "behavior_ranked_items": safe_int(summary.get("behavior_ranked_items")),
        "cold_start_proxy_items": safe_int(summary.get("cold_start_proxy_items")),
        "exploration_items": safe_int(summary.get("exploration_items")),
        "baseline_preserved": safe_int(summary.get("baseline_preserved")),
        "candidate_coverage_status": coverage_status,
        "mission_likelihood_score": safe_float(summary.get("mission_likelihood_score")),
        "brand_specificity_score": safe_float(summary.get("brand_specificity_score")),
        "compound_intent_score": safe_float(summary.get("compound_intent_score")),
        "narrow_query_score": safe_float(summary.get("narrow_query_score")),
        "repair_risk_score": safe_float(summary.get("repair_risk_score")),
        "coverage_gap_score": safe_float(summary.get("coverage_gap_score")),
        "behavior_rescue_signal": safe_float(summary.get("behavior_rescue_signal")),
        "critic_need_score": safe_float(summary.get("critic_need_score")),
        "plain_english_reason": clean_text(summary.get("plain_english_reason")),
    }


def mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def summarize_eval(rows: List[Dict[str, object]]) -> Dict[str, object]:
    total = len(rows)
    success_rows = [row for row in rows if safe_int(row.get("success")) == 1]
    failed_rows = [row for row in rows if safe_int(row.get("success")) == 0]

    route_counts = Counter(clean_text(row.get("governance_route")) for row in success_rows)
    execution_counts = Counter(clean_text(row.get("final_execution_source")) for row in success_rows)
    coverage_counts = Counter(clean_text(row.get("candidate_coverage_status")) for row in success_rows)
    query_type_counts = Counter(clean_text(row.get("query_type")) for row in success_rows)

    final_slate_sizes = [safe_float(row.get("final_slate_size")) for row in success_rows]
    unique_sub_intents = [safe_float(row.get("unique_sub_intents")) for row in success_rows]
    cold_start_items = [safe_float(row.get("cold_start_proxy_items")) for row in success_rows]
    exploration_items = [safe_float(row.get("exploration_items")) for row in success_rows]
    baseline_preserved = [safe_float(row.get("baseline_preserved")) for row in success_rows]

    supported_count = coverage_counts.get("SUPPORTED", 0)
    baseline_fallback_count = coverage_counts.get("GOVERNANCE_ONLY_OR_BASELINE_FALLBACK", 0)
    low_coverage_count = coverage_counts.get("LOW_COVERAGE", 0) + coverage_counts.get("LOW_SUB_INTENT_COVERAGE", 0)
    no_slate_count = coverage_counts.get("NO_FINAL_SLATE", 0)

    return {
        "total_queries": total,
        "successful_queries": len(success_rows),
        "failed_queries": len(failed_rows),
        "success_rate": round(len(success_rows) / max(total, 1), 4),
        "supported_queries": supported_count,
        "baseline_or_governance_only_queries": baseline_fallback_count,
        "low_coverage_queries": low_coverage_count,
        "no_final_slate_queries": no_slate_count,
        "avg_final_slate_size": round(mean(final_slate_sizes), 4),
        "avg_unique_sub_intents": round(mean(unique_sub_intents), 4),
        "total_cold_start_proxy_items": int(sum(cold_start_items)),
        "total_exploration_items": int(sum(exploration_items)),
        "baseline_preservation_rate": round(mean(baseline_preserved), 4),
        "behavior_aware_execution_count": execution_counts.get("behavior_aware", 0),
        "baseline_fallback_execution_count": execution_counts.get("baseline_fallback", 0),
        "strict_repair_execution_count": execution_counts.get("strict_repair", 0),
        "fallback_no_slate_execution_count": execution_counts.get("fallback_no_slate_available", 0),
        "top_governance_route": route_counts.most_common(1)[0][0] if route_counts else "",
        "top_execution_source": execution_counts.most_common(1)[0][0] if execution_counts else "",
        "top_query_type": query_type_counts.most_common(1)[0][0] if query_type_counts else "",
    }


def group_by_field(rows: List[Dict[str, object]], field: str) -> List[Dict[str, object]]:
    groups: Dict[str, List[Dict[str, object]]] = defaultdict(list)

    for row in rows:
        if safe_int(row.get("success")) != 1:
            continue

        key = clean_text(row.get(field)) or "unknown"
        groups[key].append(row)

    out = []

    for key, group_rows in sorted(groups.items(), key=lambda item: item[0]):
        out.append(
            {
                field: key,
                "query_count": len(group_rows),
                "avg_final_slate_size": round(mean([safe_float(r.get("final_slate_size")) for r in group_rows]), 4),
                "avg_unique_sub_intents": round(mean([safe_float(r.get("unique_sub_intents")) for r in group_rows]), 4),
                "avg_repair_risk_score": round(mean([safe_float(r.get("repair_risk_score")) for r in group_rows]), 4),
                "avg_coverage_gap_score": round(mean([safe_float(r.get("coverage_gap_score")) for r in group_rows]), 4),
                "avg_critic_need_score": round(mean([safe_float(r.get("critic_need_score")) for r in group_rows]), 4),
                "total_cold_start_proxy_items": int(sum([safe_float(r.get("cold_start_proxy_items")) for r in group_rows])),
                "baseline_preservation_rate": round(mean([safe_float(r.get("baseline_preserved")) for r in group_rows]), 4),
            }
        )

    return out


def run_scalable_eval(
    sample_size: int,
    query_mode: str,
    seed: int,
    skip_refresh: bool,
    custom_queries: List[str] | None = None,
) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    ensure_output_dir()

    queries = build_query_list(
        sample_size=sample_size,
        query_mode=query_mode,
        seed=seed,
        custom_queries=custom_queries,
    )

    print("\nScalable Governed CORTEX Evaluation")
    print("-" * 100)
    print(f"Query mode: {query_mode}")
    print(f"Requested sample size: {sample_size}")
    print(f"Actual queries: {len(queries)}")
    print(f"Skip refresh: {skip_refresh}")

    eval_rows: List[Dict[str, object]] = []

    for index, query in enumerate(queries, start=1):
        print(f"[{index}/{len(queries)}] {query}")

        success, output = run_governed_runner(
            query=query,
            skip_refresh=skip_refresh,
        )

        row = build_eval_row(
            query=query,
            success=success,
            runtime_output=output,
        )

        eval_rows.append(row)

    summary = summarize_eval(eval_rows)

    eval_fields = [
        "query",
        "success",
        "error",
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
        "candidate_coverage_status",
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

    summary_fields = [
        "total_queries",
        "successful_queries",
        "failed_queries",
        "success_rate",
        "supported_queries",
        "baseline_or_governance_only_queries",
        "low_coverage_queries",
        "no_final_slate_queries",
        "avg_final_slate_size",
        "avg_unique_sub_intents",
        "total_cold_start_proxy_items",
        "total_exploration_items",
        "baseline_preservation_rate",
        "behavior_aware_execution_count",
        "baseline_fallback_execution_count",
        "strict_repair_execution_count",
        "fallback_no_slate_execution_count",
        "top_governance_route",
        "top_execution_source",
        "top_query_type",
    ]

    by_route_rows = group_by_field(eval_rows, "governance_route")
    by_execution_source_rows = group_by_field(eval_rows, "final_execution_source")
    error_rows = [row for row in eval_rows if safe_int(row.get("success")) == 0]

    group_fields_route = [
        "governance_route",
        "query_count",
        "avg_final_slate_size",
        "avg_unique_sub_intents",
        "avg_repair_risk_score",
        "avg_coverage_gap_score",
        "avg_critic_need_score",
        "total_cold_start_proxy_items",
        "baseline_preservation_rate",
    ]

    group_fields_execution = [
        "final_execution_source",
        "query_count",
        "avg_final_slate_size",
        "avg_unique_sub_intents",
        "avg_repair_risk_score",
        "avg_coverage_gap_score",
        "avg_critic_need_score",
        "total_cold_start_proxy_items",
        "baseline_preservation_rate",
    ]

    write_csv(SCALABLE_EVAL_PATH, eval_rows, eval_fields)
    write_csv(SCALABLE_SUMMARY_PATH, [summary], summary_fields)
    write_csv(SCALABLE_BY_ROUTE_PATH, by_route_rows, group_fields_route)
    write_csv(SCALABLE_BY_EXECUTION_SOURCE_PATH, by_execution_source_rows, group_fields_execution)
    write_csv(SCALABLE_ERRORS_PATH, error_rows, eval_fields)

    return eval_rows, summary


def print_summary(summary: Dict[str, object]) -> None:
    print("\nScalable Governed Evaluation Summary")
    print("-" * 100)

    for key, value in summary.items():
        print(f"{key}: {value}")

    print("\nFiles written:")
    print(f"- {SCALABLE_EVAL_PATH}")
    print(f"- {SCALABLE_SUMMARY_PATH}")
    print(f"- {SCALABLE_BY_ROUTE_PATH}")
    print(f"- {SCALABLE_BY_EXECUTION_SOURCE_PATH}")
    print(f"- {SCALABLE_ERRORS_PATH}")


def parse_custom_queries(raw: str | None) -> List[str]:
    if not raw:
        return []

    # Supports comma-separated or pipe-separated custom queries.
    parts = []

    for chunk in raw.replace("|", ",").split(","):
        chunk = clean_text(chunk)
        if chunk:
            parts.append(chunk)

    return parts


def main() -> None:
    parser = argparse.ArgumentParser(description="MVP 19 Scalable Governed CORTEX Evaluator")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=25,
        help="Number of unique queries to evaluate.",
    )
    parser.add_argument(
        "--query-mode",
        choices=["smoke", "mission_seed", "esci"],
        default="smoke",
        help="Which query source to use.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for ESCI query sampling.",
    )
    parser.add_argument(
        "--refresh",
        action="store_true",
        help="Refresh full governance/behavior/strict dependencies. Slower but more complete.",
    )
    parser.add_argument(
        "--queries",
        default="",
        help="Optional comma-separated custom query list.",
    )

    args = parser.parse_args()

    custom_queries = parse_custom_queries(args.queries)

    rows, summary = run_scalable_eval(
        sample_size=args.sample_size,
        query_mode=args.query_mode,
        seed=args.seed,
        skip_refresh=not args.refresh,
        custom_queries=custom_queries,
    )

    print_summary(summary)


if __name__ == "__main__":
    main()