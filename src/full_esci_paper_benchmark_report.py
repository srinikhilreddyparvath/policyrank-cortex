"""
MVP 26.11: Paper-ready full ESCI benchmark report generator.

This module is intentionally read-only with respect to benchmark inputs. It
does not rerun evaluation; it consolidates existing CSV/JSON artifacts into
paper-oriented tables and a markdown report.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


DEFAULT_OUTPUT_DIR = Path("outputs/full_esci_paper_benchmark_report")

SUMMARY_FIELDS = ["metric", "value", "source"]
ROUTE_FIELDS = [
    "comparison",
    "calibrated_governance_route",
    "execution_source",
    "query_count_left",
    "query_count_right",
    "exact_rate_left",
    "exact_rate_right",
    "exact_rate_delta",
    "exact_or_sub_rate_left",
    "exact_or_sub_rate_right",
    "exact_or_sub_rate_delta",
]
QUERY_EXAMPLE_FIELDS = [
    "comparison",
    "delta_status",
    "query",
    "route_100k",
    "route_500k",
    "exact_100k",
    "exact_500k",
    "exact_or_sub_100k",
    "exact_or_sub_500k",
    "best_label_100k",
    "best_label_500k",
    "outcome_score_delta",
]


def clean_text(value: object) -> str:
    return str(value or "").strip()


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(str(value))
    except Exception:
        return default


def safe_int(value: object, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(str(value)))
    except Exception:
        return default


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


def metric_rows(path: Path) -> Dict[str, Dict[str, str]]:
    rows = read_csv(path)
    return {clean_text(row.get("metric")): row for row in rows if clean_text(row.get("metric"))}


def metric_delta(rows_by_metric: Dict[str, Dict[str, str]], metric: str) -> str:
    return clean_text(rows_by_metric.get(metric, {}).get("delta"))


def metric_right_value(rows_by_metric: Dict[str, Dict[str, str]], metric: str) -> str:
    row = rows_by_metric.get(metric, {})
    for key, value in row.items():
        if key.endswith("_value") and not key.startswith("100k_"):
            return clean_text(value)
    return ""


def normalized_query(value: object) -> str:
    return " ".join(clean_text(value).lower().split())


def label_score(label: object) -> int:
    return {"E": 4, "S": 3, "C": 2, "I": 1}.get(clean_text(label).upper(), 0)


def outcome_score(row: Dict[str, object], prefix: str) -> int:
    if safe_int(row.get(f"{prefix}_top_k_has_exact")):
        return 3
    if safe_int(row.get(f"{prefix}_top_k_has_exact_or_substitute")):
        return 2
    return label_score(row.get(f"{prefix}_best_esci_label"))


def count_status(rows: Iterable[Dict[str, str]]) -> Dict[str, int]:
    counts = {"improved": 0, "regressed": 0, "unchanged": 0}
    for row in rows:
        status = clean_text(row.get("delta_status"))
        if status in counts:
            counts[status] += 1
    return counts


def compare_raw_vs_boost(raw_rows: List[Dict[str, str]], boost_rows: List[Dict[str, str]]) -> Dict[str, int]:
    boost_by_query = {normalized_query(row.get("query")): row for row in boost_rows}
    counts = {"improved": 0, "regressed": 0, "unchanged": 0}
    for raw in raw_rows:
        query_key = normalized_query(raw.get("query"))
        boost = boost_by_query.get(query_key)
        if not boost:
            continue
        raw_score = 3 if safe_int(raw.get("top_k_has_exact")) else 2 if safe_int(raw.get("top_k_has_exact_or_substitute")) else label_score(raw.get("best_esci_label"))
        boost_score = 3 if safe_int(boost.get("top_k_has_exact")) else 2 if safe_int(boost.get("top_k_has_exact_or_substitute")) else label_score(boost.get("best_esci_label"))
        if boost_score > raw_score:
            counts["improved"] += 1
        elif boost_score < raw_score:
            counts["regressed"] += 1
        else:
            counts["unchanged"] += 1
    return counts


def summary_row(metric: str, value: object, source: str) -> Dict[str, object]:
    return {"metric": metric, "value": value, "source": source}


def build_summary(args: argparse.Namespace) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
    index_100k = read_json(Path(args.index_100k_dir) / "index_summary.json")
    index_500k = read_json(Path(args.index_500k_dir) / "index_summary.json")
    raw_summary = metric_rows(Path(args.scale_comparison_dir) / "scale_comparison_summary.csv")
    strict_summary = metric_rows(Path(args.strict_boost_comparison_dir) / "scale_comparison_third_summary.csv")

    raw_rows = read_csv(Path(args.strict_boost_comparison_dir) / "500k" / "route_retrieval_results.csv")
    boost_rows = read_csv(Path(args.strict_boost_comparison_dir) / "500k_strict_boost" / "route_retrieval_results.csv")
    raw_vs_boost = compare_raw_vs_boost(raw_rows, boost_rows)

    rows = [
        summary_row("100k_product_count", index_100k.get("total_products", ""), "data/esci_index/index_summary.json"),
        summary_row("500k_product_count", index_500k.get("total_products", ""), "data/esci_index_500k/index_summary.json"),
        summary_row("500k_examples_loaded", index_500k.get("total_example_rows_loaded", ""), "data/esci_index_500k/index_summary.json"),
        summary_row("100k_vs_500k_exact_delta", metric_delta(raw_summary, "top_k_has_exact_rate"), "scale_comparison_summary.csv"),
        summary_row(
            "100k_vs_500k_exact_or_sub_delta",
            metric_delta(raw_summary, "top_k_has_exact_or_substitute_rate"),
            "scale_comparison_summary.csv",
        ),
        summary_row("100k_vs_500k_strict_any_violation_delta", metric_delta(raw_summary, "strict_any_violation_rate"), "scale_comparison_summary.csv"),
        summary_row("100k_vs_500k_strict_hard_violation_delta", metric_delta(raw_summary, "strict_hard_violation_rate"), "scale_comparison_summary.csv"),
        summary_row("100k_vs_500k_strict_soft_violation_delta", metric_delta(raw_summary, "strict_soft_violation_rate"), "scale_comparison_summary.csv"),
        summary_row("100k_vs_500k_runtime_delta", metric_delta(raw_summary, "runtime_seconds"), "scale_comparison_summary.csv"),
        summary_row("100k_vs_500k_strict_boost_exact_delta", metric_delta(strict_summary, "top_k_has_exact_rate"), "scale_comparison_third_summary.csv"),
        summary_row(
            "100k_vs_500k_strict_boost_exact_or_sub_delta",
            metric_delta(strict_summary, "top_k_has_exact_or_substitute_rate"),
            "scale_comparison_third_summary.csv",
        ),
        summary_row("500k_strict_boost_vs_500k_raw_improved", raw_vs_boost["improved"], "row-level comparison"),
        summary_row("500k_strict_boost_vs_500k_raw_regressed", raw_vs_boost["regressed"], "row-level comparison"),
        summary_row("500k_strict_boost_vs_500k_raw_unchanged", raw_vs_boost["unchanged"], "row-level comparison"),
        summary_row("strict_boost_applied_count", metric_right_value(strict_summary, "strict_boost_applied_count"), "scale_comparison_third_summary.csv"),
        summary_row("reranked_candidate_count", metric_right_value(strict_summary, "reranked_candidate_count"), "scale_comparison_third_summary.csv"),
        summary_row("hard_violation_penalty_count", metric_right_value(strict_summary, "hard_violation_penalty_count"), "scale_comparison_third_summary.csv"),
        summary_row("soft_violation_penalty_count", metric_right_value(strict_summary, "soft_violation_penalty_count"), "scale_comparison_third_summary.csv"),
    ]
    context = {
        "index_100k": index_100k,
        "index_500k": index_500k,
        "raw_summary": raw_summary,
        "strict_summary": strict_summary,
        "raw_vs_boost": raw_vs_boost,
    }
    return rows, context


def find_key(row: Dict[str, str], suffix: str, fallback: str = "") -> str:
    if fallback and fallback in row:
        return fallback
    for key in row:
        if key.endswith(suffix):
            return key
    return ""


def normalize_route_row(row: Dict[str, str], comparison: str) -> Dict[str, object]:
    left_count_key = find_key(row, "_query_count", "100k_query_count")
    right_count_keys = [key for key in row if key.endswith("_query_count") and key != left_count_key]
    right_count_key = right_count_keys[0] if right_count_keys else ""
    left_exact_key = find_key(row, "_top_k_has_exact_rate", "100k_top_k_has_exact_rate")
    right_exact_key = [key for key in row if key.endswith("_top_k_has_exact_rate") and key != left_exact_key]
    left_eos_key = find_key(row, "_top_k_has_exact_or_substitute_rate", "100k_top_k_has_exact_or_substitute_rate")
    right_eos_key = [key for key in row if key.endswith("_top_k_has_exact_or_substitute_rate") and key != left_eos_key]
    return {
        "comparison": comparison,
        "calibrated_governance_route": row.get("calibrated_governance_route", ""),
        "execution_source": row.get("execution_source", ""),
        "query_count_left": row.get(left_count_key, ""),
        "query_count_right": row.get(right_count_key, "") if right_count_key else "",
        "exact_rate_left": row.get(left_exact_key, ""),
        "exact_rate_right": row.get(right_exact_key[0], "") if right_exact_key else "",
        "exact_rate_delta": row.get("top_k_has_exact_rate_delta", ""),
        "exact_or_sub_rate_left": row.get(left_eos_key, ""),
        "exact_or_sub_rate_right": row.get(right_eos_key[0], "") if right_eos_key else "",
        "exact_or_sub_rate_delta": row.get("top_k_has_exact_or_substitute_rate_delta", ""),
    }


def build_route_table(args: argparse.Namespace) -> List[Dict[str, object]]:
    raw_rows = read_csv(Path(args.scale_comparison_dir) / "scale_comparison_route_deltas.csv")
    strict_rows = read_csv(Path(args.strict_boost_comparison_dir) / "scale_comparison_third_route_deltas.csv")
    output = [normalize_route_row(row, "100k_vs_500k_raw") for row in raw_rows]
    output.extend(normalize_route_row(row, "100k_vs_500k_strict_boost") for row in strict_rows)
    return output


def normalize_query_example(row: Dict[str, str], comparison: str, right_name: str) -> Dict[str, object]:
    return {
        "comparison": comparison,
        "delta_status": row.get("delta_status", ""),
        "query": row.get("query", ""),
        "route_100k": row.get("100k_route", ""),
        "route_500k": row.get(f"{right_name}_route", ""),
        "exact_100k": row.get("100k_top_k_has_exact", ""),
        "exact_500k": row.get(f"{right_name}_top_k_has_exact", ""),
        "exact_or_sub_100k": row.get("100k_top_k_has_exact_or_substitute", ""),
        "exact_or_sub_500k": row.get(f"{right_name}_top_k_has_exact_or_substitute", ""),
        "best_label_100k": row.get("100k_best_esci_label", ""),
        "best_label_500k": row.get(f"{right_name}_best_esci_label", ""),
        "outcome_score_delta": row.get("outcome_score_delta", ""),
    }


def select_examples(rows: List[Dict[str, str]], comparison: str, right_name: str, limit_per_status: int = 5) -> List[Dict[str, object]]:
    output: List[Dict[str, object]] = []
    for status in ("improved", "regressed"):
        status_rows = [row for row in rows if row.get("delta_status") == status]
        status_rows.sort(key=lambda row: (safe_float(row.get("outcome_score_delta")), row.get("query", "")), reverse=(status == "improved"))
        for row in status_rows[:limit_per_status]:
            output.append(normalize_query_example(row, comparison, right_name))
    return output


def build_query_examples(args: argparse.Namespace) -> List[Dict[str, object]]:
    raw_rows = read_csv(Path(args.scale_comparison_dir) / "scale_comparison_query_deltas.csv")
    strict_rows = read_csv(Path(args.strict_boost_comparison_dir) / "scale_comparison_third_query_deltas.csv")
    output = select_examples(raw_rows, "100k_vs_500k_raw", "500k")
    output.extend(select_examples(strict_rows, "100k_vs_500k_strict_boost", "500k_strict_boost"))
    return output


def get_summary_value(summary_rows: List[Dict[str, object]], metric: str) -> str:
    for row in summary_rows:
        if row.get("metric") == metric:
            return clean_text(row.get("value"))
    return ""


def report_markdown(
    summary_rows: List[Dict[str, object]],
    route_rows: List[Dict[str, object]],
    query_rows: List[Dict[str, object]],
    context: Dict[str, object],
    args: argparse.Namespace,
) -> str:
    raw_query_counts = count_status(read_csv(Path(args.scale_comparison_dir) / "scale_comparison_query_deltas.csv"))
    strict_query_counts = count_status(read_csv(Path(args.strict_boost_comparison_dir) / "scale_comparison_third_query_deltas.csv"))
    raw_vs_boost = context.get("raw_vs_boost", {"improved": 0, "regressed": 0, "unchanged": 0})

    lines = [
        "# Full ESCI Route-Aware Retrieval Benchmark Report",
        "",
        "## Executive Summary",
        "Scaling the local ESCI FTS index from 100k examples to 500k examples increased candidate coverage but did not automatically improve route-aware retrieval quality on the current benchmark slice.",
        "The experimental strict_boost reranker was active, but it did not recover the lost quality, which supports the research claim that governed route-aware optimization requires more than naive scale and penalty-based filtering.",
        "",
        "## Dataset and Scale",
        f"- 100k products: {get_summary_value(summary_rows, '100k_product_count')}",
        f"- 500k products: {get_summary_value(summary_rows, '500k_product_count')}",
        f"- 500k examples loaded: {get_summary_value(summary_rows, '500k_examples_loaded')}",
        "",
        "## Benchmark Setup",
        "- Retrieval backend: SQLite FTS5.",
        "- Comparisons: 100k baseline vs 500k raw; 100k baseline vs 500k strict_boost.",
        "- Evaluation metric: ESCI label hit rates over top-k route-generated slates.",
        "- This report reads existing benchmark outputs only; it does not rerun evaluation.",
        "",
        "## Main Results",
        f"- 100k vs 500k exact delta: {get_summary_value(summary_rows, '100k_vs_500k_exact_delta')}",
        f"- 100k vs 500k exact_or_sub delta: {get_summary_value(summary_rows, '100k_vs_500k_exact_or_sub_delta')}",
        f"- 100k vs 500k strict_boost exact delta: {get_summary_value(summary_rows, '100k_vs_500k_strict_boost_exact_delta')}",
        f"- 100k vs 500k strict_boost exact_or_sub delta: {get_summary_value(summary_rows, '100k_vs_500k_strict_boost_exact_or_sub_delta')}",
        f"- 500k raw query outcomes: improved={raw_query_counts['improved']}, regressed={raw_query_counts['regressed']}, unchanged={raw_query_counts['unchanged']}",
        f"- 500k strict_boost query outcomes: improved={strict_query_counts['improved']}, regressed={strict_query_counts['regressed']}, unchanged={strict_query_counts['unchanged']}",
        f"- 500k strict_boost vs 500k raw: improved={raw_vs_boost['improved']}, regressed={raw_vs_boost['regressed']}, unchanged={raw_vs_boost['unchanged']}",
        "",
        "## Route-Level Analysis",
    ]
    for row in route_rows:
        lines.append(
            f"- {row.get('comparison')} / {row.get('calibrated_governance_route')}: "
            f"exact_delta={row.get('exact_rate_delta')}, exact_or_sub_delta={row.get('exact_or_sub_rate_delta')}"
        )
    lines.extend(["", "## Query-Level Examples"])
    for row in query_rows[:20]:
        lines.append(
            f"- {row.get('comparison')} {row.get('delta_status')}: {row.get('query')} "
            f"({row.get('route_100k')} -> {row.get('route_500k')}, delta={row.get('outcome_score_delta')})"
        )
    lines.extend(
        [
            "",
            "## Key Finding",
            "The 500k index introduces additional near matches that can harm strict and critic-review routes. The strict_boost experiment reduced some violation signals, but it also demoted or displaced labeled exact/substitute products, so simple forbidden-token penalties are not sufficient.",
            "",
            "## Paper-Ready Interpretation",
            "These results motivate CORTEX as a governed route-aware ranking system: retrieval quality depends on how query constraints, governance routes, and candidate generation interact. Larger lexical indexes and simple constraint penalties are useful infrastructure, but they are not enough to optimize route-specific slate quality.",
            "",
            "## Next Step",
            "Use these artifacts to motivate a learned or evaluator-guided route optimization stage that tunes candidate blending by route and query type, while preserving the existing guardrail metrics for strict constraints.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_report(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_rows, context = build_summary(args)
    route_rows = build_route_table(args)
    query_rows = build_query_examples(args)

    write_csv(output_dir / "paper_benchmark_summary.csv", summary_rows, SUMMARY_FIELDS)
    write_csv(output_dir / "paper_benchmark_route_table.csv", route_rows, ROUTE_FIELDS)
    write_csv(output_dir / "paper_benchmark_query_examples.csv", query_rows, QUERY_EXAMPLE_FIELDS)
    (output_dir / "paper_benchmark_report.md").write_text(
        report_markdown(summary_rows, route_rows, query_rows, context, args),
        encoding="utf-8",
    )

    print("\nMVP 26.11 Full ESCI Paper Benchmark Report")
    print("-" * 80)
    for row in summary_rows:
        print(f"{row['metric']}: {row['value']}")
    print("\nOutput files")
    print(output_dir / "paper_benchmark_summary.csv")
    print(output_dir / "paper_benchmark_route_table.csv")
    print(output_dir / "paper_benchmark_query_examples.csv")
    print(output_dir / "paper_benchmark_report.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MVP 26.11 paper-ready ESCI benchmark report generator")
    parser.add_argument("--scale-comparison-dir", default="outputs/full_esci_index_scale_comparison")
    parser.add_argument("--strict-boost-comparison-dir", default="outputs/full_esci_index_scale_comparison_strict_boost")
    parser.add_argument("--raw-500k-eval-dir", default="outputs/full_esci_route_retrieval_eval_500k")
    parser.add_argument("--strict-boost-eval-dir", default="outputs/full_esci_route_retrieval_eval_500k_strict_boost")
    parser.add_argument("--index-100k-dir", default="data/esci_index")
    parser.add_argument("--index-500k-dir", default="data/esci_index_500k")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    run_report(parse_args())


if __name__ == "__main__":
    main()
