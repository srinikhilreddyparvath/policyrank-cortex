"""
MVP 26.9: 100k vs 500k full ESCI route-aware retrieval comparison.

Runs the existing route retrieval evaluator logic against two ESCI indexes using
the same deterministic query list, then writes summary/query/route deltas.
"""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path
from typing import Dict, List, Tuple

from src.full_esci_route_retrieval_evaluator import (
    aggregate_summary,
    clean_text,
    grouped_summary,
    load_label_lookup,
    read_csv,
    run_query,
    safe_float,
    safe_int,
    write_outputs as write_evaluator_outputs,
    write_csv,
)


DEFAULT_OUTPUT_DIR = Path("outputs/full_esci_index_scale_comparison")
SUMMARY_METRICS = [
    "total_queries",
    "success_count",
    "failure_count",
    "avg_final_slate_size",
    "top_k_has_exact_rate",
    "top_k_has_exact_or_substitute_rate",
    "avg_label_coverage_rate",
    "fallback_rate",
    "strict_any_violation_rate",
    "strict_hard_violation_rate",
    "strict_soft_violation_rate",
    "strict_boost_applied_count",
    "reranked_candidate_count",
    "hard_violation_penalty_count",
    "soft_violation_penalty_count",
    "runtime_seconds",
]


def normalized_query(value: object) -> str:
    return " ".join(clean_text(value).lower().split())


def load_indexed_queries(index_dir: Path, sample_size: int) -> List[str]:
    rows = read_csv(index_dir / "queries.csv")
    queries = [clean_text(row.get("query")) for row in rows if clean_text(row.get("query"))]
    return queries[:sample_size]


def label_score(label: object) -> int:
    return {"E": 4, "S": 3, "C": 2, "I": 1}.get(clean_text(label).upper(), 0)


def outcome_score(row: Dict[str, object]) -> int:
    if safe_int(row.get("top_k_has_exact")):
        return 3
    if safe_int(row.get("top_k_has_exact_or_substitute")):
        return 2
    return label_score(row.get("best_esci_label"))


def run_side(
    name: str,
    index_dir: Path,
    queries: List[str],
    output_dir: Path,
    args: argparse.Namespace,
    scale_aware_rerank_mode: str,
) -> Tuple[List[Dict[str, object]], List[Dict[str, object]], Dict[str, object], List[Dict[str, object]]]:
    label_lookup = load_label_lookup(index_dir)
    result_rows: List[Dict[str, object]] = []
    query_eval_rows: List[Dict[str, object]] = []
    start = time.perf_counter()

    print(f"\nRunning {name} index")
    print("-" * 80)
    print(f"index_dir: {index_dir}")
    print(f"query_count: {len(queries)}")
    print(f"scale_aware_rerank_mode: {scale_aware_rerank_mode}")

    for index, query in enumerate(queries, start=1):
        try:
            row, eval_rows = run_query(
                query=query,
                output_dir=output_dir,
                index_dir=index_dir,
                top_k=args.top_k,
                retrieval_mode="full_esci",
                retrieval_backend=args.retrieval_backend,
                strict_filter_mode=args.strict_filter_mode,
                strict_min_clean_results=args.strict_min_clean_results,
                strict_candidate_multiplier=args.strict_candidate_multiplier,
                scale_aware_rerank_mode=scale_aware_rerank_mode,
                label_lookup=label_lookup,
            )
            result_rows.append(row)
            query_eval_rows.extend(eval_rows)
            if index <= 5 or index == len(queries):
                print(
                    f"[{index}/{len(queries)}] {query} -> "
                    f"exact={row.get('top_k_has_exact')} exact_or_sub={row.get('top_k_has_exact_or_substitute')} "
                    f"route={row.get('calibrated_governance_route')}"
                )
        except Exception as exc:
            result_rows.append({"query": query, "error_message": str(exc)[:1000]})

    runtime_seconds = time.perf_counter() - start
    summary = aggregate_summary(result_rows, runtime_seconds=runtime_seconds)
    route_rows = grouped_summary(result_rows, ["calibrated_governance_route", "execution_source"])
    write_evaluator_outputs(
        output_dir=output_dir,
        result_rows=result_rows,
        query_eval_rows=query_eval_rows,
        summary=summary,
        query_type_rows=[],
        route_source_rows=route_rows,
        issues=[],
    )
    return result_rows, query_eval_rows, summary, route_rows


def summary_delta(left_name: str, right_name: str, left: Dict[str, object], right: Dict[str, object]) -> List[Dict[str, object]]:
    rows = []
    for metric in SUMMARY_METRICS:
        left_value = left.get(metric, "")
        right_value = right.get(metric, "")
        rows.append(
            {
                "metric": metric,
                f"{left_name}_value": left_value,
                f"{right_name}_value": right_value,
                "delta": round(safe_float(right_value) - safe_float(left_value), 6),
            }
        )
    return rows


def query_deltas(left_name: str, right_name: str, left_rows: List[Dict[str, object]], right_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    right_by_query = {normalized_query(row.get("query")): row for row in right_rows}
    rows = []
    for left in left_rows:
        key = normalized_query(left.get("query"))
        right = right_by_query.get(key, {})
        left_score = outcome_score(left)
        right_score = outcome_score(right)
        if right_score > left_score:
            status = "improved"
        elif right_score < left_score:
            status = "regressed"
        else:
            status = "unchanged"
        rows.append(
            {
                "query": left.get("query"),
                "delta_status": status,
                f"{left_name}_route": left.get("calibrated_governance_route"),
                f"{right_name}_route": right.get("calibrated_governance_route", ""),
                f"{left_name}_top_k_has_exact": left.get("top_k_has_exact"),
                f"{right_name}_top_k_has_exact": right.get("top_k_has_exact", ""),
                f"{left_name}_top_k_has_exact_or_substitute": left.get("top_k_has_exact_or_substitute"),
                f"{right_name}_top_k_has_exact_or_substitute": right.get("top_k_has_exact_or_substitute", ""),
                f"{left_name}_best_esci_label": left.get("best_esci_label"),
                f"{right_name}_best_esci_label": right.get("best_esci_label", ""),
                "outcome_score_delta": right_score - left_score,
            }
        )
    return rows


def route_key(row: Dict[str, object]) -> str:
    return f"{clean_text(row.get('calibrated_governance_route'))}|{clean_text(row.get('execution_source'))}"


def route_deltas(left_name: str, right_name: str, left_rows: List[Dict[str, object]], right_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    left_by_key = {route_key(row): row for row in left_rows}
    right_by_key = {route_key(row): row for row in right_rows}
    output = []
    for key in sorted(set(left_by_key) | set(right_by_key)):
        left = left_by_key.get(key, {})
        right = right_by_key.get(key, {})
        route, source = key.split("|", 1)
        output.append(
            {
                "calibrated_governance_route": route,
                "execution_source": source,
                f"{left_name}_query_count": left.get("query_count", 0),
                f"{right_name}_query_count": right.get("query_count", 0),
                f"{left_name}_top_k_has_exact_rate": left.get("top_k_has_exact_rate", 0),
                f"{right_name}_top_k_has_exact_rate": right.get("top_k_has_exact_rate", 0),
                "top_k_has_exact_rate_delta": round(
                    safe_float(right.get("top_k_has_exact_rate")) - safe_float(left.get("top_k_has_exact_rate")),
                    6,
                ),
                f"{left_name}_top_k_has_exact_or_substitute_rate": left.get("top_k_has_exact_or_substitute_rate", 0),
                f"{right_name}_top_k_has_exact_or_substitute_rate": right.get("top_k_has_exact_or_substitute_rate", 0),
                "top_k_has_exact_or_substitute_rate_delta": round(
                    safe_float(right.get("top_k_has_exact_or_substitute_rate"))
                    - safe_float(left.get("top_k_has_exact_or_substitute_rate")),
                    6,
                ),
            }
        )
    return output


def write_report(
    path: Path,
    args: argparse.Namespace,
    summary_rows: List[Dict[str, object]],
    query_rows: List[Dict[str, object]],
    route_rows: List[Dict[str, object]],
    third_summary_rows: List[Dict[str, object]] | None = None,
    third_query_rows: List[Dict[str, object]] | None = None,
    third_route_rows: List[Dict[str, object]] | None = None,
) -> None:
    counts = {
        "improved": sum(1 for row in query_rows if row.get("delta_status") == "improved"),
        "regressed": sum(1 for row in query_rows if row.get("delta_status") == "regressed"),
        "unchanged": sum(1 for row in query_rows if row.get("delta_status") == "unchanged"),
    }
    lines = [
        "# Full ESCI Index Scale Comparison",
        "",
        f"- left baseline: {args.left_name} ({args.left_index_dir}) rerank=none",
        f"- right raw: {args.right_name} ({args.right_index_dir}) rerank={args.right_rerank_mode}",
        f"- optional third strict boost: {args.third_name} ({args.third_index_dir}) rerank={args.third_rerank_mode}",
        f"- right_rerank_mode: {args.right_rerank_mode}",
        f"- include_third: {args.include_third}",
        f"- sample_size: {args.sample_size}",
        f"- query_mode: {args.query_mode}",
        f"- top_k: {args.top_k}",
        f"- retrieval_backend: {args.retrieval_backend}",
        "",
        "## Query Outcomes",
        f"- improved: {counts['improved']}",
        f"- regressed: {counts['regressed']}",
        f"- unchanged: {counts['unchanged']}",
        "",
        "## Summary Deltas",
    ]
    for row in summary_rows:
        lines.append(f"- {row['metric']}: delta={row['delta']}")
    lines.extend(["", "## Route Deltas"])
    for row in route_rows:
        lines.append(
            f"- {row['calibrated_governance_route']} / {row['execution_source']}: "
            f"exact_delta={row['top_k_has_exact_rate_delta']}, "
            f"exact_or_sub_delta={row['top_k_has_exact_or_substitute_rate_delta']}"
        )
    if third_summary_rows is not None:
        lines.extend(["", "## Third Comparison Summary Deltas"])
        for row in third_summary_rows:
            lines.append(f"- {row['metric']}: delta={row['delta']}")
    if third_query_rows is not None:
        counts = {
            "improved": sum(1 for row in third_query_rows if row.get("delta_status") == "improved"),
            "regressed": sum(1 for row in third_query_rows if row.get("delta_status") == "regressed"),
            "unchanged": sum(1 for row in third_query_rows if row.get("delta_status") == "unchanged"),
        }
        lines.extend(
            [
                "",
                f"## Third Query Outcomes ({args.third_name})",
                f"- improved: {counts['improved']}",
                f"- regressed: {counts['regressed']}",
                f"- unchanged: {counts['unchanged']}",
            ]
        )
    if third_route_rows is not None:
        lines.extend(["", f"## Third Route Deltas ({args.third_name})"])
        for row in third_route_rows:
            lines.append(
                f"- {row['calibrated_governance_route']} / {row['execution_source']}: "
                f"exact_delta={row['top_k_has_exact_rate_delta']}, "
                f"exact_or_sub_delta={row['top_k_has_exact_or_substitute_rate_delta']}"
            )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_comparison(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    left_dir = Path(args.left_index_dir)
    right_dir = Path(args.right_index_dir)
    queries = load_indexed_queries(left_dir, args.sample_size)
    if args.query_mode != "indexed_queries":
        raise ValueError("MVP 26.9 comparison currently supports --query-mode indexed_queries.")
    if not queries:
        raise RuntimeError(f"No indexed queries found in {left_dir / 'queries.csv'}")

    print("\nFull ESCI Index Scale Comparison")
    print("-" * 80)
    print(f"left baseline: {args.left_name} -> {left_dir} rerank=none")
    print(f"right raw: {args.right_name} -> {right_dir} rerank={args.right_rerank_mode}")
    if args.include_third:
        print(f"third strict boost: {args.third_name} -> {args.third_index_dir} rerank={args.third_rerank_mode}")
    print(f"same_query_count: {len(queries)}")

    left_rows, _left_eval_rows, left_summary, left_route_rows = run_side(
        name=args.left_name,
        index_dir=left_dir,
        queries=queries,
        output_dir=output_dir / args.left_name,
        args=args,
        scale_aware_rerank_mode="none",
    )
    right_rows, _right_eval_rows, right_summary, right_route_rows = run_side(
        name=args.right_name,
        index_dir=right_dir,
        queries=queries,
        output_dir=output_dir / args.right_name,
        args=args,
        scale_aware_rerank_mode=args.right_rerank_mode,
    )

    summary_rows = summary_delta(args.left_name, args.right_name, left_summary, right_summary)
    query_rows = query_deltas(args.left_name, args.right_name, left_rows, right_rows)
    route_rows = route_deltas(args.left_name, args.right_name, left_route_rows, right_route_rows)
    third_summary_rows = None
    third_query_rows = None
    third_route_rows = None
    if args.include_third:
        third_rows, _third_eval_rows, third_summary, third_route_source_rows = run_side(
            name=args.third_name,
            index_dir=Path(args.third_index_dir),
            queries=queries,
            output_dir=output_dir / args.third_name,
            args=args,
            scale_aware_rerank_mode=args.third_rerank_mode,
        )
        third_summary_rows = summary_delta(args.left_name, args.third_name, left_summary, third_summary)
        third_query_rows = query_deltas(args.left_name, args.third_name, left_rows, third_rows)
        third_route_rows = route_deltas(args.left_name, args.third_name, left_route_rows, third_route_source_rows)

    write_csv(output_dir / "scale_comparison_summary.csv", summary_rows, list(summary_rows[0].keys()))
    write_csv(output_dir / "scale_comparison_query_deltas.csv", query_rows, list(query_rows[0].keys()) if query_rows else ["query"])
    write_csv(output_dir / "scale_comparison_route_deltas.csv", route_rows, list(route_rows[0].keys()) if route_rows else ["route"])
    if third_summary_rows is not None:
        write_csv(output_dir / "scale_comparison_third_summary.csv", third_summary_rows, list(third_summary_rows[0].keys()))
        write_csv(
            output_dir / "scale_comparison_third_query_deltas.csv",
            third_query_rows or [],
            list(third_query_rows[0].keys()) if third_query_rows else ["query"],
        )
        write_csv(
            output_dir / "scale_comparison_third_route_deltas.csv",
            third_route_rows or [],
            list(third_route_rows[0].keys()) if third_route_rows else ["route"],
        )
    write_report(
        output_dir / "scale_comparison_report.md",
        args,
        summary_rows,
        query_rows,
        route_rows,
        third_summary_rows=third_summary_rows,
        third_query_rows=third_query_rows,
        third_route_rows=third_route_rows,
    )

    print("\nComparison summary")
    print("-" * 80)
    for row in summary_rows:
        print(f"{row['metric']}: delta={row['delta']}")
    print("\nOutput files")
    print(output_dir / "scale_comparison_summary.csv")
    print(output_dir / "scale_comparison_query_deltas.csv")
    print(output_dir / "scale_comparison_route_deltas.csv")
    if third_summary_rows is not None:
        print(output_dir / "scale_comparison_third_summary.csv")
        print(output_dir / "scale_comparison_third_query_deltas.csv")
        print(output_dir / "scale_comparison_third_route_deltas.csv")
    print(output_dir / "scale_comparison_report.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MVP 26.9 Full ESCI index scale comparison")
    parser.add_argument("--left-name", default="100k")
    parser.add_argument("--left-index-dir", default="data/esci_index")
    parser.add_argument("--right-name", default="500k")
    parser.add_argument("--right-index-dir", default="data/esci_index_500k")
    parser.add_argument("--sample-size", type=int, default=100)
    parser.add_argument("--query-mode", default="indexed_queries")
    parser.add_argument("--top-k", type=int, default=12)
    parser.add_argument("--retrieval-backend", choices=["lexical", "fts"], default="fts")
    parser.add_argument("--right-rerank-mode", choices=["none", "strict_boost"], default="none")
    parser.add_argument("--include-third", action="store_true")
    parser.add_argument("--third-name", default="500k_strict_boost")
    parser.add_argument("--third-index-dir", default="data/esci_index_500k")
    parser.add_argument("--third-rerank-mode", choices=["none", "strict_boost"], default="strict_boost")
    parser.add_argument("--strict-filter-mode", choices=["remove", "demote", "hybrid"], default="hybrid")
    parser.add_argument("--strict-min-clean-results", type=int, default=8)
    parser.add_argument("--strict-candidate-multiplier", type=int, default=8)
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_size <= 0:
        raise ValueError("--sample-size must be positive")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    run_comparison(args)


if __name__ == "__main__":
    main()
