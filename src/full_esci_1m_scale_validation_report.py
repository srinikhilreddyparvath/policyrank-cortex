"""
MVP 26.12: Reproducible 1M full ESCI scale validation report.

This script does not build indexes or rerun evaluations. It reads existing
1M-scale artifacts when present and writes compact validation tables for paper
and experiment tracking.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
from pathlib import Path
from typing import Dict, List


DEFAULT_OUTPUT_DIR = Path("outputs/full_esci_paper_benchmark_report_1m")

SUMMARY_FIELDS = ["metric", "value", "source", "status"]
EVAL_METRICS = [
    "total_queries",
    "success_count",
    "top_k_has_exact_rate",
    "top_k_has_exact_or_substitute_rate",
    "fallback_rate",
    "strict_any_violation_rate",
    "strict_hard_violation_rate",
    "strict_soft_violation_rate",
    "runtime_seconds",
]


def clean_text(value: object) -> str:
    return str(value or "").strip()


def read_json(path: Path) -> Dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}


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


def summary_row(metric: str, value: object, source: Path | str, status: str = "available") -> Dict[str, object]:
    return {
        "metric": metric,
        "value": value,
        "source": str(source),
        "status": status,
    }


def missing_row(metric: str, source: Path | str) -> Dict[str, object]:
    return summary_row(metric, "", source, status="missing")


def sqlite_count(sqlite_path: Path, table: str) -> str:
    if not sqlite_path.exists():
        return ""
    try:
        with sqlite3.connect(sqlite_path) as conn:
            return str(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])
    except Exception:
        return ""


def load_eval_summary(eval_dir: Path) -> Dict[str, str]:
    rows = read_csv(eval_dir / "route_retrieval_summary.csv")
    return rows[0] if rows else {}


def metric_rows(path: Path) -> Dict[str, Dict[str, str]]:
    rows = read_csv(path)
    return {clean_text(row.get("metric")): row for row in rows if clean_text(row.get("metric"))}


def add_eval_rows(rows: List[Dict[str, object]], prefix: str, eval_dir: Path) -> None:
    summary = load_eval_summary(eval_dir)
    if not summary:
        for metric in EVAL_METRICS:
            rows.append(missing_row(f"{prefix}_{metric}", eval_dir / "route_retrieval_summary.csv"))
        return
    for metric in EVAL_METRICS:
        rows.append(summary_row(f"{prefix}_{metric}", summary.get(metric, ""), eval_dir / "route_retrieval_summary.csv"))


def add_comparison_rows(rows: List[Dict[str, object]], comparison_dir: Path) -> None:
    summary_path = comparison_dir / "scale_comparison_summary.csv"
    summary = metric_rows(summary_path)
    if not summary:
        rows.append(missing_row("comparison_to_500k", summary_path))
        return
    for metric in (
        "top_k_has_exact_rate",
        "top_k_has_exact_or_substitute_rate",
        "avg_label_coverage_rate",
        "fallback_rate",
        "strict_any_violation_rate",
        "strict_hard_violation_rate",
        "strict_soft_violation_rate",
        "runtime_seconds",
    ):
        rows.append(summary_row(f"comparison_500k_vs_1m_{metric}_delta", summary.get(metric, {}).get("delta", ""), summary_path))


def build_summary(args: argparse.Namespace) -> List[Dict[str, object]]:
    index_dir = Path(args.index_1m_dir)
    index_summary_path = index_dir / "index_summary.json"
    index_summary = read_json(index_summary_path)
    sqlite_path = index_dir / "esci_products_fts.sqlite"
    rows: List[Dict[str, object]] = []

    rows.append(summary_row("requested_max_rows", 1000000, "MVP 26.12 convention"))
    if index_summary:
        rows.extend(
            [
                summary_row("actual_examples_loaded", index_summary.get("total_example_rows_loaded", ""), index_summary_path),
                summary_row("actual_product_count", index_summary.get("total_products", ""), index_summary_path),
                summary_row("query_count", index_summary.get("total_queries", ""), index_summary_path),
                summary_row("query_product_pair_count", index_summary.get("total_query_product_pairs", ""), index_summary_path),
            ]
        )
    else:
        rows.extend(
            [
                missing_row("actual_examples_loaded", index_summary_path),
                missing_row("actual_product_count", index_summary_path),
                missing_row("query_count", index_summary_path),
                missing_row("query_product_pair_count", index_summary_path),
            ]
        )

    if sqlite_path.exists():
        rows.append(summary_row("sqlite_file_path", sqlite_path, sqlite_path))
        rows.append(summary_row("sqlite_file_size_bytes", sqlite_path.stat().st_size, sqlite_path))
        rows.append(summary_row("fts_products_count", sqlite_count(sqlite_path, "products"), sqlite_path))
        rows.append(summary_row("fts_table_exists", "1" if sqlite_count(sqlite_path, "products_fts") else "0", sqlite_path))
    else:
        rows.extend(
            [
                missing_row("sqlite_file_path", sqlite_path),
                missing_row("sqlite_file_size_bytes", sqlite_path),
                missing_row("fts_products_count", sqlite_path),
                missing_row("fts_table_exists", sqlite_path),
            ]
        )

    add_eval_rows(rows, "eval_1m", Path(args.eval_1m_dir))
    add_eval_rows(rows, "eval_1m_strict_boost", Path(args.eval_1m_strict_boost_dir))
    add_comparison_rows(rows, Path(args.comparison_1m_dir))
    return rows


def get_value(rows: List[Dict[str, object]], metric: str) -> str:
    for row in rows:
        if row.get("metric") == metric:
            return clean_text(row.get("value"))
    return ""


def report_markdown(rows: List[Dict[str, object]], args: argparse.Namespace) -> str:
    missing = [row for row in rows if row.get("status") == "missing"]
    lines = [
        "# MVP 26.12 1M Full ESCI Scale Validation",
        "",
        "## Purpose",
        "This report validates whether 1M-scale ESCI artifacts are present and summarizes their retrieval benchmark metrics without rerunning heavy jobs.",
        "",
        "## Index Status",
        f"- requested max rows: {get_value(rows, 'requested_max_rows')}",
        f"- actual examples loaded: {get_value(rows, 'actual_examples_loaded') or 'missing'}",
        f"- actual product count: {get_value(rows, 'actual_product_count') or 'missing'}",
        f"- query count: {get_value(rows, 'query_count') or 'missing'}",
        f"- FTS products count: {get_value(rows, 'fts_products_count') or 'missing'}",
        f"- SQLite file: {get_value(rows, 'sqlite_file_path') or 'missing'}",
        f"- SQLite file size bytes: {get_value(rows, 'sqlite_file_size_bytes') or 'missing'}",
        "",
        "## 1M Route Evaluation",
    ]
    for metric in EVAL_METRICS:
        lines.append(f"- {metric}: {get_value(rows, f'eval_1m_{metric}') or 'missing'}")
    lines.extend(["", "## 1M Strict Boost Evaluation"])
    for metric in EVAL_METRICS:
        lines.append(f"- {metric}: {get_value(rows, f'eval_1m_strict_boost_{metric}') or 'missing'}")
    lines.extend(["", "## 500k vs 1M Comparison"])
    comparison_metrics = [row for row in rows if clean_text(row.get("metric")).startswith("comparison_500k_vs_1m_")]
    if comparison_metrics:
        for row in comparison_metrics:
            lines.append(f"- {row.get('metric')}: {row.get('value') or 'missing'}")
    else:
        lines.append("- missing")
    lines.extend(["", "## Missing Artifacts"])
    if missing:
        for row in missing:
            lines.append(f"- {row.get('metric')}: {row.get('source')}")
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Reproducibility Notes",
            f"- index_1m_dir: {args.index_1m_dir}",
            f"- eval_1m_dir: {args.eval_1m_dir}",
            f"- eval_1m_strict_boost_dir: {args.eval_1m_strict_boost_dir}",
            f"- comparison_1m_dir: {args.comparison_1m_dir}",
            "- Heavy build and evaluation commands are intentionally not run by this report script.",
        ]
    )
    return "\n".join(lines) + "\n"


def run_report(args: argparse.Namespace) -> None:
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = build_summary(args)
    write_csv(output_dir / "scale_1m_validation_summary.csv", rows, SUMMARY_FIELDS)
    (output_dir / "scale_1m_validation_report.md").write_text(report_markdown(rows, args), encoding="utf-8")

    print("\nMVP 26.12 1M Full ESCI Scale Validation Report")
    print("-" * 80)
    for row in rows:
        print(f"{row['metric']}: {row['value']} [{row['status']}]")
    print("\nOutput files")
    print(output_dir / "scale_1m_validation_summary.csv")
    print(output_dir / "scale_1m_validation_report.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MVP 26.12 1M ESCI scale validation report")
    parser.add_argument("--index-1m-dir", default="data/esci_index_1m")
    parser.add_argument("--eval-1m-dir", default="outputs/full_esci_route_retrieval_eval_1m")
    parser.add_argument("--eval-1m-strict-boost-dir", default="outputs/full_esci_route_retrieval_eval_1m_strict_boost")
    parser.add_argument("--comparison-1m-dir", default="outputs/full_esci_index_scale_comparison_1m")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    return parser.parse_args()


def main() -> None:
    run_report(parse_args())


if __name__ == "__main__":
    main()
