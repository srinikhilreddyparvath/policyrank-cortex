"""
MVP 23E: Route-Specific Slate Quality Analyzer

Reads calibrated governed CORTEX outputs and produces route/query-type slate
quality diagnostics. Analyzer-only; it does not change calibrated execution.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


DEFAULT_INPUT_DIR = Path("outputs/calibrated_governed_cortex")
DEFAULT_OUTPUT_DIR = Path("outputs/route_specific_slate_quality")

RESULT_HINTS = ["results", "by_query"]
SUMMARY_HINTS = ["summary"]
ROUTE_HINTS = ["by_route"]

MISSION_QUERY_TYPES = {"mission_query", "setup_or_kit", "gift_or_party", "broad_discovery"}
MISSION_ROUTES = {"MISSION_REPAIR", "STRICT_REPAIR", "BEHAVIOR_AWARE_RERANK", "CRITIC_REVIEW"}
NON_MISSION_QUERY_TYPES = {
    "narrow_product",
    "numeric_model",
    "general_retail",
    "compatibility_query",
    "negation_constraint",
    "noisy_query",
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


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def csv_files(input_dir: Path) -> List[Path]:
    return sorted(input_dir.glob("*.csv"), key=lambda path: path.stat().st_mtime, reverse=True)


def has_columns(path: Path, required: Iterable[str]) -> bool:
    try:
        with path.open("r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fields = set(reader.fieldnames or [])
        return set(required).issubset(fields)
    except Exception:
        return False


def latest_matching_file(input_dir: Path, required_columns: Iterable[str], hints: Iterable[str]) -> Path | None:
    candidates = []
    hints = [hint.lower() for hint in hints]
    for path in csv_files(input_dir):
        name = path.name.lower()
        if hints and not any(hint in name for hint in hints):
            continue
        if has_columns(path, required_columns):
            candidates.append(path)
    return candidates[0] if candidates else None


def locate_input_files(input_dir: Path) -> Dict[str, Path | None]:
    result_file = latest_matching_file(
        input_dir,
        required_columns=[
            "query",
            "calibrated_governance_route",
            "calibrated_final_slate_size",
            "calibrated_fallback_used",
        ],
        hints=RESULT_HINTS,
    )
    summary_file = latest_matching_file(
        input_dir,
        required_columns=["total_queries", "success_count", "failure_count"],
        hints=SUMMARY_HINTS,
    )
    route_file = latest_matching_file(
        input_dir,
        required_columns=["calibrated_governance_route", "query_count", "avg_calibrated_final_slate_size"],
        hints=ROUTE_HINTS,
    )
    return {"results": result_file, "summary": summary_file, "route": route_file}


def avg(rows: List[Dict[str, str]], field: str) -> float:
    return round(sum(safe_float(row.get(field)) for row in rows) / max(len(rows), 1), 6)


def count_true(rows: List[Dict[str, str]], field: str) -> int:
    return sum(1 for row in rows if safe_int(row.get(field)) == 1)


def distribution(rows: List[Dict[str, str]], field: str, limit: int = 6) -> str:
    counts = Counter(clean_text(row.get(field)) or "unknown" for row in rows)
    total = len(rows)
    parts = []
    for value, count in counts.most_common(limit):
        parts.append(f"{value}:{count}({round(count / max(total, 1), 4)})")
    return " | ".join(parts)


def top_value(rows: List[Dict[str, str]], field: str) -> str:
    counts = Counter(clean_text(row.get(field)) for row in rows if clean_text(row.get(field)))
    return counts.most_common(1)[0][0] if counts else ""


def quality_label(
    route: str,
    query_type: str,
    fallback_rate: float,
    avg_slate_size: float,
    avg_sub_intents: float,
) -> str:
    if avg_slate_size <= 3:
        return "weak_slate_generation"
    if fallback_rate >= 0.5:
        return "baseline_fallback_heavy"
    if query_type in NON_MISSION_QUERY_TYPES:
        return "strong_real_execution"
    if route in MISSION_ROUTES and avg_sub_intents < 1:
        return "needs_route_adapter_improvement"
    if query_type in MISSION_QUERY_TYPES and avg_sub_intents < 2:
        return "good_but_low_diversity"
    if avg_sub_intents < 1 and route not in {"BASELINE_ONLY", "REJECT_REPAIR_NARROW_QUERY"}:
        return "needs_route_adapter_improvement"
    return "strong_real_execution"


def recommended_action(issue_type: str, route: str, query_type: str) -> str:
    if route == "BASELINE_ONLY" and issue_type != "weak_slate_generation":
        return "Keep baseline-only route stable."
    if route == "STRICT_REPAIR":
        return "Improve strict repair slate generation."
    if route == "BEHAVIOR_AWARE_RERANK":
        return "Improve behavior-aware sub-intent tagging."
    if route == "MISSION_REPAIR":
        return "Improve mission slate builder integration."
    if route == "CRITIC_REVIEW":
        return "Improve critic-review adapter path or add query normalization before routing."
    if query_type in {"gift_or_party", "setup_or_kit", "mission_query"}:
        return "Add query normalization before routing."
    if issue_type == "low_mission_diversity":
        return "Improve mission slate builder integration."
    return "No action needed."


def severity_for(issue_type: str, fallback_rate: float, avg_slate_size: float, avg_sub_intents: float) -> str:
    if issue_type == "weak_slate_generation" or fallback_rate >= 0.8:
        return "high"
    if avg_sub_intents < 1 or fallback_rate >= 0.5:
        return "medium"
    return "low"


def overall_summary(rows: List[Dict[str, str]]) -> Dict[str, object]:
    total = len(rows)
    success_count = count_true(rows, "calibrated_success")
    route_changed_count = count_true(rows, "route_changed_flag")
    fallback_count = count_true(rows, "calibrated_fallback_used")
    return {
        "total_queries": total,
        "success_count": success_count,
        "failure_count": total - success_count,
        "route_changed_count": route_changed_count,
        "route_changed_rate": round(route_changed_count / max(total, 1), 6),
        "avg_current_final_slate_size": avg(rows, "current_final_slate_size"),
        "avg_calibrated_final_slate_size": avg(rows, "calibrated_final_slate_size"),
        "avg_current_unique_sub_intents": avg(rows, "current_unique_sub_intents"),
        "avg_calibrated_unique_sub_intents": avg(rows, "calibrated_unique_sub_intents"),
        "calibrated_adapter_fallback_count": fallback_count,
        "calibrated_adapter_fallback_rate": round(fallback_count / max(total, 1), 6),
    }


def group_rows(rows: List[Dict[str, str]], fields: Tuple[str, ...]) -> Dict[Tuple[str, ...], List[Dict[str, str]]]:
    groups: Dict[Tuple[str, ...], List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = tuple(clean_text(row.get(field)) or "unknown" for field in fields)
        groups[key].append(row)
    return groups


def route_summary(rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    output = []
    for (route,), group in group_rows(rows, ("calibrated_governance_route",)).items():
        fallback_count = count_true(group, "calibrated_fallback_used")
        fallback_rate = fallback_count / max(len(group), 1)
        avg_slate = avg(group, "calibrated_final_slate_size")
        avg_intents = avg(group, "calibrated_unique_sub_intents")
        output.append(
            {
                "calibrated_governance_route": route,
                "query_count": len(group),
                "success_count": count_true(group, "calibrated_success"),
                "route_changed_count": count_true(group, "route_changed_flag"),
                "avg_current_final_slate_size": avg(group, "current_final_slate_size"),
                "avg_calibrated_final_slate_size": avg_slate,
                "avg_current_unique_sub_intents": avg(group, "current_unique_sub_intents"),
                "avg_calibrated_unique_sub_intents": avg_intents,
                "fallback_count": fallback_count,
                "fallback_rate": round(fallback_rate, 6),
                "top_calibrated_execution_source": top_value(group, "calibrated_execution_source")
                or top_value(group, "calibrated_final_execution_source"),
                "execution_source_distribution": distribution(group, "calibrated_execution_source"),
                "top_fallback_reason": top_value(group, "calibrated_fallback_reason"),
                "fallback_reason_distribution": distribution(
                    [row for row in group if safe_int(row.get("calibrated_fallback_used")) == 1],
                    "calibrated_fallback_reason",
                ),
                "quality_label": quality_label(route, "", fallback_rate, avg_slate, avg_intents),
            }
        )
    return sorted(output, key=lambda row: (-safe_int(row["query_count"]), clean_text(row["calibrated_governance_route"])))


def query_type_summary(rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    output = []
    for (query_type,), group in group_rows(rows, ("query_type",)).items():
        fallback_count = count_true(group, "calibrated_fallback_used")
        fallback_rate = fallback_count / max(len(group), 1)
        avg_slate = avg(group, "calibrated_final_slate_size")
        avg_intents = avg(group, "calibrated_unique_sub_intents")
        output.append(
            {
                "query_type": query_type,
                "query_count": len(group),
                "avg_calibrated_final_slate_size": avg_slate,
                "avg_calibrated_unique_sub_intents": avg_intents,
                "fallback_count": fallback_count,
                "fallback_rate": round(fallback_rate, 6),
                "top_calibrated_route": top_value(group, "calibrated_governance_route"),
                "top_calibrated_execution_source": top_value(group, "calibrated_execution_source")
                or top_value(group, "calibrated_final_execution_source"),
                "quality_label": quality_label("", query_type, fallback_rate, avg_slate, avg_intents),
            }
        )
    return sorted(output, key=lambda row: (-safe_int(row["query_count"]), clean_text(row["query_type"])))


def route_query_type_summary(rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    output = []
    for (route, query_type), group in group_rows(rows, ("calibrated_governance_route", "query_type")).items():
        fallback_count = count_true(group, "calibrated_fallback_used")
        fallback_rate = fallback_count / max(len(group), 1)
        avg_slate = avg(group, "calibrated_final_slate_size")
        avg_intents = avg(group, "calibrated_unique_sub_intents")
        output.append(
            {
                "calibrated_governance_route": route,
                "query_type": query_type,
                "query_count": len(group),
                "avg_calibrated_final_slate_size": avg_slate,
                "avg_calibrated_unique_sub_intents": avg_intents,
                "fallback_count": fallback_count,
                "fallback_rate": round(fallback_rate, 6),
                "top_execution_source": top_value(group, "calibrated_execution_source")
                or top_value(group, "calibrated_final_execution_source"),
                "top_fallback_reason": top_value(group, "calibrated_fallback_reason"),
                "quality_label": quality_label(route, query_type, fallback_rate, avg_slate, avg_intents),
            }
        )
    return sorted(output, key=lambda row: (-safe_int(row["query_count"]), clean_text(row["calibrated_governance_route"]), clean_text(row["query_type"])))


def issue_rows(route_query_rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    issues = []
    for row in route_query_rows:
        route = clean_text(row.get("calibrated_governance_route"))
        query_type = clean_text(row.get("query_type"))
        fallback_rate = safe_float(row.get("fallback_rate"))
        avg_slate = safe_float(row.get("avg_calibrated_final_slate_size"))
        avg_intents = safe_float(row.get("avg_calibrated_unique_sub_intents"))
        query_count = safe_int(row.get("query_count"))

        candidates = []
        if fallback_rate >= 0.5:
            candidates.append(("high_fallback_rate", f"fallback_rate={fallback_rate}; fallback_count={row.get('fallback_count')}; query_count={query_count}"))
        if avg_slate <= 3:
            candidates.append(("weak_slate_generation", f"avg_calibrated_final_slate_size={avg_slate}; query_count={query_count}"))
        if route in MISSION_ROUTES and query_type in MISSION_QUERY_TYPES and avg_intents < 1:
            candidates.append(("low_mission_diversity", f"avg_calibrated_unique_sub_intents={avg_intents}; route={route}; query_type={query_type}"))

        for issue_type, evidence in candidates:
            issues.append(
                {
                    "issue_type": issue_type,
                    "calibrated_governance_route": route,
                    "query_type": query_type,
                    "severity": severity_for(issue_type, fallback_rate, avg_slate, avg_intents),
                    "evidence": evidence,
                    "recommended_next_action": recommended_action(issue_type, route, query_type),
                }
            )
    severity_order = {"high": 0, "medium": 1, "low": 2}
    return sorted(
        issues,
        key=lambda row: (
            severity_order.get(clean_text(row["severity"]), 9),
            clean_text(row["calibrated_governance_route"]),
            clean_text(row["query_type"]),
        ),
    )


def report_markdown(
    files: Dict[str, Path | None],
    overall: Dict[str, object],
    route_rows: List[Dict[str, object]],
    query_rows: List[Dict[str, object]],
    issues: List[Dict[str, object]],
) -> str:
    lines = [
        "# MVP 23E Route-Specific Slate Quality Analyzer",
        "",
        "## Input Files",
    ]
    for key, path in files.items():
        lines.append(f"- {key}: {path if path else 'not found'}")

    lines.extend(["", "## Overall Summary"])
    for key, value in overall.items():
        lines.append(f"- {key}: {value}")

    lines.extend(["", "## Top Route Quality"])
    for row in route_rows[:10]:
        lines.append(
            f"- {row['calibrated_governance_route']}: count={row['query_count']}, "
            f"fallback_rate={row['fallback_rate']}, avg_slate={row['avg_calibrated_final_slate_size']}, "
            f"avg_intents={row['avg_calibrated_unique_sub_intents']}, label={row['quality_label']}"
        )

    lines.extend(["", "## Top Query-Type Quality"])
    for row in query_rows[:10]:
        lines.append(
            f"- {row['query_type']}: count={row['query_count']}, "
            f"fallback_rate={row['fallback_rate']}, avg_slate={row['avg_calibrated_final_slate_size']}, "
            f"avg_intents={row['avg_calibrated_unique_sub_intents']}, label={row['quality_label']}"
        )

    lines.extend(["", "## Biggest Issues"])
    for row in issues[:10]:
        lines.append(
            f"- {row['severity']} {row['issue_type']} on {row['calibrated_governance_route']} / "
            f"{row['query_type']}: {row['evidence']}. Next: {row['recommended_next_action']}"
        )

    lines.extend(["", "## Recommended Next MVP", recommended_next_mvp(issues)])
    return "\n".join(lines) + "\n"


def recommended_next_mvp(issues: List[Dict[str, object]]) -> str:
    if any(clean_text(row.get("calibrated_governance_route")) == "STRICT_REPAIR" for row in issues[:10]):
        return "MVP 23F should improve strict repair adapter materialization and sub-intent tagging."
    if any(clean_text(row.get("calibrated_governance_route")) == "MISSION_REPAIR" for row in issues[:10]):
        return "MVP 23F should improve mission slate builder integration for calibrated repair routes."
    return "No major route execution change is needed; continue monitoring calibrated route quality."


def write_report(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def print_console_report(
    files: Dict[str, Path | None],
    overall: Dict[str, object],
    route_rows: List[Dict[str, object]],
    query_rows: List[Dict[str, object]],
    issues: List[Dict[str, object]],
) -> None:
    print("\nMVP 23E Route-Specific Slate Quality Analyzer")
    print("-" * 100)
    print("Input files used:")
    for key, path in files.items():
        print(f"- {key}: {path if path else 'not found'}")

    print("\nOverall summary")
    print("-" * 100)
    for key, value in overall.items():
        print(f"{key}: {value}")

    print("\nTop route quality table")
    print("-" * 100)
    for row in route_rows[:8]:
        print(
            f"{row['calibrated_governance_route']}: n={row['query_count']} "
            f"fallback={row['fallback_rate']} slate={row['avg_calibrated_final_slate_size']} "
            f"intents={row['avg_calibrated_unique_sub_intents']} label={row['quality_label']}"
        )

    print("\nTop query type quality table")
    print("-" * 100)
    for row in query_rows[:10]:
        print(
            f"{row['query_type']}: n={row['query_count']} "
            f"fallback={row['fallback_rate']} slate={row['avg_calibrated_final_slate_size']} "
            f"intents={row['avg_calibrated_unique_sub_intents']} label={row['quality_label']}"
        )

    print("\nBiggest issues")
    print("-" * 100)
    for row in issues[:10]:
        print(
            f"{row['severity']} | {row['issue_type']} | "
            f"{row['calibrated_governance_route']} / {row['query_type']} | "
            f"{row['evidence']} | {row['recommended_next_action']}"
        )

    print("\nRecommended next MVP")
    print("-" * 100)
    print(recommended_next_mvp(issues))


def run_analyzer(input_dir: Path, output_dir: Path) -> Dict[str, object]:
    files = locate_input_files(input_dir)
    result_file = files["results"]
    if result_file is None:
        raise FileNotFoundError(f"No calibrated governed row-level results CSV found under {input_dir}")

    rows = read_csv(result_file)
    if not rows:
        raise ValueError(f"Row-level results file is empty: {result_file}")

    overall = overall_summary(rows)
    route_rows = route_summary(rows)
    query_rows = query_type_summary(rows)
    route_query_rows = route_query_type_summary(rows)
    issues = issue_rows(route_query_rows)

    ensure_output_dir(output_dir)
    write_csv(output_dir / "route_quality_overall_summary.csv", [overall], list(overall.keys()))
    write_csv(output_dir / "route_quality_summary.csv", route_rows, list(route_rows[0].keys()) if route_rows else [])
    write_csv(output_dir / "query_type_quality_summary.csv", query_rows, list(query_rows[0].keys()) if query_rows else [])
    write_csv(
        output_dir / "route_query_type_quality_summary.csv",
        route_query_rows,
        list(route_query_rows[0].keys()) if route_query_rows else [],
    )
    issue_fields = [
        "issue_type",
        "calibrated_governance_route",
        "query_type",
        "severity",
        "evidence",
        "recommended_next_action",
    ]
    write_csv(output_dir / "route_quality_issues.csv", issues, issue_fields)
    report = report_markdown(files, overall, route_rows, query_rows, issues)
    write_report(output_dir / "route_quality_report.md", report)

    print_console_report(files, overall, route_rows, query_rows, issues)
    print("\nFiles written under", output_dir)
    return overall


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MVP 23E Route-Specific Slate Quality Analyzer")
    parser.add_argument("--input-dir", default=str(DEFAULT_INPUT_DIR), help="Calibrated governed output directory.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Quality analyzer output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_analyzer(input_dir=Path(args.input_dir), output_dir=Path(args.output_dir))


if __name__ == "__main__":
    main()
