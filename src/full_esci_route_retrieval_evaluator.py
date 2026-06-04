"""
MVP 26.2: Full ESCI Route Retrieval Evaluation

Runs calibrated CORTEX route execution with full ESCI retrieval and evaluates
retrieved slates against ESCI labels by route and query type.
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from src.calibrated_route_execution_adapter import execute_calibrated_route
from src.governance_calibration_dry_run import apply_calibration, calibrate_query
from src.query_understanding_agent import SMOKE_QUERIES, console_text, understand_query


DEFAULT_INDEX_DIR = Path("data/esci_index")
DEFAULT_OUTPUT_DIR = Path("outputs/full_esci_route_retrieval_eval")

LABEL_ORDER = {"E": 4, "S": 3, "C": 2, "I": 1}
LABEL_ALIASES = {
    "exact": "E",
    "exactly": "E",
    "substitute": "S",
    "complement": "C",
    "irrelevant": "I",
}

RESULT_FIELDS = [
    "query",
    "query_type",
    "recommended_governance_bias",
    "current_governance_route",
    "calibrated_governance_route",
    "calibrated_governance_decision",
    "calibration_action",
    "execution_source",
    "final_slate_size",
    "unique_sub_intents",
    "fallback_used",
    "fallback_reason",
    "critic_review_stage",
    "review_priority",
    "calibrated_adapter_trace",
    "retrieved_count",
    "labeled_retrieved_count",
    "exact_count",
    "substitute_count",
    "complement_count",
    "irrelevant_count",
    "top_k_has_exact",
    "top_k_has_exact_or_substitute",
    "best_esci_label",
    "label_coverage_rate",
    "exact_rate_in_retrieved",
    "exact_or_substitute_rate_in_retrieved",
    "error_message",
]

QUERY_EVAL_FIELDS = [
    "query",
    "calibrated_governance_route",
    "execution_source",
    "product_id",
    "rank",
    "product_title",
    "score",
    "sub_intent",
    "esci_label",
    "critic_review_stage",
    "review_priority",
]


def clean_text(value: object) -> str:
    return str(value or "").strip()


def normalized_query(value: object) -> str:
    return " ".join(clean_text(value).lower().split())


def safe_int(value: object, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(str(value)))
    except Exception:
        return default


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(str(value))
    except Exception:
        return default


def normalize_label(label: object) -> str:
    raw = clean_text(label)
    upper = raw.upper()
    if upper in LABEL_ORDER:
        return upper
    return LABEL_ALIASES.get(raw.lower(), raw)


def best_label(labels: Iterable[str]) -> str:
    best = ""
    best_score = 0
    for label in labels:
        normalized = normalize_label(label)
        score = LABEL_ORDER.get(normalized, 0)
        if score > best_score:
            best = normalized
            best_score = score
    return best


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_indexed_queries(index_dir: Path, sample_size: int, start_index: int) -> List[str]:
    rows = read_csv(index_dir / "queries.csv")
    queries = [clean_text(row.get("query")) for row in rows if clean_text(row.get("query"))]
    return queries[start_index : start_index + sample_size]


def load_label_lookup(index_dir: Path) -> Dict[Tuple[str, str], str]:
    lookup: Dict[Tuple[str, str], str] = {}
    for row in read_csv(index_dir / "query_product_labels.csv"):
        query = normalized_query(row.get("query"))
        product_id = clean_text(row.get("product_id"))
        label = normalize_label(row.get("esci_label"))
        if query and product_id:
            lookup[(query, product_id)] = label
    return lookup


def fallback_calibration_from_rule(query: str, error: str) -> Dict[str, object]:
    understanding = understand_query(query)
    query_type = clean_text(understanding.get("query_type"))
    bias = clean_text(understanding.get("recommended_governance_bias"))
    confidence = safe_float(understanding.get("confidence_score"))
    risk = safe_float(understanding.get("risk_score"))
    current_route = "BASELINE_ONLY"
    current_decision = "PRESERVE_BASELINE"
    calibration = apply_calibration(
        query=query,
        query_type=query_type,
        bias=bias,
        confidence=confidence,
        risk=risk,
        current_route=current_route,
        current_decision=current_decision,
    )
    calibration["calibration_reason"] = (
        f"{calibration.get('calibration_reason')} Imported governance was unavailable "
        f"inside the route evaluator sandbox: {clean_text(error)[:300]}"
    )
    return {
        "query": query,
        "query_type": query_type,
        "recommended_governance_bias": bias,
        "query_understanding_confidence": confidence,
        "query_understanding_risk": risk,
        "current_governance_route": current_route,
        "current_governance_decision": current_decision,
        **calibration,
    }


def calibrate_query_in_sandbox(query: str, output_dir: Path) -> Dict[str, object]:
    import src.governance_alignment_analyzer as alignment

    (output_dir / "runtime").mkdir(parents=True, exist_ok=True)
    original_subprocess = alignment.run_governance_subprocess

    def disabled_subprocess(_query: str) -> Dict[str, object]:
        raise RuntimeError("Subprocess governance fallback is disabled for route retrieval evaluation.")

    alignment.run_governance_subprocess = disabled_subprocess
    try:
        return calibrate_query(query=query, output_dir=output_dir)
    except Exception as exc:
        return fallback_calibration_from_rule(query=query, error=str(exc))
    finally:
        alignment.run_governance_subprocess = original_subprocess


def select_queries(query_mode: str, sample_size: int, start_index: int, index_dir: Path) -> List[str]:
    if query_mode == "smoke":
        return list(SMOKE_QUERIES)[start_index : start_index + sample_size]
    if query_mode == "indexed_queries":
        return load_indexed_queries(index_dir, sample_size=sample_size, start_index=start_index)
    raise ValueError(f"Unsupported query_mode: {query_mode}")


def evaluate_slate(
    query: str,
    route: str,
    execution_source: str,
    slate_rows: List[Dict[str, object]],
    label_lookup: Dict[Tuple[str, str], str],
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    query_key = normalized_query(query)
    evaluated_rows = []
    labels = []
    critic_review_stages = []
    review_priorities = []

    for index, row in enumerate(slate_rows, start=1):
        product_id = clean_text(row.get("product_id") or row.get("item_id"))
        label = normalize_label(row.get("esci_label")) or label_lookup.get((query_key, product_id), "")
        if label:
            labels.append(label)
        critic_review_stage = clean_text(row.get("critic_review_stage"))
        review_priority = clean_text(row.get("review_priority"))
        if critic_review_stage:
            critic_review_stages.append(critic_review_stage)
        if review_priority:
            review_priorities.append(review_priority)
        evaluated_rows.append(
            {
                "query": query,
                "calibrated_governance_route": route,
                "execution_source": execution_source,
                "product_id": product_id,
                "rank": row.get("rank") or index,
                "product_title": clean_text(row.get("product_title") or row.get("title")),
                "score": row.get("score", ""),
                "sub_intent": clean_text(row.get("sub_intent")),
                "esci_label": label,
                "critic_review_stage": critic_review_stage,
                "review_priority": review_priority,
            }
        )

    counts = Counter(labels)
    retrieved_count = len(slate_rows)
    labeled_count = len(labels)
    exact_count = counts.get("E", 0)
    substitute_count = counts.get("S", 0)
    exact_or_substitute = exact_count + substitute_count

    return (
        {
            "retrieved_count": retrieved_count,
            "labeled_retrieved_count": labeled_count,
            "exact_count": exact_count,
            "substitute_count": substitute_count,
            "complement_count": counts.get("C", 0),
            "irrelevant_count": counts.get("I", 0),
            "top_k_has_exact": int(exact_count > 0),
            "top_k_has_exact_or_substitute": int(exact_or_substitute > 0),
            "best_esci_label": best_label(labels),
            "label_coverage_rate": round(labeled_count / max(retrieved_count, 1), 6),
            "exact_rate_in_retrieved": round(exact_count / max(retrieved_count, 1), 6),
            "exact_or_substitute_rate_in_retrieved": round(exact_or_substitute / max(retrieved_count, 1), 6),
            "critic_review_stage": top_counter_value(critic_review_stages),
            "review_priority": top_counter_value(review_priorities),
        },
        evaluated_rows,
    )


def run_query(
    query: str,
    output_dir: Path,
    index_dir: Path,
    top_k: int,
    retrieval_mode: str,
    label_lookup: Dict[Tuple[str, str], str],
) -> Tuple[Dict[str, object], List[Dict[str, object]]]:
    calibration = calibrate_query_in_sandbox(query=query, output_dir=output_dir)
    query_type = clean_text(calibration.get("query_type"))
    bias = clean_text(calibration.get("recommended_governance_bias"))
    route = clean_text(calibration.get("calibrated_governance_route")) or "BASELINE_ONLY"
    adapter_result = execute_calibrated_route(
        query=query,
        calibrated_route=route,
        query_understanding={
            "query_type": query_type,
            "recommended_governance_bias": bias,
            "confidence_score": calibration.get("query_understanding_confidence"),
            "risk_score": calibration.get("query_understanding_risk"),
        },
        max_items=top_k,
        retrieval_mode=retrieval_mode,
        index_dir=index_dir,
        top_k=top_k,
    )
    slate_rows = list(adapter_result.get("final_slate", []))
    execution_source = clean_text(adapter_result.get("execution_source"))
    eval_metrics, eval_rows = evaluate_slate(
        query=query,
        route=route,
        execution_source=execution_source,
        slate_rows=slate_rows,
        label_lookup=label_lookup,
    )
    result = {
        "query": query,
        "query_type": query_type,
        "recommended_governance_bias": bias,
        "current_governance_route": clean_text(calibration.get("current_governance_route")),
        "calibrated_governance_route": route,
        "calibrated_governance_decision": clean_text(calibration.get("calibrated_governance_decision")),
        "calibration_action": clean_text(calibration.get("calibration_action")),
        "execution_source": execution_source,
        "final_slate_size": safe_int(adapter_result.get("final_slate_size")),
        "unique_sub_intents": safe_int(adapter_result.get("unique_sub_intents")),
        "fallback_used": int(bool(adapter_result.get("fallback_used"))),
        "fallback_reason": clean_text(adapter_result.get("fallback_reason")),
        "critic_review_stage": eval_metrics.get("critic_review_stage", ""),
        "review_priority": eval_metrics.get("review_priority", ""),
        "calibrated_adapter_trace": clean_text(adapter_result.get("adapter_trace")),
        **eval_metrics,
        "error_message": "",
    }
    return result, eval_rows


def aggregate_summary(rows: List[Dict[str, object]], runtime_seconds: float) -> Dict[str, object]:
    total = len(rows)
    success_rows = [row for row in rows if not clean_text(row.get("error_message"))]
    fallback_count = sum(safe_int(row.get("fallback_used")) for row in success_rows)
    return {
        "total_queries": total,
        "success_count": len(success_rows),
        "failure_count": total - len(success_rows),
        "avg_final_slate_size": avg(success_rows, "final_slate_size"),
        "avg_unique_sub_intents": avg(success_rows, "unique_sub_intents"),
        "top_k_has_exact_rate": avg(success_rows, "top_k_has_exact"),
        "top_k_has_exact_or_substitute_rate": avg(success_rows, "top_k_has_exact_or_substitute"),
        "avg_label_coverage_rate": avg(success_rows, "label_coverage_rate"),
        "top_calibrated_route": top_value(success_rows, "calibrated_governance_route"),
        "top_execution_source": top_value(success_rows, "execution_source"),
        "fallback_count": fallback_count,
        "fallback_rate": round(fallback_count / max(len(success_rows), 1), 6),
        "runtime_seconds": round(runtime_seconds, 4),
    }


def avg(rows: List[Dict[str, object]], field: str) -> float:
    return round(sum(safe_float(row.get(field)) for row in rows) / max(len(rows), 1), 6)


def top_value(rows: List[Dict[str, object]], field: str) -> str:
    counts = Counter(clean_text(row.get(field)) for row in rows if clean_text(row.get(field)))
    return counts.most_common(1)[0][0] if counts else ""


def top_counter_value(values: Iterable[str]) -> str:
    counts = Counter(clean_text(value) for value in values if clean_text(value))
    return counts.most_common(1)[0][0] if counts else ""


def grouped_summary(rows: List[Dict[str, object]], group_fields: List[str]) -> List[Dict[str, object]]:
    groups: Dict[Tuple[str, ...], List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = tuple(clean_text(row.get(field)) or "unknown" for field in group_fields)
        groups[key].append(row)

    output = []
    for key, group in sorted(groups.items(), key=lambda item: (-len(item[1]), item[0])):
        fallback_count = sum(safe_int(row.get("fallback_used")) for row in group)
        route_summary = {
            "query_count": len(group),
            "avg_final_slate_size": avg(group, "final_slate_size"),
            "avg_unique_sub_intents": avg(group, "unique_sub_intents"),
            "fallback_rate": round(fallback_count / max(len(group), 1), 6),
            "top_k_has_exact_rate": avg(group, "top_k_has_exact"),
            "top_k_has_exact_or_substitute_rate": avg(group, "top_k_has_exact_or_substitute"),
            "avg_label_coverage_rate": avg(group, "label_coverage_rate"),
            "avg_exact_rate_in_retrieved": avg(group, "exact_rate_in_retrieved"),
            "avg_exact_or_substitute_rate_in_retrieved": avg(group, "exact_or_substitute_rate_in_retrieved"),
        }
        for field, value in zip(group_fields, key):
            route_summary[field] = value
        route_summary["top_route"] = top_value(group, "calibrated_governance_route")
        route_summary["top_critic_review_stage"] = top_value(group, "critic_review_stage")
        route_summary["top_review_priority"] = top_value(group, "review_priority")
        route_summary["top_quality_issue"] = top_issue(route_summary)
        output.append(route_summary)
    return output


def query_type_summary(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    summaries = grouped_summary(rows, ["query_type"])
    fields_to_keep = {
        "query_type",
        "query_count",
        "top_route",
        "avg_final_slate_size",
        "avg_unique_sub_intents",
        "fallback_rate",
        "top_k_has_exact_rate",
        "top_k_has_exact_or_substitute_rate",
        "avg_label_coverage_rate",
    }
    return [{key: row.get(key, "") for key in fields_to_keep} for row in summaries]


def top_issue(summary: Dict[str, object]) -> str:
    if (
        clean_text(summary.get("calibrated_governance_route")) == "CRITIC_REVIEW"
        and clean_text(summary.get("top_critic_review_stage")) == "retrieval_ready_pending_critic"
    ):
        return "critic_review_pending_llm"
    if safe_float(summary.get("fallback_rate")) > 0.5:
        return "high_fallback_rate"
    if safe_float(summary.get("top_k_has_exact_rate")) < 0.25:
        return "low_exact_hit_rate"
    if safe_float(summary.get("top_k_has_exact_or_substitute_rate")) < 0.5:
        return "low_exact_or_substitute_rate"
    if safe_float(summary.get("avg_label_coverage_rate")) < 0.25:
        return "low_label_coverage"
    if clean_text(summary.get("calibrated_governance_route")) == "MISSION_REPAIR" and safe_float(summary.get("avg_unique_sub_intents")) < 3:
        return "low_mission_diversity"
    if "behavior_aware_fallback" in clean_text(summary.get("execution_source")):
        return "behavior_aware_fallback"
    return ""


def issue_rows(route_summaries: List[Dict[str, object]]) -> List[Dict[str, object]]:
    issues = []
    for row in route_summaries:
        checks = [
            ("high_fallback_rate", safe_float(row.get("fallback_rate")) > 0.5),
            ("low_exact_hit_rate", safe_float(row.get("top_k_has_exact_rate")) < 0.25),
            ("low_exact_or_substitute_rate", safe_float(row.get("top_k_has_exact_or_substitute_rate")) < 0.5),
            ("low_label_coverage", safe_float(row.get("avg_label_coverage_rate")) < 0.25),
            (
                "low_mission_diversity",
                clean_text(row.get("calibrated_governance_route")) == "MISSION_REPAIR"
                and safe_float(row.get("avg_unique_sub_intents")) < 3,
            ),
            ("behavior_aware_fallback", "behavior_aware_fallback" in clean_text(row.get("execution_source"))),
            (
                "critic_review_pending_llm",
                clean_text(row.get("calibrated_governance_route")) == "CRITIC_REVIEW"
                and clean_text(row.get("top_critic_review_stage")) == "retrieval_ready_pending_critic",
            ),
        ]
        for issue_type, triggered in checks:
            if triggered:
                issues.append(
                    {
                        "issue_type": issue_type,
                        "calibrated_governance_route": row.get("calibrated_governance_route"),
                        "execution_source": row.get("execution_source"),
                        "query_count": row.get("query_count"),
                        "evidence": json.dumps(row, sort_keys=True),
                    }
                )
    return issues


def report_markdown(summary: Dict[str, object], route_summaries: List[Dict[str, object]], issues: List[Dict[str, object]]) -> str:
    lines = ["# MVP 26.2 Full ESCI Route Retrieval Evaluation", "", "## Summary"]
    for key, value in summary.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Route Summary"])
    for row in route_summaries:
        lines.append(
            f"- {row.get('calibrated_governance_route')} / {row.get('execution_source')}: "
            f"n={row.get('query_count')}, exact={row.get('top_k_has_exact_rate')}, "
            f"exact_or_sub={row.get('top_k_has_exact_or_substitute_rate')}, "
            f"coverage={row.get('avg_label_coverage_rate')}, fallback={row.get('fallback_rate')}"
        )
    lines.extend(["", "## Issues"])
    if issues:
        for issue in issues[:50]:
            lines.append(
                f"- {issue.get('issue_type')}: {issue.get('calibrated_governance_route')} / "
                f"{issue.get('execution_source')} n={issue.get('query_count')}"
            )
    else:
        lines.append("- None")
    return "\n".join(lines) + "\n"


def write_outputs(
    output_dir: Path,
    result_rows: List[Dict[str, object]],
    query_eval_rows: List[Dict[str, object]],
    summary: Dict[str, object],
    query_type_rows: List[Dict[str, object]],
    route_source_rows: List[Dict[str, object]],
    issues: List[Dict[str, object]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "route_retrieval_results.csv", result_rows, RESULT_FIELDS)
    write_csv(output_dir / "route_retrieval_query_eval.csv", query_eval_rows, QUERY_EVAL_FIELDS)
    write_csv(output_dir / "route_retrieval_summary.csv", [summary], list(summary.keys()))
    write_csv(output_dir / "query_type_retrieval_summary.csv", query_type_rows, list(query_type_rows[0].keys()) if query_type_rows else ["query_type"])
    write_csv(
        output_dir / "route_execution_source_summary.csv",
        route_source_rows,
        list(route_source_rows[0].keys()) if route_source_rows else ["calibrated_governance_route"],
    )
    write_csv(
        output_dir / "route_quality_issues.csv",
        issues,
        ["issue_type", "calibrated_governance_route", "execution_source", "query_count", "evidence"],
    )
    (output_dir / "route_retrieval_report.md").write_text(
        report_markdown(summary, route_source_rows, issues),
        encoding="utf-8",
    )


def run_evaluation(args: argparse.Namespace) -> None:
    index_dir = Path(args.index_dir)
    output_dir = Path(args.output_dir)
    label_lookup = load_label_lookup(index_dir)
    queries = select_queries(
        query_mode=args.query_mode,
        sample_size=args.sample_size,
        start_index=args.start_index,
        index_dir=index_dir,
    )

    print("\nMVP 26.2 Full ESCI Route Retrieval Evaluation")
    print("-" * 100)
    print(f"query_mode: {args.query_mode}")
    print(f"sample_size: {args.sample_size}")
    print(f"selected_query_count: {len(queries)}")
    print(f"retrieval_mode: {args.retrieval_mode}")
    print(f"index_dir: {index_dir}")
    print(f"top_k: {args.top_k}")

    result_rows: List[Dict[str, object]] = []
    query_eval_rows: List[Dict[str, object]] = []
    start = time.perf_counter()

    for index, query in enumerate(queries, start=1):
        try:
            row, eval_rows = run_query(
                query=query,
                output_dir=output_dir,
                index_dir=index_dir,
                top_k=args.top_k,
                retrieval_mode=args.retrieval_mode,
                label_lookup=label_lookup,
            )
            result_rows.append(row)
            query_eval_rows.extend(eval_rows)
            if index <= 10 or index % 100 == 0 or index == len(queries):
                print(
                    f"[{index}/{len(queries)}] {console_text(query)} -> "
                    f"route={row['calibrated_governance_route']} source={row['execution_source']} "
                    f"exact={row['top_k_has_exact']} best={row['best_esci_label']} slate={row['final_slate_size']}"
                )
        except Exception as exc:
            result_rows.append(
                {
                    "query": query,
                    "error_message": str(exc)[:1000],
                }
            )

    runtime_seconds = time.perf_counter() - start
    summary = aggregate_summary(result_rows, runtime_seconds=runtime_seconds)
    route_source_rows = grouped_summary(result_rows, ["calibrated_governance_route", "execution_source"])
    query_type_rows = query_type_summary(result_rows)
    issues = issue_rows(route_source_rows)
    write_outputs(output_dir, result_rows, query_eval_rows, summary, query_type_rows, route_source_rows, issues)

    print("\nSummary")
    print("-" * 100)
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("\nRoute summary")
    print("-" * 100)
    for row in route_source_rows:
        print(
            f"{row.get('calibrated_governance_route')}: n={row.get('query_count')} "
            f"exact={row.get('top_k_has_exact_rate')} "
            f"exact_or_sub={row.get('top_k_has_exact_or_substitute_rate')} "
            f"coverage={row.get('avg_label_coverage_rate')} fallback={row.get('fallback_rate')}"
        )
    print("\nOutput files")
    print("-" * 100)
    for name in [
        "route_retrieval_results.csv",
        "route_retrieval_query_eval.csv",
        "route_retrieval_summary.csv",
        "query_type_retrieval_summary.csv",
        "route_execution_source_summary.csv",
        "route_quality_issues.csv",
        "route_retrieval_report.md",
    ]:
        print(output_dir / name)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MVP 26.2 Full ESCI Route Retrieval Evaluation")
    parser.add_argument("--sample-size", type=int, default=100, help="Number of queries to evaluate.")
    parser.add_argument("--start-index", type=int, default=0, help="Start offset.")
    parser.add_argument(
        "--query-mode",
        choices=["smoke", "indexed_queries"],
        default="indexed_queries",
        help="Query source.",
    )
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR), help="ESCI index directory.")
    parser.add_argument(
        "--retrieval-mode",
        choices=["full_esci"],
        default="full_esci",
        help="Route execution retrieval mode.",
    )
    parser.add_argument("--top-k", type=int, default=12, help="Slate size / retrieval top-k.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_size <= 0:
        raise ValueError("--sample-size must be positive")
    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    run_evaluation(args)


if __name__ == "__main__":
    main()
