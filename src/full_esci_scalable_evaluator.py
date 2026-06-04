"""
MVP 21: Full ESCI Scalable Evaluation Engine

Runs governed CORTEX evaluation over a large unique ESCI query set with chunked,
resumable output. All evaluator-owned artifacts are written under
outputs/full_esci_eval by default.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


DEFAULT_OUTPUT_DIR = Path("outputs/full_esci_eval")

PREFERRED_ESCI_PATHS = [
    Path("external_data/esci-data/shopping_queries_dataset/shopping_queries_dataset_examples.parquet"),
    Path("esci-data/shopping_queries_dataset/shopping_queries_dataset_examples.parquet"),
    Path("data/esci_balanced_sample.csv"),
]

SAMPLE_PATH = Path("data/esci_balanced_sample.csv")

SMOKE_QUERIES = [
    "new apartment kitchen setup",
    "beach vacation packing list",
    "adidas soccer cleats",
    "world cup watch party",
    "camping trip essentials",
    "college dorm essentials",
    "baby shower decorations",
    "office desk setup",
    "birthday party supplies",
    "hiking trip snacks",
    "home gym setup",
    "moving day essentials",
    "road trip snacks",
    "bbq party supplies",
    "school lunch packing",
    "coffee maker with grinder",
    "iphone 15 case",
    "red dress for wedding guest",
    "back to school supplies",
    "kitchen storage containers",
]

EVAL_FIELDS = [
    "query",
    "query_index",
    "chunk_id",
    "success",
    "runtime_seconds",
    "final_slate_size",
    "governance_route",
    "final_execution_source",
    "baseline_preserved",
    "unique_sub_intents",
    "cold_start_proxy_items",
    "error_message",
]


def clean_text(value: object) -> str:
    return str(value or "").strip()


def lower_text(value: object) -> str:
    return clean_text(value).lower()


def console_text(value: object) -> str:
    text = clean_text(value)
    encoding = sys.stdout.encoding or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding, errors="replace")


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


def ensure_output_dirs(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "chunks").mkdir(parents=True, exist_ok=True)
    (output_dir / "runtime").mkdir(parents=True, exist_ok=True)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []

    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv_rows(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def find_query_column(columns: Iterable[str]) -> str:
    candidates = [
        "query",
        "shopping_query",
        "query_text",
        "search_query",
        "q",
    ]
    columns_list = list(columns)
    lower_to_original = {str(col).lower(): str(col) for col in columns_list}

    for candidate in candidates:
        if candidate in lower_to_original:
            return lower_to_original[candidate]

    for col in columns_list:
        if "query" in str(col).lower():
            return str(col)

    raise ValueError(f"Could not identify query column from columns: {columns_list}")


def unique_clean_queries(values: Iterable[object], max_queries: int | None = None) -> List[str]:
    queries: List[str] = []
    seen = set()

    for value in values:
        query = clean_text(value)
        if not query:
            continue

        key = query.lower()
        if key in seen:
            continue

        seen.add(key)
        queries.append(query)

        if max_queries is not None and len(queries) >= max_queries:
            break

    return queries


def load_queries_from_csv(path: Path, max_queries: int | None = None) -> List[str]:
    df = pd.read_csv(path)
    query_col = find_query_column(df.columns)
    return unique_clean_queries(df[query_col].tolist(), max_queries=max_queries)


def load_queries_from_parquet(path: Path, max_queries: int | None = None) -> List[str]:
    try:
        df = pd.read_parquet(path, columns=["query"])
        query_col = "query"
    except Exception:
        df = pd.read_parquet(path)
        query_col = find_query_column(df.columns)

    return unique_clean_queries(df[query_col].tolist(), max_queries=max_queries)


def load_queries_from_path(path: Path, max_queries: int | None = None) -> List[str]:
    if path.suffix.lower() == ".parquet":
        return load_queries_from_parquet(path, max_queries=max_queries)
    return load_queries_from_csv(path, max_queries=max_queries)


def resolve_query_source(query_mode: str) -> Path | None:
    if query_mode == "smoke":
        return None

    if query_mode == "sample":
        if SAMPLE_PATH.exists():
            return SAMPLE_PATH
        for path in PREFERRED_ESCI_PATHS:
            if path.exists():
                return path
        return None

    for path in PREFERRED_ESCI_PATHS:
        if path.exists():
            return path

    return None


def build_query_list(query_mode: str, max_queries: int) -> Tuple[List[str], str]:
    if query_mode == "smoke":
        return SMOKE_QUERIES[:max_queries], "built_in_smoke_queries"

    source = resolve_query_source(query_mode)
    if source is None:
        fallback = SMOKE_QUERIES[:max_queries]
        return fallback, "built_in_smoke_queries_fallback"

    queries = load_queries_from_path(source, max_queries=max_queries)
    return queries, str(source)


def load_all_queries(query_mode: str) -> Tuple[List[str], str]:
    if query_mode == "smoke":
        return list(SMOKE_QUERIES), "built_in_smoke_queries"

    source = resolve_query_source(query_mode)
    if source is None:
        return list(SMOKE_QUERIES), "built_in_smoke_queries_fallback"

    queries = load_queries_from_path(source, max_queries=None)
    return queries, str(source)


def select_query_window(
    all_queries: List[str],
    start_index: int,
    max_queries: int,
) -> List[str]:
    if start_index >= len(all_queries):
        return []

    return all_queries[start_index : start_index + max_queries]


def chunk_path(output_dir: Path, chunk_id: int) -> Path:
    return output_dir / "chunks" / f"chunk_{chunk_id:06d}.csv"


def chunk_ranges(total: int, chunk_size: int, start_index: int) -> List[Tuple[int, int, int]]:
    ranges = []
    if total <= 0:
        return ranges

    first_chunk = start_index // chunk_size
    for chunk_id in range(first_chunk, math.ceil(total / chunk_size)):
        start = max(chunk_id * chunk_size, start_index)
        end = min(start + (chunk_size - (start % chunk_size)), total)
        if start < end:
            ranges.append((chunk_id, start, end))

    return ranges


def clear_previous_run_outputs(output_dir: Path) -> None:
    def remove_or_truncate(path: Path) -> None:
        try:
            path.unlink()
        except PermissionError:
            path.write_text("", encoding="utf-8")

    chunk_dir = output_dir / "chunks"
    if chunk_dir.exists():
        for path in chunk_dir.glob("*.csv"):
            if path.is_file():
                remove_or_truncate(path)

    for filename in [
        "full_esci_eval_by_query.csv",
        "full_esci_eval_summary.csv",
        "full_esci_eval_by_route.csv",
        "full_esci_eval_failure_modes.csv",
    ]:
        path = output_dir / filename
        if path.exists() and path.is_file():
            remove_or_truncate(path)


def configure_governed_import_paths(output_dir: Path) -> Tuple[object, object]:
    import src.cortex_governance_agent as cga
    import src.governed_cortex_runner as gcr

    runtime_dir = output_dir / "runtime"
    governance_decisions = runtime_dir / "cortex_governance_decisions.csv"
    governance_summary = runtime_dir / "cortex_governance_summary.csv"
    governance_trace = runtime_dir / "cortex_governance_trace.csv"

    # Governance writes are evaluator-local. Strict/behavior artifacts are read
    # from the existing CORTEX outputs so skip-refresh can reuse prior artifacts.
    cga.GOVERNANCE_DECISIONS_PATH = governance_decisions
    cga.GOVERNANCE_SUMMARY_PATH = governance_summary
    cga.GOVERNANCE_TRACE_PATH = governance_trace

    gcr.GOVERNANCE_DECISIONS_PATH = governance_decisions
    gcr.GOVERNANCE_SUMMARY_PATH = governance_summary
    gcr.GOVERNANCE_TRACE_PATH = governance_trace

    return cga, gcr


def run_governed_query_imported(query: str, output_dir: Path) -> Dict[str, object]:
    cga, gcr = configure_governed_import_paths(output_dir)

    cga.run_governance(query=query, skip_refresh=True)
    governance_decision = gcr.get_governance_decision(query)
    governance_summary = gcr.get_governance_summary(query)

    final_rows, final_source = gcr.select_final_slate(
        query=query,
        governance_decision=governance_decision,
    )
    summary = gcr.summarize_final_slate(
        query=query,
        final_rows=final_rows,
        final_source=final_source,
        governance_decision=governance_decision,
        governance_summary=governance_summary,
    )

    return summary


def evaluate_query(query: str, query_index: int, chunk_id: int, output_dir: Path) -> Dict[str, object]:
    start = time.perf_counter()

    try:
        summary = run_governed_query_imported(query=query, output_dir=output_dir)
        runtime = round(time.perf_counter() - start, 4)

        if not summary:
            raise RuntimeError("Governed runner produced no summary for query.")

        return {
            "query": query,
            "query_index": query_index,
            "chunk_id": chunk_id,
            "success": 1,
            "runtime_seconds": runtime,
            "final_slate_size": safe_int(summary.get("final_slate_size")),
            "governance_route": clean_text(summary.get("governance_route")),
            "final_execution_source": clean_text(summary.get("final_execution_source")),
            "baseline_preserved": safe_int(summary.get("baseline_preserved")),
            "unique_sub_intents": safe_int(summary.get("unique_sub_intents")),
            "cold_start_proxy_items": safe_int(summary.get("cold_start_proxy_items")),
            "error_message": "",
        }

    except Exception as exc:
        runtime = round(time.perf_counter() - start, 4)
        error_message = str(exc)

        return {
            "query": query,
            "query_index": query_index,
            "chunk_id": chunk_id,
            "success": 0,
            "runtime_seconds": runtime,
            "final_slate_size": 0,
            "governance_route": "",
            "final_execution_source": "",
            "baseline_preserved": 0,
            "unique_sub_intents": 0,
            "cold_start_proxy_items": 0,
            "error_message": error_message[:1000],
        }


def evaluate_chunk(
    queries: List[str],
    chunk_id: int,
    start: int,
    end: int,
    output_dir: Path,
    global_start_index: int,
) -> List[Dict[str, object]]:
    rows = []
    total = len(queries)
    print(f"Chunk {chunk_id:06d}: evaluating query indexes {start}..{end - 1}")

    for query_index in range(start, end):
        query = queries[query_index]
        global_query_index = global_start_index + query_index
        display_index = query_index + 1
        print(f"  [{display_index}/{total}] {console_text(query)}")
        row = evaluate_query(
            query=query,
            query_index=global_query_index,
            chunk_id=chunk_id,
            output_dir=output_dir,
        )
        rows.append(row)

        status = "ok" if safe_int(row.get("success")) == 1 else "failed"
        print(
            "    "
            f"{status}; route={row.get('governance_route') or 'n/a'}; "
            f"source={row.get('final_execution_source') or 'n/a'}; "
            f"runtime={row.get('runtime_seconds')}s"
        )

    return rows


def collect_chunk_rows(output_dir: Path) -> List[Dict[str, str]]:
    chunk_files = sorted((output_dir / "chunks").glob("chunk_*.csv"))
    rows: List[Dict[str, str]] = []
    for path in chunk_files:
        rows.extend(read_csv_rows(path))

    rows.sort(key=lambda row: safe_int(row.get("query_index")))
    return rows


def mean(values: List[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def build_summary(
    rows: List[Dict[str, str]],
    source: str,
    max_queries: int,
    chunk_size: int,
    start_index: int,
    selected_query_count: int,
    total_available_unique_queries: int,
) -> Dict[str, object]:
    total = len(rows)
    success_rows = [row for row in rows if safe_int(row.get("success")) == 1]
    failed_rows = [row for row in rows if safe_int(row.get("success")) != 1]
    runtimes = [safe_float(row.get("runtime_seconds")) for row in rows]

    route_counts = Counter(clean_text(row.get("governance_route")) or "unknown" for row in success_rows)
    source_counts = Counter(clean_text(row.get("final_execution_source")) or "unknown" for row in success_rows)

    return {
        "source": source,
        "start_index": start_index,
        "requested_max_queries": max_queries,
        "selected_query_count": selected_query_count,
        "total_available_unique_queries": total_available_unique_queries,
        "chunk_size": chunk_size,
        "total_query_rows": total,
        "successful_queries": len(success_rows),
        "failed_queries": len(failed_rows),
        "success_rate": round(len(success_rows) / max(total, 1), 6),
        "avg_runtime_seconds": round(mean(runtimes), 6),
        "total_runtime_seconds": round(sum(runtimes), 6),
        "avg_final_slate_size": round(mean([safe_float(row.get("final_slate_size")) for row in success_rows]), 6),
        "baseline_preservation_rate": round(
            mean([safe_float(row.get("baseline_preserved")) for row in success_rows]),
            6,
        ),
        "avg_unique_sub_intents": round(mean([safe_float(row.get("unique_sub_intents")) for row in success_rows]), 6),
        "total_cold_start_proxy_items": int(sum(safe_float(row.get("cold_start_proxy_items")) for row in success_rows)),
        "top_governance_route": route_counts.most_common(1)[0][0] if route_counts else "",
        "top_final_execution_source": source_counts.most_common(1)[0][0] if source_counts else "",
    }


def build_by_route(rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    groups: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        if safe_int(row.get("success")) == 1:
            groups[clean_text(row.get("governance_route")) or "unknown"].append(row)

    output = []
    total_success = sum(len(group) for group in groups.values())
    for route, group in sorted(groups.items(), key=lambda item: item[0]):
        output.append(
            {
                "governance_route": route,
                "query_count": len(group),
                "query_share": round(len(group) / max(total_success, 1), 6),
                "avg_runtime_seconds": round(mean([safe_float(row.get("runtime_seconds")) for row in group]), 6),
                "avg_final_slate_size": round(mean([safe_float(row.get("final_slate_size")) for row in group]), 6),
                "baseline_preservation_rate": round(mean([safe_float(row.get("baseline_preserved")) for row in group]), 6),
                "avg_unique_sub_intents": round(mean([safe_float(row.get("unique_sub_intents")) for row in group]), 6),
                "total_cold_start_proxy_items": int(sum(safe_float(row.get("cold_start_proxy_items")) for row in group)),
            }
        )

    return output


def classify_failure(row: Dict[str, str]) -> str:
    error = lower_text(row.get("error_message"))
    if safe_int(row.get("success")) == 1:
        return "success"
    if "no governance decision" in error or "no evaluator-local summary" in error:
        return "missing_governance_or_summary"
    if "parquet" in error or "query column" in error:
        return "data_loading_error"
    if "traceback" in error or "runtimeerror" in error:
        return "runner_runtime_error"
    if not error:
        return "unknown_failure"
    return "query_execution_failure"


def build_failure_modes(rows: List[Dict[str, str]]) -> List[Dict[str, object]]:
    failure_rows = [row for row in rows if safe_int(row.get("success")) != 1]
    groups: Dict[str, List[Dict[str, str]]] = defaultdict(list)

    for row in failure_rows:
        groups[classify_failure(row)].append(row)

    output = []
    for mode, group in sorted(groups.items(), key=lambda item: item[0]):
        examples = [clean_text(row.get("query")) for row in group[:5]]
        output.append(
            {
                "failure_mode": mode,
                "query_count": len(group),
                "query_share": round(len(group) / max(len(rows), 1), 6),
                "example_queries": " | ".join(examples),
                "sample_error": clean_text(group[0].get("error_message"))[:500] if group else "",
            }
        )

    if not output:
        output.append(
            {
                "failure_mode": "none",
                "query_count": 0,
                "query_share": 0.0,
                "example_queries": "",
                "sample_error": "",
            }
        )

    return output


def aggregate_outputs(
    output_dir: Path,
    source: str,
    max_queries: int,
    chunk_size: int,
    start_index: int,
    selected_query_count: int,
    total_available_unique_queries: int,
) -> None:
    rows = collect_chunk_rows(output_dir)
    by_query_path = output_dir / "full_esci_eval_by_query.csv"
    summary_path = output_dir / "full_esci_eval_summary.csv"
    by_route_path = output_dir / "full_esci_eval_by_route.csv"
    failure_modes_path = output_dir / "full_esci_eval_failure_modes.csv"

    summary = build_summary(
        rows,
        source=source,
        max_queries=max_queries,
        chunk_size=chunk_size,
        start_index=start_index,
        selected_query_count=selected_query_count,
        total_available_unique_queries=total_available_unique_queries,
    )
    by_route = build_by_route(rows)
    failure_modes = build_failure_modes(rows)

    write_csv_rows(by_query_path, rows, EVAL_FIELDS)
    write_csv_rows(summary_path, [summary], list(summary.keys()))
    write_csv_rows(
        by_route_path,
        by_route,
        [
            "governance_route",
            "query_count",
            "query_share",
            "avg_runtime_seconds",
            "avg_final_slate_size",
            "baseline_preservation_rate",
            "avg_unique_sub_intents",
            "total_cold_start_proxy_items",
        ],
    )
    write_csv_rows(
        failure_modes_path,
        failure_modes,
        ["failure_mode", "query_count", "query_share", "example_queries", "sample_error"],
    )

    print("\nAggregate outputs written")
    print("-" * 100)
    print(f"- {by_query_path}")
    print(f"- {summary_path}")
    print(f"- {by_route_path}")
    print(f"- {failure_modes_path}")


def run_full_esci_eval(
    max_queries: int,
    chunk_size: int,
    start_index: int,
    query_mode: str,
    skip_existing: bool,
    output_dir: Path,
) -> None:
    ensure_output_dirs(output_dir)

    if not skip_existing:
        clear_previous_run_outputs(output_dir)

    all_queries, source = load_all_queries(query_mode=query_mode)
    total_available = len(all_queries)
    queries = select_query_window(
        all_queries=all_queries,
        start_index=start_index,
        max_queries=max_queries,
    )
    selected_count = len(queries)

    if total_available == 0:
        raise RuntimeError("No queries were loaded for evaluation.")

    print("\nMVP 21 Full ESCI Scalable Evaluation")
    print("-" * 100)
    print(f"Query mode: {query_mode}")
    print(f"Query source: {source}")
    print(f"total_available_unique_queries: {total_available}")
    print(f"start_index: {start_index}")
    print(f"requested_max_queries: {max_queries}")
    print(f"selected_query_count: {selected_count}")
    print(f"Chunk size: {chunk_size}")
    print(f"Skip existing chunks: {skip_existing}")
    print(f"Output dir: {output_dir}")

    if selected_count == 0:
        print("No selected queries remain after applying start-index and max-queries.")
        aggregate_outputs(
            output_dir=output_dir,
            source=source,
            max_queries=max_queries,
            chunk_size=chunk_size,
            start_index=start_index,
            selected_query_count=selected_count,
            total_available_unique_queries=total_available,
        )
        return

    print("\nFirst selected queries")
    print("-" * 100)
    for preview_index, query in enumerate(queries[:5], start=1):
        print(f"{preview_index}. {console_text(query)}")

    for chunk_id, start, end in chunk_ranges(total=selected_count, chunk_size=chunk_size, start_index=0):
        path = chunk_path(output_dir, chunk_id)

        if skip_existing and path.exists():
            print(f"Chunk {chunk_id:06d}: skipping existing file {path}")
            continue

        rows = evaluate_chunk(
            queries=queries,
            chunk_id=chunk_id,
            start=start,
            end=end,
            output_dir=output_dir,
            global_start_index=start_index,
        )
        write_csv_rows(path, rows, EVAL_FIELDS)
        print(f"Chunk {chunk_id:06d}: wrote {len(rows)} rows to {path}")

    aggregate_outputs(
        output_dir=output_dir,
        source=source,
        max_queries=max_queries,
        chunk_size=chunk_size,
        start_index=start_index,
        selected_query_count=selected_count,
        total_available_unique_queries=total_available,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MVP 21 Full ESCI Scalable Evaluation Engine")
    parser.add_argument("--max-queries", type=int, default=1000, help="Maximum unique queries to evaluate.")
    parser.add_argument("--chunk-size", type=int, default=250, help="Queries per output chunk.")
    parser.add_argument("--start-index", type=int, default=0, help="Zero-based query index to start from.")
    parser.add_argument(
        "--query-mode",
        choices=["esci", "sample", "smoke"],
        default="esci",
        help="Query source mode.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip chunk files that already exist.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Evaluator output directory.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.max_queries <= 0:
        raise ValueError("--max-queries must be positive.")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive.")
    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative.")

    run_full_esci_eval(
        max_queries=args.max_queries,
        chunk_size=args.chunk_size,
        start_index=args.start_index,
        query_mode=args.query_mode,
        skip_existing=bool(args.skip_existing),
        output_dir=Path(args.output_dir),
    )


if __name__ == "__main__":
    main()
