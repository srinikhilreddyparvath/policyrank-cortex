"""
MVP 25: Full ESCI Product Index Builder

Builds normalized local product/query index artifacts from available ESCI-like
data files. Generated index files are written under data/esci_index/ by default.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple

import pandas as pd

try:
    import pyarrow.parquet as pq

    PYARROW_AVAILABLE = True
except Exception:
    pq = None
    PYARROW_AVAILABLE = False


DEFAULT_OUTPUT_DIR = Path("data/esci_index")
DEFAULT_REPORT_DIR = Path("outputs/full_esci_product_index")

PREFERRED_ESCI_PATHS = [
    Path("data/esci_balanced_sample.csv"),
    Path("data/esci_sample_products.csv"),
    Path("external_data/esci-data/shopping_queries_dataset/shopping_queries_dataset_examples.parquet"),
    Path("external_data/esci-data/shopping_queries_dataset/shopping_queries_dataset_products.parquet"),
    Path("esci-data/shopping_queries_dataset/shopping_queries_dataset_examples.parquet"),
    Path("esci-data/shopping_queries_dataset/shopping_queries_dataset_products.parquet"),
]

SCAN_ROOTS = [
    Path("data"),
    Path("datasets"),
    Path("esci"),
    Path("external_data"),
    Path("esci-data"),
]

SUPPORTED_SUFFIXES = {".csv", ".tsv", ".parquet", ".jsonl"}

PRODUCT_ID_COLUMNS = ["product_id", "item_id", "product_id_hash", "esci_product_id", "asin", "sku"]
TITLE_COLUMNS = ["product_title", "title", "item_title"]
QUERY_COLUMNS = ["query", "search_query", "query_text", "shopping_query"]
DESCRIPTION_COLUMNS = ["product_description", "description", "item_description", "product_bullet_point"]
BRAND_COLUMNS = ["product_brand", "brand", "manufacturer"]
COLOR_COLUMNS = ["product_color", "color"]
LOCALE_COLUMNS = ["product_locale", "country", "marketplace", "locale"]
LABEL_COLUMNS = ["esci_label", "label", "relevance", "relevance_label"]
SPLIT_COLUMNS = ["split", "data_split", "dataset_split"]

PRODUCT_FIELDS = [
    "product_id",
    "product_title",
    "product_description",
    "product_brand",
    "product_color",
    "product_locale",
    "source_file",
]

QUERY_FIELDS = ["query_id", "query", "query_length", "normalized_query_basic"]

LABEL_FIELDS = ["query_id", "query", "product_id", "esci_label", "split", "source_file"]


def clean_text(value: object) -> str:
    return str(value or "").strip()


def normalized_basic(value: object) -> str:
    return re.sub(r"\s+", " ", clean_text(value).lower()).strip()


def normalized_col(name: object) -> str:
    return re.sub(r"[^a-z0-9]+", "_", clean_text(name).lower()).strip("_")


def first_present(columns: Iterable[str], candidates: List[str]) -> str:
    normalized_to_original = {normalized_col(column): str(column) for column in columns}
    for candidate in candidates:
        found = normalized_to_original.get(normalized_col(candidate))
        if found:
            return found
    for column in columns:
        col_norm = normalized_col(column)
        for candidate in candidates:
            if normalized_col(candidate) in col_norm:
                return str(column)
    return ""


def detect_schema(columns: Iterable[str]) -> Dict[str, str]:
    columns = list(columns)
    schema = {
        "product_id": first_present(columns, PRODUCT_ID_COLUMNS),
        "query": first_present(columns, QUERY_COLUMNS),
        "product_title": first_present(columns, TITLE_COLUMNS),
        "product_description": first_present(columns, DESCRIPTION_COLUMNS),
        "product_brand": first_present(columns, BRAND_COLUMNS),
        "product_color": first_present(columns, COLOR_COLUMNS),
        "product_locale": first_present(columns, LOCALE_COLUMNS),
        "esci_label": first_present(columns, LABEL_COLUMNS),
        "split": first_present(columns, SPLIT_COLUMNS),
    }
    if normalized_col(schema["query"]) == "query_id":
        schema["query"] = ""
    return schema


def path_looks_esci(path: Path) -> bool:
    lowered = str(path).lower()
    if path.suffix.lower() not in SUPPORTED_SUFFIXES:
        return False
    if "esci_index" in lowered or "outputs" in lowered:
        return False
    return any(term in lowered for term in ["esci", "shopping_queries", "shopping_query", "balanced_sample"])


def detect_candidate_files() -> List[Path]:
    candidates: List[Path] = []
    seen = set()
    seen_signature = set()

    def add_candidate(path: Path) -> None:
        if path.exists() and path.is_file():
            resolved = path.resolve()
            signature = (path.name.lower(), path.stat().st_size)
            if resolved in seen or signature in seen_signature:
                return
            candidates.append(path)
            seen.add(resolved)
            seen_signature.add(signature)

    for path in PREFERRED_ESCI_PATHS:
        add_candidate(path)

    for root in SCAN_ROOTS:
        if not root.exists():
            continue
        try:
            for path in root.rglob("*"):
                if not path.is_file() or not path_looks_esci(path):
                    continue
                add_candidate(path)
        except Exception:
            continue

    return sorted(candidates, key=file_priority)


def file_priority(path: Path) -> Tuple[int, str]:
    lowered = str(path).lower()
    if "balanced_sample" in lowered:
        return (0, lowered)
    if "sample_products" in lowered:
        return (1, lowered)
    if "examples" in lowered:
        return (2, lowered)
    if "products" in lowered:
        return (3, lowered)
    if "sources" in lowered:
        return (9, lowered)
    return (5, lowered)


def delimiter_for(path: Path) -> str:
    return "\t" if path.suffix.lower() == ".tsv" else ","


def cheap_file_info(path: Path) -> Dict[str, object]:
    info: Dict[str, object] = {
        "path": str(path),
        "suffix": path.suffix.lower(),
        "size_bytes": path.stat().st_size if path.exists() else 0,
        "columns": [],
        "schema": {},
        "sample_row_count": 0,
        "warning": "",
    }
    try:
        if path.suffix.lower() == ".parquet":
            if PYARROW_AVAILABLE and pq is not None:
                parquet_file = pq.ParquetFile(path)
                info["columns"] = [str(column) for column in parquet_file.schema_arrow.names]
                info["sample_row_count"] = parquet_file.metadata.num_rows if parquet_file.metadata else 0
            else:
                df = pd.read_parquet(path)
                info["columns"] = [str(column) for column in df.columns]
                info["sample_row_count"] = len(df)
        elif path.suffix.lower() == ".jsonl":
            with path.open("r", encoding="utf-8") as f:
                rows = []
                for _index, line in zip(range(25), f):
                    if line.strip():
                        rows.append(json.loads(line))
                columns = sorted({key for row in rows if isinstance(row, dict) for key in row.keys()})
                info["columns"] = columns
                info["sample_row_count"] = len(rows)
        else:
            df = pd.read_csv(path, sep=delimiter_for(path), nrows=25)
            info["columns"] = [str(column) for column in df.columns]
            info["sample_row_count"] = len(df)
        info["schema"] = detect_schema(info["columns"])  # type: ignore[arg-type]
    except Exception as exc:
        info["warning"] = f"{type(exc).__name__}: {exc}"
    return info


def iter_csv_rows(path: Path, max_file_rows: Optional[int], chunk_size: int) -> Iterator[Dict[str, object]]:
    processed = 0
    for chunk in pd.read_csv(path, sep=delimiter_for(path), chunksize=chunk_size):
        for row in chunk.to_dict(orient="records"):
            yield row
            processed += 1
            if max_file_rows is not None and processed >= max_file_rows:
                return


def iter_parquet_rows(path: Path, max_file_rows: Optional[int]) -> Iterator[Dict[str, object]]:
    if PYARROW_AVAILABLE and pq is not None:
        parquet_file = pq.ParquetFile(path)
        processed = 0
        for batch in parquet_file.iter_batches(batch_size=50000):
            df = batch.to_pandas()
            for row in df.to_dict(orient="records"):
                yield row
                processed += 1
                if max_file_rows is not None and processed >= max_file_rows:
                    return
        return

    df = pd.read_parquet(path)
    if max_file_rows is not None:
        df = df.head(max_file_rows)
    for row in df.to_dict(orient="records"):
        yield row


def iter_jsonl_rows(path: Path, max_file_rows: Optional[int]) -> Iterator[Dict[str, object]]:
    processed = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            parsed = json.loads(line)
            if isinstance(parsed, dict):
                yield parsed
                processed += 1
                if max_file_rows is not None and processed >= max_file_rows:
                    return


def iter_rows(path: Path, max_file_rows: Optional[int], chunk_size: int) -> Iterator[Dict[str, object]]:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        yield from iter_parquet_rows(path, max_file_rows=max_file_rows)
    elif suffix == ".jsonl":
        yield from iter_jsonl_rows(path, max_file_rows=max_file_rows)
    else:
        yield from iter_csv_rows(path, max_file_rows=max_file_rows, chunk_size=chunk_size)


def row_value(row: Dict[str, object], column: str) -> str:
    return clean_text(row.get(column)) if column else ""


def stable_product_id(row: Dict[str, object], schema: Dict[str, str], source_file: str, ordinal: int) -> str:
    explicit = row_value(row, schema.get("product_id", ""))
    if explicit:
        return explicit
    title = row_value(row, schema.get("product_title", ""))
    brand = row_value(row, schema.get("product_brand", ""))
    key = normalized_basic(f"{brand} {title}")
    if key:
        return f"derived_{hashlib.sha1(key.encode('utf-8')).hexdigest()[:16]}"
    source_key = hashlib.sha1(source_file.encode("utf-8")).hexdigest()[:10]
    return f"derived_{source_key}_{ordinal}"


def product_text(product: Dict[str, object]) -> str:
    return normalized_basic(
        " ".join(
            [
                clean_text(product.get("product_title")),
                clean_text(product.get("product_brand")),
                clean_text(product.get("product_description")),
                clean_text(product.get("product_color")),
            ]
        )
    )


def merge_product(existing: Dict[str, object], incoming: Dict[str, object]) -> Dict[str, object]:
    merged = dict(existing)
    for field in PRODUCT_FIELDS:
        if not clean_text(merged.get(field)) and clean_text(incoming.get(field)):
            merged[field] = incoming[field]
    return merged


def write_csv(path: Path, rows: Iterable[Dict[str, object]], fieldnames: List[str]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def write_jsonl(path: Path, rows: Iterable[Dict[str, object]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def product_text_rows(products: Iterable[Dict[str, object]]) -> Iterator[Dict[str, object]]:
    for product in products:
        yield {
            "product_id": clean_text(product.get("product_id")),
            "product_title": clean_text(product.get("product_title")),
            "brand": clean_text(product.get("product_brand")),
            "text": product_text(product),
            "source": "esci",
        }


def build_index(
    output_dir: Path,
    report_dir: Path,
    max_rows: Optional[int],
    full: bool,
    chunk_size: int,
) -> Dict[str, object]:
    start = time.perf_counter()
    candidates = detect_candidate_files()
    warnings: List[str] = []
    products: Dict[str, Dict[str, object]] = {}
    queries: Dict[str, Dict[str, object]] = {}
    labels: List[Dict[str, object]] = []
    columns_detected: Dict[str, Dict[str, str]] = {}
    input_files_used: List[str] = []
    total_raw_rows = 0
    query_id_by_normalized: Dict[str, str] = {}

    row_budget = None if full else max_rows
    for path in candidates:
        if row_budget is not None and total_raw_rows >= row_budget:
            break

        info = cheap_file_info(path)
        schema = info.get("schema", {}) if isinstance(info.get("schema"), dict) else {}
        columns_detected[str(path)] = dict(schema)
        if not schema.get("query") and not schema.get("product_title") and not schema.get("product_id"):
            warnings.append(f"Skipped {path}: no query/product schema detected.")
            continue

        input_files_used.append(str(path))
        remaining = None if row_budget is None else max(row_budget - total_raw_rows, 0)
        source_file = str(path)

        try:
            for ordinal, row in enumerate(iter_rows(path, max_file_rows=remaining, chunk_size=chunk_size), start=1):
                total_raw_rows += 1
                query = row_value(row, schema.get("query", ""))
                norm_query = normalized_basic(query)
                query_id = ""
                if norm_query:
                    query_id = query_id_by_normalized.get(norm_query, "")
                    if not query_id:
                        query_id = f"q_{len(query_id_by_normalized) + 1:08d}"
                        query_id_by_normalized[norm_query] = query_id
                        queries[query_id] = {
                            "query_id": query_id,
                            "query": query,
                            "query_length": len(query),
                            "normalized_query_basic": norm_query,
                        }

                product_id = stable_product_id(row, schema, source_file=source_file, ordinal=ordinal)
                product = {
                    "product_id": product_id,
                    "product_title": row_value(row, schema.get("product_title", "")),
                    "product_description": row_value(row, schema.get("product_description", "")),
                    "product_brand": row_value(row, schema.get("product_brand", "")),
                    "product_color": row_value(row, schema.get("product_color", "")),
                    "product_locale": row_value(row, schema.get("product_locale", "")),
                    "source_file": source_file,
                }
                if product_id:
                    products[product_id] = merge_product(products.get(product_id, {}), product)

                if query_id and product_id:
                    labels.append(
                        {
                            "query_id": query_id,
                            "query": query,
                            "product_id": product_id,
                            "esci_label": row_value(row, schema.get("esci_label", "")),
                            "split": row_value(row, schema.get("split", "")),
                            "source_file": source_file,
                        }
                    )

                if row_budget is not None and total_raw_rows >= row_budget:
                    break
        except Exception as exc:
            warnings.append(f"Failed reading {path}: {type(exc).__name__}: {exc}")

    output_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    products_path = output_dir / "products.csv"
    queries_path = output_dir / "queries.csv"
    labels_path = output_dir / "query_product_labels.csv"
    text_index_path = output_dir / "product_text_index.jsonl"
    summary_path = output_dir / "index_summary.json"
    report_path = report_dir / "full_esci_product_index_report.md"

    product_rows = sorted(products.values(), key=lambda row: clean_text(row.get("product_id")))
    query_rows = sorted(queries.values(), key=lambda row: clean_text(row.get("query_id")))

    write_csv(products_path, product_rows, PRODUCT_FIELDS)
    write_csv(queries_path, query_rows, QUERY_FIELDS)
    write_csv(labels_path, labels, LABEL_FIELDS)
    write_jsonl(text_index_path, product_text_rows(product_rows))

    runtime_seconds = time.perf_counter() - start
    summary = {
        "input_files_detected": [str(path) for path in candidates],
        "input_files_used": input_files_used,
        "total_raw_rows": total_raw_rows,
        "total_products": len(products),
        "total_queries": len(queries),
        "total_query_product_pairs": len(labels),
        "columns_detected": columns_detected,
        "output_files": {
            "products": str(products_path),
            "queries": str(queries_path),
            "query_product_labels": str(labels_path),
            "product_text_index": str(text_index_path),
            "index_summary": str(summary_path),
            "report": str(report_path),
        },
        "max_rows": max_rows if not full else "full",
        "runtime_seconds": round(runtime_seconds, 4),
        "warnings": warnings,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    report_path.write_text(report_markdown(summary), encoding="utf-8")
    return summary


def report_markdown(summary: Dict[str, object]) -> str:
    lines = [
        "# MVP 25 Full ESCI Product Index Builder",
        "",
        "## Files Detected",
    ]
    for path in summary.get("input_files_detected", []):
        lines.append(f"- {path}")
    lines.extend(["", "## Files Used"])
    for path in summary.get("input_files_used", []):
        lines.append(f"- {path}")
    lines.extend(
        [
            "",
            "## Output Counts",
            f"- total_raw_rows: {summary.get('total_raw_rows')}",
            f"- total_products: {summary.get('total_products')}",
            f"- total_queries: {summary.get('total_queries')}",
            f"- total_query_product_pairs: {summary.get('total_query_product_pairs')}",
            f"- runtime_seconds: {summary.get('runtime_seconds')}",
            "",
            "## Warnings",
        ]
    )
    warnings = summary.get("warnings", [])
    if warnings:
        for warning in warnings:
            lines.append(f"- {warning}")
    else:
        lines.append("- None")
    lines.extend(
        [
            "",
            "## Next Steps",
            "- Wire route execution adapter retrieval to `data/esci_index/product_text_index.jsonl`.",
            "- Add lightweight lexical retrieval over product text.",
            "- Add route-specific evaluation against ESCI labels.",
        ]
    )
    return "\n".join(lines) + "\n"


def inspect_candidates() -> List[Dict[str, object]]:
    return [cheap_file_info(path) for path in detect_candidate_files()]


def print_inspect(rows: List[Dict[str, object]]) -> None:
    print("\nMVP 25 Full ESCI Product Index Builder")
    print("-" * 100)
    print("Inspect mode")
    print(f"candidate_file_count: {len(rows)}")
    for row in rows:
        print("\nCandidate")
        print(f"path: {row.get('path')}")
        print(f"size_bytes: {row.get('size_bytes')}")
        print(f"sample_row_count: {row.get('sample_row_count')}")
        print(f"columns: {row.get('columns')}")
        print(f"schema: {row.get('schema')}")
        if row.get("warning"):
            print(f"warning: {row.get('warning')}")


def print_build_summary(summary: Dict[str, object]) -> None:
    print("\nMVP 25 Full ESCI Product Index Builder")
    print("-" * 100)
    print("Build complete")
    print(f"input_files_used: {summary.get('input_files_used')}")
    print(f"total_raw_rows: {summary.get('total_raw_rows')}")
    print(f"total_products: {summary.get('total_products')}")
    print(f"total_queries: {summary.get('total_queries')}")
    print(f"total_query_product_pairs: {summary.get('total_query_product_pairs')}")
    print(f"runtime_seconds: {summary.get('runtime_seconds')}")
    print("\nOutput files")
    for _name, path in dict(summary.get("output_files", {})).items():
        print(path)
    warnings = list(summary.get("warnings", []))
    if warnings:
        print("\nWarnings")
        for warning in warnings[:20]:
            print(f"- {warning}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MVP 25 Full ESCI Product Index Builder")
    parser.add_argument("--inspect", action="store_true", help="Inspect detected ESCI files and schemas.")
    parser.add_argument("--max-rows", type=int, default=1000, help="Maximum total rows to process.")
    parser.add_argument("--full", action="store_true", help="Process all detected rows.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Index output directory.")
    parser.add_argument("--report-dir", default=str(DEFAULT_REPORT_DIR), help="Markdown report output directory.")
    parser.add_argument("--chunk-size", type=int, default=50000, help="CSV/TSV chunk size.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_rows <= 0 and not args.full:
        raise ValueError("--max-rows must be positive unless --full is used")
    if args.chunk_size <= 0:
        raise ValueError("--chunk-size must be positive")

    if args.inspect:
        print_inspect(inspect_candidates())
        return

    summary = build_index(
        output_dir=Path(args.output_dir),
        report_dir=Path(args.report_dir),
        max_rows=args.max_rows,
        full=bool(args.full),
        chunk_size=args.chunk_size,
    )
    print_build_summary(summary)


if __name__ == "__main__":
    main()
