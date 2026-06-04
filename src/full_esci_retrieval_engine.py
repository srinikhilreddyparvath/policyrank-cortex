"""
MVP 26: Full ESCI Retrieval Engine

Local lexical retrieval over data/esci_index artifacts produced by
src.full_esci_product_index_builder.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


DEFAULT_INDEX_DIR = Path("data/esci_index")
DEFAULT_OUTPUT_DIR = Path("outputs/full_esci_retrieval")
_ENGINE_CACHE: Dict[str, object] = {}
SQLITE_FTS_NAME = "esci_products_fts.sqlite"

STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "is",
    "it",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}

LABEL_ORDER = {"E": 4, "S": 3, "C": 2, "I": 1}
LABEL_ALIASES = {
    "exact": "E",
    "substitute": "S",
    "complement": "C",
    "irrelevant": "I",
}

RESULT_FIELDS = [
    "query",
    "rank",
    "product_id",
    "product_title",
    "product_brand",
    "product_color",
    "score",
    "matched_tokens",
    "query_token_coverage",
    "retrieval_source",
    "esci_label",
]

SUMMARY_FIELDS = [
    "total_queries",
    "success_count",
    "failure_count",
    "avg_retrieved_count",
    "avg_retrieval_time_seconds",
    "top_k",
    "top_k_has_exact_rate",
    "top_k_has_exact_or_substitute_rate",
    "avg_labeled_retrieved_count",
    "avg_label_coverage_rate",
    "runtime_seconds",
]


def clean_text(value: object) -> str:
    return str(value or "").strip()


def normalized_text(value: object) -> str:
    return re.sub(r"\s+", " ", clean_text(value).lower()).strip()


def tokenize(text: object) -> List[str]:
    return [
        token
        for token in re.findall(r"[a-z0-9]+", normalized_text(text))
        if token and token not in STOPWORDS
    ]


def fts_tokens(text: object) -> List[str]:
    return [
        token
        for token in re.findall(r"[a-zA-Z0-9]+", normalized_text(text))
        if token and token.lower() not in STOPWORDS
    ]


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value: object, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(str(value)))
    except Exception:
        return default


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


def normalize_label(label: object) -> str:
    cleaned = clean_text(label)
    upper = cleaned.upper()
    if upper in LABEL_ORDER:
        return upper
    return LABEL_ALIASES.get(cleaned.lower(), cleaned)


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


class FullEsciRetrievalEngine:
    def __init__(self, index_dir: Path) -> None:
        self.index_dir = index_dir
        self.products: List[Dict[str, object]] = []
        self.queries: List[Dict[str, str]] = []
        self.labels_by_query_product: Dict[Tuple[str, str], str] = {}
        self.labels_by_query: Dict[str, Dict[str, str]] = defaultdict(dict)
        self.idf: Dict[str, float] = {}
        self.index_summary: Dict[str, object] = {}

    def required_paths(self) -> Dict[str, Path]:
        return {
            "products": self.index_dir / "products.csv",
            "queries": self.index_dir / "queries.csv",
            "query_product_labels": self.index_dir / "query_product_labels.csv",
            "product_text_index": self.index_dir / "product_text_index.jsonl",
            "index_summary": self.index_dir / "index_summary.json",
        }

    def validate_index(self) -> None:
        missing = [str(path) for path in self.required_paths().values() if not path.exists()]
        if missing:
            raise FileNotFoundError(
                "Missing ESCI index files: "
                + ", ".join(missing)
                + ". Run python -m src.full_esci_product_index_builder --source full_parquet --max-rows 10000 first"
            )

    def load(self) -> None:
        self.validate_index()
        paths = self.required_paths()
        products_by_id = {clean_text(row.get("product_id")): dict(row) for row in read_csv(paths["products"])}

        text_rows = []
        with paths["product_text_index"].open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                parsed = json.loads(line)
                if isinstance(parsed, dict):
                    text_rows.append(parsed)

        self.products = []
        doc_freq = Counter()
        for text_row in text_rows:
            product_id = clean_text(text_row.get("product_id"))
            product = dict(products_by_id.get(product_id, {}))
            product["product_id"] = product_id
            product["product_title"] = clean_text(product.get("product_title") or text_row.get("product_title"))
            product["product_brand"] = clean_text(product.get("product_brand") or text_row.get("brand"))
            product["product_color"] = clean_text(product.get("product_color"))
            product["search_text"] = clean_text(text_row.get("text")) or " ".join(
                [
                    clean_text(product.get("product_title")),
                    clean_text(product.get("product_brand")),
                    clean_text(product.get("product_description")),
                    clean_text(product.get("product_color")),
                ]
            )
            title_tokens = tokenize(product.get("product_title"))
            text_tokens = tokenize(product.get("search_text"))
            brand_tokens = tokenize(product.get("product_brand"))
            product["title_tokens"] = set(title_tokens)
            product["text_tokens"] = set(text_tokens)
            product["brand_tokens"] = set(brand_tokens)
            product["all_tokens"] = set(title_tokens).union(text_tokens).union(brand_tokens)
            if product["all_tokens"]:
                doc_freq.update(product["all_tokens"])
            self.products.append(product)

        total_docs = max(len(self.products), 1)
        self.idf = {
            token: math.log((1 + total_docs) / (1 + freq)) + 1.0
            for token, freq in doc_freq.items()
        }

        self.queries = read_csv(paths["queries"])
        for row in read_csv(paths["query_product_labels"]):
            query = normalized_text(row.get("query"))
            product_id = clean_text(row.get("product_id"))
            label = normalize_label(row.get("esci_label"))
            if query and product_id:
                self.labels_by_query_product[(query, product_id)] = label
                self.labels_by_query[query][product_id] = label

        try:
            self.index_summary = json.loads(paths["index_summary"].read_text(encoding="utf-8"))
        except Exception:
            self.index_summary = {}

    def score_product(self, query: str, product: Dict[str, object]) -> Dict[str, object]:
        query_tokens = set(tokenize(query))
        if not query_tokens:
            return {"score": 0.0, "matched_tokens": [], "coverage": 0.0}

        title_tokens = product.get("title_tokens", set())
        text_tokens = product.get("text_tokens", set())
        brand_tokens = product.get("brand_tokens", set())
        if not isinstance(title_tokens, set):
            title_tokens = set()
        if not isinstance(text_tokens, set):
            text_tokens = set()
        if not isinstance(brand_tokens, set):
            brand_tokens = set()

        title_overlap = query_tokens.intersection(title_tokens)
        text_overlap = query_tokens.intersection(text_tokens)
        brand_overlap = query_tokens.intersection(brand_tokens)
        all_overlap = query_tokens.intersection(title_tokens.union(text_tokens).union(brand_tokens))
        coverage = len(all_overlap) / max(len(query_tokens), 1)
        idf_score = sum(self.idf.get(token, 1.0) for token in all_overlap)
        query_norm = normalized_text(query)
        title_norm = normalized_text(product.get("product_title"))
        exact_phrase_in_title = int(bool(query_norm) and query_norm in title_norm)
        empty_penalty = 2.0 if not title_norm and not clean_text(product.get("search_text")) else 0.0

        score = (
            2.0 * len(title_overlap)
            + 1.0 * len(text_overlap)
            + 2.0 * exact_phrase_in_title
            + 0.5 * coverage
            + 1.5 * len(brand_overlap)
            + 0.35 * idf_score
            - empty_penalty
        )

        return {
            "score": round(score, 6),
            "matched_tokens": sorted(all_overlap),
            "coverage": round(coverage, 6),
        }

    def retrieve(self, query: str, top_k: int) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
        start = time.perf_counter()
        query_norm = normalized_text(query)
        scored = []
        labels_for_query = self.labels_by_query.get(query_norm, {})

        for product in self.products:
            score_info = self.score_product(query, product)
            score = safe_float(score_info.get("score"))
            if score <= 0:
                continue
            product_id = clean_text(product.get("product_id"))
            scored.append(
                {
                    "product": product,
                    "score": score,
                    "matched_tokens": score_info.get("matched_tokens", []),
                    "coverage": score_info.get("coverage", 0.0),
                    "esci_label": labels_for_query.get(product_id, ""),
                }
            )

        scored.sort(
            key=lambda row: (
                -safe_float(row.get("score")),
                clean_text(row["product"].get("product_title")),  # type: ignore[index]
                clean_text(row["product"].get("product_id")),  # type: ignore[index]
            )
        )

        rows = []
        for rank, row in enumerate(scored[:top_k], start=1):
            product = row["product"]  # type: ignore[assignment]
            rows.append(
                {
                    "query": query,
                    "rank": rank,
                    "product_id": clean_text(product.get("product_id")),
                    "product_title": clean_text(product.get("product_title")),
                    "product_brand": clean_text(product.get("product_brand")),
                    "product_color": clean_text(product.get("product_color")),
                    "product_text": clean_text(product.get("search_text")),
                    "score": row.get("score"),
                    "matched_tokens": "|".join(row.get("matched_tokens", [])),  # type: ignore[arg-type]
                    "query_token_coverage": row.get("coverage"),
                    "retrieval_source": "full_esci_lexical",
                    "esci_label": row.get("esci_label", ""),
                }
            )

        elapsed = time.perf_counter() - start
        evaluation = evaluate_results(query=query, results=rows, labels_for_query=labels_for_query)
        evaluation["retrieval_time_seconds"] = round(elapsed, 6)
        return rows, evaluation


class FullEsciFtsRetrievalEngine(FullEsciRetrievalEngine):
    def __init__(self, index_dir: Path, allow_fallback: bool = False) -> None:
        super().__init__(index_dir)
        self.allow_fallback = allow_fallback
        self.sqlite_path = index_dir / SQLITE_FTS_NAME
        self.conn: sqlite3.Connection | None = None
        self.lexical_fallback: FullEsciRetrievalEngine | None = None

    def validate_fts_index(self) -> None:
        if not self.sqlite_path.exists():
            raise FileNotFoundError(
                f"Missing FTS index: {self.sqlite_path}. "
                "Run python -m src.full_esci_fts_index_builder --rebuild first"
            )

    def load(self) -> None:
        self.validate_index()
        self.validate_fts_index()
        self.conn = sqlite3.connect(self.sqlite_path)
        self.conn.row_factory = sqlite3.Row
        self.queries = read_csv(self.required_paths()["queries"])
        for row in read_csv(self.required_paths()["query_product_labels"]):
            query = normalized_text(row.get("query"))
            product_id = clean_text(row.get("product_id"))
            label = normalize_label(row.get("esci_label"))
            if query and product_id:
                self.labels_by_query_product[(query, product_id)] = label
                self.labels_by_query[query][product_id] = label
        try:
            self.index_summary = json.loads(self.required_paths()["index_summary"].read_text(encoding="utf-8"))
        except Exception:
            self.index_summary = {}

    def safe_match_query(self, query: str) -> str:
        tokens = fts_tokens(query)
        if not tokens:
            return ""
        return " OR ".join(f'"{token}"' for token in tokens[:12])

    def total_products(self) -> int:
        if not self.conn:
            return 0
        return int(self.conn.execute("SELECT COUNT(*) FROM products").fetchone()[0])

    def retrieve(self, query: str, top_k: int) -> Tuple[List[Dict[str, object]], Dict[str, object]]:
        start = time.perf_counter()
        if not self.conn:
            raise RuntimeError("FTS engine is not loaded.")

        match_query = self.safe_match_query(query)
        if not match_query:
            rows: List[Dict[str, object]] = []
            evaluation = evaluate_results(query=query, results=rows, labels_for_query={})
            evaluation["retrieval_time_seconds"] = round(time.perf_counter() - start, 6)
            return rows, evaluation

        labels_for_query = self.labels_by_query.get(normalized_text(query), {})
        try:
            sql = """
                SELECT product_id, product_title, product_brand, product_color, product_text,
                       bm25(products_fts) AS bm25_score
                FROM products_fts
                WHERE products_fts MATCH ?
                ORDER BY bm25_score
                LIMIT ?
            """
            raw_rows = list(self.conn.execute(sql, (match_query, top_k)))
        except sqlite3.Error:
            if self.allow_fallback:
                if self.lexical_fallback is None:
                    self.lexical_fallback = FullEsciRetrievalEngine(self.index_dir)
                    self.lexical_fallback.load()
                return self.lexical_fallback.retrieve(query, top_k=top_k)
            raise

        query_tokens = set(tokenize(query))
        results = []
        for rank, row in enumerate(raw_rows, start=1):
            title = clean_text(row["product_title"])
            brand = clean_text(row["product_brand"])
            color = clean_text(row["product_color"])
            product_text = clean_text(row["product_text"])
            product_id = clean_text(row["product_id"])
            matched = sorted(query_tokens.intersection(set(tokenize(" ".join([title, brand, color, product_text])))))
            coverage = round(len(matched) / max(len(query_tokens), 1), 6)
            results.append(
                {
                    "query": query,
                    "rank": rank,
                    "product_id": product_id,
                    "product_title": title,
                    "product_brand": brand,
                    "product_color": color,
                    "product_text": product_text,
                    "score": round(-safe_float(row["bm25_score"]), 6),
                    "matched_tokens": "|".join(matched),
                    "query_token_coverage": coverage,
                    "retrieval_source": "full_esci_fts",
                    "esci_label": labels_for_query.get(product_id, ""),
                }
            )

        evaluation = evaluate_results(query=query, results=results, labels_for_query=labels_for_query)
        evaluation["retrieval_time_seconds"] = round(time.perf_counter() - start, 6)
        return results, evaluation


def get_engine(
    index_dir: Path = DEFAULT_INDEX_DIR,
    backend: str = "lexical",
    allow_fallback: bool = False,
) -> FullEsciRetrievalEngine:
    backend = normalized_text(backend) or "lexical"
    key = f"{backend}:{allow_fallback}:{index_dir.resolve()}"
    engine = _ENGINE_CACHE.get(key)
    if engine is None:
        if backend == "fts":
            engine = FullEsciFtsRetrievalEngine(index_dir, allow_fallback=allow_fallback)
        else:
            engine = FullEsciRetrievalEngine(index_dir)
        engine.load()
        _ENGINE_CACHE[key] = engine
    return engine  # type: ignore[return-value]


def retrieve_products(
    query: str,
    top_k: int = 12,
    index_dir: Path = DEFAULT_INDEX_DIR,
    backend: str = "lexical",
    allow_fallback: bool = False,
) -> List[Dict[str, object]]:
    engine = get_engine(index_dir, backend=backend, allow_fallback=allow_fallback)
    rows, _evaluation = engine.retrieve(query, top_k=top_k)
    return rows


def evaluate_results(query: str, results: List[Dict[str, object]], labels_for_query: Dict[str, str]) -> Dict[str, object]:
    labels = [normalize_label(row.get("esci_label")) for row in results if clean_text(row.get("esci_label"))]
    counts = Counter(labels)
    labeled_count = len(labels)
    return {
        "query": query,
        "retrieved_count": len(results),
        "retrieved_labeled_count": labeled_count,
        "exact_count": counts.get("E", 0),
        "substitute_count": counts.get("S", 0),
        "complement_count": counts.get("C", 0),
        "irrelevant_count": counts.get("I", 0),
        "top_k_has_exact": int(counts.get("E", 0) > 0),
        "top_k_has_exact_or_substitute": int(counts.get("E", 0) > 0 or counts.get("S", 0) > 0),
        "best_esci_label": best_label(labels),
        "label_coverage_rate": round(labeled_count / max(len(results), 1), 6),
        "known_label_count_for_query": len(labels_for_query),
    }


def select_indexed_queries(engine: FullEsciRetrievalEngine, sample_size: int) -> List[str]:
    queries = []
    seen = set()
    for row in engine.queries:
        query = clean_text(row.get("query"))
        key = normalized_text(query)
        if not query or key in seen:
            continue
        seen.add(key)
        queries.append(query)
        if len(queries) >= sample_size:
            break
    return queries


def summarize_batch(evaluations: List[Dict[str, object]], top_k: int, runtime_seconds: float) -> Dict[str, object]:
    total = len(evaluations)
    success_rows = [row for row in evaluations if not clean_text(row.get("error_message"))]
    return {
        "total_queries": total,
        "success_count": len(success_rows),
        "failure_count": total - len(success_rows),
        "avg_retrieved_count": round(
            sum(safe_int(row.get("retrieved_count")) for row in success_rows) / max(len(success_rows), 1),
            6,
        ),
        "avg_retrieval_time_seconds": round(
            sum(safe_float(row.get("retrieval_time_seconds")) for row in success_rows) / max(len(success_rows), 1),
            6,
        ),
        "top_k": top_k,
        "top_k_has_exact_rate": round(
            sum(safe_int(row.get("top_k_has_exact")) for row in success_rows) / max(len(success_rows), 1),
            6,
        ),
        "top_k_has_exact_or_substitute_rate": round(
            sum(safe_int(row.get("top_k_has_exact_or_substitute")) for row in success_rows) / max(len(success_rows), 1),
            6,
        ),
        "avg_labeled_retrieved_count": round(
            sum(safe_int(row.get("retrieved_labeled_count")) for row in success_rows) / max(len(success_rows), 1),
            6,
        ),
        "avg_label_coverage_rate": round(
            sum(safe_float(row.get("label_coverage_rate")) for row in success_rows) / max(len(success_rows), 1),
            6,
        ),
        "runtime_seconds": round(runtime_seconds, 4),
    }


def report_markdown(summary: Dict[str, object], evaluations: List[Dict[str, object]]) -> str:
    lines = ["# MVP 26 Full ESCI Retrieval Engine", "", "## Summary"]
    for key, value in summary.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Sample Queries"])
    for row in evaluations[:20]:
        lines.append(
            f"- {row.get('query')}: retrieved={row.get('retrieved_count')} "
            f"exact={row.get('top_k_has_exact')} best={row.get('best_esci_label')}"
        )
    return "\n".join(lines) + "\n"


def write_batch_outputs(
    output_dir: Path,
    result_rows: List[Dict[str, object]],
    evaluations: List[Dict[str, object]],
    summary: Dict[str, object],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_fields = list(evaluations[0].keys()) if evaluations else ["query"]
    write_csv(output_dir / "full_esci_retrieval_results.csv", result_rows, RESULT_FIELDS)
    write_csv(output_dir / "full_esci_retrieval_summary.csv", [summary], SUMMARY_FIELDS)
    write_csv(output_dir / "full_esci_retrieval_query_eval.csv", evaluations, eval_fields)
    (output_dir / "full_esci_retrieval_report.md").write_text(
        report_markdown(summary, evaluations),
        encoding="utf-8",
    )


def print_single(
    query: str,
    top_k: int,
    index_dir: Path,
    engine: FullEsciRetrievalEngine,
    results: List[Dict[str, object]],
    evaluation: Dict[str, object],
) -> None:
    print("\nMVP 26 Full ESCI Retrieval Engine")
    print("-" * 100)
    print(f"query: {query}")
    print(f"top_k: {top_k}")
    print(f"index_dir: {index_dir}")
    total_products = engine.total_products() if hasattr(engine, "total_products") else len(engine.products)
    print(f"total_products_loaded: {total_products}")
    print(f"retrieval_backend: {clean_text(getattr(engine, '__class__', type(engine)).__name__)}")
    print(f"retrieval_time_seconds: {evaluation.get('retrieval_time_seconds')}")
    print("\nEvaluation")
    print("-" * 100)
    for key in [
        "retrieved_labeled_count",
        "exact_count",
        "substitute_count",
        "complement_count",
        "irrelevant_count",
        "top_k_has_exact",
        "top_k_has_exact_or_substitute",
        "best_esci_label",
        "label_coverage_rate",
        "known_label_count_for_query",
    ]:
        print(f"{key}: {evaluation.get(key)}")

    print("\nTop Results")
    print("-" * 100)
    for row in results:
        title = clean_text(row.get("product_title"))
        if len(title) > 80:
            title = title[:77] + "..."
        print(
            f"{row.get('rank')}. score={row.get('score')} id={row.get('product_id')} "
            f"label={row.get('esci_label') or ''} brand={row.get('product_brand') or ''} title={title}"
        )


def run_single(args: argparse.Namespace) -> None:
    engine = get_engine(
        Path(args.index_dir),
        backend=args.backend,
        allow_fallback=args.allow_fallback,
    )
    results, evaluation = engine.retrieve(args.query, top_k=args.top_k)
    print_single(
        query=args.query,
        top_k=args.top_k,
        index_dir=Path(args.index_dir),
        engine=engine,
        results=results,
        evaluation=evaluation,
    )


def run_batch(args: argparse.Namespace) -> None:
    engine = get_engine(
        Path(args.index_dir),
        backend=args.backend,
        allow_fallback=args.allow_fallback,
    )
    if args.query_mode != "indexed_queries":
        raise ValueError("--query-mode currently supports indexed_queries")

    queries = select_indexed_queries(engine, sample_size=args.sample_size)
    print("\nMVP 26 Full ESCI Retrieval Engine")
    print("-" * 100)
    print(f"query_mode: {args.query_mode}")
    print(f"sample_size: {args.sample_size}")
    print(f"selected_query_count: {len(queries)}")
    print(f"top_k: {args.top_k}")
    print(f"index_dir: {args.index_dir}")
    print(f"backend: {args.backend}")
    total_products = engine.total_products() if hasattr(engine, "total_products") else len(engine.products)
    print(f"total_products_loaded: {total_products}")

    all_results: List[Dict[str, object]] = []
    evaluations: List[Dict[str, object]] = []
    start = time.perf_counter()
    for index, query in enumerate(queries, start=1):
        try:
            results, evaluation = engine.retrieve(query, top_k=args.top_k)
            all_results.extend(results)
            evaluations.append(evaluation)
            if index <= 10 or index % 100 == 0 or index == len(queries):
                print(
                    f"[{index}/{len(queries)}] {query} -> retrieved={evaluation['retrieved_count']} "
                    f"exact={evaluation['top_k_has_exact']} best={evaluation['best_esci_label']}"
                )
        except Exception as exc:
            evaluations.append({"query": query, "error_message": str(exc)[:1000]})

    runtime = time.perf_counter() - start
    summary = summarize_batch(evaluations, top_k=args.top_k, runtime_seconds=runtime)
    write_batch_outputs(Path(args.output_dir), all_results, evaluations, summary)

    print("\nSummary")
    print("-" * 100)
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("\nOutput files")
    print("-" * 100)
    print(Path(args.output_dir) / "full_esci_retrieval_results.csv")
    print(Path(args.output_dir) / "full_esci_retrieval_summary.csv")
    print(Path(args.output_dir) / "full_esci_retrieval_query_eval.csv")
    print(Path(args.output_dir) / "full_esci_retrieval_report.md")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MVP 26 Full ESCI Retrieval Engine")
    parser.add_argument("--query", default="", help="Single query to retrieve.")
    parser.add_argument("--top-k", type=int, default=10, help="Number of products to return.")
    parser.add_argument("--sample-size", type=int, default=10, help="Batch query sample size.")
    parser.add_argument(
        "--query-mode",
        choices=["indexed_queries"],
        default="indexed_queries",
        help="Batch query source mode.",
    )
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR), help="ESCI index directory.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Batch output directory.")
    parser.add_argument("--backend", choices=["lexical", "fts"], default="lexical", help="Retrieval backend.")
    parser.add_argument(
        "--allow-fallback",
        action="store_true",
        help="Allow FTS backend to fall back to lexical retrieval on FTS query errors.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")
    if args.sample_size <= 0:
        raise ValueError("--sample-size must be positive")

    if args.query:
        run_single(args)
        return
    run_batch(args)


if __name__ == "__main__":
    main()
