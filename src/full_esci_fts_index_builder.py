"""
MVP 26.5: SQLite FTS5 index builder for full ESCI products.

Builds data/esci_index/esci_products_fts.sqlite from the CSV/JSONL artifacts
created by the full ESCI product index builder.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sqlite3
import time
from pathlib import Path
from typing import Dict, Iterable, List


DEFAULT_INDEX_DIR = Path("data/esci_index")
SQLITE_NAME = "esci_products_fts.sqlite"


def clean_text(value: object) -> str:
    return str(value or "").strip()


def safe_int(value: object, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(str(value)))
    except Exception:
        return default


def read_csv_by_id(path: Path) -> Dict[str, Dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", newline="", encoding="utf-8") as f:
        return {clean_text(row.get("product_id")): dict(row) for row in csv.DictReader(f) if clean_text(row.get("product_id"))}


def iter_product_text_rows(path: Path) -> Iterable[Dict[str, object]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(row, dict) and clean_text(row.get("product_id")):
                yield row


def safe_fts_query(query: str) -> str:
    tokens = [token for token in re.findall(r"[a-zA-Z0-9]+", query.lower()) if token]
    if not tokens:
        return ""
    return " OR ".join(f'"{token}"' for token in tokens[:12])


def connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.execute("PRAGMA journal_mode=DELETE")
    conn.execute("PRAGMA synchronous=OFF")
    conn.execute("PRAGMA temp_store=MEMORY")
    return conn


def ensure_fts5(conn: sqlite3.Connection) -> None:
    try:
        conn.execute("CREATE VIRTUAL TABLE temp.fts5_check USING fts5(value)")
        conn.execute("DROP TABLE temp.fts5_check")
    except sqlite3.Error as exc:
        raise RuntimeError("SQLite FTS5 is not available in this Python build.") from exc


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        DROP TABLE IF EXISTS products;
        DROP TABLE IF EXISTS products_fts;

        CREATE TABLE products(
          product_id TEXT PRIMARY KEY,
          product_title TEXT,
          product_brand TEXT,
          product_color TEXT,
          product_text TEXT
        );

        CREATE VIRTUAL TABLE products_fts USING fts5(
          product_id UNINDEXED,
          product_title,
          product_brand,
          product_color,
          product_text
        );
        """
    )


def fts_table_exists(conn: sqlite3.Connection) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='products_fts'"
    ).fetchone()
    return row is not None


def batch_insert(conn: sqlite3.Connection, rows: List[Dict[str, object]]) -> None:
    metadata_rows = [
        (
            clean_text(row.get("product_id")),
            clean_text(row.get("product_title")),
            clean_text(row.get("product_brand") or row.get("brand")),
            clean_text(row.get("product_color")),
            clean_text(row.get("product_text") or row.get("text")),
        )
        for row in rows
    ]
    conn.executemany(
        "INSERT OR REPLACE INTO products(product_id, product_title, product_brand, product_color, product_text) VALUES (?, ?, ?, ?, ?)",
        metadata_rows,
    )
    conn.executemany(
        "INSERT INTO products_fts(product_id, product_title, product_brand, product_color, product_text) VALUES (?, ?, ?, ?, ?)",
        metadata_rows,
    )


def sample_query_retrieval_count(sqlite_path: Path, query: str, top_k: int) -> int:
    if not sqlite_path.exists():
        return 0
    match_query = safe_fts_query(query)
    if not match_query:
        return 0
    conn = sqlite3.connect(sqlite_path)
    try:
        count = len(
            list(
                conn.execute(
                    """
                    SELECT product_id
                    FROM products_fts
                    WHERE products_fts MATCH ?
                    LIMIT ?
                    """,
                    (match_query, top_k),
                )
            )
        )
    finally:
        conn.close()
    return count


def build_index(
    index_dir: Path,
    rebuild: bool,
    max_products: int | None = None,
    validate_query: str = "wireless mouse",
    validate_top_k: int = 10,
) -> None:
    products_csv = index_dir / "products.csv"
    product_text_jsonl = index_dir / "product_text_index.jsonl"
    sqlite_path = index_dir / SQLITE_NAME

    if sqlite_path.exists() and not rebuild:
        print(f"FTS index already exists: {sqlite_path}")
        print("Use --rebuild to recreate it.")
        return
    if not products_csv.exists() or not product_text_jsonl.exists():
        raise FileNotFoundError("Missing products.csv or product_text_index.jsonl. Build data/esci_index first.")

    start = time.perf_counter()
    index_dir.mkdir(parents=True, exist_ok=True)
    products_by_id = read_csv_by_id(products_csv)

    if sqlite_path.exists():
        sqlite_path.unlink()
    journal_path = sqlite_path.with_name(sqlite_path.name + "-journal")
    if journal_path.exists():
        journal_path.unlink()

    conn = connect(sqlite_path)
    ensure_fts5(conn)
    create_schema(conn)

    count = 0
    batch: List[Dict[str, object]] = []
    for text_row in iter_product_text_rows(product_text_jsonl):
        if max_products is not None and count + len(batch) >= max_products:
            break
        product_id = clean_text(text_row.get("product_id"))
        product = dict(products_by_id.get(product_id, {}))
        product["product_id"] = product_id
        product["product_title"] = clean_text(product.get("product_title") or text_row.get("product_title"))
        product["product_brand"] = clean_text(product.get("product_brand") or text_row.get("brand"))
        product["product_color"] = clean_text(product.get("product_color"))
        product["product_text"] = clean_text(text_row.get("text"))
        batch.append(product)
        if len(batch) >= 5000:
            batch_insert(conn, batch)
            conn.commit()
            count += len(batch)
            print(f"inserted: {count}")
            batch = []

    if batch:
        batch_insert(conn, batch)
        conn.commit()
        count += len(batch)

    conn.execute("INSERT INTO products_fts(products_fts) VALUES('optimize')")
    conn.commit()
    table_exists = fts_table_exists(conn)
    conn.close()

    runtime = time.perf_counter() - start
    sample_count = sample_query_retrieval_count(sqlite_path, validate_query, validate_top_k)
    print("\nMVP 26.5 SQLite FTS Index Builder")
    print("-" * 80)
    print(f"index_dir: {index_dir}")
    print(f"sqlite_path: {sqlite_path}")
    print(f"requested_product_limit: {max_products if max_products is not None else 'all'}")
    print(f"products_indexed: {count}")
    print(f"fts_table_exists: {int(table_exists)}")
    print(f"sample_query: {validate_query}")
    print(f"sample_query_retrieval_count: {sample_count}")
    print(f"runtime_seconds: {runtime:.4f}")


def inspect_index(index_dir: Path, validate_query: str = "wireless mouse", validate_top_k: int = 10) -> None:
    sqlite_path = index_dir / SQLITE_NAME
    print("\nMVP 26.5 SQLite FTS Index Inspect")
    print("-" * 80)
    print(f"sqlite_path: {sqlite_path}")
    if not sqlite_path.exists():
        print("FTS index missing. Run python -m src.full_esci_fts_index_builder --index-dir data/esci_index --rebuild")
        return

    conn = sqlite3.connect(sqlite_path)
    exists = fts_table_exists(conn)
    print(f"fts_table_exists: {int(exists)}")
    for table in ["products", "products_fts"]:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"{table}_count: {count}")
    print("\nSample rows")
    for row in conn.execute(
        "SELECT product_id, product_title, product_brand FROM products ORDER BY product_id LIMIT 5"
    ):
        print(f"- id={row[0]} brand={row[2]} title={row[1]}")
    conn.close()
    print(f"\nsample_query: {validate_query}")
    print(f"sample_query_retrieval_count: {sample_query_retrieval_count(sqlite_path, validate_query, validate_top_k)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SQLite FTS5 ESCI product index.")
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR), help="ESCI index directory.")
    parser.add_argument("--rebuild", action="store_true", help="Recreate the SQLite FTS index.")
    parser.add_argument("--inspect", action="store_true", help="Inspect existing SQLite FTS index.")
    parser.add_argument(
        "--max-products",
        type=int,
        default=0,
        help="Optional maximum products to insert into FTS. Use 500000 for a capped 500k build; 0 means all products in the index.",
    )
    parser.add_argument("--validate-query", default="wireless mouse", help="Sample query for post-build/inspect validation.")
    parser.add_argument("--validate-top-k", type=int, default=10, help="Sample validation retrieval limit.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index_dir = Path(args.index_dir)
    max_products = safe_int(args.max_products)
    if max_products < 0:
        raise ValueError("--max-products must be non-negative")
    if args.validate_top_k <= 0:
        raise ValueError("--validate-top-k must be positive")
    if args.inspect:
        inspect_index(index_dir, validate_query=args.validate_query, validate_top_k=args.validate_top_k)
        return
    build_index(
        index_dir=index_dir,
        rebuild=args.rebuild,
        max_products=max_products or None,
        validate_query=args.validate_query,
        validate_top_k=args.validate_top_k,
    )


if __name__ == "__main__":
    main()
