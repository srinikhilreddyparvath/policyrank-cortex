"""
MVP 26.5: SQLite FTS5 index builder for full ESCI products.

Builds data/esci_index/esci_products_fts.sqlite from the CSV/JSONL artifacts
created by the full ESCI product index builder.
"""

from __future__ import annotations

import argparse
import csv
import json
import sqlite3
import time
from pathlib import Path
from typing import Dict, Iterable, List


DEFAULT_INDEX_DIR = Path("data/esci_index")
SQLITE_NAME = "esci_products_fts.sqlite"


def clean_text(value: object) -> str:
    return str(value or "").strip()


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


def build_index(index_dir: Path, rebuild: bool) -> None:
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
    conn.close()

    runtime = time.perf_counter() - start
    print("\nMVP 26.5 SQLite FTS Index Builder")
    print("-" * 80)
    print(f"index_dir: {index_dir}")
    print(f"sqlite_path: {sqlite_path}")
    print(f"products_indexed: {count}")
    print(f"runtime_seconds: {runtime:.4f}")


def inspect_index(index_dir: Path) -> None:
    sqlite_path = index_dir / SQLITE_NAME
    print("\nMVP 26.5 SQLite FTS Index Inspect")
    print("-" * 80)
    print(f"sqlite_path: {sqlite_path}")
    if not sqlite_path.exists():
        print("FTS index missing. Run python -m src.full_esci_fts_index_builder --index-dir data/esci_index --rebuild")
        return

    conn = sqlite3.connect(sqlite_path)
    for table in ["products", "products_fts"]:
        count = conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0]
        print(f"{table}_count: {count}")
    print("\nSample rows")
    for row in conn.execute(
        "SELECT product_id, product_title, product_brand FROM products ORDER BY product_id LIMIT 5"
    ):
        print(f"- id={row[0]} brand={row[2]} title={row[1]}")
    conn.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build SQLite FTS5 ESCI product index.")
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR), help="ESCI index directory.")
    parser.add_argument("--rebuild", action="store_true", help="Recreate the SQLite FTS index.")
    parser.add_argument("--inspect", action="store_true", help="Inspect existing SQLite FTS index.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    index_dir = Path(args.index_dir)
    if args.inspect:
        inspect_index(index_dir)
        return
    build_index(index_dir=index_dir, rebuild=args.rebuild)


if __name__ == "__main__":
    main()
