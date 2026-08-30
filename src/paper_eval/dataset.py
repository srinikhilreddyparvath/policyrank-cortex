from __future__ import annotations
import hashlib, json, subprocess
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

REQUIRED_EXAMPLE_COLUMNS = {"query", "query_id", "product_id", "product_locale", "esci_label", "small_version", "large_version", "split"}

def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest=hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""): digest.update(chunk)
    return digest.hexdigest()

@dataclass(frozen=True)
class DatasetPaths:
    root: Path; examples: Path; products: Path; sources: Path

def resolve_dataset(root: Path) -> DatasetPaths:
    root=root.resolve()
    paths=DatasetPaths(root, root/"shopping_queries_dataset_examples.parquet", root/"shopping_queries_dataset_products.parquet", root/"shopping_queries_dataset_sources.csv")
    missing=[str(p) for p in (paths.examples, paths.products, paths.sources) if not p.exists()]
    if missing: raise FileNotFoundError("Missing required ESCI files: " + ", ".join(missing))
    return paths

def load_examples(root: Path) -> pd.DataFrame:
    paths=resolve_dataset(root)
    frame=pd.read_parquet(paths.examples, columns=sorted(REQUIRED_EXAMPLE_COLUMNS))
    missing=REQUIRED_EXAMPLE_COLUMNS-set(frame.columns)
    if missing: raise ValueError(f"Examples parquet missing columns: {sorted(missing)}")
    frame["esci_label"]=frame["esci_label"].astype(str).str.upper()
    return frame

def task_examples(examples: pd.DataFrame, *, task_version: str="small", split: str="test", locale: str|None=None) -> pd.DataFrame:
    version_col={"small":"small_version", "large":"large_version"}.get(task_version)
    if version_col is None: raise ValueError("task_version must be small or large")
    selected=examples[(examples[version_col]==1)&(examples["split"]==split)].copy()
    if locale: selected=selected[selected["product_locale"]==locale]
    return selected.sort_values(["query_id","product_id"], kind="stable").reset_index(drop=True)

def select_query_ids(frame: pd.DataFrame, *, max_queries: int|None, seed: int) -> list[int]:
    ids=sorted(int(v) for v in frame["query_id"].unique())
    if max_queries is None or max_queries >= len(ids): return ids
    # Seeded ordering is deterministic and sampling remains query-level.
    import random
    shuffled=ids.copy(); random.Random(seed).shuffle(shuffled)
    return sorted(shuffled[:max_queries])

def preserve_selected_queries(frame: pd.DataFrame, query_ids: list[int]) -> pd.DataFrame:
    return frame[frame["query_id"].isin(query_ids)].sort_values(["query_id","product_id"], kind="stable").reset_index(drop=True)

def development_query_split(frame: pd.DataFrame, *, seed: int, train_fraction: float = 0.8) -> tuple[list[int], list[int]]:
    if not 0 < train_fraction < 1: raise ValueError("train_fraction must be between zero and one")
    import random
    ids=sorted(int(value) for value in frame["query_id"].unique()); random.Random(seed).shuffle(ids)
    boundary=int(len(ids)*train_fraction); train_fit=sorted(ids[:boundary]); validation=sorted(ids[boundary:])
    if set(train_fit)&set(validation): raise AssertionError("development query split overlap")
    return train_fit,validation

def policy_calibration_split(train_fit_ids: list[int], *, seed: int, policy_fraction: float = 0.8) -> tuple[list[int], list[int]]:
    if not 0 < policy_fraction < 1: raise ValueError("policy_fraction must be between zero and one")
    import random
    ids=sorted(int(value) for value in train_fit_ids); random.Random(seed).shuffle(ids)
    boundary=int(len(ids)*policy_fraction); policy_train=sorted(ids[:boundary]); calibration=sorted(ids[boundary:])
    if set(policy_train)&set(calibration): raise AssertionError("policy/calibration query split overlap")
    return policy_train,calibration

def load_products_for_candidates(root: Path, product_ids: set[str]) -> pd.DataFrame:
    paths=resolve_dataset(root)
    columns=["product_id","product_locale","product_title","product_description","product_bullet_point","product_brand","product_color"]
    products=pd.read_parquet(paths.products, columns=columns)
    return products[products["product_id"].astype(str).isin(product_ids)].drop_duplicates(["product_locale","product_id"], keep="first")

def _counts(series: pd.Series) -> dict[str,int]: return {str(k):int(v) for k,v in series.value_counts(dropna=False).sort_index().items()}

def build_dataset_manifest(root: Path, examples: pd.DataFrame|None=None, development_split: dict|None=None) -> dict:
    paths=resolve_dataset(root); examples=load_examples(root) if examples is None else examples
    product_keys=pd.read_parquet(paths.products, columns=["product_id","product_locale"])
    try: version=subprocess.check_output(["git","rev-parse","HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception: version=None
    return {
      "dataset_root":str(paths.root), "created_at_utc":datetime.now(timezone.utc).isoformat(), "code_version":version,
      "files":{
        "examples":{"path":str(paths.examples),"sha256":sha256_file(paths.examples),"row_count":int(len(examples))},
        "products":{"path":str(paths.products),"sha256":sha256_file(paths.products),"row_count":int(len(product_keys))},
        "sources":{"path":str(paths.sources),"sha256":sha256_file(paths.sources),"row_count":sum(1 for _ in paths.sources.open(encoding="utf-8"))-1},
      },
      "unique_query_count":int(examples["query_id"].nunique()), "unique_product_count":int(product_keys["product_id"].nunique()),
      "unique_product_locale_key_count":int(product_keys.drop_duplicates(["product_locale","product_id"]).shape[0]),
      "split_counts":_counts(examples["split"]), "locale_counts":_counts(examples["product_locale"]), "label_counts":_counts(examples["esci_label"]),
      "small_version_counts":_counts(examples["small_version"]), "large_version_counts":_counts(examples["large_version"]),
      "task1_small_test":{"row_count":int(((examples.small_version==1)&(examples.split=="test")).sum()), "unique_query_count":int(examples[(examples.small_version==1)&(examples.split=="test")]["query_id"].nunique())},
      "development_split":development_split,
    }

def write_json(path: Path, value: dict): path.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
