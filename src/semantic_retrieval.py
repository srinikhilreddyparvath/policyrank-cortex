import os
import pickle
import hashlib
from typing import Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity


MODEL_NAME = "all-MiniLM-L6-v2"
MODEL_DIR = "models"
CACHE_PATH = os.path.join(MODEL_DIR, "semantic_embeddings_cache.pkl")


def ensure_model_dir():
    os.makedirs(MODEL_DIR, exist_ok=True)


def get_search_text_column(df: pd.DataFrame) -> pd.Series:
    """
    Builds searchable text for semantic retrieval.

    Uses search_text if available.
    Otherwise falls back to product_title + category + brand + query.
    """

    if "search_text" in df.columns:
        return df["search_text"].fillna("").astype(str)

    text_parts = []

    for col in ["product_title", "product_description", "category", "brand", "query"]:
        if col in df.columns:
            text_parts.append(df[col].fillna("").astype(str))

    if not text_parts:
        return pd.Series([""] * len(df))

    combined = text_parts[0]

    for part in text_parts[1:]:
        combined = combined + " " + part

    return combined.fillna("").astype(str)


def compute_dataset_signature(df: pd.DataFrame) -> str:
    """
    Creates a lightweight dataset signature so we know when embeddings are stale.

    This prevents the old 5,000-row embedding cache from being reused on the new
    51,077-row balanced sample.
    """

    row_count = len(df)

    cols = "|".join(list(df.columns))

    sample_text = ""

    if len(df) > 0:
        search_text = get_search_text_column(df)
        sample_text = " ".join(search_text.head(50).tolist())

    raw = f"{row_count}|{cols}|{sample_text}"
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def load_cache():
    if not os.path.exists(CACHE_PATH):
        return None

    try:
        with open(CACHE_PATH, "rb") as file:
            return pickle.load(file)
    except Exception:
        return None


def save_cache(cache_data):
    ensure_model_dir()

    with open(CACHE_PATH, "wb") as file:
        pickle.dump(cache_data, file)


def build_embeddings(df: pd.DataFrame) -> Tuple[SentenceTransformer, np.ndarray]:
    """
    Builds embeddings for all products in the current dataframe.
    """

    model = SentenceTransformer(MODEL_NAME)

    search_text = get_search_text_column(df).tolist()

    embeddings = model.encode(
        search_text,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    return model, embeddings


def get_model_and_embeddings(df: pd.DataFrame) -> Tuple[SentenceTransformer, np.ndarray]:
    """
    Loads cached embeddings if valid.
    Rebuilds them if:
    - no cache exists
    - dataframe row count changed
    - dataset signature changed
    - embedding length does not match dataframe length
    """

    ensure_model_dir()

    dataset_signature = compute_dataset_signature(df)
    cache = load_cache()

    if cache is not None:
        cached_signature = cache.get("dataset_signature")
        cached_row_count = cache.get("row_count")
        embeddings = cache.get("embeddings")

        cache_is_valid = (
            cached_signature == dataset_signature
            and cached_row_count == len(df)
            and embeddings is not None
            and len(embeddings) == len(df)
        )

        if cache_is_valid:
            model = SentenceTransformer(MODEL_NAME)
            return model, embeddings

    model, embeddings = build_embeddings(df)

    save_cache(
        {
            "dataset_signature": dataset_signature,
            "row_count": len(df),
            "model_name": MODEL_NAME,
            "embeddings": embeddings,
        }
    )

    return model, embeddings


def semantic_search(query: str, products_df: pd.DataFrame, top_k: int = 20) -> pd.DataFrame:
    """
    Semantic product retrieval using Sentence Transformers.

    Returns a dataframe with:
    - semantic_score
    - baseline_score
    """

    if products_df is None or len(products_df) == 0:
        return pd.DataFrame()

    working_df = products_df.copy().reset_index(drop=True)

    model, product_embeddings = get_model_and_embeddings(working_df)

    if len(product_embeddings) != len(working_df):
        raise ValueError(
            f"Embedding length mismatch. Embeddings={len(product_embeddings)}, "
            f"rows={len(working_df)}. Delete models/semantic_embeddings_cache.pkl and retry."
        )

    query_embedding = model.encode(
        [query],
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    similarities = cosine_similarity(query_embedding, product_embeddings)[0]

    working_df["semantic_score"] = similarities
    working_df["baseline_score"] = working_df["semantic_score"]

    result_df = working_df.sort_values(
        by="semantic_score",
        ascending=False,
    ).head(top_k)

    return result_df.reset_index(drop=True)