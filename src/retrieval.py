import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def build_product_text(row: pd.Series) -> str:
    """
    Combines product fields into one searchable text field.
    Later, Amazon ESCI title/description/brand/category fields will go here.
    """
    return f"{row['product_title']} {row['category']}"


def tfidf_search(query_text: str, df: pd.DataFrame, top_k: int = 10) -> pd.DataFrame:
    """
    Retrieves products using TF-IDF cosine similarity.
    """
    result_df = df.copy()

    result_df["search_text"] = result_df.apply(build_product_text, axis=1)

    documents = result_df["search_text"].tolist()
    documents.append(query_text)

    vectorizer = TfidfVectorizer(
        lowercase=True,
        stop_words="english",
        ngram_range=(1, 2)
    )

    tfidf_matrix = vectorizer.fit_transform(documents)

    product_vectors = tfidf_matrix[:-1]
    query_vector = tfidf_matrix[-1]

    similarities = cosine_similarity(query_vector, product_vectors).flatten()

    result_df["retrieval_score"] = similarities
    result_df["baseline_score"] = result_df["retrieval_score"]

    result_df = result_df.sort_values(
        by=["retrieval_score", "rating"],
        ascending=False
    )

    return result_df.head(top_k)