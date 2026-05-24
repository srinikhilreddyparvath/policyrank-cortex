import pandas as pd


ESCI_SCORE_MAP = {
    "E": 1.0,
    "S": 0.7,
    "C": 0.4,
    "I": 0.0
}


def normalize_series(series: pd.Series) -> pd.Series:
    min_value = series.min()
    max_value = series.max()

    if max_value == min_value:
        return series * 0.0

    return (series - min_value) / (max_value - min_value)


def add_diversity_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Simple diversity proxy:
    - Rewards items that have less repetitive product titles.
    - Later we will replace this with true multi-agent slate diversification.
    """

    result_df = df.copy()

    seen_tokens = set()
    diversity_scores = []

    for _, row in result_df.iterrows():
        title = str(row["product_title"]).lower()
        tokens = set(title.split())

        if len(tokens) == 0:
            diversity_scores.append(0.0)
            continue

        overlap = len(tokens.intersection(seen_tokens)) / len(tokens)
        diversity_score = 1.0 - overlap

        diversity_scores.append(diversity_score)
        seen_tokens.update(tokens)

    result_df["diversity_score"] = diversity_scores

    return result_df


def compile_and_apply_policy(
    df: pd.DataFrame,
    action_weights: dict,
    retrieval_mode: str
) -> pd.DataFrame:
    """
    Converts the selected RL action into an executable ranking formula.
    """

    result_df = df.copy()

    if "retrieval_score" not in result_df.columns:
        result_df["retrieval_score"] = result_df.get("baseline_score", 0.0)

    if "semantic_score" not in result_df.columns:
        result_df["semantic_score"] = result_df.get("baseline_score", 0.0)

    result_df["esci_score"] = result_df["esci_label"].map(ESCI_SCORE_MAP).fillna(0.0)
    result_df["rating_score"] = result_df["rating"] / 5.0

    result_df = add_diversity_score(result_df)

    score_columns = [
        "retrieval_score",
        "semantic_score",
        "esci_score",
        "rating_score",
        "diversity_score"
    ]

    for col in score_columns:
        result_df[f"{col}_norm"] = normalize_series(result_df[col])

    result_df["policy_rank_score"] = (
        action_weights["retrieval_weight"] * result_df["retrieval_score_norm"]
        + action_weights["semantic_weight"] * result_df["semantic_score_norm"]
        + action_weights["esci_weight"] * result_df["esci_score_norm"]
        + action_weights["rating_weight"] * result_df["rating_score_norm"]
        + action_weights["diversity_weight"] * result_df["diversity_score_norm"]
    )

    result_df = result_df.sort_values(
        by=["policy_rank_score", "rating"],
        ascending=False
    )

    result_df["selected_action_weights"] = str(action_weights)

    return result_df