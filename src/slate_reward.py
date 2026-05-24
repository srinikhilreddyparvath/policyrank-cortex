import pandas as pd


ESCI_REWARD_MAP = {
    "E": 1.0,
    "S": 0.7,
    "C": 0.4,
    "I": 0.0
}


def compute_position_weight(rank: int) -> float:
    """
    Higher-ranked results matter more.
    rank is 1-indexed.
    """
    return 1.0 / rank


def compute_slate_reward(results_df: pd.DataFrame, top_k: int = 5) -> float:
    """
    Computes a slate-level reward from the top-k ranked results.

    Reward combines:
    - ESCI relevance
    - simulated click reward if available
    - position weight
    """

    if results_df is None or len(results_df) == 0:
        return 0.0

    slate = results_df.head(top_k).copy()
    total_reward = 0.0

    for idx, (_, row) in enumerate(slate.iterrows(), start=1):
        label_reward = ESCI_REWARD_MAP.get(row["esci_label"], 0.0)
        click_reward = float(row.get("simulated_reward", 0.0))

        position_weight = compute_position_weight(idx)

        item_reward = (0.70 * label_reward) + (0.30 * click_reward)

        total_reward += position_weight * item_reward

    normalized_reward = total_reward / sum(
        compute_position_weight(rank) for rank in range(1, len(slate) + 1)
    )

    return float(normalized_reward)