import pandas as pd


ESCI_REWARD_MAP = {
    "E": 1.0,
    "S": 0.7,
    "C": 0.4,
    "I": 0.0
}


def apply_policy_rank(df: pd.DataFrame, user_preference: str) -> pd.DataFrame:
    result_df = df.copy()

    result_df["policy_boost"] = 0.0

    if user_preference == "Best rating":
        result_df["policy_boost"] = result_df["rating"] / 5.0

    elif user_preference == "Lowest price":
        max_price = result_df["price"].max()

        if max_price > 0:
            result_df["policy_boost"] = 1 - (result_df["price"] / max_price)
        else:
            result_df["policy_boost"] = 0.0

    elif user_preference == "Most relevant":
        result_df["policy_boost"] = result_df["esci_label"].map(ESCI_REWARD_MAP)

    result_df["policy_rank_score"] = (
        result_df["baseline_score"] + result_df["policy_boost"]
    )

    result_df = result_df.sort_values(
        by=["policy_rank_score", "rating"],
        ascending=False
    )

    return result_df