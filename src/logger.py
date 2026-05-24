import os
from datetime import datetime

import pandas as pd


FEEDBACK_LOG_PATH = "storage/feedback_log.csv"


BASE_COLUMNS = [
    "timestamp",
    "query",
    "ranking_objective",
    "product_id",
    "product_title",
    "price",
    "rating",
    "esci_label",
    "baseline_score",
    "policy_boost",
    "policy_rank_score",
    "simulated_click",
    "simulated_reward"
]


EXTRA_COLUMNS = [
    "retrieval_score",
    "semantic_score",
    "esci_score",
    "rating_score",
    "diversity_score",
    "selected_action_weights"
]


ALL_COLUMNS = BASE_COLUMNS + EXTRA_COLUMNS


def ensure_storage_exists():
    os.makedirs("storage", exist_ok=True)

    if not os.path.exists(FEEDBACK_LOG_PATH):
        empty_log = pd.DataFrame(columns=ALL_COLUMNS)
        empty_log.to_csv(FEEDBACK_LOG_PATH, index=False)


def log_feedback(query: str, ranking_objective: str, feedback_df: pd.DataFrame):
    ensure_storage_exists()

    log_df = feedback_df.copy()
    log_df["timestamp"] = datetime.now().isoformat(timespec="seconds")
    log_df["query"] = query
    log_df["ranking_objective"] = ranking_objective

    # Add any missing columns so old modes and RL modes both work.
    for col in ALL_COLUMNS:
        if col not in log_df.columns:
            log_df[col] = None

    log_df = log_df[ALL_COLUMNS]

    log_df.to_csv(
        FEEDBACK_LOG_PATH,
        mode="a",
        header=False,
        index=False
    )


def load_feedback_log() -> pd.DataFrame:
    ensure_storage_exists()
    return pd.read_csv(FEEDBACK_LOG_PATH)