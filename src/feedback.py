import random
import pandas as pd


ESCI_REWARD_MAP = {
    "E": 1.0,
    "S": 0.7,
    "C": 0.4,
    "I": 0.0
}


def get_simulated_reward(esci_label: str) -> float:
    return ESCI_REWARD_MAP.get(esci_label, 0.0)


def simulate_user_clicks(results_df: pd.DataFrame) -> pd.DataFrame:
    """
    Simulates click behavior using ESCI labels.
    Higher relevance labels have higher click probability.
    """
    result_df = results_df.copy()

    click_prob_map = {
        "E": 0.75,
        "S": 0.45,
        "C": 0.25,
        "I": 0.05
    }

    clicks = []
    rewards = []

    for _, row in result_df.iterrows():
        label = row["esci_label"]
        click_prob = click_prob_map.get(label, 0.05)

        clicked = 1 if random.random() < click_prob else 0
        reward = get_simulated_reward(label) if clicked else 0.0

        clicks.append(clicked)
        rewards.append(reward)

    result_df["simulated_click"] = clicks
    result_df["simulated_reward"] = rewards

    return result_df


def summarize_feedback(feedback_df: pd.DataFrame) -> dict:
    if len(feedback_df) == 0:
        return {
            "total_results": 0,
            "total_clicks": 0,
            "total_reward": 0.0,
            "avg_reward": 0.0
        }

    return {
        "total_results": len(feedback_df),
        "total_clicks": int(feedback_df["simulated_click"].sum()),
        "total_reward": float(feedback_df["simulated_reward"].sum()),
        "avg_reward": float(feedback_df["simulated_reward"].mean())
    }