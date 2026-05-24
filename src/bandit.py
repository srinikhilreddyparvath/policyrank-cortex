import random
import pandas as pd


POLICIES = [
    "Most relevant",
    "Best rating",
    "Lowest price"
]


def summarize_policy_rewards(feedback_log: pd.DataFrame) -> pd.DataFrame:
    """
    Summarizes average reward per ranking policy.
    """
    if feedback_log is None or len(feedback_log) == 0:
        return pd.DataFrame({
            "ranking_objective": POLICIES,
            "times_used": [0, 0, 0],
            "total_reward": [0.0, 0.0, 0.0],
            "avg_reward": [0.0, 0.0, 0.0]
        })

    summary = (
        feedback_log
        .groupby("ranking_objective")
        .agg(
            times_used=("query", "count"),
            total_reward=("simulated_reward", "sum"),
            avg_reward=("simulated_reward", "mean")
        )
        .reset_index()
    )

    all_policies_df = pd.DataFrame({"ranking_objective": POLICIES})

    summary = all_policies_df.merge(
        summary,
        on="ranking_objective",
        how="left"
    )

    summary["times_used"] = summary["times_used"].fillna(0).astype(int)
    summary["total_reward"] = summary["total_reward"].fillna(0.0)
    summary["avg_reward"] = summary["avg_reward"].fillna(0.0)

    return summary


def select_policy_epsilon_greedy(
    feedback_log: pd.DataFrame,
    epsilon: float = 0.2
) -> str:
    """
    Epsilon-greedy selector:
    - With probability epsilon, explore a random policy.
    - Otherwise, exploit the policy with the best historical average reward.
    """
    summary = summarize_policy_rewards(feedback_log)

    if feedback_log is None or len(feedback_log) < 5:
        return random.choice(POLICIES)

    explore = random.random() < epsilon

    if explore:
        return random.choice(POLICIES)

    best_row = summary.sort_values(
        by=["avg_reward", "total_reward"],
        ascending=False
    ).iloc[0]

    return best_row["ranking_objective"]