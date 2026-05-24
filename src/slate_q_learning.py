import os
import json
import random
from typing import Dict, Tuple

import pandas as pd


Q_TABLE_PATH = "storage/slate_q_table.json"


ACTIONS = {
    "relevance_heavy": {
        "retrieval_weight": 0.25,
        "semantic_weight": 0.25,
        "esci_weight": 0.40,
        "rating_weight": 0.10,
        "diversity_weight": 0.00
    },
    "semantic_heavy": {
        "retrieval_weight": 0.15,
        "semantic_weight": 0.55,
        "esci_weight": 0.20,
        "rating_weight": 0.10,
        "diversity_weight": 0.00
    },
    "quality_heavy": {
        "retrieval_weight": 0.25,
        "semantic_weight": 0.20,
        "esci_weight": 0.20,
        "rating_weight": 0.35,
        "diversity_weight": 0.00
    },
    "exploration_heavy": {
        "retrieval_weight": 0.20,
        "semantic_weight": 0.25,
        "esci_weight": 0.20,
        "rating_weight": 0.10,
        "diversity_weight": 0.25
    },
    "balanced": {
        "retrieval_weight": 0.25,
        "semantic_weight": 0.25,
        "esci_weight": 0.25,
        "rating_weight": 0.15,
        "diversity_weight": 0.10
    }
}


ESCI_SCORE_MAP = {
    "E": 1.0,
    "S": 0.7,
    "C": 0.4,
    "I": 0.0
}


def ensure_q_table_exists():
    os.makedirs("storage", exist_ok=True)

    if not os.path.exists(Q_TABLE_PATH):
        with open(Q_TABLE_PATH, "w") as f:
            json.dump({}, f, indent=2)


def load_q_table() -> Dict:
    ensure_q_table_exists()

    with open(Q_TABLE_PATH, "r") as f:
        return json.load(f)


def save_q_table(q_table: Dict):
    ensure_q_table_exists()

    with open(Q_TABLE_PATH, "w") as f:
        json.dump(q_table, f, indent=2)


def get_contract_state(contract: Dict, retrieval_confidence: float) -> str:
    """
    Converts the search contract into a simple discrete RL state.

    Example state:
    electronics_high_quality_high_conf
    """

    intent = contract.get("intent", {})

    category = intent.get("detected_category", "general")
    quality_preference = intent.get("quality_preference", "medium")
    price_sensitivity = intent.get("price_sensitivity", "medium")

    if retrieval_confidence >= 0.50:
        confidence_bucket = "high_conf"
    elif retrieval_confidence >= 0.20:
        confidence_bucket = "medium_conf"
    else:
        confidence_bucket = "low_conf"

    state = f"{category}_{quality_preference}_quality_{price_sensitivity}_price_{confidence_bucket}"

    return state


def initialize_state_if_needed(q_table: Dict, state: str):
    if state not in q_table:
        q_table[state] = {}

    for action_name in ACTIONS.keys():
        if action_name not in q_table[state]:
            q_table[state][action_name] = 0.0


def select_q_learning_action(
    state: str,
    epsilon: float = 0.25
) -> Tuple[str, Dict]:
    """
    Epsilon-greedy Q-learning action selector.

    The action is a ranking compiler strategy.
    """

    q_table = load_q_table()
    initialize_state_if_needed(q_table, state)
    save_q_table(q_table)

    should_explore = random.random() < epsilon

    if should_explore:
        selected_action = random.choice(list(ACTIONS.keys()))
    else:
        state_actions = q_table[state]
        selected_action = max(state_actions, key=state_actions.get)

    return selected_action, ACTIONS[selected_action]


def update_q_value(
    state: str,
    action: str,
    reward: float,
    next_state: str = None,
    learning_rate: float = 0.15,
    discount_factor: float = 0.90
):
    """
    Standard Q-learning update:

    Q(s,a) = Q(s,a) + alpha * [reward + gamma * max Q(s',a') - Q(s,a)]
    """

    q_table = load_q_table()

    initialize_state_if_needed(q_table, state)

    if next_state is None:
        next_state = state

    initialize_state_if_needed(q_table, next_state)

    current_q = q_table[state][action]
    max_next_q = max(q_table[next_state].values())

    updated_q = current_q + learning_rate * (
        reward + discount_factor * max_next_q - current_q
    )

    q_table[state][action] = updated_q

    save_q_table(q_table)


def get_q_table_as_dataframe() -> pd.DataFrame:
    q_table = load_q_table()

    rows = []

    for state, actions in q_table.items():
        for action, q_value in actions.items():
            rows.append({
                "state": state,
                "action": action,
                "q_value": q_value
            })

    if len(rows) == 0:
        return pd.DataFrame(columns=["state", "action", "q_value"])

    return pd.DataFrame(rows)