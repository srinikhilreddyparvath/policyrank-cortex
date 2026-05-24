import json
import os
from datetime import datetime
from typing import Dict, Optional

import pandas as pd


STORAGE_DIR = "storage"
EXPERIMENT_LOG_PATH = os.path.join(STORAGE_DIR, "experiment_runs.csv")


def ensure_experiment_storage_exists():
    os.makedirs(STORAGE_DIR, exist_ok=True)


def safe_json_dumps(value) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return "{}"


def get_top_products_json(results_df: pd.DataFrame, top_k: int = 5) -> str:
    if results_df is None or len(results_df) == 0:
        return "[]"

    cols = [
        "position_agent",
        "product_id",
        "product_title",
        "brand",
        "category",
        "esci_label",
        "contract_match_score",
        "final_contract_score",
        "brand_preference_score",
        "simulated_reward",
    ]

    existing_cols = [col for col in cols if col in results_df.columns]

    top_rows = results_df.head(top_k)[existing_cols].copy()

    return top_rows.to_json(orient="records", force_ascii=False)


def get_top_product_title(results_df: pd.DataFrame) -> str:
    if results_df is None or len(results_df) == 0:
        return ""

    if "product_title" not in results_df.columns:
        return ""

    return str(results_df.iloc[0].get("product_title", ""))


def get_top_product_id(results_df: pd.DataFrame) -> str:
    if results_df is None or len(results_df) == 0:
        return ""

    if "product_id" not in results_df.columns:
        return ""

    return str(results_df.iloc[0].get("product_id", ""))


def log_experiment_run(
    query: str,
    retrieval_mode: str,
    contract_mode: str,
    policy_mode: str,
    selected_policy: str,
    contract: Dict,
    enforcement_report: Optional[Dict],
    final_results: pd.DataFrame,
    slate_reward: float,
):
    """
    Saves one CORTEX experiment run.

    This is separate from feedback logging.
    Feedback logging stores row-level simulated clicks/rewards.
    Experiment logging stores one summary row per query run.
    """

    ensure_experiment_storage_exists()

    enforcement_report = enforcement_report or {}

    intent = contract.get("intent", {}) if isinstance(contract, dict) else {}

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "query": query,
        "retrieval_mode": retrieval_mode,
        "contract_mode": contract_mode,
        "policy_mode": policy_mode,
        "selected_policy": selected_policy,
        "contract_source": contract.get("contract_source", "") if isinstance(contract, dict) else "",
        "fallback_used": contract.get("fallback_used", "") if isinstance(contract, dict) else "",
        "detected_category": intent.get("detected_category", ""),
        "product_type": intent.get("product_type", ""),
        "price_sensitivity": intent.get("price_sensitivity", ""),
        "quality_preference": intent.get("quality_preference", ""),
        "ambiguity_level": intent.get("ambiguity_level", ""),
        "enforcement_status": enforcement_report.get("enforcement_status", ""),
        "preferred_brand": enforcement_report.get("preferred_brand", ""),
        "positive_contract_rows": enforcement_report.get("positive_contract_rows", ""),
        "blocked_rows": enforcement_report.get("blocked_rows", ""),
        "low_coverage": enforcement_report.get("low_coverage", ""),
        "enforcement_warning": enforcement_report.get("warning", ""),
        "top_product_id": get_top_product_id(final_results),
        "top_product_title": get_top_product_title(final_results),
        "top_5_products_json": get_top_products_json(final_results, top_k=5),
        "slate_reward_at_5": slate_reward,
        "contract_json": safe_json_dumps(contract),
        "enforcement_report_json": safe_json_dumps(enforcement_report),
    }

    new_row_df = pd.DataFrame([row])

    if os.path.exists(EXPERIMENT_LOG_PATH):
        old_df = pd.read_csv(EXPERIMENT_LOG_PATH)
        combined = pd.concat([old_df, new_row_df], ignore_index=True)
    else:
        combined = new_row_df

    combined.to_csv(EXPERIMENT_LOG_PATH, index=False)


def load_experiment_runs() -> pd.DataFrame:
    ensure_experiment_storage_exists()

    if not os.path.exists(EXPERIMENT_LOG_PATH):
        return pd.DataFrame()

    try:
        return pd.read_csv(EXPERIMENT_LOG_PATH)
    except Exception:
        return pd.DataFrame()