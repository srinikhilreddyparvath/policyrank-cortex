import argparse
import json
import os
import re
from datetime import datetime
from typing import Dict, List

import matplotlib.pyplot as plt
import pandas as pd

from src.semantic_retrieval import semantic_search
from src.llm_contract_agent import generate_llm_search_contract
from src.contract_filters import apply_contract_candidate_filter
from src.final_slate_enforcer import enforce_final_slate_contract
from src.slate_q_learning import get_contract_state, select_q_learning_action
from src.policy_compiler import compile_and_apply_policy
from src.multi_agent_diversifier import multi_agent_diversify_slate
from src.feedback import simulate_user_clicks
from src.slate_reward import compute_slate_reward
from src.baseline_preservation_gate import (
    select_slate_with_baseline_preservation_gate,
    summarize_gate_for_logging,
)


PRODUCTS_PATH = "data/esci_balanced_sample.csv"
OUTPUT_DIR = "outputs"
STORAGE_DIR = "storage"
CONTRACT_CACHE_PATH = os.path.join(STORAGE_DIR, "contract_cache.json")

TOP_K_RETRIEVAL = 20
TOP_K_REWARD = 5
RANDOM_SEED = 42


def ensure_dirs():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(STORAGE_DIR, exist_ok=True)


def safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def safe_json_dumps(value) -> str:
    try:
        return json.dumps(value, ensure_ascii=False)
    except Exception:
        return "{}"


def normalize_query_text(query: str) -> str:
    if query is None:
        return ""

    query = str(query).strip().lower()
    query = re.sub(r"\s+", " ", query)

    return query


def tokenize_query(query: str) -> List[str]:
    query = normalize_query_text(query)

    tokens = re.findall(r"[a-zA-Z0-9]+", query)

    stopwords = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "for",
        "with",
        "to",
        "of",
        "in",
        "on",
        "at",
        "by",
        "from",
        "is",
        "are",
        "was",
        "were",
        "be",
        "this",
        "that",
        "these",
        "those",
        "my",
        "your",
        "his",
        "her",
        "their",
        "our",
    }

    clean_tokens = [
        token
        for token in tokens
        if token and token not in stopwords and len(token) > 1
    ]

    return clean_tokens


def infer_product_type_from_query(query: str) -> str:
    tokens = tokenize_query(query)

    if not tokens:
        return "unknown"

    # Usually the last meaningful term is a useful product-type proxy.
    return tokens[-1]


def build_local_fallback_contract(
    query: str,
    sample_products: List[Dict],
    error_message: str = "",
) -> Dict:
    """
    Local fallback contract used when the LLM contract agent fails.

    This prevents long scalable evaluations from collapsing when:
    - the OpenAI client closes,
    - the API temporarily fails,
    - rate limits happen,
    - malformed/non-English queries cause contract issues.

    The fallback is intentionally simple and conservative:
    - Uses query terms as must-have terms.
    - Avoids blocked terms unless the LLM provides them later.
    - Keeps enough schema fields for downstream filters, policy, and logging.
    """

    clean_query = normalize_query_text(query)
    query_terms = tokenize_query(clean_query)
    product_type = infer_product_type_from_query(clean_query)

    sample_titles = []

    for product in sample_products:
        title = product.get("product_title", "")
        if title:
            sample_titles.append(str(title))

    fallback_contract = {
        "contract_source": "local_fallback_contract",
        "fallback_used": True,
        "fallback_reason": error_message,
        "query": query,
        "intent": {
            "detected_category": product_type,
            "product_type": product_type,
            "price_sensitivity": "unknown",
            "quality_preference": "unknown",
            "ambiguity_level": "medium",
        },
        "dynamic_filters": {
            "must_have_terms": query_terms[:6],
            "blocked_terms": [],
            "preferred_terms": query_terms[:6],
        },
        "required_terms": query_terms[:6],
        "preferred_terms": query_terms[:6],
        "excluded_terms": [],
        "brand_preferences": [],
        "notes": {
            "source": "Generated locally because LLM contract generation failed.",
            "sample_titles": sample_titles[:5],
        },
    }

    return fallback_contract


def load_products() -> pd.DataFrame:
    if not os.path.exists(PRODUCTS_PATH):
        raise FileNotFoundError(f"Missing product file: {PRODUCTS_PATH}")

    products = pd.read_csv(PRODUCTS_PATH)

    if len(products) == 0:
        raise ValueError("Product file is empty.")

    return products


def build_query_pool(products: pd.DataFrame, sample_size: int) -> List[str]:
    """
    Samples unique queries from the balanced ESCI sample.
    """

    if "query" not in products.columns:
        raise ValueError("Product file must contain a query column.")

    query_pool = products["query"].dropna().astype(str).str.strip()
    query_pool = query_pool[query_pool.str.len() > 0].drop_duplicates()

    if sample_size > len(query_pool):
        sample_size = len(query_pool)

    sampled_queries = query_pool.sample(
        n=sample_size,
        random_state=RANDOM_SEED,
    ).tolist()

    return sampled_queries


def get_output_paths(sample_size: int) -> Dict[str, str]:
    """
    MVP 13.3.1 resilient evaluator output paths.

    Uses a new filename suffix so we do not confuse this robust run with the
    earlier MVP 13.3 run that had LLM-client-closed errors.
    """

    return {
        "eval": os.path.join(OUTPUT_DIR, f"scalable_eval_mvp13_3_1_resilient_{sample_size}.csv"),
        "summary": os.path.join(OUTPUT_DIR, f"scalable_eval_summary_mvp13_3_1_resilient_{sample_size}.csv"),
        "by_query": os.path.join(OUTPUT_DIR, f"scalable_eval_by_query_mvp13_3_1_resilient_{sample_size}.csv"),
        "chart": os.path.join(OUTPUT_DIR, f"scalable_eval_lift_chart_mvp13_3_1_resilient_{sample_size}.png"),
        "errors": os.path.join(OUTPUT_DIR, f"scalable_eval_errors_mvp13_3_1_resilient_{sample_size}.csv"),
    }


def load_contract_cache() -> Dict:
    if not os.path.exists(CONTRACT_CACHE_PATH):
        return {}

    try:
        with open(CONTRACT_CACHE_PATH, "r", encoding="utf-8") as file:
            return json.load(file)
    except Exception:
        return {}


def save_contract_cache(cache: Dict):
    with open(CONTRACT_CACHE_PATH, "w", encoding="utf-8") as file:
        json.dump(cache, file, ensure_ascii=False, indent=2)


def get_cached_or_generate_contract(
    query: str,
    sample_products: List[Dict],
    cache: Dict,
    disable_llm_cache: bool = False,
    allow_fallback_contract: bool = True,
) -> Dict:
    """
    Uses cached LLM contract when available.

    If no cached contract exists:
    - Try to generate a fresh LLM contract.
    - If the LLM call fails, use a local fallback contract instead of failing the query.

    This is important for scalable evaluation because one closed OpenAI client should not
    invalidate the rest of the run.
    """

    cache_key = query.strip().lower()

    if not disable_llm_cache and cache_key in cache:
        cached_contract = cache[cache_key]

        if isinstance(cached_contract, dict):
            cached_contract["cache_hit"] = True

        return cached_contract

    try:
        contract = generate_llm_search_contract(
            query=query,
            sample_products=sample_products,
        )

        if not isinstance(contract, dict):
            raise ValueError("LLM contract generator returned a non-dictionary contract.")

        contract["cache_hit"] = False

        cache[cache_key] = contract
        save_contract_cache(cache)

        return contract

    except Exception as error:
        error_message = str(error)

        if not allow_fallback_contract:
            raise

        fallback_contract = build_local_fallback_contract(
            query=query,
            sample_products=sample_products,
            error_message=error_message,
        )

        fallback_contract["cache_hit"] = False

        # Cache the fallback too. This prevents repeated failures for the same query.
        cache[cache_key] = fallback_contract
        save_contract_cache(cache)

        print(f"WARNING: LLM contract failed for query '{query}'. Using local fallback. Error: {error_message}")

        return fallback_contract


def get_top_product_title(df: pd.DataFrame) -> str:
    if df is None or len(df) == 0 or "product_title" not in df.columns:
        return ""

    return str(df.iloc[0].get("product_title", ""))


def get_top_product_id(df: pd.DataFrame) -> str:
    if df is None or len(df) == 0 or "product_id" not in df.columns:
        return ""

    return str(df.iloc[0].get("product_id", ""))


def get_top_5_titles(df: pd.DataFrame) -> str:
    if df is None or len(df) == 0 or "product_title" not in df.columns:
        return ""

    return " || ".join(df.head(TOP_K_REWARD)["product_title"].astype(str).tolist())


def evaluate_baseline(query: str, products: pd.DataFrame) -> Dict:
    baseline_results = semantic_search(
        query=query,
        products_df=products,
        top_k=TOP_K_RETRIEVAL,
    )

    baseline_feedback = simulate_user_clicks(baseline_results)
    baseline_reward = compute_slate_reward(baseline_feedback, top_k=TOP_K_REWARD)

    return {
        "baseline_results": baseline_feedback,
        "baseline_reward_at_5": baseline_reward,
        "baseline_top_product_id": get_top_product_id(baseline_feedback),
        "baseline_top_product_title": get_top_product_title(baseline_feedback),
        "baseline_top_5_titles": get_top_5_titles(baseline_feedback),
    }


def evaluate_cortex(
    query: str,
    baseline_results: pd.DataFrame,
    contract_cache: Dict,
    disable_llm_cache: bool = False,
    allow_fallback_contract: bool = True,
) -> Dict:
    """
    MVP 13.3.1 resilient version.

    Pipeline:
    1. Start from semantic baseline.
    2. Generate/load LLM search contract.
       - If LLM fails, use local fallback contract.
    3. Apply contract filters.
    4. Apply Q-learning policy.
    5. Enforce final contract.
    6. Diversify with multi-agent slate.
    7. Apply Baseline Preservation Gate:
       - preserve baseline
       - light rerank
       - full CORTEX
    8. Compute final gated reward.
    """

    sample_products = baseline_results.head(8).to_dict(orient="records")

    contract = get_cached_or_generate_contract(
        query=query,
        sample_products=sample_products,
        cache=contract_cache,
        disable_llm_cache=disable_llm_cache,
        allow_fallback_contract=allow_fallback_contract,
    )

    contract_filtered_results = apply_contract_candidate_filter(
        df=baseline_results,
        contract=contract,
        min_match_score=0.50,
        strict_top_k=TOP_K_RETRIEVAL,
    )

    if contract_filtered_results is not None and len(contract_filtered_results) > 0:
        ranking_input_results = contract_filtered_results
    else:
        ranking_input_results = baseline_results

    retrieval_confidence = 0.0

    if len(baseline_results) > 0 and "baseline_score" in baseline_results.columns:
        retrieval_confidence = safe_float(baseline_results["baseline_score"].max())

    contract_state = get_contract_state(
        contract=contract,
        retrieval_confidence=retrieval_confidence,
    )

    selected_rl_action, selected_action_weights = select_q_learning_action(
        state=contract_state,
        epsilon=0.0,
    )

    policy_results = compile_and_apply_policy(
        df=ranking_input_results,
        action_weights=selected_action_weights,
        retrieval_mode="Semantic",
    )

    contract_enforced_results, enforcement_report = enforce_final_slate_contract(
        df=policy_results,
        contract=contract,
        query=query,
        top_k=TOP_K_RETRIEVAL,
    )

    diversified_results, agent_trace = multi_agent_diversify_slate(
        ranked_df=contract_enforced_results,
        top_k=10,
    )

    # Full CORTEX result, kept for diagnostics.
    full_cortex_feedback = simulate_user_clicks(diversified_results)
    full_cortex_reward = compute_slate_reward(
        full_cortex_feedback,
        top_k=TOP_K_REWARD,
    )

    # MVP 13.3.1 gated final slate.
    gated_slate, gate_decision = select_slate_with_baseline_preservation_gate(
        baseline_df=baseline_results,
        cortex_df=diversified_results,
        contract=contract,
        top_k=TOP_K_REWARD,
    )

    gated_feedback = simulate_user_clicks(gated_slate)
    gated_reward = compute_slate_reward(
        gated_feedback,
        top_k=TOP_K_REWARD,
    )

    gate_log = summarize_gate_for_logging(gate_decision)

    intent = contract.get("intent", {})
    dynamic_filters = contract.get("dynamic_filters", {})

    return {
        "cortex_results": gated_feedback,
        "cortex_reward_at_5": gated_reward,
        "full_cortex_reward_at_5": full_cortex_reward,
        "gate_delta_vs_full_cortex": gated_reward - full_cortex_reward,
        "cortex_top_product_id": get_top_product_id(gated_feedback),
        "cortex_top_product_title": get_top_product_title(gated_feedback),
        "cortex_top_5_titles": get_top_5_titles(gated_feedback),
        "full_cortex_top_product_id": get_top_product_id(full_cortex_feedback),
        "full_cortex_top_product_title": get_top_product_title(full_cortex_feedback),
        "full_cortex_top_5_titles": get_top_5_titles(full_cortex_feedback),
        "contract_source": contract.get("contract_source", ""),
        "fallback_used": contract.get("fallback_used", ""),
        "fallback_reason": contract.get("fallback_reason", ""),
        "cache_hit": contract.get("cache_hit", ""),
        "detected_category": intent.get("detected_category", ""),
        "product_type": intent.get("product_type", ""),
        "price_sensitivity": intent.get("price_sensitivity", ""),
        "quality_preference": intent.get("quality_preference", ""),
        "ambiguity_level": intent.get("ambiguity_level", ""),
        "must_have_terms": ", ".join(dynamic_filters.get("must_have_terms", [])),
        "blocked_terms": ", ".join(dynamic_filters.get("blocked_terms", [])),
        "selected_rl_action": selected_rl_action,
        "contract_state": contract_state,
        "enforcement_status": enforcement_report.get("enforcement_status", ""),
        "preferred_brand": enforcement_report.get("preferred_brand", ""),
        "positive_contract_rows": enforcement_report.get("positive_contract_rows", ""),
        "blocked_rows": enforcement_report.get("blocked_rows", ""),
        "low_coverage": enforcement_report.get("low_coverage", ""),
        "enforcement_warning": enforcement_report.get("warning", ""),
        "gate_decision": gate_log.get("gate_decision", ""),
        "gate_reason": gate_log.get("gate_reason", ""),
        "gate_baseline_confidence_score": gate_log.get("gate_baseline_confidence_score", ""),
        "gate_contract_alignment_score": gate_log.get("gate_contract_alignment_score", ""),
        "gate_final_gate_score": gate_log.get("gate_final_gate_score", ""),
        "gate_exclusion_violation_rate": gate_log.get("gate_exclusion_violation_rate", ""),
        "gate_required_term_coverage": gate_log.get("gate_required_term_coverage", ""),
        "gate_preferred_term_coverage": gate_log.get("gate_preferred_term_coverage", ""),
        "gate_brand_preference_coverage": gate_log.get("gate_brand_preference_coverage", ""),
        "gate_top1_score": gate_log.get("gate_top1_score", ""),
        "gate_top5_mean_score": gate_log.get("gate_top5_mean_score", ""),
        "gate_label_quality_score": gate_log.get("gate_label_quality_score", ""),
    }


def evaluate_query(
    query: str,
    products: pd.DataFrame,
    contract_cache: Dict,
    disable_llm_cache: bool = False,
    allow_fallback_contract: bool = True,
) -> Dict:
    baseline = evaluate_baseline(query, products)

    cortex = evaluate_cortex(
        query=query,
        baseline_results=baseline["baseline_results"],
        contract_cache=contract_cache,
        disable_llm_cache=disable_llm_cache,
        allow_fallback_contract=allow_fallback_contract,
    )

    baseline_reward = safe_float(baseline["baseline_reward_at_5"])
    cortex_reward = safe_float(cortex["cortex_reward_at_5"])
    full_cortex_reward = safe_float(cortex["full_cortex_reward_at_5"])

    absolute_lift = cortex_reward - baseline_reward
    percentage_lift = 0.0 if baseline_reward == 0 else absolute_lift / baseline_reward

    full_cortex_absolute_lift = full_cortex_reward - baseline_reward
    full_cortex_percentage_lift = (
        0.0 if baseline_reward == 0 else full_cortex_absolute_lift / baseline_reward
    )

    if absolute_lift > 0.0001:
        winner = "CORTEX_MVP13_3_1_Gated"
    elif absolute_lift < -0.0001:
        winner = "Baseline"
    else:
        winner = "Tie"

    if full_cortex_absolute_lift > 0.0001:
        full_cortex_winner = "Full_CORTEX_MVP13_2"
    elif full_cortex_absolute_lift < -0.0001:
        full_cortex_winner = "Baseline"
    else:
        full_cortex_winner = "Tie"

    baseline_loss_rescued = (
        full_cortex_winner == "Baseline"
        and winner != "Baseline"
    )

    over_rerank_prevented = (
        cortex["gate_decision"] in ["preserve_baseline", "light_rerank"]
        and cortex_reward >= full_cortex_reward
    )

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "query": query,
        "baseline_reward_at_5": baseline_reward,
        "cortex_reward_at_5": cortex_reward,
        "full_cortex_reward_at_5": full_cortex_reward,
        "absolute_lift": absolute_lift,
        "percentage_lift": percentage_lift,
        "full_cortex_absolute_lift": full_cortex_absolute_lift,
        "full_cortex_percentage_lift": full_cortex_percentage_lift,
        "gate_delta_vs_full_cortex": cortex["gate_delta_vs_full_cortex"],
        "winner": winner,
        "full_cortex_winner": full_cortex_winner,
        "baseline_loss_rescued": baseline_loss_rescued,
        "over_rerank_prevented": over_rerank_prevented,
        "baseline_top_product_id": baseline["baseline_top_product_id"],
        "baseline_top_product_title": baseline["baseline_top_product_title"],
        "cortex_top_product_id": cortex["cortex_top_product_id"],
        "cortex_top_product_title": cortex["cortex_top_product_title"],
        "full_cortex_top_product_id": cortex["full_cortex_top_product_id"],
        "full_cortex_top_product_title": cortex["full_cortex_top_product_title"],
        "baseline_top_5_titles": baseline["baseline_top_5_titles"],
        "cortex_top_5_titles": cortex["cortex_top_5_titles"],
        "full_cortex_top_5_titles": cortex["full_cortex_top_5_titles"],
        "contract_source": cortex["contract_source"],
        "fallback_used": cortex["fallback_used"],
        "fallback_reason": cortex["fallback_reason"],
        "cache_hit": cortex["cache_hit"],
        "detected_category": cortex["detected_category"],
        "product_type": cortex["product_type"],
        "price_sensitivity": cortex["price_sensitivity"],
        "quality_preference": cortex["quality_preference"],
        "ambiguity_level": cortex["ambiguity_level"],
        "must_have_terms": cortex["must_have_terms"],
        "blocked_terms": cortex["blocked_terms"],
        "selected_rl_action": cortex["selected_rl_action"],
        "contract_state": cortex["contract_state"],
        "enforcement_status": cortex["enforcement_status"],
        "preferred_brand": cortex["preferred_brand"],
        "positive_contract_rows": cortex["positive_contract_rows"],
        "blocked_rows": cortex["blocked_rows"],
        "low_coverage": cortex["low_coverage"],
        "enforcement_warning": cortex["enforcement_warning"],
        "gate_decision": cortex["gate_decision"],
        "gate_reason": cortex["gate_reason"],
        "gate_baseline_confidence_score": cortex["gate_baseline_confidence_score"],
        "gate_contract_alignment_score": cortex["gate_contract_alignment_score"],
        "gate_final_gate_score": cortex["gate_final_gate_score"],
        "gate_exclusion_violation_rate": cortex["gate_exclusion_violation_rate"],
        "gate_required_term_coverage": cortex["gate_required_term_coverage"],
        "gate_preferred_term_coverage": cortex["gate_preferred_term_coverage"],
        "gate_brand_preference_coverage": cortex["gate_brand_preference_coverage"],
        "gate_top1_score": cortex["gate_top1_score"],
        "gate_top5_mean_score": cortex["gate_top5_mean_score"],
        "gate_label_quality_score": cortex["gate_label_quality_score"],
        "error": "",
    }


def append_row_to_csv(row: Dict, path: str):
    row_df = pd.DataFrame([row])

    if os.path.exists(path):
        old_df = pd.read_csv(path)
        combined = pd.concat([old_df, row_df], ignore_index=True)
    else:
        combined = row_df

    combined.to_csv(path, index=False)


def load_completed_queries(path: str) -> set:
    if not os.path.exists(path):
        return set()

    try:
        df = pd.read_csv(path)
    except Exception:
        return set()

    if "query" not in df.columns:
        return set()

    return set(df["query"].dropna().astype(str).str.strip().str.lower().tolist())


def build_summary(eval_df: pd.DataFrame) -> pd.DataFrame:
    clean_df = eval_df.dropna(
        subset=["baseline_reward_at_5", "cortex_reward_at_5"]
    ).copy()

    total_queries = len(clean_df)

    cortex_wins = int((clean_df["winner"] == "CORTEX_MVP13_3_1_Gated").sum())
    baseline_wins = int((clean_df["winner"] == "Baseline").sum())
    ties = int((clean_df["winner"] == "Tie").sum())

    full_cortex_wins = int((clean_df["full_cortex_winner"] == "Full_CORTEX_MVP13_2").sum())
    full_cortex_baseline_wins = int((clean_df["full_cortex_winner"] == "Baseline").sum())
    full_cortex_ties = int((clean_df["full_cortex_winner"] == "Tie").sum())

    preserve_count = int((clean_df["gate_decision"] == "preserve_baseline").sum())
    light_rerank_count = int((clean_df["gate_decision"] == "light_rerank").sum())
    full_cortex_count = int((clean_df["gate_decision"] == "full_cortex").sum())

    fallback_contract_count = int((clean_df["fallback_used"].astype(str).str.lower() == "true").sum())
    llm_contract_count = total_queries - fallback_contract_count

    baseline_loss_rescued_count = int(clean_df["baseline_loss_rescued"].fillna(False).sum())
    over_rerank_prevented_count = int(clean_df["over_rerank_prevented"].fillna(False).sum())

    avg_baseline_reward = (
        float(clean_df["baseline_reward_at_5"].mean()) if total_queries else 0.0
    )

    avg_cortex_reward = (
        float(clean_df["cortex_reward_at_5"].mean()) if total_queries else 0.0
    )

    avg_full_cortex_reward = (
        float(clean_df["full_cortex_reward_at_5"].mean()) if total_queries else 0.0
    )

    avg_absolute_lift = (
        float(clean_df["absolute_lift"].mean()) if total_queries else 0.0
    )

    avg_full_cortex_absolute_lift = (
        float(clean_df["full_cortex_absolute_lift"].mean()) if total_queries else 0.0
    )

    avg_percentage_lift = (
        float(clean_df["percentage_lift"].mean()) if total_queries else 0.0
    )

    avg_full_cortex_percentage_lift = (
        float(clean_df["full_cortex_percentage_lift"].mean()) if total_queries else 0.0
    )

    median_absolute_lift = (
        float(clean_df["absolute_lift"].median()) if total_queries else 0.0
    )

    median_full_cortex_absolute_lift = (
        float(clean_df["full_cortex_absolute_lift"].median()) if total_queries else 0.0
    )

    avg_gate_delta_vs_full_cortex = (
        float(clean_df["gate_delta_vs_full_cortex"].mean()) if total_queries else 0.0
    )

    cortex_win_rate = cortex_wins / total_queries if total_queries else 0.0
    full_cortex_win_rate = full_cortex_wins / total_queries if total_queries else 0.0

    return pd.DataFrame(
        [
            {
                "total_queries": total_queries,
                "mvp13_3_1_gated_wins": cortex_wins,
                "baseline_wins_vs_mvp13_3_1": baseline_wins,
                "ties_vs_mvp13_3_1": ties,
                "mvp13_3_1_gated_win_rate": cortex_win_rate,
                "mvp13_2_full_cortex_wins": full_cortex_wins,
                "baseline_wins_vs_mvp13_2_full_cortex": full_cortex_baseline_wins,
                "ties_vs_mvp13_2_full_cortex": full_cortex_ties,
                "mvp13_2_full_cortex_win_rate": full_cortex_win_rate,
                "preserve_baseline_count": preserve_count,
                "light_rerank_count": light_rerank_count,
                "full_cortex_count": full_cortex_count,
                "llm_contract_count": llm_contract_count,
                "fallback_contract_count": fallback_contract_count,
                "baseline_loss_rescued_count": baseline_loss_rescued_count,
                "over_rerank_prevented_count": over_rerank_prevented_count,
                "avg_baseline_reward_at_5": avg_baseline_reward,
                "avg_mvp13_3_1_reward_at_5": avg_cortex_reward,
                "avg_mvp13_2_full_cortex_reward_at_5": avg_full_cortex_reward,
                "avg_mvp13_3_1_absolute_lift": avg_absolute_lift,
                "avg_mvp13_2_full_cortex_absolute_lift": avg_full_cortex_absolute_lift,
                "avg_mvp13_3_1_percentage_lift": avg_percentage_lift,
                "avg_mvp13_2_full_cortex_percentage_lift": avg_full_cortex_percentage_lift,
                "median_mvp13_3_1_absolute_lift": median_absolute_lift,
                "median_mvp13_2_full_cortex_absolute_lift": median_full_cortex_absolute_lift,
                "avg_gate_delta_vs_full_cortex": avg_gate_delta_vs_full_cortex,
            }
        ]
    )


def build_by_query_table(eval_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "query",
        "winner",
        "full_cortex_winner",
        "baseline_loss_rescued",
        "over_rerank_prevented",
        "gate_decision",
        "gate_reason",
        "fallback_used",
        "contract_source",
        "fallback_reason",
        "baseline_reward_at_5",
        "cortex_reward_at_5",
        "full_cortex_reward_at_5",
        "absolute_lift",
        "full_cortex_absolute_lift",
        "gate_delta_vs_full_cortex",
        "percentage_lift",
        "full_cortex_percentage_lift",
        "gate_baseline_confidence_score",
        "gate_contract_alignment_score",
        "gate_final_gate_score",
        "baseline_top_product_title",
        "cortex_top_product_title",
        "full_cortex_top_product_title",
        "product_type",
        "selected_rl_action",
        "preferred_brand",
        "enforcement_status",
        "low_coverage",
        "error",
    ]

    existing_cols = [col for col in cols if col in eval_df.columns]
    out = eval_df[existing_cols].copy()

    if "absolute_lift" in out.columns:
        out = out.sort_values(by="absolute_lift", ascending=False)

    return out


def save_lift_chart(eval_df: pd.DataFrame, chart_path: str):
    clean_df = eval_df.dropna(subset=["absolute_lift"]).copy()

    if len(clean_df) == 0:
        return

    chart_df = clean_df.sort_values(by="absolute_lift", ascending=True).copy()

    # If too many queries, plot only worst 25 and best 25 to keep chart readable.
    if len(chart_df) > 50:
        chart_df = pd.concat(
            [
                chart_df.head(25),
                chart_df.tail(25),
            ],
            ignore_index=True,
        )

    plt.figure(figsize=(14, 10))
    plt.barh(chart_df["query"], chart_df["absolute_lift"])
    plt.axvline(0, color="black", linewidth=1)
    plt.title("MVP 13.3.1 Resilient Gated CORTEX Lift over Semantic Baseline by Query")
    plt.xlabel("Absolute Lift in SlateReward@5")
    plt.ylabel("Query")
    plt.tight_layout()
    plt.savefig(chart_path, dpi=200)
    plt.close()


def finalize_outputs(sample_size: int, paths: Dict[str, str]):
    if not os.path.exists(paths["eval"]):
        print("No eval output found yet.")
        return

    eval_df = pd.read_csv(paths["eval"])

    if len(eval_df) == 0:
        print("Eval output is empty.")
        return

    numeric_cols = [
        "baseline_reward_at_5",
        "cortex_reward_at_5",
        "full_cortex_reward_at_5",
        "absolute_lift",
        "percentage_lift",
        "full_cortex_absolute_lift",
        "full_cortex_percentage_lift",
        "gate_delta_vs_full_cortex",
        "gate_baseline_confidence_score",
        "gate_contract_alignment_score",
        "gate_final_gate_score",
    ]

    for col in numeric_cols:
        if col in eval_df.columns:
            eval_df[col] = pd.to_numeric(eval_df[col], errors="coerce")

    summary_df = build_summary(eval_df)
    by_query_df = build_by_query_table(eval_df)

    summary_df.to_csv(paths["summary"], index=False)
    by_query_df.to_csv(paths["by_query"], index=False)

    save_lift_chart(eval_df, paths["chart"])

    print()
    print("Final MVP 13.3.1 resilient summary:")
    print(summary_df.to_string(index=False))
    print()
    print("Best lift queries:")
    print(by_query_df.head(10).to_string(index=False))
    print()
    print("Worst lift queries:")
    print(by_query_df.tail(10).to_string(index=False))


def run_scalable_eval(
    sample_size: int,
    resume: bool = True,
    disable_llm_cache: bool = False,
    allow_fallback_contract: bool = True,
):
    ensure_dirs()

    paths = get_output_paths(sample_size)

    products = load_products()
    queries = build_query_pool(products, sample_size=sample_size)
    contract_cache = load_contract_cache()

    completed_queries = load_completed_queries(paths["eval"]) if resume else set()

    if not resume:
        for path in paths.values():
            if os.path.exists(path):
                os.remove(path)

    print(f"Loaded products: {len(products)}")
    print(f"Sample size requested: {sample_size}")
    print(f"Queries selected: {len(queries)}")
    print(f"Resume mode: {resume}")
    print(f"Completed queries already in output: {len(completed_queries)}")
    print(f"Contract cache entries: {len(contract_cache)}")
    print(f"Fallback contracts enabled: {allow_fallback_contract}")
    print(f"Eval output: {paths['eval']}")
    print("MVP: 13.3.1 Resilient Baseline-Aware CORTEX Gate")
    print()

    for idx, query in enumerate(queries, start=1):
        query_key = query.strip().lower()

        if resume and query_key in completed_queries:
            print(f"[{idx}/{len(queries)}] Skipping completed: {query}")
            continue

        print(f"[{idx}/{len(queries)}] Evaluating: {query}")

        try:
            row = evaluate_query(
                query=query,
                products=products,
                contract_cache=contract_cache,
                disable_llm_cache=disable_llm_cache,
                allow_fallback_contract=allow_fallback_contract,
            )

            append_row_to_csv(row, paths["eval"])

        except Exception as error:
            print(f"ERROR on query '{query}': {error}")

            error_row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "query": query,
                "baseline_reward_at_5": None,
                "cortex_reward_at_5": None,
                "full_cortex_reward_at_5": None,
                "absolute_lift": None,
                "percentage_lift": None,
                "full_cortex_absolute_lift": None,
                "full_cortex_percentage_lift": None,
                "winner": "Error",
                "full_cortex_winner": "Error",
                "baseline_loss_rescued": False,
                "over_rerank_prevented": False,
                "fallback_used": "",
                "contract_source": "",
                "gate_decision": "",
                "gate_reason": "",
                "error": str(error),
            }

            append_row_to_csv(error_row, paths["eval"])
            append_row_to_csv(error_row, paths["errors"])

    finalize_outputs(sample_size=sample_size, paths=paths)

    print()
    print("Scalable evaluation complete.")
    print("Files written:")

    for name, path in paths.items():
        print(f"- {name}: {path}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--sample-size",
        type=int,
        default=100,
        help="Number of unique queries to evaluate.",
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh and overwrite prior MVP 13.3.1 resilient output for this sample size.",
    )

    parser.add_argument(
        "--disable-llm-cache",
        action="store_true",
        help="Force new LLM contracts instead of using storage/contract_cache.json.",
    )

    parser.add_argument(
        "--no-fallback-contract",
        action="store_true",
        help="Disable local fallback contracts and fail when LLM contract generation fails.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    run_scalable_eval(
        sample_size=args.sample_size,
        resume=not args.no_resume,
        disable_llm_cache=args.disable_llm_cache,
        allow_fallback_contract=not args.no_fallback_contract,
    )