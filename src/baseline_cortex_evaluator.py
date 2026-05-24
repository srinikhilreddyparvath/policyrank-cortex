import os
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


PRODUCTS_PATH = "data/esci_balanced_sample.csv"
EXPERIMENT_RUNS_PATH = "storage/experiment_runs.csv"
OUTPUT_DIR = "outputs"

EVAL_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "baseline_vs_cortex_eval.csv")
SUMMARY_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "baseline_vs_cortex_summary.csv")
BY_QUERY_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "baseline_vs_cortex_by_query.csv")
CHART_OUTPUT_PATH = os.path.join(OUTPUT_DIR, "baseline_vs_cortex_lift_chart.png")

TOP_K_RETRIEVAL = 20
TOP_K_REWARD = 5


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_products() -> pd.DataFrame:
    if not os.path.exists(PRODUCTS_PATH):
        raise FileNotFoundError(f"Missing product file: {PRODUCTS_PATH}")

    products = pd.read_csv(PRODUCTS_PATH)

    if len(products) == 0:
        raise ValueError("Product file is empty.")

    return products


def load_eval_queries() -> List[str]:
    if not os.path.exists(EXPERIMENT_RUNS_PATH):
        raise FileNotFoundError(
            f"Missing experiment runs file: {EXPERIMENT_RUNS_PATH}. "
            "Save experiment snapshots in the app first."
        )

    runs = pd.read_csv(EXPERIMENT_RUNS_PATH)

    if "query" not in runs.columns:
        raise ValueError("experiment_runs.csv does not contain a query column.")

    queries = (
        runs["query"]
        .dropna()
        .astype(str)
        .str.strip()
        .drop_duplicates()
        .tolist()
    )

    queries = [q for q in queries if len(q) > 0]

    if not queries:
        raise ValueError("No valid queries found in experiment_runs.csv.")

    return queries


def get_top_product_title(df: pd.DataFrame) -> str:
    if df is None or len(df) == 0:
        return ""

    if "product_title" not in df.columns:
        return ""

    return str(df.iloc[0].get("product_title", ""))


def get_top_product_id(df: pd.DataFrame) -> str:
    if df is None or len(df) == 0:
        return ""

    if "product_id" not in df.columns:
        return ""

    return str(df.iloc[0].get("product_id", ""))


def get_top_5_titles(df: pd.DataFrame) -> str:
    if df is None or len(df) == 0:
        return ""

    if "product_title" not in df.columns:
        return ""

    titles = df.head(TOP_K_REWARD)["product_title"].astype(str).tolist()
    return " || ".join(titles)


def evaluate_baseline(query: str, products: pd.DataFrame) -> Dict:
    """
    Baseline = semantic retrieval only.

    We simulate feedback and calculate SlateReward@5 directly on the semantic top results.
    """

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


def evaluate_cortex(query: str, products: pd.DataFrame, baseline_results: pd.DataFrame) -> Dict:
    """
    CORTEX = LLM contract + dynamic filter + Slate Q action + policy compiler
    + final slate enforcement + multi-agent diversification.
    """

    sample_products = baseline_results.head(8).to_dict(orient="records")

    contract = generate_llm_search_contract(
        query=query,
        sample_products=sample_products,
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

    if len(baseline_results) > 0 and "baseline_score" in baseline_results.columns:
        retrieval_confidence = float(baseline_results["baseline_score"].max())
    else:
        retrieval_confidence = 0.0

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

    cortex_feedback = simulate_user_clicks(diversified_results)
    cortex_reward = compute_slate_reward(cortex_feedback, top_k=TOP_K_REWARD)

    intent = contract.get("intent", {})
    dynamic_filters = contract.get("dynamic_filters", {})

    return {
        "cortex_results": cortex_feedback,
        "cortex_reward_at_5": cortex_reward,
        "cortex_top_product_id": get_top_product_id(cortex_feedback),
        "cortex_top_product_title": get_top_product_title(cortex_feedback),
        "cortex_top_5_titles": get_top_5_titles(cortex_feedback),
        "contract_source": contract.get("contract_source", ""),
        "fallback_used": contract.get("fallback_used", ""),
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
    }


def evaluate_query(query: str, products: pd.DataFrame) -> Dict:
    print(f"Evaluating query: {query}")

    baseline = evaluate_baseline(query, products)

    cortex = evaluate_cortex(
        query=query,
        products=products,
        baseline_results=baseline["baseline_results"],
    )

    baseline_reward = float(baseline["baseline_reward_at_5"])
    cortex_reward = float(cortex["cortex_reward_at_5"])

    absolute_lift = cortex_reward - baseline_reward

    if baseline_reward == 0:
        percentage_lift = 0.0
    else:
        percentage_lift = absolute_lift / baseline_reward

    if absolute_lift > 0.0001:
        winner = "CORTEX"
    elif absolute_lift < -0.0001:
        winner = "Baseline"
    else:
        winner = "Tie"

    return {
        "query": query,
        "baseline_reward_at_5": baseline_reward,
        "cortex_reward_at_5": cortex_reward,
        "absolute_lift": absolute_lift,
        "percentage_lift": percentage_lift,
        "winner": winner,
        "baseline_top_product_id": baseline["baseline_top_product_id"],
        "baseline_top_product_title": baseline["baseline_top_product_title"],
        "cortex_top_product_id": cortex["cortex_top_product_id"],
        "cortex_top_product_title": cortex["cortex_top_product_title"],
        "baseline_top_5_titles": baseline["baseline_top_5_titles"],
        "cortex_top_5_titles": cortex["cortex_top_5_titles"],
        "contract_source": cortex["contract_source"],
        "fallback_used": cortex["fallback_used"],
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
    }


def build_summary(eval_df: pd.DataFrame) -> pd.DataFrame:
    total_queries = len(eval_df)

    cortex_wins = int((eval_df["winner"] == "CORTEX").sum())
    baseline_wins = int((eval_df["winner"] == "Baseline").sum())
    ties = int((eval_df["winner"] == "Tie").sum())

    avg_baseline_reward = float(eval_df["baseline_reward_at_5"].mean())
    avg_cortex_reward = float(eval_df["cortex_reward_at_5"].mean())
    avg_absolute_lift = float(eval_df["absolute_lift"].mean())
    avg_percentage_lift = float(eval_df["percentage_lift"].mean())

    median_absolute_lift = float(eval_df["absolute_lift"].median())

    win_rate = cortex_wins / total_queries if total_queries else 0.0

    summary = pd.DataFrame(
        [
            {
                "total_queries": total_queries,
                "cortex_wins": cortex_wins,
                "baseline_wins": baseline_wins,
                "ties": ties,
                "cortex_win_rate": win_rate,
                "avg_baseline_reward_at_5": avg_baseline_reward,
                "avg_cortex_reward_at_5": avg_cortex_reward,
                "avg_absolute_lift": avg_absolute_lift,
                "avg_percentage_lift": avg_percentage_lift,
                "median_absolute_lift": median_absolute_lift,
            }
        ]
    )

    return summary


def build_by_query_table(eval_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "query",
        "winner",
        "baseline_reward_at_5",
        "cortex_reward_at_5",
        "absolute_lift",
        "percentage_lift",
        "baseline_top_product_title",
        "cortex_top_product_title",
        "product_type",
        "selected_rl_action",
        "preferred_brand",
        "enforcement_status",
        "low_coverage",
    ]

    existing_cols = [col for col in cols if col in eval_df.columns]

    return eval_df[existing_cols].sort_values(
        by="absolute_lift",
        ascending=False,
    )


def save_lift_chart(eval_df: pd.DataFrame):
    chart_df = eval_df.sort_values(by="absolute_lift", ascending=True).copy()

    plt.figure(figsize=(14, 8))
    plt.barh(chart_df["query"], chart_df["absolute_lift"])
    plt.axvline(0, color="black", linewidth=1)
    plt.title("CORTEX Lift over Semantic Baseline by Query")
    plt.xlabel("Absolute Lift in SlateReward@5")
    plt.ylabel("Query")
    plt.tight_layout()

    plt.savefig(CHART_OUTPUT_PATH, dpi=200)
    plt.close()


def run_evaluation():
    ensure_output_dir()

    products = load_products()
    queries = load_eval_queries()

    print(f"Loaded products: {len(products)}")
    print(f"Loaded unique eval queries: {len(queries)}")

    rows = []

    for idx, query in enumerate(queries, start=1):
        print(f"[{idx}/{len(queries)}] {query}")

        try:
            row = evaluate_query(query=query, products=products)
            rows.append(row)
        except Exception as error:
            print(f"ERROR on query '{query}': {error}")

            rows.append(
                {
                    "query": query,
                    "baseline_reward_at_5": None,
                    "cortex_reward_at_5": None,
                    "absolute_lift": None,
                    "percentage_lift": None,
                    "winner": "Error",
                    "error": str(error),
                }
            )

    eval_df = pd.DataFrame(rows)

    eval_df.to_csv(EVAL_OUTPUT_PATH, index=False)

    clean_eval_df = eval_df.dropna(
        subset=["baseline_reward_at_5", "cortex_reward_at_5"]
    ).copy()

    summary_df = build_summary(clean_eval_df)
    by_query_df = build_by_query_table(clean_eval_df)

    summary_df.to_csv(SUMMARY_OUTPUT_PATH, index=False)
    by_query_df.to_csv(BY_QUERY_OUTPUT_PATH, index=False)

    save_lift_chart(clean_eval_df)

    print()
    print("Evaluation complete.")
    print("Saved:", EVAL_OUTPUT_PATH)
    print("Saved:", SUMMARY_OUTPUT_PATH)
    print("Saved:", BY_QUERY_OUTPUT_PATH)
    print("Saved:", CHART_OUTPUT_PATH)
    print()
    print("Summary:")
    print(summary_df.to_string(index=False))
    print()
    print("Top lift queries:")
    print(by_query_df.head(10).to_string(index=False))
    print()
    print("Worst lift queries:")
    print(by_query_df.tail(10).to_string(index=False))


if __name__ == "__main__":
    run_evaluation()