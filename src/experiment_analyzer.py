import json
import os
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import pandas as pd


EXPERIMENT_LOG_PATH = "storage/experiment_runs.csv"
OUTPUT_DIR = "outputs"


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def safe_load_json(value: Any):
    if not isinstance(value, str):
        return None

    try:
        return json.loads(value)
    except Exception:
        return None


def parse_top_5_products(top_5_json: str) -> List[Dict]:
    parsed = safe_load_json(top_5_json)

    if isinstance(parsed, list):
        return parsed

    return []


def load_experiment_runs() -> pd.DataFrame:
    if not os.path.exists(EXPERIMENT_LOG_PATH):
        raise FileNotFoundError(
            f"Could not find {EXPERIMENT_LOG_PATH}. Save experiment snapshots in the app first."
        )

    df = pd.read_csv(EXPERIMENT_LOG_PATH)

    if len(df) == 0:
        raise ValueError("experiment_runs.csv exists, but it has no rows.")

    return df


def create_overall_summary(df: pd.DataFrame) -> pd.DataFrame:
    total_runs = len(df)

    avg_reward = df["slate_reward_at_5"].mean() if "slate_reward_at_5" in df.columns else 0.0
    median_reward = df["slate_reward_at_5"].median() if "slate_reward_at_5" in df.columns else 0.0
    max_reward = df["slate_reward_at_5"].max() if "slate_reward_at_5" in df.columns else 0.0
    min_reward = df["slate_reward_at_5"].min() if "slate_reward_at_5" in df.columns else 0.0

    low_coverage_rate = 0.0
    if "low_coverage" in df.columns:
        low_coverage_bool = df["low_coverage"].astype(str).str.lower().isin(["true", "1", "yes"])
        low_coverage_rate = low_coverage_bool.mean()

    unique_queries = df["query"].nunique() if "query" in df.columns else 0
    unique_product_types = df["product_type"].nunique() if "product_type" in df.columns else 0

    summary = pd.DataFrame(
        [
            {
                "total_runs": total_runs,
                "unique_queries": unique_queries,
                "unique_product_types": unique_product_types,
                "average_slate_reward_at_5": avg_reward,
                "median_slate_reward_at_5": median_reward,
                "max_slate_reward_at_5": max_reward,
                "min_slate_reward_at_5": min_reward,
                "low_coverage_rate": low_coverage_rate,
            }
        ]
    )

    return summary


def create_query_results_table(df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "timestamp",
        "query",
        "retrieval_mode",
        "contract_mode",
        "policy_mode",
        "selected_policy",
        "contract_source",
        "fallback_used",
        "detected_category",
        "product_type",
        "price_sensitivity",
        "quality_preference",
        "ambiguity_level",
        "preferred_brand",
        "enforcement_status",
        "positive_contract_rows",
        "blocked_rows",
        "low_coverage",
        "top_product_id",
        "top_product_title",
        "slate_reward_at_5",
        "enforcement_warning",
    ]

    existing_cols = [col for col in cols if col in df.columns]

    result = df[existing_cols].copy()

    if "slate_reward_at_5" in result.columns:
        result = result.sort_values(by="slate_reward_at_5", ascending=False)

    return result


def create_reward_by_product_type(df: pd.DataFrame) -> pd.DataFrame:
    if "product_type" not in df.columns or "slate_reward_at_5" not in df.columns:
        return pd.DataFrame()

    grouped = (
        df.groupby("product_type", dropna=False)
        .agg(
            runs=("query", "count"),
            unique_queries=("query", "nunique"),
            avg_slate_reward_at_5=("slate_reward_at_5", "mean"),
            max_slate_reward_at_5=("slate_reward_at_5", "max"),
            min_slate_reward_at_5=("slate_reward_at_5", "min"),
        )
        .reset_index()
        .sort_values(by="avg_slate_reward_at_5", ascending=False)
    )

    return grouped


def create_reward_by_policy(df: pd.DataFrame) -> pd.DataFrame:
    if "selected_policy" not in df.columns or "slate_reward_at_5" not in df.columns:
        return pd.DataFrame()

    grouped = (
        df.groupby("selected_policy", dropna=False)
        .agg(
            runs=("query", "count"),
            unique_queries=("query", "nunique"),
            avg_slate_reward_at_5=("slate_reward_at_5", "mean"),
            max_slate_reward_at_5=("slate_reward_at_5", "max"),
            min_slate_reward_at_5=("slate_reward_at_5", "min"),
        )
        .reset_index()
        .sort_values(by="avg_slate_reward_at_5", ascending=False)
    )

    return grouped


def create_low_coverage_summary(df: pd.DataFrame) -> pd.DataFrame:
    if "low_coverage" not in df.columns:
        return pd.DataFrame()

    work = df.copy()
    work["low_coverage_bool"] = work["low_coverage"].astype(str).str.lower().isin(
        ["true", "1", "yes"]
    )

    grouped = (
        work.groupby("low_coverage_bool")
        .agg(
            runs=("query", "count"),
            unique_queries=("query", "nunique"),
            avg_slate_reward_at_5=("slate_reward_at_5", "mean"),
        )
        .reset_index()
    )

    grouped = grouped.rename(columns={"low_coverage_bool": "low_coverage"})

    return grouped


def create_top_products_expanded_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for _, row in df.iterrows():
        query = row.get("query", "")
        timestamp = row.get("timestamp", "")
        selected_policy = row.get("selected_policy", "")
        product_type = row.get("product_type", "")
        preferred_brand = row.get("preferred_brand", "")
        slate_reward = row.get("slate_reward_at_5", "")

        top_products = parse_top_5_products(row.get("top_5_products_json", ""))

        for rank, product in enumerate(top_products, start=1):
            rows.append(
                {
                    "timestamp": timestamp,
                    "query": query,
                    "rank": rank,
                    "selected_policy": selected_policy,
                    "product_type": product_type,
                    "preferred_brand": preferred_brand,
                    "slate_reward_at_5": slate_reward,
                    "product_id": product.get("product_id", ""),
                    "product_title": product.get("product_title", ""),
                    "brand": product.get("brand", ""),
                    "category": product.get("category", ""),
                    "esci_label": product.get("esci_label", ""),
                    "contract_match_score": product.get("contract_match_score", ""),
                    "final_contract_score": product.get("final_contract_score", ""),
                    "brand_preference_score": product.get("brand_preference_score", ""),
                    "simulated_reward": product.get("simulated_reward", ""),
                }
            )

    return pd.DataFrame(rows)


def save_reward_chart(df: pd.DataFrame):
    if "slate_reward_at_5" not in df.columns:
        return

    plt.figure(figsize=(12, 6))
    plt.plot(range(len(df)), df["slate_reward_at_5"], marker="o")
    plt.title("SlateReward@5 Across Saved Experiment Runs")
    plt.xlabel("Experiment Run Index")
    plt.ylabel("SlateReward@5")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    output_path = os.path.join(OUTPUT_DIR, "slate_reward_chart.png")
    plt.savefig(output_path, dpi=200)
    plt.close()


def run_analysis():
    ensure_output_dir()

    df = load_experiment_runs()

    if "slate_reward_at_5" in df.columns:
        df["slate_reward_at_5"] = pd.to_numeric(df["slate_reward_at_5"], errors="coerce")

    overall_summary = create_overall_summary(df)
    query_results = create_query_results_table(df)
    reward_by_product_type = create_reward_by_product_type(df)
    reward_by_policy = create_reward_by_policy(df)
    low_coverage_summary = create_low_coverage_summary(df)
    top_products_expanded = create_top_products_expanded_table(df)

    overall_summary.to_csv(os.path.join(OUTPUT_DIR, "experiment_summary.csv"), index=False)
    query_results.to_csv(os.path.join(OUTPUT_DIR, "query_results_table.csv"), index=False)
    reward_by_product_type.to_csv(os.path.join(OUTPUT_DIR, "reward_by_product_type.csv"), index=False)
    reward_by_policy.to_csv(os.path.join(OUTPUT_DIR, "reward_by_policy.csv"), index=False)
    low_coverage_summary.to_csv(os.path.join(OUTPUT_DIR, "low_coverage_summary.csv"), index=False)
    top_products_expanded.to_csv(os.path.join(OUTPUT_DIR, "top_products_expanded.csv"), index=False)

    save_reward_chart(df)

    print("Analysis complete.")
    print("Input:", EXPERIMENT_LOG_PATH)
    print("Output directory:", OUTPUT_DIR)
    print()
    print("Overall summary:")
    print(overall_summary.to_string(index=False))
    print()
    print("Top query results:")
    print(query_results.head(10).to_string(index=False))


if __name__ == "__main__":
    run_analysis()