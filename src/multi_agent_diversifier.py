import re
from typing import List, Tuple

import pandas as pd


def normalize_text(value) -> str:
    if not isinstance(value, str):
        return ""

    value = value.lower()
    value = re.sub(r"[^a-z0-9\s\-]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def get_product_text(row: pd.Series) -> str:
    parts = []

    for col in ["product_title", "brand", "category", "query"]:
        if col in row.index:
            parts.append(str(row.get(col, "")))

    return normalize_text(" ".join(parts))


def token_set(text: str) -> set:
    return set(normalize_text(text).split())


def jaccard_similarity(text_a: str, text_b: str) -> float:
    tokens_a = token_set(text_a)
    tokens_b = token_set(text_b)

    if not tokens_a or not tokens_b:
        return 0.0

    return len(tokens_a.intersection(tokens_b)) / len(tokens_a.union(tokens_b))


def compute_diversity_score_against_selected(
    candidate_row: pd.Series,
    selected_rows: List[pd.Series],
) -> float:
    """
    Returns high score when candidate is different from already selected products.
    """

    if not selected_rows:
        return 1.0

    candidate_text = get_product_text(candidate_row)

    max_similarity = 0.0

    for selected_row in selected_rows:
        selected_text = get_product_text(selected_row)
        max_similarity = max(
            max_similarity,
            jaccard_similarity(candidate_text, selected_text),
        )

    diversity_score = 1.0 - max_similarity

    return max(0.0, min(1.0, diversity_score))


def safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def get_contract_priority_score(row: pd.Series) -> float:
    """
    Contract-priority score.

    This function makes the multi-agent slate builder respect the final
    contract enforcement layer.

    Priority order:
    1. final_contract_score if available
    2. contract_match_score
    3. policy_rank_score
    4. baseline_score
    """

    if "final_contract_score" in row.index:
        return safe_float(row.get("final_contract_score", 0.0))

    if "contract_match_score" in row.index:
        return safe_float(row.get("contract_match_score", 0.0))

    if "policy_rank_score" in row.index:
        return safe_float(row.get("policy_rank_score", 0.0))

    if "baseline_score" in row.index:
        return safe_float(row.get("baseline_score", 0.0))

    return 0.0


def is_blocked_candidate(row: pd.Series) -> bool:
    """
    Prevent known blocked candidates from being selected when cleaner options exist.
    """

    if "is_blocked_by_contract" in row.index:
        try:
            if bool(row.get("is_blocked_by_contract")):
                return True
        except Exception:
            pass

    reason = str(row.get("contract_filter_reason", "")).lower()

    if "blocked=" in reason:
        return True

    contract_score = safe_float(row.get("contract_match_score", 0.0))

    if contract_score <= 0.0:
        return True

    return False


def compute_agent_score(
    candidate_row: pd.Series,
    selected_rows: List[pd.Series],
    position: int,
) -> Tuple[float, float]:
    """
    Computes the score used by each rank-position agent.

    Earlier MVPs over-weighted diversity. This version makes contract quality
    the dominant signal.

    For top positions:
    - contract quality dominates heavily.
    - diversity is only a secondary signal.

    For later positions:
    - diversity becomes slightly more important.
    """

    contract_priority_score = get_contract_priority_score(candidate_row)

    diversity_score = compute_diversity_score_against_selected(
        candidate_row,
        selected_rows,
    )

    brand_score = safe_float(candidate_row.get("brand_preference_score", 0.0))

    blocked_penalty = 0.0

    if is_blocked_candidate(candidate_row):
        blocked_penalty = 0.75

    # Top ranks should be very contract-faithful.
    if position <= 3:
        contract_weight = 0.78
        diversity_weight = 0.07
        brand_weight = 0.15

    # Middle ranks can diversify a little more.
    elif position <= 6:
        contract_weight = 0.68
        diversity_weight = 0.17
        brand_weight = 0.15

    # Lower ranks can diversify more, but still respect the contract.
    else:
        contract_weight = 0.58
        diversity_weight = 0.27
        brand_weight = 0.15

    agent_score = (
        contract_weight * contract_priority_score
        + diversity_weight * diversity_score
        + brand_weight * brand_score
        - blocked_penalty
    )

    agent_score = max(0.0, min(1.0, agent_score))

    return agent_score, diversity_score


def multi_agent_diversify_slate(
    ranked_df: pd.DataFrame,
    top_k: int = 10,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Contract-priority multi-agent slate construction.

    Each rank-position agent selects one product.

    Important behavior:
    - final_contract_score dominates.
    - brand_preference_score boosts brand-intent queries like "adidas soccer cleats".
    - blocked or zero contract-match products are avoided.
    - diversity is used only after contract quality is satisfied.
    """

    if ranked_df is None or len(ranked_df) == 0:
        return ranked_df, pd.DataFrame()

    working_df = ranked_df.copy().reset_index(drop=True)

    if "final_contract_score" not in working_df.columns:
        if "contract_match_score" in working_df.columns:
            working_df["final_contract_score"] = working_df["contract_match_score"].astype(float)
        elif "policy_rank_score" in working_df.columns:
            working_df["final_contract_score"] = working_df["policy_rank_score"].astype(float)
        elif "baseline_score" in working_df.columns:
            working_df["final_contract_score"] = working_df["baseline_score"].astype(float)
        else:
            working_df["final_contract_score"] = 0.5

    if "brand_preference_score" not in working_df.columns:
        working_df["brand_preference_score"] = 0.0

    if "is_blocked_by_contract" not in working_df.columns:
        working_df["is_blocked_by_contract"] = working_df.apply(
            is_blocked_candidate,
            axis=1,
        )

    # If we have enough non-blocked candidates, remove blocked rows completely.
    non_blocked_df = working_df[~working_df["is_blocked_by_contract"]].copy()

    if len(non_blocked_df) >= min(top_k, 5):
        candidate_pool = non_blocked_df.copy()
    else:
        candidate_pool = working_df.copy()

    selected_rows = []
    trace_rows = []

    remaining_df = candidate_pool.copy().reset_index(drop=True)

    for position in range(1, min(top_k, len(remaining_df)) + 1):
        scored_candidates = []

        for idx, candidate_row in remaining_df.iterrows():
            agent_score, diversity_score = compute_agent_score(
                candidate_row=candidate_row,
                selected_rows=selected_rows,
                position=position,
            )

            scored_candidates.append(
                {
                    "idx": idx,
                    "agent_score": agent_score,
                    "diversity_score_against_selected": diversity_score,
                    "contract_priority_score": get_contract_priority_score(candidate_row),
                    "brand_preference_score": safe_float(
                        candidate_row.get("brand_preference_score", 0.0)
                    ),
                    "is_blocked_by_contract": bool(
                        candidate_row.get("is_blocked_by_contract", False)
                    ),
                }
            )

        scored_df = pd.DataFrame(scored_candidates)

        scored_df = scored_df.sort_values(
            by=[
                "agent_score",
                "contract_priority_score",
                "brand_preference_score",
                "diversity_score_against_selected",
            ],
            ascending=False,
        ).reset_index(drop=True)

        best_idx = int(scored_df.iloc[0]["idx"])

        selected_row = remaining_df.loc[best_idx].copy()

        selected_row["position_agent"] = position
        selected_row["agent_score"] = float(scored_df.iloc[0]["agent_score"])
        selected_row["diversity_score_against_selected"] = float(
            scored_df.iloc[0]["diversity_score_against_selected"]
        )
        selected_row["contract_priority_score"] = float(
            scored_df.iloc[0]["contract_priority_score"]
        )

        selected_rows.append(selected_row)

        trace_rows.append(
            {
                "position_agent": position,
                "selected_product_id": selected_row.get("product_id", ""),
                "selected_product_title": selected_row.get("product_title", ""),
                "agent_score": selected_row["agent_score"],
                "contract_priority_score": selected_row["contract_priority_score"],
                "brand_preference_score": selected_row.get("brand_preference_score", 0.0),
                "diversity_score_against_selected": selected_row[
                    "diversity_score_against_selected"
                ],
                "is_blocked_by_contract": selected_row.get("is_blocked_by_contract", False),
                "final_contract_score": selected_row.get("final_contract_score", None),
                "contract_match_score": selected_row.get("contract_match_score", None),
                "final_slate_enforcement_reason": selected_row.get(
                    "final_slate_enforcement_reason",
                    "",
                ),
            }
        )

        remaining_df = remaining_df.drop(index=best_idx).reset_index(drop=True)

        if len(remaining_df) == 0:
            break

    diversified_df = pd.DataFrame(selected_rows).reset_index(drop=True)
    agent_trace_df = pd.DataFrame(trace_rows)

    return diversified_df, agent_trace_df


def compute_slate_quality_metrics(
    slate_df: pd.DataFrame,
    top_k: int = 5,
) -> dict:
    """
    Computes simple slate-quality metrics for the top-k results.
    """

    if slate_df is None or len(slate_df) == 0:
        return {
            "exact_count_at_k": 0,
            "substitute_count_at_k": 0,
            "complement_count_at_k": 0,
            "irrelevant_count_at_k": 0,
            "diversity_at_k": 0.0,
        }

    top_df = slate_df.head(top_k).copy()

    if "esci_label" not in top_df.columns:
        return {
            "exact_count_at_k": 0,
            "substitute_count_at_k": 0,
            "complement_count_at_k": 0,
            "irrelevant_count_at_k": 0,
            "diversity_at_k": 0.0,
        }

    labels = top_df["esci_label"].astype(str).str.upper()

    exact_count = int((labels == "E").sum())
    substitute_count = int((labels == "S").sum())
    complement_count = int((labels == "C").sum())
    irrelevant_count = int((labels == "I").sum())

    product_texts = top_df.apply(get_product_text, axis=1).tolist()

    pairwise_diversities = []

    for i in range(len(product_texts)):
        for j in range(i + 1, len(product_texts)):
            similarity = jaccard_similarity(product_texts[i], product_texts[j])
            pairwise_diversities.append(1.0 - similarity)

    if pairwise_diversities:
        diversity_at_k = sum(pairwise_diversities) / len(pairwise_diversities)
    else:
        diversity_at_k = 0.0

    return {
        "exact_count_at_k": exact_count,
        "substitute_count_at_k": substitute_count,
        "complement_count_at_k": complement_count,
        "irrelevant_count_at_k": irrelevant_count,
        "diversity_at_k": diversity_at_k,
    }