import re
from typing import Dict, Tuple

import pandas as pd


KNOWN_BRANDS = [
    "adidas",
    "nike",
    "puma",
    "reebok",
    "under armour",
    "new balance",
    "asics",
    "skechers",
    "sony",
    "bose",
    "apple",
    "samsung",
    "panasonic",
    "dell",
    "hp",
    "lenovo",
]


def normalize_text(value) -> str:
    if not isinstance(value, str):
        return ""

    value = value.lower()
    value = re.sub(r"[^a-z0-9\s\-]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def detect_brand_preference(query: str) -> str:
    query_text = normalize_text(query)

    for brand in KNOWN_BRANDS:
        if brand in query_text:
            return brand

    return ""


def get_row_text(row: pd.Series) -> str:
    parts = []

    for col in ["product_title", "brand", "category", "query"]:
        if col in row.index:
            parts.append(str(row.get(col, "")))

    return normalize_text(" ".join(parts))


def is_blocked_row(row: pd.Series) -> bool:
    reason = str(row.get("contract_filter_reason", "")).lower()

    if "blocked=" in reason:
        return True

    if float(row.get("contract_match_score", 0.0)) <= 0.0:
        return True

    return False


def compute_brand_preference_score(row: pd.Series, preferred_brand: str) -> float:
    if not preferred_brand:
        return 0.0

    row_text = get_row_text(row)

    if preferred_brand in row_text:
        return 1.0

    return 0.0


def get_base_rank_score(row: pd.Series) -> float:
    """
    Uses the strongest available ranking score.
    """

    for col in ["policy_rank_score", "agent_score", "baseline_score", "semantic_score", "retrieval_score"]:
        if col in row.index:
            try:
                return float(row.get(col, 0.0))
            except Exception:
                continue

    return 0.0


def min_max_normalize(series: pd.Series) -> pd.Series:
    series = series.astype(float)

    min_val = series.min()
    max_val = series.max()

    if max_val == min_val:
        return pd.Series([0.5] * len(series), index=series.index)

    return (series - min_val) / (max_val - min_val)


def enforce_final_slate_contract(
    df: pd.DataFrame,
    contract: Dict,
    query: str,
    top_k: int = 20,
) -> Tuple[pd.DataFrame, Dict]:
    """
    Final contract gate before slate diversification.

    This is intentionally stricter than earlier candidate filtering.

    Rules:
    1. Blocked rows should not enter the final slate if enough non-blocked rows exist.
    2. Rows with contract_match_score = 0 should be removed when better rows exist.
    3. Brand preference should boost rows matching the brand mentioned in the query.
    4. If exact/core coverage is weak, return a warning instead of pretending results are perfect.
    """

    if df is None or len(df) == 0:
        return df, {
            "enforcement_status": "empty_input",
            "warning": "No candidates were available for final slate enforcement.",
            "preferred_brand": "",
            "input_rows": 0,
            "output_rows": 0,
            "positive_contract_rows": 0,
            "blocked_rows": 0,
            "low_coverage": True,
        }

    working_df = df.copy().reset_index(drop=True)

    if "contract_match_score" not in working_df.columns:
        working_df["contract_match_score"] = 0.5

    if "contract_filter_reason" not in working_df.columns:
        working_df["contract_filter_reason"] = "no_filter_reason_available"

    preferred_brand = detect_brand_preference(query)

    working_df["is_blocked_by_contract"] = working_df.apply(is_blocked_row, axis=1)

    working_df["brand_preference_score"] = working_df.apply(
        lambda row: compute_brand_preference_score(row, preferred_brand),
        axis=1,
    )

    working_df["base_rank_score_for_enforcement"] = working_df.apply(
        get_base_rank_score,
        axis=1,
    )

    working_df["normalized_base_rank_score"] = min_max_normalize(
        working_df["base_rank_score_for_enforcement"]
    )

    positive_contract_df = working_df[
        (working_df["contract_match_score"].astype(float) > 0.0)
        & (~working_df["is_blocked_by_contract"])
    ].copy()

    non_blocked_df = working_df[
        ~working_df["is_blocked_by_contract"]
    ].copy()

    blocked_count = int(working_df["is_blocked_by_contract"].sum())
    positive_count = len(positive_contract_df)

    low_coverage = positive_count < 3

    enforcement_notes = []

    # Strongest case: enough positive contract-aligned rows.
    if positive_count >= min(5, top_k):
        candidate_df = positive_contract_df.copy()
        enforcement_status = "strict_positive_contract_only"
        enforcement_notes.append("Using only positive contract-aligned candidates.")

    # Medium case: enough non-blocked rows but not enough strong positives.
    elif len(non_blocked_df) >= min(5, top_k):
        candidate_df = non_blocked_df.copy()
        enforcement_status = "non_blocked_only_low_core_coverage"
        enforcement_notes.append(
            "Using non-blocked candidates, but product-core coverage is weak."
        )

    # Weak case: not enough clean rows. Keep all but penalize blocked rows heavily.
    else:
        candidate_df = working_df.copy()
        enforcement_status = "fallback_low_inventory"
        enforcement_notes.append(
            "Not enough clean candidates found; retaining weak candidates with penalties."
        )

    # Final score: contract alignment dominates.
    candidate_df["final_contract_score"] = (
        0.50 * candidate_df["contract_match_score"].astype(float)
        + 0.30 * candidate_df["normalized_base_rank_score"].astype(float)
        + 0.20 * candidate_df["brand_preference_score"].astype(float)
    )

    # Heavily demote blocked rows if they are retained due to low inventory.
    candidate_df.loc[
        candidate_df["is_blocked_by_contract"], "final_contract_score"
    ] = candidate_df.loc[
        candidate_df["is_blocked_by_contract"], "final_contract_score"
    ] * 0.10

    candidate_df["final_slate_enforcement_reason"] = candidate_df.apply(
        lambda row: build_enforcement_reason(row, preferred_brand),
        axis=1,
    )

    candidate_df = candidate_df.sort_values(
        by=[
            "final_contract_score",
            "contract_match_score",
            "brand_preference_score",
            "normalized_base_rank_score",
        ],
        ascending=False,
    ).head(top_k).reset_index(drop=True)

    report = {
        "enforcement_status": enforcement_status,
        "warning": build_warning(
            low_coverage=low_coverage,
            positive_count=positive_count,
            input_count=len(working_df),
            enforcement_status=enforcement_status,
        ),
        "preferred_brand": preferred_brand if preferred_brand else "none",
        "input_rows": len(working_df),
        "output_rows": len(candidate_df),
        "positive_contract_rows": positive_count,
        "blocked_rows": blocked_count,
        "low_coverage": low_coverage,
        "notes": enforcement_notes,
    }

    return candidate_df, report


def build_enforcement_reason(row: pd.Series, preferred_brand: str) -> str:
    reasons = []

    if row.get("is_blocked_by_contract", False):
        reasons.append("blocked_or_zero_contract_match")

    contract_score = float(row.get("contract_match_score", 0.0))

    if contract_score > 0:
        reasons.append(f"contract_score={round(contract_score, 3)}")

    brand_score = float(row.get("brand_preference_score", 0.0))

    if preferred_brand and brand_score > 0:
        reasons.append(f"brand_match={preferred_brand}")

    if not reasons:
        reasons.append("weak_final_contract_alignment")

    return " | ".join(reasons)


def build_warning(
    low_coverage: bool,
    positive_count: int,
    input_count: int,
    enforcement_status: str,
) -> str:
    if not low_coverage:
        return "Final slate has enough contract-aligned candidates."

    return (
        f"Low product-core coverage: only {positive_count} of {input_count} candidates "
        f"strongly matched the search contract. Enforcement status: {enforcement_status}."
    )