"""
MVP 13.3.1: Tuned Baseline Preservation Gate

Purpose:
--------
Prevents CORTEX from over-reranking when the semantic baseline top-5 is already strong.

MVP 13.3 result:
----------------
The first gate worked technically, but it was too conservative:
- Too many preserve/light-rerank decisions
- Too few full CORTEX decisions
- Average lift dropped compared to full CORTEX

MVP 13.3.1 change:
------------------
Make the gate stricter.

Full CORTEX should remain the default path unless the baseline is very strong.
Light rerank now slightly favors CORTEX instead of baseline.

Gate decisions:
---------------
1. preserve_baseline
   - Baseline top-5 appears very strong and contract-aligned.
   - Keep baseline order.

2. light_rerank
   - Baseline is strong enough to partially trust.
   - Blend baseline rank and CORTEX rank, slightly favoring CORTEX.

3. full_cortex
   - Baseline appears weak, uncertain, or not clearly superior.
   - Use full CORTEX reranking.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import math
import re

import pandas as pd


# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

DEFAULT_TOP_K = 5

# MVP 13.3.1 tuned thresholds:
# Full CORTEX should remain the default path unless baseline is very strong.
PRESERVE_CONFIDENCE_THRESHOLD = 0.88
PRESERVE_ALIGNMENT_THRESHOLD = 0.85

LIGHT_CONFIDENCE_THRESHOLD = 0.78
LIGHT_ALIGNMENT_THRESHOLD = 0.75

# Light reranking should still favor CORTEX slightly because full CORTEX wins most queries.
BASELINE_WEIGHT_LIGHT_RERANK = 0.45
CORTEX_WEIGHT_LIGHT_RERANK = 0.55


# ---------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------

@dataclass
class BaselineGateSignals:
    top_k: int
    num_candidates: int

    score_column: Optional[str]
    id_column: Optional[str]
    title_column: Optional[str]
    brand_column: Optional[str]
    label_column: Optional[str]

    top1_score: float
    top5_mean_score: float
    top5_min_score: float
    score_gap_top1_top5: float
    score_decay_strength: float

    label_quality_score: float
    contract_alignment_score: float
    exclusion_violation_rate: float
    required_term_coverage: float
    preferred_term_coverage: float
    brand_preference_coverage: float

    baseline_confidence_score: float
    final_gate_score: float


@dataclass
class BaselineGateDecision:
    decision: str
    reason: str
    signals: BaselineGateSignals

    def to_dict(self) -> Dict[str, Any]:
        out = {
            "gate_decision": self.decision,
            "gate_reason": self.reason,
        }

        signal_dict = asdict(self.signals)

        for key, value in signal_dict.items():
            out[f"gate_{key}"] = value

        return out


# ---------------------------------------------------------------------
# Column inference helpers
# ---------------------------------------------------------------------

def _first_existing_column(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col

    return None


def infer_id_column(df: pd.DataFrame) -> Optional[str]:
    return _first_existing_column(
        df,
        [
            "product_id",
            "item_id",
            "doc_id",
            "asin",
            "id",
            "candidate_id",
            "esci_id",
        ],
    )


def infer_score_column(df: pd.DataFrame) -> Optional[str]:
    return _first_existing_column(
        df,
        [
            "semantic_score",
            "retrieval_score",
            "similarity_score",
            "cosine_similarity",
            "cosine_score",
            "score",
            "rank_score",
            "baseline_score",
            "final_score",
        ],
    )


def infer_title_column(df: pd.DataFrame) -> Optional[str]:
    return _first_existing_column(
        df,
        [
            "product_title",
            "item_title",
            "title",
            "name",
            "product_name",
            "description",
        ],
    )


def infer_brand_column(df: pd.DataFrame) -> Optional[str]:
    return _first_existing_column(
        df,
        [
            "brand",
            "brand_name",
            "product_brand",
            "manufacturer",
        ],
    )


def infer_label_column(df: pd.DataFrame) -> Optional[str]:
    return _first_existing_column(
        df,
        [
            "esci_label",
            "label",
            "relevance_label",
            "relevance",
            "grade",
        ],
    )


# ---------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------

def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default

        if isinstance(value, float) and math.isnan(value):
            return default

        return float(value)

    except Exception:
        return default


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def _normalize_score_series(series: pd.Series) -> pd.Series:
    """
    Converts arbitrary score columns into roughly 0..1 scale.

    Handles:
    - Already normalized scores
    - Cosine similarities around -1..1
    - Arbitrary positive rank scores
    """

    numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)

    if len(numeric) == 0:
        return numeric

    min_val = float(numeric.min())
    max_val = float(numeric.max())

    # Already looks like 0..1
    if min_val >= 0.0 and max_val <= 1.0:
        return numeric.clip(0.0, 1.0)

    # Looks like cosine similarity -1..1
    if min_val >= -1.0 and max_val <= 1.0:
        return ((numeric + 1.0) / 2.0).clip(0.0, 1.0)

    # Generic min-max
    if abs(max_val - min_val) < 1e-9:
        return pd.Series([0.5] * len(numeric), index=series.index)

    return ((numeric - min_val) / (max_val - min_val)).clip(0.0, 1.0)


def _contains_any(text: str, terms: List[str]) -> bool:
    text_lower = str(text).lower()

    return any(
        str(term).lower() in text_lower
        for term in terms
        if str(term).strip()
    )


def _coverage(texts: List[str], terms: List[str]) -> float:
    clean_terms = [
        str(term).lower().strip()
        for term in terms
        if str(term).strip()
    ]

    if not clean_terms:
        return 1.0

    combined_text = " ".join([str(text).lower() for text in texts])

    matched = sum(
        1
        for term in clean_terms
        if term in combined_text
    )

    return matched / max(len(clean_terms), 1)


# ---------------------------------------------------------------------
# Contract parsing helpers
# ---------------------------------------------------------------------

def _extract_contract_terms(contract: Optional[Dict[str, Any]]) -> Dict[str, List[str]]:
    """
    Supports flexible contract schemas.

    Expected possible fields:
    - required_terms
    - must_have_terms
    - product_type_terms
    - category_terms
    - preferred_terms
    - nice_to_have_terms
    - brand_preferences
    - preferred_brands
    - excluded_terms
    - avoid_terms

    Also supports your current dynamic_filters style:
    - contract["dynamic_filters"]["must_have_terms"]
    - contract["dynamic_filters"]["blocked_terms"]
    """

    if not isinstance(contract, dict):
        return {
            "required_terms": [],
            "preferred_terms": [],
            "brand_terms": [],
            "excluded_terms": [],
        }

    dynamic_filters = contract.get("dynamic_filters", {})
    intent = contract.get("intent", {})

    def collect_from_object(obj: Any) -> List[str]:
        values: List[str] = []

        if obj is None:
            return values

        if isinstance(obj, str):
            values.append(obj)

        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, str):
                    values.append(item)
                elif isinstance(item, dict):
                    for value in item.values():
                        values.extend(collect_from_object(value))
                else:
                    values.append(str(item))

        elif isinstance(obj, dict):
            for value in obj.values():
                values.extend(collect_from_object(value))

        else:
            values.append(str(obj))

        return values

    def collect(keys: List[str]) -> List[str]:
        values: List[str] = []

        for key in keys:
            if key in contract:
                values.extend(collect_from_object(contract.get(key)))

            if isinstance(dynamic_filters, dict) and key in dynamic_filters:
                values.extend(collect_from_object(dynamic_filters.get(key)))

            if isinstance(intent, dict) and key in intent:
                values.extend(collect_from_object(intent.get(key)))

        cleaned = [
            str(value).strip()
            for value in values
            if str(value).strip()
        ]

        return list(dict.fromkeys(cleaned))

    required_terms = collect(
        [
            "required_terms",
            "must_have_terms",
            "must_include",
            "product_type_terms",
            "category_terms",
            "hard_constraints",
            "product_type",
            "detected_category",
        ]
    )

    preferred_terms = collect(
        [
            "preferred_terms",
            "nice_to_have_terms",
            "soft_constraints",
            "attributes",
            "intent_terms",
            "quality_preference",
            "price_sensitivity",
        ]
    )

    brand_terms = collect(
        [
            "brand_preferences",
            "preferred_brands",
            "brands",
            "brand_terms",
            "preferred_brand",
        ]
    )

    excluded_terms = collect(
        [
            "excluded_terms",
            "avoid_terms",
            "negative_terms",
            "must_not_include",
            "blocked_terms",
        ]
    )

    return {
        "required_terms": required_terms,
        "preferred_terms": preferred_terms,
        "brand_terms": brand_terms,
        "excluded_terms": excluded_terms,
    }


# ---------------------------------------------------------------------
# Label quality
# ---------------------------------------------------------------------

def _label_to_score(label: Any) -> float:
    """
    ESCI-style label scoring.

    E = exact       -> 1.00
    S = substitute  -> 0.70
    C = complement  -> 0.40
    I = irrelevant  -> 0.00
    """

    if label is None:
        return 0.5

    value = str(label).strip().lower()

    mapping = {
        "e": 1.0,
        "exact": 1.0,
        "exactly": 1.0,
        "s": 0.7,
        "substitute": 0.7,
        "substitutable": 0.7,
        "c": 0.4,
        "complement": 0.4,
        "complementary": 0.4,
        "i": 0.0,
        "irrelevant": 0.0,
    }

    if value in mapping:
        return mapping[value]

    try:
        numeric = float(value)
        return _clip(numeric)
    except Exception:
        return 0.5


def _compute_label_quality(top_df: pd.DataFrame, label_col: Optional[str]) -> float:
    if label_col is None:
        return 0.5

    if label_col not in top_df.columns:
        return 0.5

    if len(top_df) == 0:
        return 0.5

    scores = [_label_to_score(value) for value in top_df[label_col].tolist()]

    return float(sum(scores) / max(len(scores), 1))


# ---------------------------------------------------------------------
# Gate signal computation
# ---------------------------------------------------------------------

def compute_baseline_gate_signals(
    baseline_df: pd.DataFrame,
    contract: Optional[Dict[str, Any]] = None,
    top_k: int = DEFAULT_TOP_K,
) -> BaselineGateSignals:
    """
    Computes confidence and alignment signals for the baseline top-k.

    This function does not change the dataframe.
    """

    if baseline_df is None or len(baseline_df) == 0:
        return BaselineGateSignals(
            top_k=top_k,
            num_candidates=0,
            score_column=None,
            id_column=None,
            title_column=None,
            brand_column=None,
            label_column=None,
            top1_score=0.0,
            top5_mean_score=0.0,
            top5_min_score=0.0,
            score_gap_top1_top5=0.0,
            score_decay_strength=0.0,
            label_quality_score=0.0,
            contract_alignment_score=0.0,
            exclusion_violation_rate=1.0,
            required_term_coverage=0.0,
            preferred_term_coverage=0.0,
            brand_preference_coverage=0.0,
            baseline_confidence_score=0.0,
            final_gate_score=0.0,
        )

    df = baseline_df.copy().reset_index(drop=True)

    id_col = infer_id_column(df)
    score_col = infer_score_column(df)
    title_col = infer_title_column(df)
    brand_col = infer_brand_column(df)
    label_col = infer_label_column(df)

    if score_col is not None:
        df["_gate_norm_score"] = _normalize_score_series(df[score_col])
    else:
        # If no score exists, assume order is meaningful but less certain.
        n = len(df)

        df["_gate_norm_score"] = [
            1.0 - (index / max(n - 1, 1)) * 0.5
            for index in range(n)
        ]

    top_df = df.head(top_k).copy()

    scores = top_df["_gate_norm_score"].tolist()

    top1_score = _safe_float(scores[0], 0.0) if scores else 0.0
    top5_mean_score = float(sum(scores) / max(len(scores), 1)) if scores else 0.0
    top5_min_score = float(min(scores)) if scores else 0.0

    score_gap_top1_top5 = top1_score - top5_min_score

    # Strong baseline usually has stable top-k scores, not a sharp cliff.
    score_decay_strength = 1.0 - _clip(score_gap_top1_top5)

    label_quality_score = _compute_label_quality(top_df, label_col)

    contract_terms = _extract_contract_terms(contract)

    titles: List[str] = []
    if title_col is not None and title_col in top_df.columns:
        titles = [str(value) for value in top_df[title_col].fillna("").tolist()]

    brands: List[str] = []
    if brand_col is not None and brand_col in top_df.columns:
        brands = [str(value) for value in top_df[brand_col].fillna("").tolist()]

    searchable_texts = titles + brands

    required_term_coverage = _coverage(
        searchable_texts,
        contract_terms["required_terms"],
    )

    preferred_term_coverage = _coverage(
        searchable_texts,
        contract_terms["preferred_terms"],
    )

    if contract_terms["brand_terms"]:
        brand_preference_coverage = _coverage(
            brands + titles,
            contract_terms["brand_terms"],
        )
    else:
        brand_preference_coverage = 1.0

    excluded_terms = contract_terms["excluded_terms"]

    if excluded_terms and searchable_texts:
        violations = 0

        for text in searchable_texts:
            if _contains_any(text, excluded_terms):
                violations += 1

        exclusion_violation_rate = violations / max(len(searchable_texts), 1)
    else:
        exclusion_violation_rate = 0.0

    no_exclusion_score = 1.0 - exclusion_violation_rate

    contract_alignment_score = (
        0.45 * required_term_coverage
        + 0.25 * preferred_term_coverage
        + 0.20 * brand_preference_coverage
        + 0.10 * no_exclusion_score
    )

    contract_alignment_score = _clip(contract_alignment_score)

    baseline_confidence_score = (
        0.45 * top5_mean_score
        + 0.20 * top1_score
        + 0.15 * score_decay_strength
        + 0.20 * label_quality_score
    )

    baseline_confidence_score = _clip(baseline_confidence_score)

    final_gate_score = (
        0.60 * baseline_confidence_score
        + 0.40 * contract_alignment_score
    )

    final_gate_score = _clip(final_gate_score)

    return BaselineGateSignals(
        top_k=top_k,
        num_candidates=len(df),
        score_column=score_col,
        id_column=id_col,
        title_column=title_col,
        brand_column=brand_col,
        label_column=label_col,
        top1_score=round(top1_score, 6),
        top5_mean_score=round(top5_mean_score, 6),
        top5_min_score=round(top5_min_score, 6),
        score_gap_top1_top5=round(score_gap_top1_top5, 6),
        score_decay_strength=round(score_decay_strength, 6),
        label_quality_score=round(label_quality_score, 6),
        contract_alignment_score=round(contract_alignment_score, 6),
        exclusion_violation_rate=round(exclusion_violation_rate, 6),
        required_term_coverage=round(required_term_coverage, 6),
        preferred_term_coverage=round(preferred_term_coverage, 6),
        brand_preference_coverage=round(brand_preference_coverage, 6),
        baseline_confidence_score=round(baseline_confidence_score, 6),
        final_gate_score=round(final_gate_score, 6),
    )


# ---------------------------------------------------------------------
# Gate decision
# ---------------------------------------------------------------------

def decide_baseline_gate(
    baseline_df: pd.DataFrame,
    contract: Optional[Dict[str, Any]] = None,
    top_k: int = DEFAULT_TOP_K,
) -> BaselineGateDecision:
    """
    Returns preserve_baseline, light_rerank, or full_cortex.
    """

    signals = compute_baseline_gate_signals(
        baseline_df=baseline_df,
        contract=contract,
        top_k=top_k,
    )

    if signals.num_candidates == 0:
        return BaselineGateDecision(
            decision="full_cortex",
            reason="Baseline has no candidates.",
            signals=signals,
        )

    if signals.exclusion_violation_rate > 0.25:
        return BaselineGateDecision(
            decision="full_cortex",
            reason="Baseline top-k has too many contract exclusion violations.",
            signals=signals,
        )

    if (
        signals.baseline_confidence_score >= PRESERVE_CONFIDENCE_THRESHOLD
        and signals.contract_alignment_score >= PRESERVE_ALIGNMENT_THRESHOLD
    ):
        return BaselineGateDecision(
            decision="preserve_baseline",
            reason="Baseline top-k appears very strong and contract-aligned.",
            signals=signals,
        )

    if (
        signals.baseline_confidence_score >= LIGHT_CONFIDENCE_THRESHOLD
        and signals.contract_alignment_score >= LIGHT_ALIGNMENT_THRESHOLD
    ):
        return BaselineGateDecision(
            decision="light_rerank",
            reason="Baseline is strong enough for conservative blended reranking.",
            signals=signals,
        )

    return BaselineGateDecision(
        decision="full_cortex",
        reason="Baseline is not strong enough to override full CORTEX.",
        signals=signals,
    )


# ---------------------------------------------------------------------
# Slate selection
# ---------------------------------------------------------------------

def _add_rank_score(
    df: pd.DataFrame,
    rank_col_name: str,
    score_col_name: str,
) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)

    n = len(out)

    if n == 0:
        out[rank_col_name] = []
        out[score_col_name] = []
        return out

    out[rank_col_name] = range(1, n + 1)

    # Rank score: rank 1 gets 1.0, last gets smaller value.
    out[score_col_name] = [
        1.0 - ((rank - 1) / max(n - 1, 1))
        for rank in out[rank_col_name]
    ]

    return out


def _dedupe_keep_order(df: pd.DataFrame, id_col: Optional[str]) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame()

    if id_col is None or id_col not in df.columns:
        return df.reset_index(drop=True)

    return df.drop_duplicates(
        subset=[id_col],
        keep="first",
    ).reset_index(drop=True)


def preserve_baseline_slate(
    baseline_df: pd.DataFrame,
    cortex_df: Optional[pd.DataFrame] = None,
    top_k: int = DEFAULT_TOP_K,
) -> pd.DataFrame:
    """
    Keeps baseline order for top-k.

    If more rows are needed after top-k, appends CORTEX candidates not already used.
    """

    baseline = baseline_df.copy().reset_index(drop=True)
    id_col = infer_id_column(baseline)

    baseline = _dedupe_keep_order(baseline, id_col)

    selected = baseline.head(top_k).copy()

    if cortex_df is not None and len(selected) < top_k:
        cortex = cortex_df.copy().reset_index(drop=True)
        cortex_id_col = infer_id_column(cortex)

        if (
            id_col is not None
            and cortex_id_col == id_col
            and id_col in selected.columns
        ):
            used = set(selected[id_col].astype(str).tolist())
            cortex = cortex[~cortex[id_col].astype(str).isin(used)]

        selected = pd.concat(
            [selected, cortex],
            ignore_index=True,
        ).head(top_k)

    selected["mvp13_3_gate_mode"] = "preserve_baseline"

    return selected.reset_index(drop=True)


def full_cortex_slate(
    baseline_df: Optional[pd.DataFrame],
    cortex_df: pd.DataFrame,
    top_k: int = DEFAULT_TOP_K,
) -> pd.DataFrame:
    """
    Uses full CORTEX output.
    """

    cortex = cortex_df.copy().reset_index(drop=True)
    id_col = infer_id_column(cortex)

    cortex = _dedupe_keep_order(cortex, id_col)

    selected = cortex.head(top_k).copy()
    selected["mvp13_3_gate_mode"] = "full_cortex"

    return selected.reset_index(drop=True)


def light_rerank_slate(
    baseline_df: pd.DataFrame,
    cortex_df: pd.DataFrame,
    top_k: int = DEFAULT_TOP_K,
    baseline_weight: float = BASELINE_WEIGHT_LIGHT_RERANK,
    cortex_weight: float = CORTEX_WEIGHT_LIGHT_RERANK,
) -> pd.DataFrame:
    """
    Conservative blend:
    - Baseline is considered, but CORTEX receives slightly more weight in MVP 13.3.1.
    - This prevents the gate from suppressing CORTEX too aggressively.
    """

    baseline = baseline_df.copy().reset_index(drop=True)
    cortex = cortex_df.copy().reset_index(drop=True)

    id_col = infer_id_column(baseline)
    cortex_id_col = infer_id_column(cortex)

    if id_col is None or cortex_id_col is None or id_col != cortex_id_col:
        # Fallback: if we cannot safely merge, use CORTEX because full CORTEX is the default path.
        selected = cortex.head(top_k).copy()
        selected["mvp13_3_gate_mode"] = "light_rerank_fallback_full_cortex"
        selected["mvp13_3_light_score"] = 1.0
        return selected.reset_index(drop=True)

    baseline = _dedupe_keep_order(baseline, id_col)
    cortex = _dedupe_keep_order(cortex, id_col)

    baseline = _add_rank_score(
        baseline,
        rank_col_name="_baseline_rank",
        score_col_name="_baseline_rank_score",
    )

    cortex = _add_rank_score(
        cortex,
        rank_col_name="_cortex_rank",
        score_col_name="_cortex_rank_score",
    )

    cortex_columns = [
        id_col,
        "_cortex_rank",
        "_cortex_rank_score",
    ]

    merged = baseline.merge(
        cortex[cortex_columns],
        on=id_col,
        how="outer",
    )

    # Missing candidates get low rank scores, not zero, to avoid harsh deletion.
    merged["_baseline_rank_score"] = merged["_baseline_rank_score"].fillna(0.10)
    merged["_cortex_rank_score"] = merged["_cortex_rank_score"].fillna(0.10)

    merged["mvp13_3_light_score"] = (
        baseline_weight * merged["_baseline_rank_score"]
        + cortex_weight * merged["_cortex_rank_score"]
    )

    merged = merged.sort_values(
        by=[
            "mvp13_3_light_score",
            "_cortex_rank_score",
            "_baseline_rank_score",
        ],
        ascending=[False, False, False],
    ).reset_index(drop=True)

    selected_ids = merged[id_col].astype(str).head(top_k).tolist()

    # Reconstruct full rows from CORTEX first, then baseline if candidate only exists there.
    # MVP 13.3.1 intentionally favors CORTEX metadata/order in light rerank.
    baseline["_id_str"] = baseline[id_col].astype(str)
    cortex["_id_str"] = cortex[id_col].astype(str)

    rows = []

    for candidate_id in selected_ids:
        c_match = cortex[cortex["_id_str"] == candidate_id]

        if len(c_match) > 0:
            row = c_match.iloc[0].drop(labels=["_id_str"]).to_dict()
        else:
            b_match = baseline[baseline["_id_str"] == candidate_id]
            row = b_match.iloc[0].drop(labels=["_id_str"]).to_dict()

        score_match = merged[merged[id_col].astype(str) == candidate_id]

        if len(score_match) > 0:
            row["mvp13_3_light_score"] = float(score_match.iloc[0]["mvp13_3_light_score"])
            row["mvp13_3_baseline_rank_score"] = float(score_match.iloc[0]["_baseline_rank_score"])
            row["mvp13_3_cortex_rank_score"] = float(score_match.iloc[0]["_cortex_rank_score"])

        rows.append(row)

    selected = pd.DataFrame(rows)
    selected["mvp13_3_gate_mode"] = "light_rerank"

    return selected.reset_index(drop=True)


def select_slate_with_baseline_preservation_gate(
    baseline_df: pd.DataFrame,
    cortex_df: pd.DataFrame,
    contract: Optional[Dict[str, Any]] = None,
    top_k: int = DEFAULT_TOP_K,
) -> Tuple[pd.DataFrame, BaselineGateDecision]:
    """
    Main function to call from app/evaluator.

    Inputs:
    -------
    baseline_df:
        Semantic baseline candidates in current baseline order.

    cortex_df:
        CORTEX reranked candidates in final CORTEX order.

    contract:
        Optional LLM search contract.

    top_k:
        Slate size.

    Returns:
    --------
    final_slate:
        Final selected slate after gate decision.

    decision:
        Gate decision object with diagnostic signals.
    """

    decision = decide_baseline_gate(
        baseline_df=baseline_df,
        contract=contract,
        top_k=top_k,
    )

    if decision.decision == "preserve_baseline":
        final_slate = preserve_baseline_slate(
            baseline_df=baseline_df,
            cortex_df=cortex_df,
            top_k=top_k,
        )

    elif decision.decision == "light_rerank":
        final_slate = light_rerank_slate(
            baseline_df=baseline_df,
            cortex_df=cortex_df,
            top_k=top_k,
        )

    else:
        final_slate = full_cortex_slate(
            baseline_df=baseline_df,
            cortex_df=cortex_df,
            top_k=top_k,
        )

    gate_metadata = decision.to_dict()

    for key, value in gate_metadata.items():
        final_slate[key] = value

    return final_slate.reset_index(drop=True), decision


# ---------------------------------------------------------------------
# Convenience function for evaluator rows
# ---------------------------------------------------------------------

def summarize_gate_for_logging(decision: BaselineGateDecision) -> Dict[str, Any]:
    """
    Converts gate decision into flat dictionary for CSV logging.
    """

    return decision.to_dict()


# ---------------------------------------------------------------------
# Manual smoke test
# ---------------------------------------------------------------------

if __name__ == "__main__":
    sample_baseline = pd.DataFrame(
        [
            {
                "item_id": "A",
                "title": "nike running shoes men",
                "brand": "Nike",
                "semantic_score": 0.94,
                "esci_label": "E",
            },
            {
                "item_id": "B",
                "title": "nike training shoes men",
                "brand": "Nike",
                "semantic_score": 0.91,
                "esci_label": "E",
            },
            {
                "item_id": "C",
                "title": "adidas running shoes men",
                "brand": "Adidas",
                "semantic_score": 0.88,
                "esci_label": "S",
            },
            {
                "item_id": "D",
                "title": "men athletic sneakers",
                "brand": "Puma",
                "semantic_score": 0.85,
                "esci_label": "S",
            },
            {
                "item_id": "E",
                "title": "black running shoes",
                "brand": "Asics",
                "semantic_score": 0.82,
                "esci_label": "S",
            },
        ]
    )

    sample_cortex = pd.DataFrame(
        [
            {
                "item_id": "C",
                "title": "adidas running shoes men",
                "brand": "Adidas",
                "semantic_score": 0.88,
            },
            {
                "item_id": "A",
                "title": "nike running shoes men",
                "brand": "Nike",
                "semantic_score": 0.94,
            },
            {
                "item_id": "B",
                "title": "nike training shoes men",
                "brand": "Nike",
                "semantic_score": 0.91,
            },
            {
                "item_id": "E",
                "title": "black running shoes",
                "brand": "Asics",
                "semantic_score": 0.82,
            },
            {
                "item_id": "D",
                "title": "men athletic sneakers",
                "brand": "Puma",
                "semantic_score": 0.85,
            },
        ]
    )

    sample_contract = {
        "required_terms": ["running shoes"],
        "preferred_terms": ["men"],
        "brand_preferences": ["nike"],
        "excluded_terms": ["women", "kids"],
    }

    slate, gate = select_slate_with_baseline_preservation_gate(
        baseline_df=sample_baseline,
        cortex_df=sample_cortex,
        contract=sample_contract,
        top_k=5,
    )

    print("Gate decision:", gate.decision)
    print("Reason:", gate.reason)
    print("Signals:", gate.signals)
    print(slate)