import re
from typing import Dict, List, Set

import pandas as pd


SOFT_INTENT_WORDS = {
    "cheap",
    "budget",
    "affordable",
    "best",
    "premium",
    "top",
    "rated",
    "wireless",
    "quiet",
    "durable",
    "for",
    "with",
    "and",
    "the",
    "a",
    "an",
    "of",
    "to",
    "in",
    "on",
    "by",
    "from",
    "please",
    "not",
}


BROAD_CONTEXT_TERMS = {
    "dog",
    "dogs",
    "pet",
    "pets",
    "canine",
    "cat",
    "cats",
    "animal",
    "animals",
    "baby",
    "kids",
    "men",
    "women",
    "adult",
    "adults",
    "home",
    "house",
    "outdoor",
    "indoor",
}


PRODUCT_CORE_SYNONYMS = {
    "leash": ["leash", "leashes", "lead", "leads", "dog leash", "pet leash"],
    "leashes": ["leash", "leashes", "lead", "leads", "dog leash", "pet leash"],
    "collar": ["collar", "collars", "dog collar", "pet collar"],
    "harness": ["harness", "harnesses", "dog harness", "pet harness"],
    "bag": ["bag", "bags"],
    "bags": ["bag", "bags"],
    "waste bag": ["waste bag", "waste bags", "poop bag", "poop bags", "dog waste", "pet waste"],
    "poop bag": ["waste bag", "waste bags", "poop bag", "poop bags", "dog waste", "pet waste"],
    "headphone": ["headphone", "headphones", "earbud", "earbuds", "headset", "earphone", "earphones"],
    "headphones": ["headphone", "headphones", "earbud", "earbuds", "headset", "earphone", "earphones"],
    "earbud": ["earbud", "earbuds", "headphone", "headphones", "earphone", "earphones"],
    "earbuds": ["earbud", "earbuds", "headphone", "headphones", "earphone", "earphones"],
    "fan": ["fan", "fans", "ventilation", "exhaust", "blower"],
    "fans": ["fan", "fans", "ventilation", "exhaust", "blower"],
    "shoe": ["shoe", "shoes", "sneaker", "sneakers", "trainer", "trainers", "footwear"],
    "shoes": ["shoe", "shoes", "sneaker", "sneakers", "trainer", "trainers", "footwear"],
    "coffee maker": ["coffee maker", "coffee machine", "drip coffee", "espresso machine", "brewer"],
    "vitamin c serum": ["vitamin c serum", "serum", "face serum", "skin serum"],
    "laptop stand": ["laptop stand", "notebook stand", "computer stand"],
}


def normalize_text(value) -> str:
    if not isinstance(value, str):
        return ""

    value = value.lower()
    value = re.sub(r"[^a-z0-9\s\-]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def simple_singular(term: str) -> str:
    term = normalize_text(term)

    if term.endswith("ies") and len(term) > 4:
        return term[:-3] + "y"

    if term.endswith("es") and len(term) > 4:
        return term[:-2]

    if term.endswith("s") and len(term) > 3:
        return term[:-1]

    return term


def expand_term(term: str) -> List[str]:
    term = normalize_text(term)
    singular = simple_singular(term)

    expanded = set()

    for candidate in [term, singular]:
        if candidate:
            expanded.add(candidate)

        if candidate in PRODUCT_CORE_SYNONYMS:
            for synonym in PRODUCT_CORE_SYNONYMS[candidate]:
                expanded.add(normalize_text(synonym))

    return sorted(expanded)


def get_product_text(row: pd.Series) -> str:
    parts = []

    for col in ["product_title", "category", "query"]:
        if col in row.index:
            parts.append(str(row.get(col, "")))

    return normalize_text(" ".join(parts))


def keyword_present(text: str, keyword: str) -> bool:
    keyword = normalize_text(keyword)

    if not keyword:
        return False

    if " " in keyword or "-" in keyword:
        return keyword in text

    tokens = set(text.split())

    keyword_singular = simple_singular(keyword)

    for token in tokens:
        token_singular = simple_singular(token)

        if token == keyword:
            return True

        if token_singular == keyword_singular:
            return True

        if token.startswith(keyword) and len(keyword) >= 4:
            return True

        if keyword.startswith(token) and len(token) >= 4:
            return True

    return False


def clean_terms(terms: List[str]) -> List[str]:
    cleaned = []

    if not isinstance(terms, list):
        return cleaned

    for term in terms:
        term = normalize_text(term)

        if not term:
            continue

        if term in SOFT_INTENT_WORDS:
            continue

        if len(term) < 3:
            continue

        if term not in cleaned:
            cleaned.append(term)

    return cleaned


def extract_query_core_terms(query: str) -> List[str]:
    query = normalize_text(query)
    tokens = query.split()

    core_terms = []

    # Add phrase-level special cases first.
    phrase_candidates = [
        "dog leash",
        "pet leash",
        "dog leashes",
        "pet leashes",
        "waste bag",
        "poop bag",
        "coffee maker",
        "vitamin c serum",
        "laptop stand",
        "bathroom fan",
        "wireless headphones",
        "running shoes",
    ]

    for phrase in phrase_candidates:
        if phrase in query:
            core_terms.append(phrase)

    for token in tokens:
        if token in SOFT_INTENT_WORDS:
            continue

        if token in BROAD_CONTEXT_TERMS:
            continue

        if len(token) < 3:
            continue

        core_terms.append(token)

    # Expand synonyms.
    expanded = []
    for term in core_terms:
        for expanded_term in expand_term(term):
            if expanded_term not in expanded:
                expanded.append(expanded_term)

    return expanded


def get_dynamic_filters(contract: Dict) -> Dict:
    dynamic_filters = contract.get("dynamic_filters", {})
    query = contract.get("query", "")

    must_have_terms = clean_terms(dynamic_filters.get("must_have_terms", []))
    nice_to_have_terms = clean_terms(dynamic_filters.get("nice_to_have_terms", []))
    blocked_terms = clean_terms(dynamic_filters.get("blocked_terms", []))
    allowed_substitutes = clean_terms(dynamic_filters.get("allowed_substitutes", []))
    allowed_complements = clean_terms(dynamic_filters.get("allowed_complements", []))

    query_core_terms = extract_query_core_terms(query)

    # Separate broad context from actual product-core terms.
    context_terms = []
    product_core_terms = []

    for term in must_have_terms:
        if term in BROAD_CONTEXT_TERMS:
            context_terms.append(term)
        else:
            product_core_terms.append(term)

    # Query-derived product core terms are mandatory hints.
    for term in query_core_terms:
        if term not in product_core_terms:
            product_core_terms.append(term)

    # Expand product core terms with synonyms.
    expanded_product_core_terms = []
    for term in product_core_terms:
        for expanded in expand_term(term):
            if expanded not in expanded_product_core_terms:
                expanded_product_core_terms.append(expanded)

    # If no product core terms exist, use must-have terms as fallback.
    if not expanded_product_core_terms:
        expanded_product_core_terms = must_have_terms

    return {
        "must_have_terms": must_have_terms,
        "product_core_terms": expanded_product_core_terms,
        "context_terms": context_terms,
        "nice_to_have_terms": nice_to_have_terms,
        "blocked_terms": blocked_terms,
        "allowed_substitutes": allowed_substitutes,
        "allowed_complements": allowed_complements,
        "query_core_terms": query_core_terms,
    }


def get_matched_terms(product_text: str, terms: List[str]) -> List[str]:
    return [term for term in terms if keyword_present(product_text, term)]


def compute_dynamic_contract_match_score(row: pd.Series, contract: Dict) -> float:
    product_text = get_product_text(row)
    filters = get_dynamic_filters(contract)

    core_hits = get_matched_terms(product_text, filters["product_core_terms"])
    context_hits = get_matched_terms(product_text, filters["context_terms"])
    must_hits = get_matched_terms(product_text, filters["must_have_terms"])
    nice_hits = get_matched_terms(product_text, filters["nice_to_have_terms"])
    blocked_hits = get_matched_terms(product_text, filters["blocked_terms"])
    substitute_hits = get_matched_terms(product_text, filters["allowed_substitutes"])
    complement_hits = get_matched_terms(product_text, filters["allowed_complements"])

    if blocked_hits:
        return 0.00

    score = 0.05

    # Core product noun is the strongest signal.
    # Example: for dog leashes, leash/leashes must matter more than dog.
    if core_hits:
        score += 0.65
    else:
        # Context-only matches should not pass strongly.
        if context_hits or must_hits:
            score += 0.15

    # Substitutes are useful only if they also have product-core alignment
    # or are explicitly close substitutes.
    if substitute_hits:
        score += 0.20

    # Complements should rank lower than true matches.
    if complement_hits:
        score += 0.10

    # Nice terms should only provide a small boost.
    score += min(0.15, len(nice_hits) * 0.04)

    return max(0.0, min(1.0, float(score)))


def explain_dynamic_filter(row: pd.Series, contract: Dict) -> str:
    product_text = get_product_text(row)
    filters = get_dynamic_filters(contract)

    core_hits = get_matched_terms(product_text, filters["product_core_terms"])
    context_hits = get_matched_terms(product_text, filters["context_terms"])
    must_hits = get_matched_terms(product_text, filters["must_have_terms"])
    nice_hits = get_matched_terms(product_text, filters["nice_to_have_terms"])
    blocked_hits = get_matched_terms(product_text, filters["blocked_terms"])
    substitute_hits = get_matched_terms(product_text, filters["allowed_substitutes"])
    complement_hits = get_matched_terms(product_text, filters["allowed_complements"])

    reasons = []

    if blocked_hits:
        reasons.append(f"blocked={blocked_hits[:3]}")

    if core_hits:
        reasons.append(f"core_match={core_hits[:3]}")

    if context_hits:
        reasons.append(f"context_match={context_hits[:3]}")

    if must_hits:
        reasons.append(f"must_match={must_hits[:3]}")

    if substitute_hits:
        reasons.append(f"substitute_match={substitute_hits[:3]}")

    if complement_hits:
        reasons.append(f"complement_match={complement_hits[:3]}")

    if nice_hits:
        reasons.append(f"nice_match={nice_hits[:3]}")

    if not reasons:
        reasons.append("weak_dynamic_filter_match")

    return " | ".join(reasons)


def apply_contract_candidate_filter(
    df: pd.DataFrame,
    contract: Dict,
    min_match_score: float = 0.50,
    strict_top_k: int = 20,
) -> pd.DataFrame:
    """
    Dynamic LLM contract filter with core product-term enforcement.

    This fixes cases like:
    - query = dog leashes
    - weak match = any product containing dog
    - strong match = product containing leash/leashes/lead/pet leash

    The product core term must dominate broad context terms.
    """

    if df is None or len(df) == 0:
        return df

    filtered_df = df.copy()

    filtered_df["contract_match_score"] = filtered_df.apply(
        lambda row: compute_dynamic_contract_match_score(row, contract),
        axis=1,
    )

    filtered_df["contract_filter_reason"] = filtered_df.apply(
        lambda row: explain_dynamic_filter(row, contract),
        axis=1,
    )

    # Strong matches are true product-type candidates.
    strong_df = filtered_df[
        filtered_df["contract_match_score"] >= min_match_score
    ].copy()

    # If at least one strong match exists, put strong matches first.
    # Keep weaker candidates only after strong matches so the UI still has enough rows.
    if len(strong_df) > 0:
        weak_df = filtered_df[
            filtered_df["contract_match_score"] < min_match_score
        ].copy()

        filtered_df = pd.concat([strong_df, weak_df], ignore_index=True)

    # Blend contract score into baseline score.
    if "baseline_score" in filtered_df.columns:
        filtered_df["baseline_score"] = (
            0.50 * filtered_df["baseline_score"].astype(float)
            + 0.50 * filtered_df["contract_match_score"].astype(float)
        )

    sort_cols = []

    if "contract_match_score" in filtered_df.columns:
        sort_cols.append("contract_match_score")

    if "baseline_score" in filtered_df.columns:
        sort_cols.append("baseline_score")

    if sort_cols:
        filtered_df = filtered_df.sort_values(
            by=sort_cols,
            ascending=False,
        )

    return filtered_df.head(strict_top_k).reset_index(drop=True)