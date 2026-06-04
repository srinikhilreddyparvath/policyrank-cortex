"""
MVP 22: Query Understanding Agent

Standalone pre-governance query understanding layer for CORTEX / PolicyRank-RL.
This module does not change governance behavior. It classifies query intent with
deterministic heuristics and writes only under outputs/query_understanding/.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd


OUTPUT_DIR = Path("outputs/query_understanding")

PREFERRED_ESCI_PATHS = [
    Path("external_data/esci-data/shopping_queries_dataset/shopping_queries_dataset_examples.parquet"),
    Path("esci-data/shopping_queries_dataset/shopping_queries_dataset_examples.parquet"),
    Path("data/esci_balanced_sample.csv"),
]

SMOKE_QUERIES = [
    "adidas soccer cleats",
    "beach vacation packing list",
    "new apartment kitchen setup",
    "gift basket for new mom",
    "charger for canon camera",
    "running shoes without laces",
    "64gb micro sd",
    "a7r iii",
    "baby shower decorations",
    "office desk setup",
]

QUERY_TYPE_ORDER = [
    "narrow_product",
    "numeric_model",
    "mission_query",
    "setup_or_kit",
    "gift_or_party",
    "compatibility_query",
    "negation_constraint",
    "noisy_query",
    "broad_discovery",
    "general_retail",
]

BIAS_ORDER = [
    "preserve_baseline",
    "allow_mission_repair",
    "allow_behavior_aware",
    "strict_guardrails",
    "reject_aggressive_repair",
    "send_to_critic",
]

BRAND_TERMS = {
    "adidas",
    "amazon",
    "apple",
    "barbie",
    "bissell",
    "canon",
    "carhartt",
    "dewalt",
    "dyson",
    "hp",
    "keurig",
    "lego",
    "lenovo",
    "lg",
    "macbook",
    "microsoft",
    "nike",
    "ninja",
    "otterbox",
    "puma",
    "samsung",
    "sony",
    "under",
    "xbox",
}

PRODUCT_TERMS = {
    "bag",
    "bottle",
    "camera",
    "case",
    "charger",
    "cleats",
    "cover",
    "dress",
    "headphones",
    "jacket",
    "keyboard",
    "laptop",
    "mouse",
    "phone",
    "shirt",
    "shoes",
    "sneakers",
    "tv",
}

MISSION_TERMS = {
    "camping",
    "college",
    "dorm",
    "essentials",
    "moving",
    "packing",
    "registry",
    "supplies",
    "travel",
    "trip",
    "vacation",
}

SETUP_TERMS = {
    "setup",
    "kit",
    "bundle",
    "set",
    "starter kit",
}

EVENT_TERMS = {
    "baby shower",
    "barbecue",
    "bbq",
    "birthday",
    "christmas",
    "graduation",
    "halloween",
    "party",
    "shower",
    "tailgate",
    "thanksgiving",
    "wedding",
}

GIFT_TERMS = {
    "basket",
    "favor",
    "favors",
    "gift",
    "gifts",
    "present",
    "stocking",
}

ATTRIBUTE_TERMS = {
    "black",
    "blue",
    "cotton",
    "large",
    "leather",
    "lightweight",
    "metal",
    "mini",
    "pink",
    "portable",
    "red",
    "small",
    "stainless",
    "waterproof",
    "white",
    "wireless",
    "women",
    "men",
    "kids",
}


@dataclass
class QueryUnderstandingResult:
    query: str
    normalized_query: str
    token_count: int
    char_count: int
    query_type: str
    intent_bucket: str
    is_narrow_product: bool
    is_brand_specific: bool
    is_numeric_model: bool
    is_mission_query: bool
    is_setup_or_kit: bool
    is_gift_or_party: bool
    is_event_query: bool
    is_compatibility_query: bool
    has_negation_constraint: bool
    has_attribute_constraint: bool
    repair_eligibility: str
    recommended_governance_bias: str
    confidence_score: float
    risk_score: float
    plain_english_reason: str


RESULT_FIELDS = list(QueryUnderstandingResult.__dataclass_fields__.keys())


def clean_text(value: object) -> str:
    return str(value or "").strip()


def normalize_query(query: str) -> str:
    return re.sub(r"\s+", " ", clean_text(query).lower()).strip()


def console_text(value: object) -> str:
    text = clean_text(value)
    encoding = sys.stdout.encoding or "utf-8"
    return text.encode(encoding, errors="replace").decode(encoding, errors="replace")


def tokenize_query(query: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", normalize_query(query))


def has_term(normalized: str, tokens: set[str], terms: set[str]) -> bool:
    for term in terms:
        term = term.lower()
        if " " in term:
            if term in normalized:
                return True
        elif term in tokens:
            return True
    return False


def has_numeric_or_model_token(tokens: List[str]) -> bool:
    return any(
        any(ch.isdigit() for ch in token)
        or bool(re.search(r"[a-z]+\d+|\d+[a-z]+", token))
        for token in tokens
    )


def extract_features(query: str) -> Dict[str, object]:
    normalized = normalize_query(query)
    tokens = tokenize_query(query)
    token_set = set(tokens)

    return {
        "normalized_query": normalized,
        "tokens": tokens,
        "token_count": len(tokens),
        "char_count": len(clean_text(query)),
        "is_brand_specific": any(token in BRAND_TERMS for token in tokens),
        "is_numeric_model": has_numeric_or_model_token(tokens),
        "is_mission_query": has_term(normalized, token_set, MISSION_TERMS),
        "is_setup_or_kit": has_term(normalized, token_set, SETUP_TERMS),
        "is_gift_or_party": has_term(normalized, token_set, GIFT_TERMS) or has_term(normalized, token_set, EVENT_TERMS),
        "is_event_query": has_term(normalized, token_set, EVENT_TERMS),
        "is_compatibility_query": "for" in token_set,
        "has_negation_constraint": any(token in {"without", "not", "no", "non"} for token in tokens),
        "has_attribute_constraint": has_term(normalized, token_set, ATTRIBUTE_TERMS) or bool(re.search(r"\b\d+\s?(inch|in|ft|oz|gb|tb|mm|cm|xl)\b", normalized)),
        "starts_with_symbol": bool(clean_text(query)) and not clean_text(query)[0].isalnum(),
        "has_product_term": any(token in PRODUCT_TERMS for token in tokens),
    }


def infer_query_type(features: Dict[str, object]) -> str:
    token_count = int(features["token_count"])

    if bool(features["starts_with_symbol"]):
        return "noisy_query"
    if bool(features["has_negation_constraint"]):
        return "negation_constraint"
    if bool(features["is_gift_or_party"]):
        return "gift_or_party"
    if bool(features["is_setup_or_kit"]):
        return "setup_or_kit"
    if bool(features["is_mission_query"]):
        return "mission_query"
    if bool(features["is_compatibility_query"]):
        return "compatibility_query"
    if bool(features["is_numeric_model"]):
        return "numeric_model"
    if bool(features["is_brand_specific"]) or (token_count <= 3 and bool(features["has_product_term"])):
        return "narrow_product"
    if token_count >= 5 or int(features["char_count"]) >= 40:
        return "broad_discovery"
    return "general_retail"


def infer_bias(query_type: str, features: Dict[str, object]) -> str:
    if query_type in {"narrow_product", "numeric_model"}:
        return "preserve_baseline"
    if query_type == "noisy_query":
        return "send_to_critic"
    if query_type in {"negation_constraint", "compatibility_query"}:
        return "strict_guardrails"
    if query_type in {"mission_query", "setup_or_kit", "gift_or_party"}:
        if bool(features["has_negation_constraint"]):
            return "strict_guardrails"
        if query_type == "setup_or_kit":
            return "allow_behavior_aware"
        return "allow_mission_repair"
    if query_type == "broad_discovery":
        return "allow_behavior_aware"
    return "reject_aggressive_repair"


def infer_repair_eligibility(query_type: str, bias: str) -> str:
    if bias == "preserve_baseline":
        return "not_repair_eligible"
    if bias == "reject_aggressive_repair":
        return "baseline_preferred"
    if bias == "strict_guardrails":
        return "repair_only_with_strict_guardrails"
    if bias == "send_to_critic":
        return "critic_review_before_repair"
    if query_type in {"mission_query", "setup_or_kit", "gift_or_party", "broad_discovery"}:
        return "repair_eligible"
    return "limited_repair_eligible"


def confidence_score(query_type: str, features: Dict[str, object]) -> float:
    score = 0.55
    if query_type in {"setup_or_kit", "gift_or_party", "negation_constraint", "numeric_model"}:
        score += 0.20
    if query_type in {"mission_query", "compatibility_query", "narrow_product"}:
        score += 0.15
    if bool(features["is_brand_specific"]):
        score += 0.05
    if bool(features["has_attribute_constraint"]):
        score += 0.04
    if bool(features["starts_with_symbol"]):
        score -= 0.10
    return round(max(0.05, min(score, 0.98)), 4)


def risk_score(query_type: str, features: Dict[str, object]) -> float:
    risk = 0.10
    if query_type in {"numeric_model", "narrow_product"}:
        risk += 0.20
    if bool(features["has_negation_constraint"]):
        risk += 0.25
    if query_type == "noisy_query":
        risk += 0.30
    if bool(features["is_compatibility_query"]):
        risk += 0.15
    if bool(features["is_mission_query"]) or bool(features["is_setup_or_kit"]):
        risk -= 0.05
    return round(max(0.0, min(risk, 1.0)), 4)


def explain(query: str, query_type: str, bias: str, features: Dict[str, object]) -> str:
    reasons = []
    if bool(features["is_brand_specific"]):
        reasons.append("brand-specific terms")
    if bool(features["is_numeric_model"]):
        reasons.append("numeric/model-like tokens")
    if bool(features["is_mission_query"]):
        reasons.append("mission keywords")
    if bool(features["is_setup_or_kit"]):
        reasons.append("setup/kit language")
    if bool(features["is_gift_or_party"]):
        reasons.append("gift, event, or party language")
    if bool(features["is_compatibility_query"]):
        reasons.append("compatibility 'for' structure")
    if bool(features["has_negation_constraint"]):
        reasons.append("negation constraint")
    if bool(features["starts_with_symbol"]):
        reasons.append("symbol-leading noisy shape")

    reason_text = ", ".join(reasons) if reasons else "general retail wording"
    return (
        f"'{query}' is classified as {query_type} because it contains {reason_text}. "
        f"The recommended governance bias is {bias}."
    )


def understand_query(query: str) -> QueryUnderstandingResult:
    features = extract_features(query)
    query_type = infer_query_type(features)
    bias = infer_bias(query_type, features)
    eligibility = infer_repair_eligibility(query_type, bias)

    is_narrow_product = query_type == "narrow_product" or (
        bool(features["is_brand_specific"])
        and int(features["token_count"]) <= 5
        and not bool(features["is_mission_query"])
    )

    return QueryUnderstandingResult(
        query=clean_text(query),
        normalized_query=str(features["normalized_query"]),
        token_count=int(features["token_count"]),
        char_count=int(features["char_count"]),
        query_type=query_type,
        intent_bucket=query_type,
        is_narrow_product=is_narrow_product,
        is_brand_specific=bool(features["is_brand_specific"]),
        is_numeric_model=bool(features["is_numeric_model"]),
        is_mission_query=bool(features["is_mission_query"]),
        is_setup_or_kit=bool(features["is_setup_or_kit"]),
        is_gift_or_party=bool(features["is_gift_or_party"]),
        is_event_query=bool(features["is_event_query"]),
        is_compatibility_query=bool(features["is_compatibility_query"]),
        has_negation_constraint=bool(features["has_negation_constraint"]),
        has_attribute_constraint=bool(features["has_attribute_constraint"]),
        repair_eligibility=eligibility,
        recommended_governance_bias=bias,
        confidence_score=confidence_score(query_type, features),
        risk_score=risk_score(query_type, features),
        plain_english_reason=explain(query, query_type, bias, features),
    )


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def find_query_column(columns: Iterable[str]) -> str:
    columns_list = [str(col) for col in columns]
    lower_to_original = {col.lower(): col for col in columns_list}
    for candidate in ["query", "shopping_query", "query_text", "search_query", "q"]:
        if candidate in lower_to_original:
            return lower_to_original[candidate]
    for col in columns_list:
        if "query" in col.lower():
            return col
    raise ValueError(f"Could not identify query column from columns: {columns_list}")


def unique_clean_queries(values: Iterable[object]) -> List[str]:
    queries = []
    seen = set()
    for value in values:
        query = clean_text(value)
        if not query:
            continue
        key = query.lower()
        if key in seen:
            continue
        seen.add(key)
        queries.append(query)
    return queries


def load_queries_from_path(path: Path) -> List[str]:
    if path.suffix.lower() == ".parquet":
        try:
            df = pd.read_parquet(path, columns=["query"])
            query_col = "query"
        except Exception:
            df = pd.read_parquet(path)
            query_col = find_query_column(df.columns)
    else:
        df = pd.read_csv(path)
        query_col = find_query_column(df.columns)
    return unique_clean_queries(df[query_col].tolist())


def resolve_esci_source() -> Path | None:
    for path in PREFERRED_ESCI_PATHS:
        if path.exists():
            return path
    return None


def load_all_queries(query_mode: str) -> Tuple[List[str], str]:
    if query_mode == "smoke":
        return list(SMOKE_QUERIES), "built_in_smoke_queries"

    source = resolve_esci_source()
    if source is None:
        return list(SMOKE_QUERIES), "built_in_smoke_queries_fallback"

    return load_queries_from_path(source), str(source)


def bucket_records(queries: List[str]) -> Dict[str, List[Tuple[int, str]]]:
    buckets: Dict[str, List[Tuple[int, str]]] = {query_type: [] for query_type in QUERY_TYPE_ORDER}
    for index, query in enumerate(queries):
        query_type = understand_query(query).query_type
        buckets.setdefault(query_type, []).append((index, query))
    return buckets


def select_queries(
    queries: List[str],
    query_mode: str,
    sample_size: int,
    start_index: int,
) -> List[Tuple[int, str]]:
    if query_mode == "stratified_esci":
        buckets = bucket_records(queries)
        bucket_offsets = {bucket: rows[start_index:] for bucket, rows in buckets.items()}
        selected: List[Tuple[int, str]] = []
        seen = set()

        while len(selected) < sample_size:
            added = 0
            for bucket in QUERY_TYPE_ORDER:
                rows = bucket_offsets.get(bucket, [])
                if not rows:
                    continue
                index, query = rows.pop(0)
                key = query.lower()
                if key in seen:
                    continue
                seen.add(key)
                selected.append((index, query))
                added += 1
                if len(selected) >= sample_size:
                    break
            if added == 0:
                break
        return selected

    return [
        (start_index + offset, query)
        for offset, query in enumerate(queries[start_index : start_index + sample_size])
    ]


def summarize_results(rows: List[Dict[str, object]], query_mode: str, source: str, total_available: int) -> Dict[str, object]:
    type_counts = Counter(str(row["query_type"]) for row in rows)
    bias_counts = Counter(str(row["recommended_governance_bias"]) for row in rows)
    avg_confidence = sum(float(row["confidence_score"]) for row in rows) / max(len(rows), 1)
    avg_risk = sum(float(row["risk_score"]) for row in rows) / max(len(rows), 1)

    return {
        "query_mode": query_mode,
        "source": source,
        "total_available_unique_queries": total_available,
        "selected_query_count": len(rows),
        "query_type_count": len(type_counts),
        "dominant_query_type": type_counts.most_common(1)[0][0] if type_counts else "",
        "dominant_recommended_governance_bias": bias_counts.most_common(1)[0][0] if bias_counts else "",
        "avg_confidence_score": round(avg_confidence, 6),
        "avg_risk_score": round(avg_risk, 6),
    }


def group_counts(rows: List[Dict[str, object]], field: str, ordered_values: List[str]) -> List[Dict[str, object]]:
    counts = Counter(str(row[field]) for row in rows)
    total = len(rows)
    output = []
    for value in ordered_values + sorted(set(counts) - set(ordered_values)):
        count = counts.get(value, 0)
        if count:
            output.append(
                {
                    field: value,
                    "query_count": count,
                    "query_share": round(count / max(total, 1), 6),
                }
            )
    return output


def write_outputs(rows: List[Dict[str, object]], query_mode: str, source: str, total_available: int) -> None:
    ensure_output_dir()
    summary = summarize_results(rows, query_mode=query_mode, source=source, total_available=total_available)
    by_type = group_counts(rows, "query_type", QUERY_TYPE_ORDER)
    by_bias = group_counts(rows, "recommended_governance_bias", BIAS_ORDER)

    write_csv(OUTPUT_DIR / "query_understanding_results.csv", rows, RESULT_FIELDS)
    write_csv(OUTPUT_DIR / "query_understanding_summary.csv", [summary], list(summary.keys()))
    write_csv(OUTPUT_DIR / "query_understanding_by_type.csv", by_type, ["query_type", "query_count", "query_share"])
    write_csv(
        OUTPUT_DIR / "query_understanding_by_bias.csv",
        by_bias,
        ["recommended_governance_bias", "query_count", "query_share"],
    )


def run_single_query(query: str) -> None:
    result = understand_query(query)
    rows = [asdict(result)]
    write_outputs(rows, query_mode="single_query", source="cli_query", total_available=1)

    print("\nQuery Understanding Result")
    print("-" * 100)
    for key, value in asdict(result).items():
        print(f"{key}: {console_text(value)}")
    print("\nFiles written under outputs/query_understanding/")


def run_batch(sample_size: int, query_mode: str, start_index: int) -> None:
    all_queries, source = load_all_queries(query_mode)
    selected = select_queries(
        queries=all_queries,
        query_mode=query_mode,
        sample_size=sample_size,
        start_index=start_index,
    )

    print("\nMVP 22 Query Understanding Agent")
    print("-" * 100)
    print(f"query_mode: {query_mode}")
    print(f"source: {source}")
    print(f"total_available_unique_queries: {len(all_queries)}")
    print(f"start_index: {start_index}")
    print(f"sample_size: {sample_size}")
    print(f"selected_query_count: {len(selected)}")

    rows = []
    for selected_index, (query_index, query) in enumerate(selected, start=1):
        result = understand_query(query)
        row = asdict(result)
        rows.append(row)
        if selected_index <= 10:
            print(
                f"{selected_index}. {console_text(query)} -> "
                f"{result.query_type} / {result.recommended_governance_bias}"
            )

    write_outputs(rows, query_mode=query_mode, source=source, total_available=len(all_queries))

    type_counts = group_counts(rows, "query_type", QUERY_TYPE_ORDER)
    bias_counts = group_counts(rows, "recommended_governance_bias", BIAS_ORDER)

    print("\nDistribution by query_type")
    print("-" * 100)
    for row in type_counts:
        print(f"{row['query_type']}: {row['query_count']} ({row['query_share']})")

    print("\nDistribution by recommended_governance_bias")
    print("-" * 100)
    for row in bias_counts:
        print(f"{row['recommended_governance_bias']}: {row['query_count']} ({row['query_share']})")

    print("\nFiles written under outputs/query_understanding/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MVP 22 Query Understanding Agent")
    parser.add_argument("--query", default="", help="Single query to classify.")
    parser.add_argument("--sample-size", type=int, default=100, help="Batch sample size.")
    parser.add_argument(
        "--query-mode",
        choices=["smoke", "esci", "stratified_esci"],
        default="smoke",
        help="Batch query source mode.",
    )
    parser.add_argument("--start-index", type=int, default=0, help="Start offset for batch selection.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.query:
        run_single_query(args.query)
        return

    if args.sample_size <= 0:
        raise ValueError("--sample-size must be positive.")
    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative.")

    run_batch(
        sample_size=args.sample_size,
        query_mode=args.query_mode,
        start_index=args.start_index,
    )


if __name__ == "__main__":
    main()
