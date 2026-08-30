from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional

from src.paper_eval.adapter import FORBIDDEN_RANKER_COLUMNS, CandidateBoundaryError

PARSER_VERSION = "search_contract_v1.0.0"
_STOP = frozenset({"a", "an", "and", "for", "in", "of", "on", "the", "to", "with"})
_NEGATORS = frozenset({"without", "excluding", "exclude", "except", "no", "not"})
_PRICE = frozenset({"budget", "cheap", "affordable", "premium", "luxury", "under", "below", "over"})
_BRANDS = frozenset({"adidas", "apple", "nike", "samsung", "sony", "lego", "puma", "reebok"})


@dataclass(frozen=True)
class SearchContractV1:
    query_text: str
    normalized_query: str
    product_type: Optional[str]
    positive_terms: tuple[str, ...]
    negative_terms: tuple[str, ...]
    must_have: tuple[str, ...]
    must_not_have: tuple[str, ...]
    brand_signal: Optional[str]
    price_signal: Optional[str]
    constraint_strength: str
    ambiguity: str
    contract_status: str
    parser_version: str
    parser_route: str
    parser_reason: str
    fallback_used: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_metadata(metadata: Optional[Mapping[str, Any]]) -> None:
    if metadata is None:
        return
    forbidden = FORBIDDEN_RANKER_COLUMNS.intersection(metadata.keys())
    if forbidden:
        raise CandidateBoundaryError(f"Evaluator judgments reached SearchContractV1: {sorted(forbidden)}")


def build_search_contract_v1(query_text: str, metadata: Optional[Mapping[str, Any]] = None) -> SearchContractV1:
    """Build a deterministic contract using query text and label-free metadata only."""
    _validate_metadata(metadata)
    normalized = " ".join(re.findall(r"[a-z0-9]+(?:[.'-][a-z0-9]+)?", str(query_text).lower()))
    tokens = normalized.split()
    positive: list[str] = []
    negative: list[str] = []
    negating = False
    for token in tokens:
        if token in _NEGATORS:
            negating = True
            continue
        if token in _STOP:
            continue
        target = negative if negating else positive
        if token not in target:
            target.append(token)
        if negating:
            negating = False
    brands = [token for token in positive if token in _BRANDS]
    prices = [token for token in tokens if token in _PRICE or re.fullmatch(r"\$?\d+(?:\.\d+)?", token)]
    product_tokens = [token for token in positive if token not in _BRANDS and token not in _PRICE and not token.isdigit()]
    product_type = " ".join(product_tokens[-2:]) if product_tokens else None
    if not normalized or not (positive or negative):
        status, ambiguity, reason = "unresolved", "high", "no_meaningful_query_terms"
    elif product_type and negative:
        status, ambiguity, reason = "resolved", "low", "product_type_and_explicit_exclusion"
    elif product_type:
        status, ambiguity, reason = "resolved", "medium", "product_type_with_no_explicit_constraint"
    else:
        status, ambiguity, reason = "partial", "high", "signals_detected_without_product_type"
    return SearchContractV1(
        query_text=str(query_text), normalized_query=normalized, product_type=product_type,
        positive_terms=tuple(positive), negative_terms=tuple(negative), must_have=tuple(),
        must_not_have=tuple(negative), brand_signal=brands[0] if brands else None,
        price_signal=prices[0] if prices else None,
        constraint_strength="hard" if negative else ("soft" if prices or brands else "none"),
        ambiguity=ambiguity, contract_status=status, parser_version=PARSER_VERSION,
        parser_route="deterministic_lexical_v1", parser_reason=reason, fallback_used=False,
    )


def exceptional_fallback_contract(query_text: str, reason: str) -> SearchContractV1:
    """Exceptional failure representation; never the normal schema path."""
    return SearchContractV1(str(query_text), "", None, (), (), (), (), None, None, "unknown", "unknown",
                            "unresolved", PARSER_VERSION, "exception_fallback", reason, True)
