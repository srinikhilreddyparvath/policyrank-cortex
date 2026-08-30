from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd

PILOT_VERSION = "contract_esci_pilot_v1.0.0"
SCHEMA_VERSION = "contract_esci_annotation_v1.0.0"
SELECTION_SEED = 29
PILOT_SIZE = 250

# Quotas are mutually exclusive because classify_query returns the first match.
STRATUM_QUOTAS = {
    "explicit_negation": 30,
    "price_or_value": 25,
    "brand_constraint": 25,
    "must_have_attribute": 35,
    "multi_attribute": 30,
    "soft_preference": 25,
    "ambiguous_natural_language": 20,
    "long_query": 25,
    "negative_control": 35,
}

FORBIDDEN_ANNOTATION_FIELDS = {
    "esci_label", "method", "method_name", "route", "route_selected",
    "oracle", "oracle_route", "route_regret", "ndcg", "mrr", "exact",
    "cortex_wins", "ranking_outcome",
}

NEGATION = re.compile(r"\b(no|not|without|excluding|except|minus|free[- ]of|sin)\b", re.I)
PRICE = re.compile(r"\b(cheap|cheaper|cheapest|affordable|budget|value|under|below|less than|low price|inexpensive)\b|\$\s*\d", re.I)
MUST_HAVE = re.compile(r"\b(with|including|include|has|have|featuring|made of|compatible with|for)\b", re.I)
SOFT = re.compile(r"\b(prefer|preferred|ideally|nice|comfortable|stylish|best|better|lightweight|premium)\b", re.I)
AMBIGUOUS = re.compile(r"\b(good|nice|best|thing|stuff|gift|normal|quality)\b", re.I)
ATTRIBUTES = re.compile(r"\b(red|blue|black|white|green|small|medium|large|xl|inch|inches|mm|cm|cotton|metal|wood|plastic|wireless|waterproof|rechargeable|organic|unscented|men|women|kids)\b", re.I)
BRANDS = re.compile(r"\b(nike|adidas|apple|samsung|sony|lego|disney|dell|hp|lenovo|canon|keurig|dewalt|milwaukee|stanley|kitchenaid|nintendo|xbox|playstation|pampers|gillette|colgate)\b", re.I)


def stable_order_key(query_id: int, seed: int = SELECTION_SEED) -> str:
    return hashlib.sha256(f"{seed}:{int(query_id)}".encode()).hexdigest()


def classify_query(query: str) -> str:
    text = " ".join(str(query).lower().split())
    tokens = re.findall(r"[a-z0-9]+", text)
    attribute_hits = len(set(match.group(0).lower() for match in ATTRIBUTES.finditer(text)))
    connector_count = len(re.findall(r"\b(and|with|for|without|under|in|made of)\b", text))
    if NEGATION.search(text): return "explicit_negation"
    if PRICE.search(text): return "price_or_value"
    if BRANDS.search(text): return "brand_constraint"
    if MUST_HAVE.search(text) and attribute_hits: return "must_have_attribute"
    if attribute_hits >= 2 or connector_count >= 2: return "multi_attribute"
    if SOFT.search(text): return "soft_preference"
    if AMBIGUOUS.search(text) or len(tokens) <= 1: return "ambiguous_natural_language"
    if len(tokens) >= 7 or len(text) >= 48: return "long_query"
    return "negative_control"


def select_pilot_queries(queries: pd.DataFrame, *, seed: int = SELECTION_SEED) -> pd.DataFrame:
    required = {"query_id", "query_text", "source_partition"}
    missing = required - set(queries.columns)
    if missing: raise ValueError(f"Missing pilot query columns: {sorted(missing)}")
    if queries.query_id.duplicated().any(): raise ValueError("Input query IDs must be unique")
    if not set(queries.source_partition).issubset({"policy_train", "calibration"}):
        raise ValueError("Pilot may use only policy_train and calibration")
    work = queries.copy()
    work["selection_stratum"] = work.query_text.map(classify_query)
    work["_order"] = work.query_id.map(lambda value: stable_order_key(int(value), seed))
    selected = []
    for stratum, quota in STRATUM_QUOTAS.items():
        candidates = work[work.selection_stratum == stratum].sort_values("_order")
        if len(candidates) < quota: raise ValueError(f"Insufficient {stratum} candidates: {len(candidates)} < {quota}")
        selected.append(candidates.head(quota))
    result = pd.concat(selected).sort_values("query_id").drop(columns="_order").reset_index(drop=True)
    if len(result) != PILOT_SIZE or result.query_id.nunique() != PILOT_SIZE: raise AssertionError("Pilot size/uniqueness failure")
    return result


def assert_no_esci_test_queries(selected_ids: list[int], examples: pd.DataFrame) -> None:
    test_ids=set(examples.loc[examples["split"]=="test","query_id"].astype(int)); overlap=set(map(int,selected_ids))&test_ids
    if overlap: raise ValueError(f"ContractESCI pilot contains ESCI test query IDs: {sorted(overlap)[:10]}")


def validate_requirement(requirement: dict[str, Any]) -> None:
    required = {"requirement_id", "type", "attribute", "operator", "value", "strength", "evidence_text"}
    missing = required - set(requirement)
    if missing: raise ValueError(f"Missing requirement fields: {sorted(missing)}")
    if requirement["type"] not in {"inclusion", "exclusion", "comparison", "preference"}: raise ValueError("Invalid requirement type")
    if requirement["operator"] not in {"equal", "not_equal", "contains", "not_contains", "at_most", "at_least", "lower_preferred", "higher_preferred", "compatible_with"}: raise ValueError("Invalid operator")
    if requirement["strength"] not in {"hard", "soft"}: raise ValueError("Invalid strength")
    if not str(requirement["evidence_text"]).strip(): raise ValueError("Requirement evidence_text is required")


def validate_compliance_state(state: str) -> None:
    if state not in {"satisfied", "violated", "unknown", "not_applicable"}: raise ValueError("Invalid compliance state")


def assert_blind_artifact(record: dict[str, Any]) -> None:
    overlap = FORBIDDEN_ANNOTATION_FIELDS & set(record)
    if overlap: raise ValueError(f"Forbidden evaluation fields in annotation artifact: {sorted(overlap)}")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
