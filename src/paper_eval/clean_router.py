from __future__ import annotations

from dataclasses import dataclass

from src.paper_eval.contracts_v1 import SearchContractV1

ROUTER_VERSION = "clean_contract_router_v1.0.0"
ROUTE_FAMILY_VERSION = "clean_candidate_routes_v1"
ALLOWED_ROUTES = frozenset({"PRESERVE", "STRICT_FILTER"})
ROUTER_V2_VERSION = "clean_contract_router_v2.0.0"
ROUTE_FAMILY_V2_VERSION = "clean_candidate_routes_v2"
ALLOWED_ROUTES_V2 = frozenset({"PRESERVE", "STRICT_FILTER", "CONTRACT_RERANK"})


@dataclass(frozen=True)
class RouterDecision:
    route: str
    reason_code: str
    router_version: str = ROUTER_VERSION


def route_contract_v1(contract: SearchContractV1) -> RouterDecision:
    if contract.constraint_strength == "hard" and contract.must_not_have:
        return RouterDecision("STRICT_FILTER", "explicit_hard_exclusion")
    return RouterDecision("PRESERVE", "no_supported_hard_exclusion")


def route_contract_v2(contract: SearchContractV1) -> RouterDecision:
    if contract.constraint_strength == "hard" and contract.must_not_have:
        return RouterDecision("STRICT_FILTER", "explicit_hard_exclusion", ROUTER_V2_VERSION)
    meaningful_soft = bool(contract.brand_signal or contract.price_signal or len(contract.positive_terms) >= 3)
    if contract.contract_status != "unresolved" and meaningful_soft:
        return RouterDecision("CONTRACT_RERANK", "meaningful_soft_or_multi_term_contract", ROUTER_V2_VERSION)
    return RouterDecision("PRESERVE", "simple_or_unresolved_contract", ROUTER_V2_VERSION)
