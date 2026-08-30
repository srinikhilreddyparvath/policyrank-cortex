from __future__ import annotations
from dataclasses import asdict, dataclass, field
from typing import Optional

@dataclass
class QueryResult:
    run_id: str; method: str; method_version: str; query_id: int; query: str; locale: str; split: str
    candidate_count: int = 0; judged_candidate_count: int = 0; unjudged_candidate_count: int = 0
    E_count: int = 0; S_count: int = 0; C_count: int = 0; I_count: int = 0
    final_product_ids: list[str] = field(default_factory=list); final_labels: list[str] = field(default_factory=list); final_scores: list[float] = field(default_factory=list)
    official_esci_ndcg: Optional[float] = None; ndcg_5: Optional[float] = None; ndcg_10: Optional[float] = None; mrr_10: Optional[float] = None
    exact_1: Optional[float] = None; exact_5: Optional[float] = None; exact_10: Optional[float] = None
    exact_or_substitute_5: Optional[float] = None; exact_or_substitute_10: Optional[float] = None; irrelevant_5: Optional[float] = None; irrelevant_10: Optional[float] = None
    label_coverage_5: Optional[float] = None; label_coverage_10: Optional[float] = None
    route: Optional[str] = None; intervened: Optional[bool] = None; preserved: Optional[bool] = None; fallback: Optional[bool] = None
    route_selected: Optional[str] = None; ranking_changed: Optional[bool] = None; explicit_preserve: Optional[bool] = None
    intervention_attempted: Optional[bool] = None; intervention_effective: Optional[bool] = None
    contract_status: Optional[str] = None; detected_constraint_count: Optional[int] = None
    negative_constraint_count: Optional[int] = None; positive_constraint_count: Optional[int] = None
    hard_constraint_count: Optional[int] = None; soft_constraint_count: Optional[int] = None
    contract_ambiguity: Optional[str] = None; contract_parser_route: Optional[str] = None; contract_parser_reason: Optional[str] = None
    hard_compatibility_changes: Optional[int] = None; soft_preference_matches: Optional[int] = None
    must_have_matches: Optional[int] = None; must_not_have_violations: Optional[int] = None
    brand_matches: Optional[int] = None; price_signal_matches: Optional[int] = None; product_type_matches: Optional[int] = None
    contract_score_range: Optional[float] = None; rerank_changed: Optional[bool] = None
    mean_candidate_movement: Optional[float] = None; top_1_changed: Optional[bool] = None
    gate_decision: Optional[str] = None; gate_reason: Optional[str] = None; diagnostics_json: Optional[str] = None
    baseline_reward: Optional[float] = None; method_reward: Optional[float] = None; reward_delta: Optional[float] = None; oracle_reward: Optional[float] = None; route_regret: Optional[float] = None; route_regret_unavailable_reason: Optional[str] = None
    hard_violation_count: Optional[int] = None; soft_violation_count: Optional[int] = None; any_violation_count: Optional[int] = None; clean_count: Optional[int] = None; removed_count: Optional[int] = None; demoted_count: Optional[int] = None
    retrieval_latency_ms: Optional[float] = None; ranking_latency_ms: Optional[float] = None; total_latency_ms: Optional[float] = None
    success: bool = True; failure_stage: Optional[str] = None; error_type: Optional[str] = None; error_message: Optional[str] = None
    def to_dict(self): return asdict(self)
