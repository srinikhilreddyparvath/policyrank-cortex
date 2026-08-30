from __future__ import annotations

import math
from typing import Iterable, Optional, Sequence

OFFICIAL_RELEVANCE_LEVELS = {"E": 4, "C": 3, "S": 2, "I": 1}
OFFICIAL_GAINS = {"E": 1.0, "C": 0.1, "S": 0.01, "I": 0.0}
CORTEX_NDCG_GAINS = {"E": 1.0, "S": 0.7, "C": 0.4, "I": 0.0}
CORTEX_REWARD_GAINS = CORTEX_NDCG_GAINS

def _labels(labels: Iterable[object]) -> list[str]:
    return [str(value).strip().upper() for value in labels]

def dcg(labels: Sequence[str], gains: dict[str, float], k: Optional[int] = None) -> float:
    values = labels if k is None else labels[:k]
    return sum(gains.get(label, 0.0) / math.log2(rank + 1) for rank, label in enumerate(values, 1))

def ndcg(labels: Iterable[object], gains: dict[str, float], k: Optional[int] = None, ideal_labels: Optional[Iterable[object]] = None) -> Optional[float]:
    observed = _labels(labels)
    ideal_source = _labels(ideal_labels) if ideal_labels is not None else observed
    if not ideal_source:
        return None
    ideal = sorted(ideal_source, key=lambda label: gains.get(label, 0.0), reverse=True)
    denominator = dcg(ideal, gains, k)
    return dcg(observed, gains, k) / denominator if denominator > 0 else 0.0

def official_esci_ndcg(labels: Iterable[object], ideal_labels: Optional[Iterable[object]] = None) -> Optional[float]:
    return ndcg(labels, OFFICIAL_GAINS, ideal_labels=ideal_labels)

def official_esci_ndcg_at_k(labels: Iterable[object], k: int, ideal_labels: Optional[Iterable[object]] = None) -> Optional[float]:
    if k < 1:
        raise ValueError("k must be positive")
    return ndcg(labels, OFFICIAL_GAINS, k, ideal_labels)

def cortex_ndcg_at_k(labels: Iterable[object], k: int, ideal_labels: Optional[Iterable[object]] = None) -> Optional[float]:
    return ndcg(labels, CORTEX_NDCG_GAINS, k, ideal_labels)

def reciprocal_rank_at_k(labels: Iterable[object], k: int = 10, relevant: frozenset[str] = frozenset({"E"})) -> Optional[float]:
    observed = _labels(labels)
    if not observed:
        return None
    for rank, label in enumerate(observed[:k], 1):
        if label in relevant:
            return 1.0 / rank
    return 0.0

def any_label_at_k(labels: Iterable[object], accepted: set[str], k: int) -> Optional[float]:
    observed = _labels(labels)
    if not observed:
        return None
    return float(any(label in accepted for label in observed[:k]))

def exact_at_k(labels: Iterable[object], k: int) -> Optional[float]: return any_label_at_k(labels, {"E"}, k)
def exact_or_substitute_at_k(labels: Iterable[object], k: int) -> Optional[float]: return any_label_at_k(labels, {"E", "S"}, k)
def irrelevant_at_k(labels: Iterable[object], k: int) -> Optional[float]: return any_label_at_k(labels, {"I"}, k)

def label_coverage_at_k(labels: Iterable[object], k: int) -> Optional[float]:
    observed = list(labels)[:k]
    if not observed:
        return None
    return sum(str(value).strip().upper() in OFFICIAL_GAINS for value in observed) / len(observed)

def cortex_reward(labels: Iterable[object], k: int = 5) -> Optional[float]:
    observed = _labels(labels)[:k]
    if not observed:
        return None
    weights = [1.0 / rank for rank in range(1, len(observed) + 1)]
    return sum(CORTEX_REWARD_GAINS.get(label, 0.0) * weight for label, weight in zip(observed, weights)) / sum(weights)

def safe_rate(numerator: Optional[float], denominator: Optional[float]) -> Optional[float]:
    if numerator is None or denominator is None or denominator <= 0: return None
    return numerator / denominator

def reward_delta_vs_baseline(method_reward: Optional[float], baseline_reward: Optional[float]) -> Optional[float]:
    return None if method_reward is None or baseline_reward is None else method_reward - baseline_reward

def route_regret(selected_reward: Optional[float], route_rewards: Optional[dict[str, Optional[float]]]) -> tuple[Optional[float], Optional[float], str]:
    if selected_reward is None: return None, None, "selected_route_reward_unavailable"
    if not route_rewards: return None, None, "all_route_outcomes_unavailable"
    if any(value is None for value in route_rewards.values()): return None, None, "incomplete_route_outcomes"
    oracle = max(float(value) for value in route_rewards.values() if value is not None)
    return oracle, oracle - selected_reward, ""

def aggregate_rate(rows: Iterable[Optional[bool]]) -> Optional[float]:
    values = [value for value in rows if value is not None]
    return None if not values else sum(bool(value) for value in values) / len(values)

def intervention_rate(values: Iterable[Optional[bool]]) -> Optional[float]: return aggregate_rate(values)
def preserve_rate(values: Iterable[Optional[bool]]) -> Optional[float]: return aggregate_rate(values)
def fallback_rate(values: Iterable[Optional[bool]]) -> Optional[float]: return aggregate_rate(values)

def beneficial_intervention_rate(intervened: Iterable[Optional[bool]], reward_deltas: Iterable[Optional[float]]) -> Optional[float]:
    values=[float(delta)>0 for flag,delta in zip(intervened,reward_deltas) if flag is True and delta is not None]
    return aggregate_rate(values)

def harmful_intervention_rate(intervened: Iterable[Optional[bool]], reward_deltas: Iterable[Optional[float]]) -> Optional[float]:
    values=[float(delta)<0 for flag,delta in zip(intervened,reward_deltas) if flag is True and delta is not None]
    return aggregate_rate(values)
def neutral_intervention_rate(intervened: Iterable[Optional[bool]], reward_deltas: Iterable[Optional[float]]) -> Optional[float]:
    values=[math.isclose(float(delta),0.0,abs_tol=1e-12) for flag,delta in zip(intervened,reward_deltas) if flag is True and delta is not None]
    return aggregate_rate(values)

def classify_intervention(intervened: Optional[bool], method_reward: Optional[float], baseline_reward: Optional[float]) -> Optional[str]:
    if intervened is not True or method_reward is None or baseline_reward is None: return None
    delta=method_reward-baseline_reward
    return "beneficial" if delta>0 else "harmful" if delta<0 else "neutral"

def violation_rate(violation_count: Optional[float], evaluated_count: Optional[float]) -> Optional[float]: return safe_rate(violation_count,evaluated_count)
def hard_violation_rate(violation_count: Optional[float], evaluated_count: Optional[float]) -> Optional[float]: return violation_rate(violation_count,evaluated_count)
def soft_violation_rate(violation_count: Optional[float], evaluated_count: Optional[float]) -> Optional[float]: return violation_rate(violation_count,evaluated_count)
def any_violation_rate(violation_count: Optional[float], evaluated_count: Optional[float]) -> Optional[float]: return violation_rate(violation_count,evaluated_count)
