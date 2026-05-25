"""
MVP 17: CORTEX Governance Agent

This module adds a top-level governance decision layer for the CORTEX / PolicyRank-RL project.

Purpose:
- Decide whether a query should use baseline-only ranking, mission building, repair, strict repair,
  behavior-aware reranking, critic review, or repair rejection.
- Avoid blindly running every CORTEX module on every query.
- Provide a traceable, explainable route decision for each query.

The Governance Agent uses existing MVP 15 and MVP 16 outputs:
- Mission strict repair summary
- Mission strict repaired slate
- Mission strict repair rejections
- Mission strict repair candidates
- Behavior-aware CORTEX summary
- Behavior-aware CORTEX slate
- Behavior-aware policy reasons

This is not a CTR model. It does not fabricate engagement.
It uses query structure, mission/repair signals, behavior-aware policy signals, and guardrail outcomes.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple


OUTPUT_DIR = Path("outputs")

STRICT_SUMMARY_PATH = OUTPUT_DIR / "mission_strict_repair_summary.csv"
STRICT_SLATE_PATH = OUTPUT_DIR / "mission_strict_repair_slate.csv"
STRICT_REJECTIONS_PATH = OUTPUT_DIR / "mission_strict_repair_rejections.csv"
STRICT_CANDIDATES_PATH = OUTPUT_DIR / "mission_strict_repair_candidates.csv"

BEHAVIOR_SUMMARY_PATH = OUTPUT_DIR / "behavior_aware_cortex_summary.csv"
BEHAVIOR_SLATE_PATH = OUTPUT_DIR / "behavior_aware_cortex_slate.csv"
BEHAVIOR_REASONS_PATH = OUTPUT_DIR / "behavior_aware_cortex_policy_reasons.csv"

GOVERNANCE_DECISIONS_PATH = OUTPUT_DIR / "cortex_governance_decisions.csv"
GOVERNANCE_SUMMARY_PATH = OUTPUT_DIR / "cortex_governance_summary.csv"
GOVERNANCE_TRACE_PATH = OUTPUT_DIR / "cortex_governance_trace.csv"


BRAND_TERMS = {
    "adidas",
    "nike",
    "puma",
    "reebok",
    "new balance",
    "under armour",
    "apple",
    "samsung",
    "sony",
    "lg",
    "dyson",
    "keurig",
    "ninja",
    "instant pot",
    "lego",
    "barbie",
    "oakley",
    "tommy bahama",
}


MISSION_TERMS = {
    "setup",
    "essentials",
    "packing list",
    "starter kit",
    "kit",
    "bundle",
    "party",
    "watch party",
    "vacation",
    "trip",
    "camping",
    "apartment",
    "kitchen setup",
    "dorm",
    "moving",
    "new home",
    "registry",
    "wedding",
    "birthday",
    "bbq",
    "tailgate",
    "world cup",
    "beach",
}


NARROW_PRODUCT_TERMS = {
    "cleats",
    "shoes",
    "sneakers",
    "shirt",
    "hoodie",
    "jacket",
    "laptop",
    "phone",
    "headphones",
    "monitor",
    "tv",
    "keyboard",
    "mouse",
    "bottle",
    "charger",
    "case",
}


ROUTE_BASELINE_ONLY = "BASELINE_ONLY"
ROUTE_MISSION_BUILD = "MISSION_BUILD"
ROUTE_MISSION_REPAIR = "MISSION_REPAIR"
ROUTE_STRICT_REPAIR = "STRICT_REPAIR"
ROUTE_BEHAVIOR_AWARE_RERANK = "BEHAVIOR_AWARE_RERANK"
ROUTE_CRITIC_REVIEW = "CRITIC_REVIEW"
ROUTE_REJECT_REPAIR_NARROW_QUERY = "REJECT_REPAIR_NARROW_QUERY"

DECISION_BASELINE_ONLY = "USE_BASELINE_ONLY"
DECISION_MISSION_AND_BEHAVIOR = "RUN_MISSION_REPAIR_AND_BEHAVIOR_AWARE"
DECISION_STRICT_AND_BEHAVIOR = "RUN_STRICT_REPAIR_AND_BEHAVIOR_AWARE"
DECISION_CRITIC_REVIEW = "SEND_TO_CRITIC_REVIEW"
DECISION_REJECT_REPAIR = "REJECT_REPAIR_USE_BASELINE"


@dataclass
class GovernanceSignals:
    query: str
    query_token_count: int
    query_type: str
    brand_specificity_score: float
    compound_intent_score: float
    mission_likelihood_score: float
    narrow_query_score: float
    strict_input_quality_slate_size: int
    strict_input_repair_rows: int
    strict_accepted_repair_rows: int
    strict_rejected_repair_rows: int
    strict_final_slate_size: int
    behavior_final_slate_size: int
    behavior_high_confidence_items: int
    behavior_medium_confidence_items: int
    behavior_low_confidence_items: int
    behavior_cold_start_rescue_items: int
    behavior_exploration_items: int
    behavior_unique_sub_intents: int
    coverage_gap_score: float
    repair_risk_score: float
    behavior_rescue_signal: float
    critic_need_score: float


@dataclass
class GovernanceDecision:
    query: str
    query_type: str
    mission_likelihood_score: float
    brand_specificity_score: float
    compound_intent_score: float
    narrow_query_score: float
    repair_risk_score: float
    coverage_gap_score: float
    behavior_rescue_signal: float
    critic_need_score: float
    recommended_route: str
    allowed_modules: str
    blocked_modules: str
    governance_decision: str
    plain_english_reason: str


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def clean_text(value: object) -> str:
    return str(value or "").strip()


def lower_text(value: object) -> str:
    return clean_text(value).lower()


def safe_int(value: object, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


def tokenize(query: str) -> List[str]:
    return re.findall(r"[a-z0-9]+", query.lower())


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []

    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    ensure_output_dir()

    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def filter_rows_by_query(rows: List[Dict[str, str]], query: str) -> List[Dict[str, str]]:
    if not rows:
        return []

    if "query" not in rows[0]:
        return rows

    query_lower = query.lower()
    return [row for row in rows if lower_text(row.get("query")) == query_lower]


def latest_query_row(path: Path, query: str) -> Dict[str, str]:
    rows = filter_rows_by_query(read_csv(path), query)
    if not rows:
        return {}
    return rows[-1]


def run_dependency(module_name: str, query: str) -> Tuple[bool, str]:
    cmd = [
        sys.executable,
        "-u",
        "-m",
        module_name,
        "--query",
        query,
    ]

    result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
    )

    combined_output = ""

    if result.stdout:
        combined_output += result.stdout

    if result.stderr:
        combined_output += "\n\nSTDERR:\n" + result.stderr

    return result.returncode == 0, combined_output


def refresh_pipeline_outputs(query: str, skip_refresh: bool = False) -> Dict[str, str]:
    """
    Runs the latest strict repair and behavior-aware modules before governance scoring.

    src.behavior_aware_cortex already refreshes strict repair internally, but we call it directly
    so this Governance Agent always has the latest MVP 15.9 and MVP 16 outputs.
    """
    outputs = {
        "strict_repair_ok": "skipped",
        "strict_repair_output": "",
        "behavior_aware_ok": "skipped",
        "behavior_aware_output": "",
    }

    if skip_refresh:
        return outputs

    behavior_ok, behavior_output = run_dependency("src.behavior_aware_cortex", query)

    outputs["behavior_aware_ok"] = str(behavior_ok)
    outputs["behavior_aware_output"] = behavior_output

    # behavior_aware_cortex internally invokes strict repair, so this reflects both stages.
    outputs["strict_repair_ok"] = str(behavior_ok)
    outputs["strict_repair_output"] = "Refreshed through src.behavior_aware_cortex."

    if not behavior_ok:
        raise RuntimeError(f"Behavior-aware dependency failed for query: {query}\n{behavior_output}")

    return outputs


def brand_specificity_score(query: str) -> float:
    query_lower = query.lower()
    tokens = tokenize(query)

    score = 0.0

    for brand in BRAND_TERMS:
        if brand in query_lower:
            score += 0.65
            break

    if len(tokens) <= 3:
        score += 0.20

    if any(term in query_lower for term in NARROW_PRODUCT_TERMS):
        score += 0.15

    return round(clamp(score), 4)


def compound_intent_score(query: str) -> float:
    query_lower = query.lower()
    tokens = tokenize(query)

    score = 0.0

    if len(tokens) >= 4:
        score += 0.20

    if len(tokens) >= 6:
        score += 0.15

    for term in MISSION_TERMS:
        if term in query_lower:
            score += 0.22

    # Cap phrase accumulation so broad mission queries are high, not always maxed.
    return round(clamp(score), 4)


def mission_likelihood_score(query: str) -> float:
    query_lower = query.lower()
    tokens = tokenize(query)

    score = 0.0

    for term in MISSION_TERMS:
        if term in query_lower:
            score += 0.20

    if any(term in query_lower for term in ["setup", "essentials", "packing list", "watch party", "trip"]):
        score += 0.30

    if len(tokens) >= 4:
        score += 0.15

    if brand_specificity_score(query) >= 0.65:
        score -= 0.35

    return round(clamp(score), 4)


def narrow_query_score(query: str) -> float:
    query_lower = query.lower()
    tokens = tokenize(query)

    score = 0.0

    if len(tokens) <= 3:
        score += 0.30

    if any(term in query_lower for term in NARROW_PRODUCT_TERMS):
        score += 0.30

    if brand_specificity_score(query) >= 0.65:
        score += 0.35

    if mission_likelihood_score(query) >= 0.65:
        score -= 0.35

    return round(clamp(score), 4)


def infer_query_type(query: str) -> str:
    brand_score = brand_specificity_score(query)
    mission_score = mission_likelihood_score(query)
    compound_score = compound_intent_score(query)
    narrow_score = narrow_query_score(query)

    if brand_score >= 0.65 and narrow_score >= 0.60:
        return "narrow_branded_product_query"

    if mission_score >= 0.75 and compound_score >= 0.60:
        return "compound_mission_query"

    if mission_score >= 0.55:
        return "mission_like_query"

    if compound_score >= 0.45:
        return "broad_multi_intent_query"

    return "standard_product_query"


def compute_coverage_gap_score(
    strict_summary: Dict[str, str],
    behavior_summary: Dict[str, str],
) -> float:
    strict_size = safe_int(strict_summary.get("final_strict_slate_size"))
    unique_sub_intents = safe_int(behavior_summary.get("unique_sub_intents"))

    if strict_size <= 0:
        return 1.0

    # A good mission slate should usually have at least 4-6 distinct sub-intents.
    if unique_sub_intents >= 6:
        return 0.05

    if unique_sub_intents >= 4:
        return 0.25

    if unique_sub_intents >= 2:
        return 0.55

    return 0.85


def compute_repair_risk_score(
    strict_summary: Dict[str, str],
    query: str,
) -> float:
    input_repair_rows = safe_int(strict_summary.get("input_repair_rows"))
    accepted = safe_int(strict_summary.get("strict_accepted_repair_rows"))
    rejected = safe_int(strict_summary.get("strict_rejected_repair_rows"))
    final_size = safe_int(strict_summary.get("final_strict_slate_size"))

    score = 0.0

    if input_repair_rows > 0:
        rejection_rate = rejected / max(input_repair_rows, 1)
        score += 0.45 * rejection_rate

    if final_size == 0:
        score += 0.25

    if narrow_query_score(query) >= 0.65:
        score += 0.30

    if accepted == 0 and input_repair_rows > 0:
        score += 0.15

    return round(clamp(score), 4)


def compute_behavior_rescue_signal(behavior_summary: Dict[str, str]) -> float:
    final_size = safe_int(behavior_summary.get("final_behavior_slate_size"))
    rescue_items = safe_int(behavior_summary.get("cold_start_rescue_items"))
    exploration_items = safe_int(behavior_summary.get("exploration_items"))

    if final_size <= 0:
        return 0.0

    score = 0.70 * (rescue_items / max(final_size, 1)) + 0.30 * (exploration_items / max(final_size, 1))
    return round(clamp(score), 4)


def compute_critic_need_score(
    coverage_gap_score: float,
    repair_risk_score: float,
    strict_summary: Dict[str, str],
    behavior_summary: Dict[str, str],
) -> float:
    rejected = safe_int(strict_summary.get("strict_rejected_repair_rows"))
    low_conf = safe_int(behavior_summary.get("low_confidence_items"))
    final_size = safe_int(behavior_summary.get("final_behavior_slate_size"))

    score = 0.0
    score += 0.40 * coverage_gap_score
    score += 0.35 * repair_risk_score

    if rejected > 0:
        score += 0.12

    if final_size > 0:
        score += 0.13 * min(1.0, low_conf / max(final_size, 1))

    return round(clamp(score), 4)


def build_governance_signals(query: str) -> GovernanceSignals:
    strict_summary = latest_query_row(STRICT_SUMMARY_PATH, query)
    behavior_summary = latest_query_row(BEHAVIOR_SUMMARY_PATH, query)

    coverage_gap = compute_coverage_gap_score(strict_summary, behavior_summary)
    repair_risk = compute_repair_risk_score(strict_summary, query)
    rescue_signal = compute_behavior_rescue_signal(behavior_summary)
    critic_need = compute_critic_need_score(
        coverage_gap,
        repair_risk,
        strict_summary,
        behavior_summary,
    )

    return GovernanceSignals(
        query=query,
        query_token_count=len(tokenize(query)),
        query_type=infer_query_type(query),
        brand_specificity_score=brand_specificity_score(query),
        compound_intent_score=compound_intent_score(query),
        mission_likelihood_score=mission_likelihood_score(query),
        narrow_query_score=narrow_query_score(query),
        strict_input_quality_slate_size=safe_int(strict_summary.get("input_quality_slate_size")),
        strict_input_repair_rows=safe_int(strict_summary.get("input_repair_rows")),
        strict_accepted_repair_rows=safe_int(strict_summary.get("strict_accepted_repair_rows")),
        strict_rejected_repair_rows=safe_int(strict_summary.get("strict_rejected_repair_rows")),
        strict_final_slate_size=safe_int(strict_summary.get("final_strict_slate_size")),
        behavior_final_slate_size=safe_int(behavior_summary.get("final_behavior_slate_size")),
        behavior_high_confidence_items=safe_int(behavior_summary.get("high_confidence_items")),
        behavior_medium_confidence_items=safe_int(behavior_summary.get("medium_confidence_items")),
        behavior_low_confidence_items=safe_int(behavior_summary.get("low_confidence_items")),
        behavior_cold_start_rescue_items=safe_int(behavior_summary.get("cold_start_rescue_items")),
        behavior_exploration_items=safe_int(behavior_summary.get("exploration_items")),
        behavior_unique_sub_intents=safe_int(behavior_summary.get("unique_sub_intents")),
        coverage_gap_score=coverage_gap,
        repair_risk_score=repair_risk,
        behavior_rescue_signal=rescue_signal,
        critic_need_score=critic_need,
    )


def allowed_and_blocked_modules_for_decision(decision: str) -> Tuple[List[str], List[str]]:
    if decision == DECISION_REJECT_REPAIR:
        return (
            [
                "Baseline Retrieval",
                "Contract Filtering",
                "Baseline Preservation Gate",
            ],
            [
                "Mission Repair Loop",
                "Strict Repair Expansion",
                "Behavior-Aware Rerank",
            ],
        )

    if decision == DECISION_BASELINE_ONLY:
        return (
            [
                "Baseline Retrieval",
                "Contract Filtering",
                "Baseline Preservation Gate",
            ],
            [
                "Mission Repair Loop",
                "Behavior-Aware Rerank",
            ],
        )

    if decision == DECISION_CRITIC_REVIEW:
        return (
            [
                "Mission Agent",
                "Mission Slate Builder",
                "Mission Coverage Analyzer",
                "Critic Agent",
                "Strict Repair Rules",
                "Behavior-Aware CORTEX",
            ],
            [
                "Blind Repair Without Critic",
            ],
        )

    if decision == DECISION_STRICT_AND_BEHAVIOR:
        return (
            [
                "Mission Agent",
                "Mission Slate Builder",
                "Repair Quality Guardrails",
                "Strict Repair Rules",
                "Behavior-Aware CORTEX",
            ],
            [
                "Baseline-Only Route",
                "Uncontrolled Repair",
            ],
        )

    return (
        [
            "Mission Agent",
            "Mission Slate Builder",
            "Mission Coverage Analyzer",
            "Mission Repair Loop",
            "Strict Repair Rules",
            "Behavior-Aware CORTEX",
        ],
        [
            "Baseline-Only Route",
        ],
    )


def choose_route_and_decision(signals: GovernanceSignals) -> Tuple[str, str]:
    if signals.query_type == "narrow_branded_product_query":
        return ROUTE_REJECT_REPAIR_NARROW_QUERY, DECISION_REJECT_REPAIR

    if signals.behavior_final_slate_size == 0 and signals.narrow_query_score >= 0.50:
        return ROUTE_BASELINE_ONLY, DECISION_BASELINE_ONLY

    if signals.critic_need_score >= 0.70:
        return ROUTE_CRITIC_REVIEW, DECISION_CRITIC_REVIEW

    if signals.strict_accepted_repair_rows > 0 and signals.strict_rejected_repair_rows > 0:
        return ROUTE_STRICT_REPAIR, DECISION_STRICT_AND_BEHAVIOR

    if signals.mission_likelihood_score >= 0.65 and signals.behavior_final_slate_size > 0:
        return ROUTE_BEHAVIOR_AWARE_RERANK, DECISION_MISSION_AND_BEHAVIOR

    if signals.compound_intent_score >= 0.50:
        return ROUTE_MISSION_REPAIR, DECISION_MISSION_AND_BEHAVIOR

    return ROUTE_BASELINE_ONLY, DECISION_BASELINE_ONLY


def build_plain_english_reason(signals: GovernanceSignals, route: str, decision: str) -> str:
    if decision == DECISION_REJECT_REPAIR:
        return (
            f"'{signals.query}' appears to be a narrow branded/product-specific query. "
            f"The governance agent blocks mission repair to avoid over-expanding the slate and recommends baseline-style handling."
        )

    if decision == DECISION_BASELINE_ONLY:
        return (
            f"'{signals.query}' does not show enough compound mission evidence for repair. "
            f"The governance agent recommends preserving the baseline route."
        )

    if decision == DECISION_CRITIC_REVIEW:
        return (
            f"'{signals.query}' has elevated governance risk because coverage gap or repair risk is high. "
            f"The governance agent recommends critic review before trusting aggressive repair."
        )

    if decision == DECISION_STRICT_AND_BEHAVIOR:
        return (
            f"'{signals.query}' is mission-like and repair was useful, but some repaired candidates were rejected. "
            f"The governance agent recommends strict repair plus behavior-aware reranking with guardrails."
        )

    return (
        f"'{signals.query}' is a mission-like or compound-intent query. "
        f"The governance agent recommends mission repair and behavior-aware reranking to preserve coverage, "
        f"rescue useful low-evidence items, and avoid click-greedy ranking."
    )


def build_governance_decision(signals: GovernanceSignals) -> GovernanceDecision:
    route, decision = choose_route_and_decision(signals)
    allowed, blocked = allowed_and_blocked_modules_for_decision(decision)

    return GovernanceDecision(
        query=signals.query,
        query_type=signals.query_type,
        mission_likelihood_score=signals.mission_likelihood_score,
        brand_specificity_score=signals.brand_specificity_score,
        compound_intent_score=signals.compound_intent_score,
        narrow_query_score=signals.narrow_query_score,
        repair_risk_score=signals.repair_risk_score,
        coverage_gap_score=signals.coverage_gap_score,
        behavior_rescue_signal=signals.behavior_rescue_signal,
        critic_need_score=signals.critic_need_score,
        recommended_route=route,
        allowed_modules=json.dumps(allowed),
        blocked_modules=json.dumps(blocked),
        governance_decision=decision,
        plain_english_reason=build_plain_english_reason(signals, route, decision),
    )


def print_governance_result(signals: GovernanceSignals, decision: GovernanceDecision) -> None:
    print("\nCORTEX Governance Agent Decision")
    print("-" * 100)
    print(f"Query: {decision.query}")
    print(f"Query type: {decision.query_type}")
    print(f"Recommended route: {decision.recommended_route}")
    print(f"Governance decision: {decision.governance_decision}")

    print("\nKey Signals")
    print("-" * 100)
    print(f"Mission likelihood: {decision.mission_likelihood_score}")
    print(f"Compound intent: {decision.compound_intent_score}")
    print(f"Brand specificity: {decision.brand_specificity_score}")
    print(f"Narrow query score: {decision.narrow_query_score}")
    print(f"Repair risk: {decision.repair_risk_score}")
    print(f"Coverage gap: {decision.coverage_gap_score}")
    print(f"Behavior rescue signal: {decision.behavior_rescue_signal}")
    print(f"Critic need: {decision.critic_need_score}")

    print("\nAllowed Modules")
    print("-" * 100)
    for module in json.loads(decision.allowed_modules):
        print(f"- {module}")

    print("\nBlocked Modules")
    print("-" * 100)
    for module in json.loads(decision.blocked_modules):
        print(f"- {module}")

    print("\nExplanation")
    print("-" * 100)
    print(decision.plain_english_reason)


def append_or_replace_by_query(
    path: Path,
    new_row: Dict[str, object],
    fieldnames: List[str],
    query: str,
) -> None:
    existing = read_csv(path)
    existing = [row for row in existing if lower_text(row.get("query")) != query.lower()]
    write_csv(path, existing + [new_row], fieldnames)


def write_governance_outputs(
    signals: GovernanceSignals,
    decision: GovernanceDecision,
    dependency_trace: Dict[str, str],
) -> None:
    decision_row = asdict(decision)
    signal_row = asdict(signals)

    trace_row = {
        "query": signals.query,
        "strict_repair_ok": dependency_trace.get("strict_repair_ok", ""),
        "behavior_aware_ok": dependency_trace.get("behavior_aware_ok", ""),
        "strict_repair_output_preview": dependency_trace.get("strict_repair_output", "")[:1000],
        "behavior_aware_output_preview": dependency_trace.get("behavior_aware_output", "")[:1000],
        "recommended_route": decision.recommended_route,
        "governance_decision": decision.governance_decision,
    }

    summary_row = {
        "query": signals.query,
        "query_type": signals.query_type,
        "recommended_route": decision.recommended_route,
        "governance_decision": decision.governance_decision,
        "mission_likelihood_score": signals.mission_likelihood_score,
        "compound_intent_score": signals.compound_intent_score,
        "brand_specificity_score": signals.brand_specificity_score,
        "narrow_query_score": signals.narrow_query_score,
        "repair_risk_score": signals.repair_risk_score,
        "coverage_gap_score": signals.coverage_gap_score,
        "behavior_rescue_signal": signals.behavior_rescue_signal,
        "critic_need_score": signals.critic_need_score,
        "strict_final_slate_size": signals.strict_final_slate_size,
        "behavior_final_slate_size": signals.behavior_final_slate_size,
        "behavior_cold_start_rescue_items": signals.behavior_cold_start_rescue_items,
        "plain_english_reason": decision.plain_english_reason,
    }

    append_or_replace_by_query(
        GOVERNANCE_DECISIONS_PATH,
        decision_row,
        list(decision_row.keys()),
        signals.query,
    )

    append_or_replace_by_query(
        GOVERNANCE_SUMMARY_PATH,
        summary_row,
        list(summary_row.keys()),
        signals.query,
    )

    append_or_replace_by_query(
        GOVERNANCE_TRACE_PATH,
        trace_row,
        list(trace_row.keys()),
        signals.query,
    )


def run_governance(query: str, skip_refresh: bool = False) -> Tuple[GovernanceSignals, GovernanceDecision]:
    ensure_output_dir()

    dependency_trace = refresh_pipeline_outputs(query, skip_refresh=skip_refresh)
    signals = build_governance_signals(query)
    decision = build_governance_decision(signals)

    write_governance_outputs(signals, decision, dependency_trace)

    return signals, decision


def main() -> None:
    parser = argparse.ArgumentParser(description="MVP 17 CORTEX Governance Agent")
    parser.add_argument("--query", required=True, help="Shopping query to govern.")
    parser.add_argument(
        "--skip-refresh",
        action="store_true",
        help="Use existing MVP 15.9 and MVP 16 output files without rerunning dependencies.",
    )

    args = parser.parse_args()

    query = args.query.strip()

    signals, decision = run_governance(query=query, skip_refresh=args.skip_refresh)

    print_governance_result(signals, decision)

    print("\nFiles written:")
    print(f"- {GOVERNANCE_DECISIONS_PATH}")
    print(f"- {GOVERNANCE_SUMMARY_PATH}")
    print(f"- {GOVERNANCE_TRACE_PATH}")


if __name__ == "__main__":
    main()