
"""
MVP 16: Behavior-Aware CORTEX

Adds behavior-aware policy scoring on top of the strict repaired mission slate.

Core ideas:
- Behavior score proxy
- Behavior confidence
- Mission stage
- Cold-start rescue
- Exploration flag
- Over-concentration penalty
- Policy reason codes

This does not train a real CTR model yet. It creates an explainable behavior-aware
ranking layer using mission context, slate source, item title signals, and proxy behavior.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple


OUTPUT_DIR = Path("outputs")

STRICT_SLATE_PATH = OUTPUT_DIR / "mission_strict_repair_slate.csv"

BEHAVIOR_SLATE_PATH = OUTPUT_DIR / "behavior_aware_cortex_slate.csv"
BEHAVIOR_SUMMARY_PATH = OUTPUT_DIR / "behavior_aware_cortex_summary.csv"
BEHAVIOR_REASONS_PATH = OUTPUT_DIR / "behavior_aware_cortex_policy_reasons.csv"


MISSION_STAGE_BY_ROLE = {
    "cooking": "core",
    "serveware": "core",
    "prep": "supporting",
    "storage": "completion",
    "cleaning": "completion",
    "protection": "core",
    "comfort": "supporting",
    "apparel": "core",
    "accessory": "supporting",
    "footwear": "core",
    "hydration": "supporting",
    "food storage": "completion",
    "sports": "core",
    "brand_exact": "core",
}


ROLE_BASE_BEHAVIOR = {
    "core": 0.82,
    "supporting": 0.68,
    "completion": 0.58,
    "optional": 0.42,
    "unknown": 0.50,
}


HIGH_INTENT_TITLE_TERMS = {
    "set": 0.06,
    "pack": 0.04,
    "kit": 0.06,
    "portable": 0.03,
    "waterproof": 0.05,
    "stainless": 0.03,
    "bpa": 0.03,
    "spf": 0.06,
    "beach": 0.05,
    "soccer": 0.08,
    "cleats": 0.10,
    "adidas": 0.10,
    "kitchen": 0.04,
    "dish": 0.04,
    "drying": 0.05,
    "sandals": 0.07,
    "sandal": 0.07,
    "flip": 0.05,
    "flop": 0.05,
}


LOW_QUALITY_TITLE_TERMS = {
    "tapestry": -0.10,
    "wall hanging": -0.10,
    "air conditioner": -0.12,
    "fan": -0.06,
    "craft knife": -0.05,
    "toiletry": -0.04,
}


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def run_strict_repair_if_needed(query: str) -> None:
    """
    Runs MVP 15.9 first so MVP 16 always has the latest strict slate for the query.
    """
    cmd = [
        sys.executable,
        "-m",
        "src.mission_strict_repair_rules",
        "--query",
        query,
    ]

    result = subprocess.run(
        cmd,
        check=False,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(result.stdout)
        print(result.stderr)
        raise RuntimeError(f"Strict repair rules failed for query: {query}")


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


def clean_text(value: object) -> str:
    return str(value or "").strip()


def lower_text(value: object) -> str:
    return clean_text(value).lower()


def parse_rank(row: Dict[str, str]) -> int:
    for key in ["final_strict_rank", "rank", "final_rank"]:
        value = row.get(key)
        if value is not None and str(value).strip().isdigit():
            return int(value)
    return 999


def get_query_rows(rows: List[Dict[str, str]], query: str) -> List[Dict[str, str]]:
    if not rows:
        return []

    if "query" not in rows[0]:
        return rows

    filtered = [r for r in rows if lower_text(r.get("query")) == query.lower()]
    return filtered


def infer_mission_stage(row: Dict[str, str]) -> str:
    role = lower_text(row.get("sub_intent_role"))
    sub_intent = lower_text(row.get("sub_intent"))

    if role in MISSION_STAGE_BY_ROLE:
        return MISSION_STAGE_BY_ROLE[role]

    if sub_intent in MISSION_STAGE_BY_ROLE:
        return MISSION_STAGE_BY_ROLE[sub_intent]

    title = lower_text(row.get("product_title"))

    if any(term in title for term in ["adidas", "soccer cleat", "cleats"]):
        return "core"

    if any(term in title for term in ["bag", "organizer", "container", "rack"]):
        return "completion"

    if any(term in title for term in ["knife", "board", "towel", "sunglasses"]):
        return "supporting"

    return "unknown"


def title_signal_score(title: str) -> float:
    score = 0.0
    text = title.lower()

    for term, boost in HIGH_INTENT_TITLE_TERMS.items():
        if term in text:
            score += boost

    for term, penalty in LOW_QUALITY_TITLE_TERMS.items():
        if term in text:
            score += penalty

    return max(-0.20, min(0.25, score))


def source_signal(row: Dict[str, str]) -> float:
    source = lower_text(row.get("slate_source"))

    if "repair_loop" in source:
        return 0.03

    if "original_guarded" in source:
        return 0.05

    return 0.0


def cold_start_proxy(row: Dict[str, str]) -> bool:
    """
    Proxy for cold-start behavior:
    repaired rows and lower-ranked mission-completion rows are treated as lower-evidence items.
    """
    source = lower_text(row.get("slate_source"))
    rank = parse_rank(row)
    stage = infer_mission_stage(row)

    if "repair_loop" in source:
        return True

    if rank >= 9 and stage in {"completion", "supporting"}:
        return True

    return False


def behavior_score(row: Dict[str, str]) -> float:
    stage = infer_mission_stage(row)
    base = ROLE_BASE_BEHAVIOR.get(stage, ROLE_BASE_BEHAVIOR["unknown"])
    title = clean_text(row.get("product_title"))

    score = base
    score += title_signal_score(title)
    score += source_signal(row)

    if cold_start_proxy(row):
        score -= 0.08

    return round(max(0.05, min(0.99, score)), 4)


def behavior_confidence(score: float, row: Dict[str, str]) -> str:
    is_cold = cold_start_proxy(row)

    if is_cold and score >= 0.55:
        return "low_behavior_but_mission_relevant"

    if score >= 0.78:
        return "high"

    if score >= 0.60:
        return "medium"

    return "low"


def coverage_contribution(row: Dict[str, str], sub_intent_counts: Counter) -> float:
    sub_intent = lower_text(row.get("sub_intent"))
    stage = infer_mission_stage(row)

    contribution = 0.0

    if sub_intent and sub_intent_counts[sub_intent] == 1:
        contribution += 0.10

    if stage == "core":
        contribution += 0.08
    elif stage == "supporting":
        contribution += 0.06
    elif stage == "completion":
        contribution += 0.05

    return round(contribution, 4)


def concentration_penalty(row: Dict[str, str], sub_intent_counts: Counter) -> float:
    sub_intent = lower_text(row.get("sub_intent"))

    if not sub_intent:
        return 0.0

    count = sub_intent_counts[sub_intent]

    if count <= 2:
        return 0.0

    return round(min(0.18, 0.04 * (count - 2)), 4)


def exploration_flag(row: Dict[str, str], score: float, coverage: float) -> bool:
    if not cold_start_proxy(row):
        return False

    stage = infer_mission_stage(row)

    return stage in {"core", "supporting", "completion"} and coverage >= 0.05 and score >= 0.45


def policy_reason(row: Dict[str, str], score: float, confidence: str, coverage: float, penalty: float, explore: bool) -> str:
    stage = infer_mission_stage(row)
    source = lower_text(row.get("slate_source"))
    title = lower_text(row.get("product_title"))

    if "adidas" in title and "cleat" in title:
        return "KEEP_BRANDED_EXACT_MATCH"

    if explore:
        return "RESCUE_COLD_START_RELEVANT"

    if penalty > 0:
        return "PENALIZE_OVER_CONCENTRATION"

    if confidence == "high" and stage == "core":
        return "BOOST_BEHAVIOR_CORE_ITEM"

    if stage == "completion" and coverage >= 0.05:
        return "BOOST_MISSION_COMPLETION"

    if "repair_loop" in source and confidence in {"medium", "low_behavior_but_mission_relevant"}:
        return "ACCEPT_REPAIR_WITH_BEHAVIOR_GUARDRAIL"

    if confidence == "low":
        return "KEEP_LOW_CONFIDENCE_REVIEW"

    return "KEEP_BALANCED_MISSION_ITEM"


def base_relevance_proxy(row: Dict[str, str]) -> float:
    rank = parse_rank(row)

    if rank == 999:
        return 0.50

    # Smooth rank decay. Rank 1 receives strongest base score.
    return round(1.0 / math.sqrt(rank), 4)


def final_policy_score(
    row: Dict[str, str],
    score: float,
    coverage: float,
    penalty: float,
    explore: bool,
) -> float:
    base = base_relevance_proxy(row)
    stage = infer_mission_stage(row)

    stage_boost = {
        "core": 0.14,
        "supporting": 0.10,
        "completion": 0.08,
        "optional": 0.02,
        "unknown": 0.00,
    }.get(stage, 0.00)

    exploration_boost = 0.08 if explore else 0.0

    final = (
        base * 0.45
        + score * 0.35
        + coverage
        + stage_boost
        + exploration_boost
        - penalty
    )

    return round(max(0.0, min(1.5, final)), 4)


def behavior_aware_rank(query: str, rows: List[Dict[str, str]]) -> Tuple[List[Dict[str, object]], Dict[str, object], List[Dict[str, object]]]:
    query_rows = get_query_rows(rows, query)

    if not query_rows:
        summary = {
            "query": query,
            "input_strict_slate_size": 0,
            "final_behavior_slate_size": 0,
            "high_confidence_items": 0,
            "medium_confidence_items": 0,
            "low_confidence_items": 0,
            "cold_start_rescue_items": 0,
            "exploration_items": 0,
            "unique_sub_intents": 0,
            "final_decision": "no_strict_slate",
            "explanation": "No strict repaired slate was available for behavior-aware scoring.",
        }
        return [], summary, []

    sub_intent_counts = Counter(lower_text(r.get("sub_intent")) for r in query_rows)

    scored_rows: List[Dict[str, object]] = []
    reason_rows: List[Dict[str, object]] = []

    for row in query_rows:
        b_score = behavior_score(row)
        confidence = behavior_confidence(b_score, row)
        stage = infer_mission_stage(row)
        coverage = coverage_contribution(row, sub_intent_counts)
        penalty = concentration_penalty(row, sub_intent_counts)
        explore = exploration_flag(row, b_score, coverage)
        reason = policy_reason(row, b_score, confidence, coverage, penalty, explore)
        final_score = final_policy_score(row, b_score, coverage, penalty, explore)

        out = {
            "query": query,
            "behavior_rank": None,
            "original_strict_rank": parse_rank(row),
            "slate_source": clean_text(row.get("slate_source")),
            "sub_intent": clean_text(row.get("sub_intent")),
            "sub_intent_role": clean_text(row.get("sub_intent_role")),
            "mission_stage": stage,
            "product_title": clean_text(row.get("product_title")),
            "base_relevance_proxy": base_relevance_proxy(row),
            "behavior_score": b_score,
            "behavior_confidence": confidence,
            "coverage_contribution": coverage,
            "cold_start_proxy": cold_start_proxy(row),
            "exploration_flag": explore,
            "over_concentration_penalty": penalty,
            "final_policy_score": final_score,
            "policy_reason": reason,
        }

        scored_rows.append(out)

        reason_rows.append({
            "query": query,
            "product_title": clean_text(row.get("product_title")),
            "sub_intent": clean_text(row.get("sub_intent")),
            "mission_stage": stage,
            "behavior_score": b_score,
            "behavior_confidence": confidence,
            "policy_reason": reason,
            "explanation": explain_reason(reason, stage, confidence, explore, penalty),
        })

    scored_rows.sort(
        key=lambda r: (
            float(r["final_policy_score"]),
            float(r["behavior_score"]),
            float(r["coverage_contribution"]),
        ),
        reverse=True,
    )

    for idx, row in enumerate(scored_rows, start=1):
        row["behavior_rank"] = idx

    confidence_counts = Counter(clean_text(r["behavior_confidence"]) for r in scored_rows)
    reason_counts = Counter(clean_text(r["policy_reason"]) for r in scored_rows)

    summary = {
        "query": query,
        "input_strict_slate_size": len(query_rows),
        "final_behavior_slate_size": len(scored_rows),
        "high_confidence_items": confidence_counts.get("high", 0),
        "medium_confidence_items": confidence_counts.get("medium", 0),
        "low_confidence_items": confidence_counts.get("low", 0),
        "cold_start_rescue_items": reason_counts.get("RESCUE_COLD_START_RELEVANT", 0),
        "exploration_items": sum(1 for r in scored_rows if r["exploration_flag"] is True),
        "unique_sub_intents": len({lower_text(r.get("sub_intent")) for r in query_rows if lower_text(r.get("sub_intent"))}),
        "final_decision": "behavior_aware_slate_generated",
        "explanation": (
            f"Behavior-Aware CORTEX ranked {len(scored_rows)} items for '{query}' using "
            f"behavior proxy, mission stage, coverage contribution, cold-start rescue, "
            f"exploration flags, and policy reason codes."
        ),
    }

    return scored_rows, summary, reason_rows


def explain_reason(reason: str, stage: str, confidence: str, explore: bool, penalty: float) -> str:
    explanations = {
        "KEEP_BRANDED_EXACT_MATCH": "Kept because the item strongly matches a narrow branded product query.",
        "RESCUE_COLD_START_RELEVANT": "Rescued because the item has lower behavior evidence but contributes to the mission slate.",
        "PENALIZE_OVER_CONCENTRATION": "Penalized because too many items share the same sub-intent.",
        "BOOST_BEHAVIOR_CORE_ITEM": "Boosted because it is a core mission-stage item with strong behavior confidence.",
        "BOOST_MISSION_COMPLETION": "Boosted because it helps complete the shopping mission.",
        "ACCEPT_REPAIR_WITH_BEHAVIOR_GUARDRAIL": "Kept repaired item after behavior-aware guardrail review.",
        "KEEP_LOW_CONFIDENCE_REVIEW": "Kept but marked as low behavior confidence for review.",
        "KEEP_BALANCED_MISSION_ITEM": "Kept because it balances behavior evidence and mission coverage.",
    }

    return explanations.get(
        reason,
        f"Policy reason={reason}; stage={stage}; confidence={confidence}; explore={explore}; penalty={penalty}.",
    )


def print_outputs(summary: Dict[str, object], slate: List[Dict[str, object]], reasons: List[Dict[str, object]]) -> None:
    print("\nBehavior-Aware CORTEX Summary")
    print("-" * 100)
    for k, v in summary.items():
        print(f"{k}: {v}")

    if slate:
        print("\nTop Behavior-Aware Slate")
        print("-" * 100)
        for row in slate[:15]:
            print(
                f"{row['behavior_rank']:>2}. "
                f"score={row['final_policy_score']} "
                f"behavior={row['behavior_score']} "
                f"confidence={row['behavior_confidence']} "
                f"stage={row['mission_stage']} "
                f"reason={row['policy_reason']} "
                f"| {row['sub_intent']} | {row['product_title']}"
            )
    else:
        print("\nNo behavior-aware slate generated.")


def main() -> None:
    parser = argparse.ArgumentParser(description="MVP 16 Behavior-Aware CORTEX")
    parser.add_argument("--query", required=True, help="Shopping query to score.")
    parser.add_argument(
        "--skip-strict-refresh",
        action="store_true",
        help="Skip running MVP 15.9 first and use existing strict output CSV.",
    )

    args = parser.parse_args()
    query = args.query.strip()

    ensure_output_dir()

    if not args.skip_strict_refresh:
        run_strict_repair_if_needed(query)

    strict_rows = read_csv(STRICT_SLATE_PATH)

    slate, summary, reasons = behavior_aware_rank(query, strict_rows)

    slate_fields = [
        "query",
        "behavior_rank",
        "original_strict_rank",
        "slate_source",
        "sub_intent",
        "sub_intent_role",
        "mission_stage",
        "product_title",
        "base_relevance_proxy",
        "behavior_score",
        "behavior_confidence",
        "coverage_contribution",
        "cold_start_proxy",
        "exploration_flag",
        "over_concentration_penalty",
        "final_policy_score",
        "policy_reason",
    ]

    summary_fields = [
        "query",
        "input_strict_slate_size",
        "final_behavior_slate_size",
        "high_confidence_items",
        "medium_confidence_items",
        "low_confidence_items",
        "cold_start_rescue_items",
        "exploration_items",
        "unique_sub_intents",
        "final_decision",
        "explanation",
    ]

    reason_fields = [
        "query",
        "product_title",
        "sub_intent",
        "mission_stage",
        "behavior_score",
        "behavior_confidence",
        "policy_reason",
        "explanation",
    ]

    existing_slate = read_csv(BEHAVIOR_SLATE_PATH)
    existing_summary = read_csv(BEHAVIOR_SUMMARY_PATH)
    existing_reasons = read_csv(BEHAVIOR_REASONS_PATH)

    existing_slate = [r for r in existing_slate if lower_text(r.get("query")) != query.lower()]
    existing_summary = [r for r in existing_summary if lower_text(r.get("query")) != query.lower()]
    existing_reasons = [r for r in existing_reasons if lower_text(r.get("query")) != query.lower()]

    write_csv(BEHAVIOR_SLATE_PATH, existing_slate + slate, slate_fields)
    write_csv(BEHAVIOR_SUMMARY_PATH, existing_summary + [summary], summary_fields)
    write_csv(BEHAVIOR_REASONS_PATH, existing_reasons + reasons, reason_fields)

    print_outputs(summary, slate, reasons)

    print("\nFiles written:")
    print(f"- {BEHAVIOR_SLATE_PATH}")
    print(f"- {BEHAVIOR_SUMMARY_PATH}")
    print(f"- {BEHAVIOR_REASONS_PATH}")


if __name__ == "__main__":
    main()
