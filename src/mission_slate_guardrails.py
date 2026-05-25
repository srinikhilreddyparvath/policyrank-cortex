"""
MVP 15.3: Mission Slate Relevance Guardrails

Purpose:
--------
Clean the mission-aware slate produced by MVP 15.2 by removing weak or noisy
sub-intent/product matches.

Why this exists:
----------------
MVP 15.2 can retrieve products for each mission sub-intent, but lightweight text
matching can sometimes include noisy products. Example:
- "camping trip essentials" → "cooler" → irrelevant apparel product

MVP 15.3 adds guardrails:
1. Title-level relevance check
2. Role-specific required keyword checks
3. Bad-match penalties
4. Weak candidate rejection
5. Honest missing-sub-intent reporting

Inputs:
-------
src.mission_slate_builder.build_mission_slate

Outputs:
--------
outputs/mission_slate_guarded_demo.csv
outputs/mission_slate_guarded_summary.csv
outputs/mission_slate_guarded_rejections.csv
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import pandas as pd

from src.mission_agent import analyze_mission_query
from src.mission_slate_builder import build_mission_slate


OUTPUT_DIR = "outputs"

GUARDED_SLATE_OUTPUT = os.path.join(OUTPUT_DIR, "mission_slate_guarded_demo.csv")
GUARDED_SUMMARY_OUTPUT = os.path.join(OUTPUT_DIR, "mission_slate_guarded_summary.csv")
GUARDED_REJECTIONS_OUTPUT = os.path.join(OUTPUT_DIR, "mission_slate_guarded_rejections.csv")


ROLE_REQUIRED_KEYWORDS = {
    "food": ["snack", "chips", "popcorn", "cookies", "candy", "food"],
    "beverage": ["drink", "soda", "water", "juice", "beverage", "bottle"],
    "serveware": ["plate", "plates", "cup", "cups", "napkin", "fork", "spoon", "dinnerware"],
    "decor": ["decor", "decoration", "banner", "balloon", "party", "soccer", "coaster"],
    "apparel": ["shirt", "jersey", "dress", "wear", "apparel", "clothing", "pants", "sweatpants"],
    "utility": ["cooler", "storage", "bag", "container", "utility", "organizer"],
    "entertainment": ["speaker", "bluetooth", "audio", "music", "game"],
    "shelter": ["tent", "canopy", "shelter"],
    "sleep": ["sleeping", "blanket", "pillow", "mattress", "bed"],
    "lighting": ["lantern", "light", "flashlight"],
    "food storage": ["cooler", "beverage", "refrigerator", "fridge", "ice chest", "storage"],
    "safety": ["first aid", "bandage", "safety", "medical", "emergency"],
    "protection": ["sunscreen", "bug", "spray", "repellent", "protection"],
    "hydration": ["water", "bottle", "hydration"],
    "comfort": ["chair", "towel", "blanket", "comfort", "cushion"],
    "storage": ["bag", "storage", "container", "luggage", "organizer"],
    "footwear": ["shoe", "shoes", "sandals", "boots", "sneaker"],
    "cooking": ["cookware", "pan", "pot", "skillet", "burner", "cooking"],
    "prep": ["knife", "cutting", "board", "prep"],
    "cleaning": ["clean", "towel", "rack", "dish", "trash"],
}


ROLE_BAD_KEYWORDS = {
    "food storage": ["sweatpants", "jogger", "shirt", "dress", "blanket"],
    "serveware": ["shirt", "sweatpants", "shoe", "blanket"],
    "shelter": ["shirt", "pants", "snack", "drink"],
    "lighting": ["shirt", "pants", "snack", "drink"],
    "safety": ["shirt", "pants", "decor"],
    "hydration": ["shirt", "pants", "decor"],
    "footwear": ["plate", "cup", "snack", "decor"],
    "cooking": ["shirt", "pants", "decor"],
    "prep": ["shirt", "pants", "snack"],
    "cleaning": ["shirt", "pants", "snack"],
}


@dataclass
class GuardrailSummary:
    query: str
    is_mission: bool
    mission_type: str
    mission_name: str
    mission_confidence: float
    requested_sub_intent_count: int
    raw_slate_size: int
    guarded_slate_size: int
    raw_covered_sub_intent_count: int
    guarded_covered_sub_intent_count: int
    rejected_candidate_count: int
    missing_after_guardrails_count: int
    guarded_coverage_score: float
    avg_guarded_mission_score: float
    missing_after_guardrails: str
    guardrail_notes: str


def ensure_output_dir() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def normalize_text(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    normalized = normalize_text(text)
    stopwords = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "for",
        "with",
        "of",
        "to",
        "in",
        "on",
        "by",
        "at",
        "from",
        "set",
        "pack",
    }

    return [
        token
        for token in normalized.split()
        if token not in stopwords and len(token) >= 2
    ]


def text_overlap_score(query: str, title: str) -> float:
    query_tokens = tokenize(query)
    title_tokens = set(tokenize(title))

    if not query_tokens or not title_tokens:
        return 0.0

    exact_matches = sum(1 for token in query_tokens if token in title_tokens)
    partial_matches = 0.0

    for token in query_tokens:
        if token in title_tokens:
            continue

        for title_token in title_tokens:
            if token in title_token or title_token in token:
                partial_matches += 0.5
                break

    score = (exact_matches + partial_matches) / max(len(query_tokens), 1)

    return round(min(score, 1.0), 4)


def keyword_hit_count(keywords: List[str], title: str) -> int:
    normalized_title = normalize_text(title)
    return sum(1 for keyword in keywords if normalize_text(keyword) in normalized_title)


def has_bad_role_keyword(role: str, title: str) -> bool:
    role = normalize_text(role)
    bad_keywords = ROLE_BAD_KEYWORDS.get(role, [])
    return keyword_hit_count(bad_keywords, title) > 0


def role_required_score(role: str, title: str) -> float:
    role = normalize_text(role)
    required_keywords = ROLE_REQUIRED_KEYWORDS.get(role, [])

    if not required_keywords:
        return 0.0

    hits = keyword_hit_count(required_keywords, title)

    return round(min(hits * 0.25, 1.0), 4)


def evaluate_guardrail(row: pd.Series) -> Tuple[bool, str, float, float, float]:
    sub_intent = str(row.get("sub_intent", ""))
    role = str(row.get("sub_intent_role", ""))
    title = str(row.get("product_title", ""))

    text_score = float(row.get("product_text_score", 0.0))
    role_score = float(row.get("product_role_score", 0.0))
    final_score = float(row.get("final_mission_score", 0.0))

    title_intent_score = text_overlap_score(sub_intent, title)
    title_role_score = role_required_score(role, title)

    if has_bad_role_keyword(role, title):
        return (
            False,
            "Rejected because product title contains a bad keyword for the sub-intent role.",
            title_intent_score,
            title_role_score,
            final_score,
        )

    strong_title_match = title_intent_score >= 0.40
    strong_role_match = title_role_score >= 0.25
    strong_existing_score = text_score >= 0.50 and role_score >= 0.15
    high_final_score = final_score >= 1.00

    if strong_title_match:
        return (
            True,
            "Accepted because title strongly matches the sub-intent.",
            title_intent_score,
            title_role_score,
            final_score,
        )

    if strong_role_match and text_score >= 0.35:
        return (
            True,
            "Accepted because title matches the expected role and candidate text score is adequate.",
            title_intent_score,
            title_role_score,
            final_score,
        )

    if strong_existing_score and high_final_score:
        return (
            True,
            "Accepted because builder score and role score are both strong.",
            title_intent_score,
            title_role_score,
            final_score,
        )

    return (
        False,
        "Rejected because title/sub-intent/role alignment was too weak.",
        title_intent_score,
        title_role_score,
        final_score,
    )


def apply_guardrails(
    raw_slate_df: pd.DataFrame,
    max_per_sub_intent: int = 2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if raw_slate_df.empty:
        return raw_slate_df.copy(), pd.DataFrame()

    accepted_rows = []
    rejected_rows = []

    for _, row in raw_slate_df.iterrows():
        accepted, reason, title_intent_score, title_role_score, raw_final_score = evaluate_guardrail(row)

        enriched = row.to_dict()
        enriched["guardrail_accepted"] = accepted
        enriched["guardrail_reason"] = reason
        enriched["title_intent_score"] = title_intent_score
        enriched["title_role_score"] = title_role_score
        enriched["raw_final_mission_score"] = raw_final_score

        if accepted:
            accepted_rows.append(enriched)
        else:
            rejected_rows.append(enriched)

    accepted_df = pd.DataFrame(accepted_rows)
    rejected_df = pd.DataFrame(rejected_rows)

    if accepted_df.empty:
        return accepted_df, rejected_df

    accepted_df = accepted_df.sort_values(
        by=[
            "sub_intent_priority",
            "final_mission_score",
            "title_intent_score",
            "title_role_score",
        ],
        ascending=[False, False, False, False],
    )

    accepted_df = (
        accepted_df.groupby(["query", "sub_intent"], as_index=False, group_keys=False)
        .head(max_per_sub_intent)
        .copy()
    )

    accepted_df = accepted_df.sort_values(
        by=["sub_intent_priority", "final_mission_score"],
        ascending=[False, False],
    ).reset_index(drop=True)

    accepted_df["guarded_rank"] = range(1, len(accepted_df) + 1)

    return accepted_df, rejected_df


def build_guarded_mission_slate(
    query: str,
    slate_size: int = 10,
    max_per_sub_intent: int = 2,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensure_output_dir()

    mission = analyze_mission_query(query)
    raw_slate_df, raw_summary_df = build_mission_slate(
        query=query,
        slate_size=slate_size,
        max_per_sub_intent=max_per_sub_intent,
    )

    if not mission.is_mission:
        summary = GuardrailSummary(
            query=query,
            is_mission=False,
            mission_type=mission.mission_type,
            mission_name=mission.mission_name,
            mission_confidence=mission.confidence,
            requested_sub_intent_count=0,
            raw_slate_size=0,
            guarded_slate_size=0,
            raw_covered_sub_intent_count=0,
            guarded_covered_sub_intent_count=0,
            rejected_candidate_count=0,
            missing_after_guardrails_count=0,
            guarded_coverage_score=0.0,
            avg_guarded_mission_score=0.0,
            missing_after_guardrails="[]",
            guardrail_notes="Single-product query; no mission slate needed.",
        )

        return pd.DataFrame(), pd.DataFrame([asdict(summary)]), pd.DataFrame()

    guarded_df, rejected_df = apply_guardrails(
        raw_slate_df=raw_slate_df,
        max_per_sub_intent=max_per_sub_intent,
    )

    requested_sub_intents = [
        str(item.get("name", ""))
        for item in mission.sub_intents
    ]

    raw_covered = set(raw_slate_df["sub_intent"].unique()) if not raw_slate_df.empty else set()
    guarded_covered = set(guarded_df["sub_intent"].unique()) if not guarded_df.empty else set()

    missing_after_guardrails = [
        sub_intent
        for sub_intent in requested_sub_intents
        if sub_intent not in guarded_covered
    ]

    guarded_slate_size = len(guarded_df)
    requested_count = len(requested_sub_intents)
    guarded_covered_count = len(guarded_covered)

    guarded_coverage_score = (
        round(guarded_covered_count / requested_count, 4)
        if requested_count
        else 0.0
    )

    avg_guarded_score = (
        round(float(guarded_df["final_mission_score"].mean()), 4)
        if not guarded_df.empty
        else 0.0
    )

    summary = GuardrailSummary(
        query=query,
        is_mission=True,
        mission_type=mission.mission_type,
        mission_name=mission.mission_name,
        mission_confidence=mission.confidence,
        requested_sub_intent_count=requested_count,
        raw_slate_size=len(raw_slate_df),
        guarded_slate_size=guarded_slate_size,
        raw_covered_sub_intent_count=len(raw_covered),
        guarded_covered_sub_intent_count=guarded_covered_count,
        rejected_candidate_count=len(rejected_df),
        missing_after_guardrails_count=len(missing_after_guardrails),
        guarded_coverage_score=guarded_coverage_score,
        avg_guarded_mission_score=avg_guarded_score,
        missing_after_guardrails=json.dumps(missing_after_guardrails),
        guardrail_notes=(
            "Applied title-level intent checks, role keyword checks, and bad-match filters. "
            "Weak products are rejected instead of being forced into the mission slate."
        ),
    )

    return guarded_df, pd.DataFrame([asdict(summary)]), rejected_df


def run_demo() -> None:
    ensure_output_dir()

    demo_queries = [
        "world cup watch party",
        "camping trip essentials",
        "new apartment kitchen setup",
        "beach vacation packing list",
    ]

    guarded_rows = []
    summary_rows = []
    rejection_rows = []

    for query in demo_queries:
        guarded_df, summary_df, rejected_df = build_guarded_mission_slate(
            query=query,
            slate_size=10,
            max_per_sub_intent=2,
        )

        if not guarded_df.empty:
            guarded_rows.append(guarded_df)

        if not rejected_df.empty:
            rejection_rows.append(rejected_df)

        summary_rows.append(summary_df)

    final_guarded_df = (
        pd.concat(guarded_rows, ignore_index=True)
        if guarded_rows
        else pd.DataFrame()
    )
    final_summary_df = pd.concat(summary_rows, ignore_index=True)
    final_rejections_df = (
        pd.concat(rejection_rows, ignore_index=True)
        if rejection_rows
        else pd.DataFrame()
    )

    final_guarded_df.to_csv(GUARDED_SLATE_OUTPUT, index=False)
    final_summary_df.to_csv(GUARDED_SUMMARY_OUTPUT, index=False)
    final_rejections_df.to_csv(GUARDED_REJECTIONS_OUTPUT, index=False)

    print()
    print("MVP 15.3 Mission Slate Relevance Guardrails Demo")
    print("=" * 100)

    print()
    print("Guarded Summary")
    print("-" * 100)
    print(final_summary_df.to_string(index=False))

    if not final_guarded_df.empty:
        print()
        print("Guarded Mission Slate Sample")
        print("-" * 100)
        cols = [
            "query",
            "guarded_rank",
            "sub_intent",
            "sub_intent_role",
            "product_title",
            "title_intent_score",
            "title_role_score",
            "final_mission_score",
            "guardrail_reason",
        ]
        print(final_guarded_df[cols].head(30).to_string(index=False))

    if not final_rejections_df.empty:
        print()
        print("Rejected Candidate Sample")
        print("-" * 100)
        cols = [
            "query",
            "sub_intent",
            "sub_intent_role",
            "product_title",
            "title_intent_score",
            "title_role_score",
            "final_mission_score",
            "guardrail_reason",
        ]
        print(final_rejections_df[cols].head(20).to_string(index=False))

    print()
    print("Files written")
    print("-" * 100)
    print(f"- guarded slate: {GUARDED_SLATE_OUTPUT}")
    print(f"- guarded summary: {GUARDED_SUMMARY_OUTPUT}")
    print(f"- rejected candidates: {GUARDED_REJECTIONS_OUTPUT}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Mission query to build and guardrail-check.",
    )

    parser.add_argument(
        "--slate-size",
        type=int,
        default=10,
        help="Maximum raw slate size before guardrails.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.query:
        guarded, summary, rejected = build_guarded_mission_slate(
            query=args.query,
            slate_size=args.slate_size,
        )

        print()
        print("Guarded Summary")
        print("-" * 100)
        print(summary.to_string(index=False))

        if guarded.empty:
            print()
            print("No guarded mission slate generated.")
        else:
            print()
            print("Guarded Mission Slate")
            print("-" * 100)
            print(
                guarded[
                    [
                        "guarded_rank",
                        "sub_intent",
                        "sub_intent_role",
                        "product_title",
                        "title_intent_score",
                        "title_role_score",
                        "final_mission_score",
                        "guardrail_reason",
                    ]
                ].to_string(index=False)
            )

        if not rejected.empty:
            print()
            print("Rejected Candidates")
            print("-" * 100)
            print(
                rejected[
                    [
                        "sub_intent",
                        "sub_intent_role",
                        "product_title",
                        "title_intent_score",
                        "title_role_score",
                        "final_mission_score",
                        "guardrail_reason",
                    ]
                ].head(20).to_string(index=False)
            )
    else:
        run_demo()