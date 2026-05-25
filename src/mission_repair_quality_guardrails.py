"""
MVP 15.8: Repair Quality Guardrails

Purpose:
--------
Improve the Mission Repair Loop by applying stricter quality checks to repaired
products before accepting them into the final mission slate.

Why this exists:
----------------
MVP 15.7 successfully repairs missing mission needs, but some repair candidates
can be weak because broad role matching may accept products that are not truly
aligned with the missing sub-intent.

Examples of weak repairs:
- soccer jersey -> random floral dress
- cooler -> candle making kit
- kitchen towels -> wall file holder

MVP 15.8 adds:
1. Sub-intent-specific required keyword checks
2. Strict title/sub-intent matching
3. Repair quality labels
4. Rejection of role-only weak matches
5. Cleaner repaired slate output

Outputs:
--------
outputs/mission_repair_quality_guarded_slate.csv
outputs/mission_repair_quality_summary.csv
outputs/mission_repair_quality_candidates.csv
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import pandas as pd

from src.mission_repair_loop import build_repaired_mission_slate


OUTPUT_DIR = "outputs"

QUALITY_GUARDED_SLATE_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "mission_repair_quality_guarded_slate.csv",
)
QUALITY_SUMMARY_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "mission_repair_quality_summary.csv",
)
QUALITY_CANDIDATES_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "mission_repair_quality_candidates.csv",
)


SUB_INTENT_REQUIRED_KEYWORDS = {
    "soccer jersey": ["soccer", "jersey", "football", "fifa", "team", "shirt"],
    "cooler": ["cooler", "ice chest", "insulated", "cooler bag", "beverage cooler"],
    "portable speaker": ["speaker", "bluetooth", "audio", "boombox", "sound"],
    "first aid kit": ["first aid", "medical", "emergency", "bandage", "trauma"],
    "bug spray": ["bug", "insect", "mosquito", "repellent", "spray"],
    "water bottle": ["water", "bottle", "hydration", "canteen"],
    "dish drying rack": ["dish", "drying", "rack", "drainboard"],
    "trash can": ["trash", "garbage", "waste", "bin", "can"],
    "kitchen towels": ["kitchen", "towel", "dish towel", "tea towel"],
    "sandals": ["sandals", "slides", "flip flop", "footwear"],
    "portable cooler": ["cooler", "portable", "insulated", "cooler bag", "ice chest"],
}


SUB_INTENT_BAD_KEYWORDS = {
    "soccer jersey": [
        "dress",
        "sari",
        "saree",
        "sandals",
        "socks",
        "watch band",
        "hat",
        "cap",
        "sunglasses",
    ],
    "cooler": [
        "candle",
        "wax",
        "mug",
        "tablet",
        "ssd",
        "socks",
        "pants",
        "shirt",
        "dress",
    ],
    "portable speaker": [
        "shirt",
        "pants",
        "dress",
        "plate",
        "cup",
        "decor",
    ],
    "kitchen towels": [
        "file holder",
        "wall file",
        "rack organizer",
        "office",
        "paper holder",
    ],
    "trash can": [
        "bags only",
        "trash bags",
    ],
}


@dataclass
class RepairQualitySummary:
    query: str
    original_final_slate_size: int
    original_repair_count: int
    quality_accepted_repair_count: int
    quality_rejected_repair_count: int
    final_quality_guarded_slate_size: int
    accepted_repaired_sub_intents: str
    rejected_repaired_sub_intents: str
    final_quality_decision: str
    explanation: str


def ensure_output_dir() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def normalize_text(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    text = normalize_text(text)
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
        "pcs",
        "piece",
        "pieces",
    }
    return [
        token
        for token in text.split()
        if token not in stopwords and len(token) >= 2
    ]


def keyword_hit_count(keywords: List[str], title: str) -> int:
    normalized_title = normalize_text(title)
    return sum(1 for keyword in keywords if normalize_text(keyword) in normalized_title)


def token_overlap_score(a: str, b: str) -> float:
    a_tokens = tokenize(a)
    b_tokens = set(tokenize(b))

    if not a_tokens or not b_tokens:
        return 0.0

    exact = sum(1 for token in a_tokens if token in b_tokens)

    partial = 0.0
    for token in a_tokens:
        if token in b_tokens:
            continue

        for other in b_tokens:
            if token in other or other in token:
                partial += 0.5
                break

    return round(min((exact + partial) / max(len(a_tokens), 1), 1.0), 4)


def has_bad_sub_intent_keyword(sub_intent: str, title: str) -> bool:
    sub_intent = normalize_text(sub_intent)
    bad_keywords = SUB_INTENT_BAD_KEYWORDS.get(sub_intent, [])
    return keyword_hit_count(bad_keywords, title) > 0


def required_sub_intent_score(sub_intent: str, title: str) -> float:
    sub_intent = normalize_text(sub_intent)
    keywords = SUB_INTENT_REQUIRED_KEYWORDS.get(sub_intent, [])

    if not keywords:
        return 0.0

    hits = keyword_hit_count(keywords, title)
    return round(min(hits * 0.35, 1.0), 4)


def classify_repair_quality(row: pd.Series) -> Tuple[str, bool, str, float, float]:
    sub_intent = str(row.get("sub_intent", ""))
    role = str(row.get("sub_intent_role", row.get("role", "")))
    title = str(row.get("product_title", ""))
    score = float(row.get("final_mission_score", row.get("candidate_score", 0.0)))

    title_sub_intent_score = token_overlap_score(sub_intent, title)
    required_score = required_sub_intent_score(sub_intent, title)
    bad_keyword = has_bad_sub_intent_keyword(sub_intent, title)

    if bad_keyword:
        return (
            "rejected",
            False,
            "Rejected because title contains a bad keyword for this repair sub-intent.",
            title_sub_intent_score,
            required_score,
        )

    # Strong exact/sub-intent match
    if title_sub_intent_score >= 0.50 and required_score >= 0.35:
        return (
            "strong",
            True,
            "Accepted as strong repair: title matches the missing sub-intent and required keywords.",
            title_sub_intent_score,
            required_score,
        )

    # Strong required keyword match even if phrase overlap is imperfect
    if required_score >= 0.70:
        return (
            "strong",
            True,
            "Accepted as strong repair: title contains multiple required sub-intent keywords.",
            title_sub_intent_score,
            required_score,
        )

    # Acceptable but not perfect
    if title_sub_intent_score >= 0.50 and score >= 0.80:
        return (
            "acceptable",
            True,
            "Accepted as acceptable repair: title aligns with sub-intent and score is adequate.",
            title_sub_intent_score,
            required_score,
        )

    # For unknown sub-intents, allow only if title overlap is strong
    if normalize_text(sub_intent) not in SUB_INTENT_REQUIRED_KEYWORDS:
        if title_sub_intent_score >= 0.60 and score >= 0.80:
            return (
                "acceptable",
                True,
                "Accepted as acceptable repair for unregistered sub-intent due to strong title overlap.",
                title_sub_intent_score,
                required_score,
            )

    return (
        "weak",
        False,
        "Rejected because repair candidate relies on weak or role-only alignment.",
        title_sub_intent_score,
        required_score,
    )


def apply_repair_quality_guardrails(
    repaired_slate_df: pd.DataFrame,
    repair_candidates_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if repaired_slate_df.empty:
        return repaired_slate_df.copy(), repair_candidates_df.copy()

    final_rows = []

    for _, row in repaired_slate_df.iterrows():
        row_dict = row.to_dict()
        slate_source = str(row.get("slate_source", ""))

        if slate_source != "repair_loop":
            row_dict["repair_quality_label"] = "original"
            row_dict["repair_quality_accepted"] = True
            row_dict["repair_quality_reason"] = "Original guarded slate row; repair quality guardrail not needed."
            row_dict["repair_title_sub_intent_score"] = None
            row_dict["repair_required_keyword_score"] = None
            final_rows.append(row_dict)
            continue

        (
            quality_label,
            accepted,
            reason,
            title_sub_intent_score,
            required_score,
        ) = classify_repair_quality(row)

        row_dict["repair_quality_label"] = quality_label
        row_dict["repair_quality_accepted"] = accepted
        row_dict["repair_quality_reason"] = reason
        row_dict["repair_title_sub_intent_score"] = title_sub_intent_score
        row_dict["repair_required_keyword_score"] = required_score

        if accepted:
            final_rows.append(row_dict)

    final_slate_df = pd.DataFrame(final_rows)

    if not final_slate_df.empty:
        final_slate_df = final_slate_df.reset_index(drop=True)
        final_slate_df["final_quality_rank"] = range(1, len(final_slate_df) + 1)

    if repair_candidates_df.empty:
        return final_slate_df, repair_candidates_df.copy()

    candidate_rows = []
    for _, row in repair_candidates_df.iterrows():
        candidate_like = pd.Series(
            {
                "sub_intent": row.get("sub_intent", ""),
                "sub_intent_role": row.get("role", ""),
                "product_title": row.get("product_title", ""),
                "final_mission_score": row.get("candidate_score", 0.0),
            }
        )

        (
            quality_label,
            accepted,
            reason,
            title_sub_intent_score,
            required_score,
        ) = classify_repair_quality(candidate_like)

        row_dict = row.to_dict()
        row_dict["repair_quality_label"] = quality_label
        row_dict["repair_quality_accepted"] = accepted
        row_dict["repair_quality_reason"] = reason
        row_dict["repair_title_sub_intent_score"] = title_sub_intent_score
        row_dict["repair_required_keyword_score"] = required_score
        candidate_rows.append(row_dict)

    enriched_candidates_df = pd.DataFrame(candidate_rows)

    return final_slate_df, enriched_candidates_df


def build_quality_guarded_repair_slate(
    query: str,
    slate_size: int = 10,
    max_repairs: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensure_output_dir()

    repaired_slate_df, repair_summary_df, repair_candidates_df = build_repaired_mission_slate(
        query=query,
        slate_size=slate_size,
        max_repairs=max_repairs,
    )

    final_slate_df, enriched_candidates_df = apply_repair_quality_guardrails(
        repaired_slate_df=repaired_slate_df,
        repair_candidates_df=repair_candidates_df,
    )

    if repaired_slate_df.empty:
        summary = RepairQualitySummary(
            query=query,
            original_final_slate_size=0,
            original_repair_count=0,
            quality_accepted_repair_count=0,
            quality_rejected_repair_count=0,
            final_quality_guarded_slate_size=0,
            accepted_repaired_sub_intents="[]",
            rejected_repaired_sub_intents="[]",
            final_quality_decision="no_mission_slate",
            explanation="No mission slate was generated.",
        )
        return final_slate_df, pd.DataFrame([asdict(summary)]), enriched_candidates_df

    original_repairs = repaired_slate_df[
        repaired_slate_df["slate_source"] == "repair_loop"
    ].copy()

    accepted_repairs = final_slate_df[
        final_slate_df.get("slate_source", pd.Series(dtype=str)) == "repair_loop"
    ].copy() if not final_slate_df.empty else pd.DataFrame()

    accepted_sub_intents = (
        accepted_repairs["sub_intent"].astype(str).drop_duplicates().tolist()
        if not accepted_repairs.empty
        else []
    )

    original_repaired_sub_intents = (
        original_repairs["sub_intent"].astype(str).drop_duplicates().tolist()
        if not original_repairs.empty
        else []
    )

    rejected_sub_intents = [
        item for item in original_repaired_sub_intents if item not in accepted_sub_intents
    ]

    accepted_count = len(accepted_sub_intents)
    original_count = len(original_repaired_sub_intents)
    rejected_count = max(original_count - accepted_count, 0)

    if original_count == 0:
        decision = "accept_original_guarded_slate"
    elif rejected_count == 0:
        decision = "accept_quality_guarded_repaired_slate"
    elif accepted_count > 0:
        decision = "accept_with_some_repair_rejections"
    else:
        decision = "reject_repairs_keep_guarded_slate"

    explanation = (
        f"CORTEX quality guardrails reviewed {original_count} repaired sub-intents for '{query}'. "
        f"Accepted: {accepted_sub_intents if accepted_sub_intents else 'none'}. "
        f"Rejected: {rejected_sub_intents if rejected_sub_intents else 'none'}."
    )

    summary = RepairQualitySummary(
        query=query,
        original_final_slate_size=len(repaired_slate_df),
        original_repair_count=original_count,
        quality_accepted_repair_count=accepted_count,
        quality_rejected_repair_count=rejected_count,
        final_quality_guarded_slate_size=len(final_slate_df),
        accepted_repaired_sub_intents=json.dumps(accepted_sub_intents),
        rejected_repaired_sub_intents=json.dumps(rejected_sub_intents),
        final_quality_decision=decision,
        explanation=explanation,
    )

    return final_slate_df, pd.DataFrame([asdict(summary)]), enriched_candidates_df


def run_demo() -> None:
    ensure_output_dir()

    demo_queries = [
        "world cup watch party",
        "camping trip essentials",
        "new apartment kitchen setup",
        "beach vacation packing list",
    ]

    all_slates = []
    all_summaries = []
    all_candidates = []

    for query in demo_queries:
        final_slate_df, summary_df, candidates_df = build_quality_guarded_repair_slate(
            query=query,
            slate_size=10,
            max_repairs=5,
        )

        if not final_slate_df.empty:
            all_slates.append(final_slate_df)

        all_summaries.append(summary_df)

        if not candidates_df.empty:
            all_candidates.append(candidates_df)

    final_all_slates = (
        pd.concat(all_slates, ignore_index=True)
        if all_slates
        else pd.DataFrame()
    )
    final_all_summaries = pd.concat(all_summaries, ignore_index=True)
    final_all_candidates = (
        pd.concat(all_candidates, ignore_index=True)
        if all_candidates
        else pd.DataFrame()
    )

    final_all_slates.to_csv(QUALITY_GUARDED_SLATE_OUTPUT, index=False)
    final_all_summaries.to_csv(QUALITY_SUMMARY_OUTPUT, index=False)
    final_all_candidates.to_csv(QUALITY_CANDIDATES_OUTPUT, index=False)

    print()
    print("MVP 15.8 Repair Quality Guardrails Demo")
    print("=" * 100)

    print()
    print("Repair Quality Summary")
    print("-" * 100)
    print(final_all_summaries.to_string(index=False))

    if not final_all_slates.empty:
        print()
        print("Quality-Guarded Repaired Slate Sample")
        print("-" * 100)
        cols = [
            "query",
            "final_quality_rank",
            "slate_source",
            "sub_intent",
            "sub_intent_role",
            "product_title",
            "repair_quality_label",
            "repair_quality_reason",
        ]
        print(final_all_slates[cols].head(40).to_string(index=False))

    if not final_all_candidates.empty:
        print()
        print("Repair Quality Candidate Sample")
        print("-" * 100)
        cols = [
            "query",
            "repair_query",
            "sub_intent",
            "product_title",
            "candidate_score",
            "repair_quality_label",
            "repair_quality_accepted",
            "repair_quality_reason",
        ]
        print(final_all_candidates[cols].head(40).to_string(index=False))

    print()
    print("Files written")
    print("-" * 100)
    print(f"- quality guarded slate: {QUALITY_GUARDED_SLATE_OUTPUT}")
    print(f"- quality summary: {QUALITY_SUMMARY_OUTPUT}")
    print(f"- quality candidates: {QUALITY_CANDIDATES_OUTPUT}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Mission query to quality-check after repair.",
    )

    parser.add_argument(
        "--slate-size",
        type=int,
        default=10,
        help="Maximum raw mission slate size.",
    )

    parser.add_argument(
        "--max-repairs",
        type=int,
        default=5,
        help="Maximum repair products to consider.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.query:
        slate, summary, candidates = build_quality_guarded_repair_slate(
            query=args.query,
            slate_size=args.slate_size,
            max_repairs=args.max_repairs,
        )

        print()
        print("Repair Quality Summary")
        print("-" * 100)
        print(summary.to_string(index=False))

        if slate.empty:
            print()
            print("No quality-guarded slate generated.")
        else:
            print()
            print("Quality-Guarded Repaired Slate")
            print("-" * 100)
            print(
                slate[
                    [
                        "final_quality_rank",
                        "slate_source",
                        "sub_intent",
                        "sub_intent_role",
                        "product_title",
                        "repair_quality_label",
                        "repair_quality_reason",
                    ]
                ].to_string(index=False)
            )

        if not candidates.empty:
            print()
            print("Repair Quality Candidates")
            print("-" * 100)
            print(
                candidates[
                    [
                        "repair_query",
                        "sub_intent",
                        "product_title",
                        "candidate_score",
                        "repair_quality_label",
                        "repair_quality_accepted",
                        "repair_quality_reason",
                    ]
                ].head(40).to_string(index=False)
            )
    else:
        run_demo()