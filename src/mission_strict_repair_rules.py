"""
MVP 15.9: Strict Compound-Intent Repair Rules

Purpose:
--------
Apply stricter compound-intent rules on top of MVP 15.8 repair quality guardrails.

Why this exists:
----------------
MVP 15.8 improved repair quality, but some compound repair intents were still too loose.

Examples:
- portable cooler -> portable air conditioner
- portable cooler -> portable microphone
- soccer jersey -> generic shirt or dress
- kitchen towels -> paper towel holder or sink product
- water bottle -> generic bottle-like product

MVP 15.9 adds strict sub-intent-specific rules:
- portable cooler must contain cooler/fridge/ice chest/insulated
- soccer jersey must contain soccer/football/fifa/team/jersey
- water bottle must contain water bottle/hydration bottle/tumbler/canteen
- kitchen towels must contain kitchen/dish/tea/flour sack + towel/cloth
- trash can must contain trash/garbage/waste/bin/can and must not be only trash bags

Outputs:
--------
outputs/mission_strict_repair_slate.csv
outputs/mission_strict_repair_summary.csv
outputs/mission_strict_repair_rejections.csv
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Dict, List, Tuple

import pandas as pd

from src.mission_repair_quality_guardrails import build_quality_guarded_repair_slate


OUTPUT_DIR = "outputs"

STRICT_REPAIR_SLATE_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "mission_strict_repair_slate.csv",
)
STRICT_REPAIR_SUMMARY_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "mission_strict_repair_summary.csv",
)
STRICT_REPAIR_REJECTIONS_OUTPUT = os.path.join(
    OUTPUT_DIR,
    "mission_strict_repair_rejections.csv",
)


@dataclass
class StrictRepairSummary:
    query: str
    input_quality_slate_size: int
    input_repair_rows: int
    strict_accepted_repair_rows: int
    strict_rejected_repair_rows: int
    final_strict_slate_size: int
    accepted_repaired_sub_intents: str
    rejected_repaired_sub_intents: str
    final_strict_decision: str
    explanation: str


def ensure_output_dir() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def normalize_text(value: object) -> str:
    text = "" if pd.isna(value) else str(value)
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def has_any(text: str, keywords: List[str]) -> bool:
    text = normalize_text(text)
    return any(normalize_text(keyword) in text for keyword in keywords)


def has_all_groups(text: str, groups: List[List[str]]) -> bool:
    return all(has_any(text, group) for group in groups)


def strict_rule_for_sub_intent(sub_intent: str, title: str) -> Tuple[bool, str]:
    sub_intent_norm = normalize_text(sub_intent)
    title_norm = normalize_text(title)

    # Soccer jersey must be truly soccer/team/jersey-related.
    if sub_intent_norm == "soccer jersey":
        bad = [
            "dress",
            "sari",
            "saree",
            "sandals",
            "socks",
            "watch band",
            "hat",
            "cap",
            "sunglasses",
            "wedding",
            "floral",
        ]
        if has_any(title_norm, bad):
            return False, "Strict reject: soccer jersey repair contains apparel/accessory terms that are not jersey-specific."

        allowed_groups = [
            ["soccer", "football", "fifa", "team", "club", "sports", "athletic"],
            ["jersey", "shirt", "tee", "t-shirt"],
        ]

        if has_all_groups(title_norm, allowed_groups):
            return True, "Strict accept: soccer jersey repair contains team/soccer context and shirt/jersey context."

        if has_any(title_norm, ["jersey"]) and has_any(title_norm, ["sports", "team", "soccer", "football"]):
            return True, "Strict accept: jersey repair has sports/team context."

        return False, "Strict reject: soccer jersey repair does not clearly contain soccer/team/jersey context."

    # Cooler must be truly cooler/fridge/insulated/ice-chest related.
    if sub_intent_norm in {"cooler", "portable cooler"}:
        bad = [
            "air conditioner",
            "air cooler fan",
            "microphone",
            "power inverter",
            "tablet",
            "ssd",
            "candle",
            "wax",
            "mug",
            "socks",
            "pants",
            "shirt",
            "dress",
            "chapstick",
            "lipstick",
            "speaker",
        ]
        if has_any(title_norm, bad):
            return False, "Strict reject: cooler repair contains non-cooler product terms."

        required = [
            "cooler",
            "ice chest",
            "insulated",
            "cooler bag",
            "fridge",
            "refrigerator",
            "thermoelectric",
            "can cooler",
        ]

        if has_any(title_norm, required):
            return True, "Strict accept: cooler repair contains cooler/insulated/fridge terms."

        return False, "Strict reject: cooler repair does not contain cooler/insulated/fridge terms."

    # Water bottle must be hydration container, not just any bottle/bag.
    if sub_intent_norm == "water bottle":
        bad = [
            "p-trap",
            "sink",
            "diffuser",
            "humidifier",
            "toiletry",
            "bag",
            "backpack",
            "protein skimmer",
            "window film",
            "cooler bag",
        ]
        if has_any(title_norm, bad) and not has_any(title_norm, ["water bottle", "hydration bottle", "tumbler"]):
            return False, "Strict reject: water bottle repair appears to be unrelated container/bag/plumbing product."

        required = [
            "water bottle",
            "hydration bottle",
            "hydration pack",
            "tumbler",
            "canteen",
            "bottle",
            "blender bottle",
            "water jug",
        ]

        if has_any(title_norm, required) and has_any(title_norm, ["water", "hydration", "bottle", "tumbler", "jug"]):
            return True, "Strict accept: water bottle repair contains hydration/bottle terms."

        return False, "Strict reject: water bottle repair does not clearly indicate hydration bottle."

    # Kitchen towels must be actual towel/cloth items.
    if sub_intent_norm == "kitchen towels":
        bad = [
            "file holder",
            "wall file",
            "office",
            "trash can",
            "sink",
            "coffee canister",
            "rack",
            "paper towel holder",
        ]
        if has_any(title_norm, bad):
            return False, "Strict reject: kitchen towel repair contains non-towel product terms."

        towel_terms = ["towel", "towels", "dishcloth", "dish cloth", "tea towel", "flour sack", "cloth"]
        kitchen_context = ["kitchen", "dish", "tea", "flour sack", "cotton", "cleaning"]

        if has_any(title_norm, towel_terms) and has_any(title_norm, kitchen_context):
            return True, "Strict accept: kitchen towel repair contains towel/cloth and kitchen/dish context."

        return False, "Strict reject: kitchen towel repair does not clearly indicate kitchen towel or dish cloth."

    # Trash can must be actual bin/can, not only bags.
    if sub_intent_norm == "trash can":
        if has_any(title_norm, ["trash bags", "garbage bags", "bags"]) and not has_any(
            title_norm,
            ["trash can", "garbage can", "wastebasket", "waste bin", "garbage bin"],
        ):
            return False, "Strict reject: trash can repair appears to be trash bags only."

        required = [
            "trash can",
            "garbage can",
            "wastebasket",
            "waste basket",
            "waste bin",
            "garbage bin",
            "step trash can",
        ]

        if has_any(title_norm, required):
            return True, "Strict accept: trash can repair contains trash/garbage/waste bin terms."

        return False, "Strict reject: trash can repair does not clearly indicate a trash can or waste bin."

    # Dish drying rack must be actual drying rack.
    if sub_intent_norm == "dish drying rack":
        bad = [
            "shoe rack",
            "wig",
            "knife holder",
            "sink single bowl",
            "trash can",
            "baking mat",
            "file holder",
        ]
        if has_any(title_norm, bad):
            return False, "Strict reject: dish drying rack repair contains unrelated rack/household terms."

        required_groups = [
            ["dish", "dishes"],
            ["drying rack", "dish rack", "drainboard", "roll up rack"],
        ]

        if has_all_groups(title_norm, required_groups):
            return True, "Strict accept: dish drying rack repair contains dish and drying-rack context."

        return False, "Strict reject: dish drying rack repair does not clearly indicate dish drying rack."

    # First aid kit should be medical/emergency kit.
    if sub_intent_norm == "first aid kit":
        bad = ["dress", "skirt", "clothing tape"]
        if has_any(title_norm, bad):
            return False, "Strict reject: first aid repair contains unrelated apparel terms."

        if has_any(title_norm, ["first aid", "medical", "emergency", "trauma", "bandage", "survival kit"]):
            return True, "Strict accept: first aid repair contains medical/emergency terms."

        return False, "Strict reject: first aid repair does not clearly indicate medical/emergency kit."

    # Bug spray should be repellent/insect/mosquito.
    if sub_intent_norm == "bug spray":
        bad = [
            "headlamp",
            "gas mask",
            "patio mat",
            "dry bag",
            "sunshade",
        ]
        if has_any(title_norm, bad) and not has_any(title_norm, ["bug", "insect", "mosquito", "repellent"]):
            return False, "Strict reject: bug spray repair contains unrelated camping accessory terms."

        if has_any(title_norm, ["bug", "insect", "mosquito", "repellent", "deet", "picaridin"]):
            return True, "Strict accept: bug spray repair contains bug/insect/repellent terms."

        return False, "Strict reject: bug spray repair does not clearly indicate repellent."

    # Sandals should be actual sandals/slides/flip-flops.
    if sub_intent_norm == "sandals":
        bad = [
            "watch",
            "shoe stretcher",
            "shoe dryer",
            "arch support inserts",
        ]
        if has_any(title_norm, bad):
            return False, "Strict reject: sandals repair contains accessory/non-sandal footwear terms."

        if has_any(title_norm, ["sandal", "sandals", "slides", "flip flop", "flip-flop", "slipper"]):
            return True, "Strict accept: sandals repair contains sandal/slide/flip-flop terms."

        return False, "Strict reject: sandals repair does not clearly indicate sandals."

    # Default: keep previous MVP 15.8 decision.
    return True, "Strict rule not registered for this sub-intent; preserving MVP 15.8 quality decision."


def apply_strict_repair_rules(quality_slate_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if quality_slate_df.empty:
        return quality_slate_df.copy(), pd.DataFrame()

    accepted_rows = []
    rejected_rows = []

    for _, row in quality_slate_df.iterrows():
        row_dict = row.to_dict()
        slate_source = str(row.get("slate_source", ""))

        if slate_source != "repair_loop":
            row_dict["strict_repair_accepted"] = True
            row_dict["strict_repair_reason"] = "Original guarded row; strict repair rule not needed."
            accepted_rows.append(row_dict)
            continue

        sub_intent = str(row.get("sub_intent", ""))
        title = str(row.get("product_title", ""))

        accepted, reason = strict_rule_for_sub_intent(sub_intent, title)

        row_dict["strict_repair_accepted"] = accepted
        row_dict["strict_repair_reason"] = reason

        if accepted:
            accepted_rows.append(row_dict)
        else:
            rejected_rows.append(row_dict)

    accepted_df = pd.DataFrame(accepted_rows)
    rejected_df = pd.DataFrame(rejected_rows)

    if not accepted_df.empty:
        accepted_df = accepted_df.reset_index(drop=True)
        accepted_df["final_strict_rank"] = range(1, len(accepted_df) + 1)

    return accepted_df, rejected_df


def apply_strict_candidate_rules(candidates_df: pd.DataFrame) -> pd.DataFrame:
    if candidates_df.empty:
        return candidates_df.copy()

    rows = []

    for _, row in candidates_df.iterrows():
        row_dict = row.to_dict()
        sub_intent = str(row.get("sub_intent", ""))
        title = str(row.get("product_title", ""))

        accepted, reason = strict_rule_for_sub_intent(sub_intent, title)

        row_dict["strict_repair_accepted"] = accepted
        row_dict["strict_repair_reason"] = reason

        rows.append(row_dict)

    return pd.DataFrame(rows)


def build_strict_repair_slate(
    query: str,
    slate_size: int = 10,
    max_repairs: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ensure_output_dir()

    quality_slate_df, quality_summary_df, quality_candidates_df = build_quality_guarded_repair_slate(
        query=query,
        slate_size=slate_size,
        max_repairs=max_repairs,
    )

    strict_slate_df, strict_rejections_df = apply_strict_repair_rules(quality_slate_df)
    strict_candidates_df = apply_strict_candidate_rules(quality_candidates_df)

    if quality_slate_df.empty:
        summary = StrictRepairSummary(
            query=query,
            input_quality_slate_size=0,
            input_repair_rows=0,
            strict_accepted_repair_rows=0,
            strict_rejected_repair_rows=0,
            final_strict_slate_size=0,
            accepted_repaired_sub_intents="[]",
            rejected_repaired_sub_intents="[]",
            final_strict_decision="no_quality_slate",
            explanation="No quality-guarded repair slate was generated.",
        )
        return strict_slate_df, pd.DataFrame([asdict(summary)]), strict_rejections_df

    repair_rows = quality_slate_df[
        quality_slate_df["slate_source"] == "repair_loop"
    ].copy()

    accepted_repair_rows = strict_slate_df[
        strict_slate_df.get("slate_source", pd.Series(dtype=str)) == "repair_loop"
    ].copy() if not strict_slate_df.empty else pd.DataFrame()

    rejected_repair_rows = strict_rejections_df.copy()

    accepted_sub_intents = (
        accepted_repair_rows["sub_intent"].astype(str).drop_duplicates().tolist()
        if not accepted_repair_rows.empty
        else []
    )

    rejected_sub_intents = (
        rejected_repair_rows["sub_intent"].astype(str).drop_duplicates().tolist()
        if not rejected_repair_rows.empty
        else []
    )

    input_repair_count = len(repair_rows)
    accepted_count = len(accepted_repair_rows)
    rejected_count = len(rejected_repair_rows)

    if input_repair_count == 0:
        decision = "accept_original_guarded_slate"
    elif rejected_count == 0:
        decision = "accept_strict_repaired_slate"
    elif accepted_count > 0:
        decision = "accept_with_strict_repair_gaps"
    else:
        decision = "reject_all_repairs_keep_original_guarded_slate"

    explanation = (
        f"Strict repair rules reviewed {input_repair_count} repaired rows for '{query}'. "
        f"Accepted {accepted_count}; rejected {rejected_count}. "
        f"Accepted sub-intents: {accepted_sub_intents if accepted_sub_intents else 'none'}. "
        f"Rejected sub-intents: {rejected_sub_intents if rejected_sub_intents else 'none'}."
    )

    summary = StrictRepairSummary(
        query=query,
        input_quality_slate_size=len(quality_slate_df),
        input_repair_rows=input_repair_count,
        strict_accepted_repair_rows=accepted_count,
        strict_rejected_repair_rows=rejected_count,
        final_strict_slate_size=len(strict_slate_df),
        accepted_repaired_sub_intents=json.dumps(accepted_sub_intents),
        rejected_repaired_sub_intents=json.dumps(rejected_sub_intents),
        final_strict_decision=decision,
        explanation=explanation,
    )

    if not strict_candidates_df.empty:
        strict_candidates_df.to_csv(QUALITY_CANDIDATE_TEMP_PATH(), index=False)

    return strict_slate_df, pd.DataFrame([asdict(summary)]), strict_rejections_df


def QUALITY_CANDIDATE_TEMP_PATH() -> str:
    return os.path.join(OUTPUT_DIR, "mission_strict_repair_candidates.csv")


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
    all_rejections = []
    all_candidates = []

    for query in demo_queries:
        quality_slate_df, quality_summary_df, quality_candidates_df = build_quality_guarded_repair_slate(
            query=query,
            slate_size=10,
            max_repairs=5,
        )

        strict_slate_df, strict_rejections_df = apply_strict_repair_rules(quality_slate_df)
        strict_candidates_df = apply_strict_candidate_rules(quality_candidates_df)

        repair_rows = quality_slate_df[
            quality_slate_df["slate_source"] == "repair_loop"
        ].copy() if not quality_slate_df.empty else pd.DataFrame()

        accepted_repair_rows = strict_slate_df[
            strict_slate_df.get("slate_source", pd.Series(dtype=str)) == "repair_loop"
        ].copy() if not strict_slate_df.empty else pd.DataFrame()

        accepted_sub_intents = (
            accepted_repair_rows["sub_intent"].astype(str).drop_duplicates().tolist()
            if not accepted_repair_rows.empty
            else []
        )

        rejected_sub_intents = (
            strict_rejections_df["sub_intent"].astype(str).drop_duplicates().tolist()
            if not strict_rejections_df.empty
            else []
        )

        input_repair_count = len(repair_rows)
        accepted_count = len(accepted_repair_rows)
        rejected_count = len(strict_rejections_df)

        if input_repair_count == 0:
            decision = "accept_original_guarded_slate"
        elif rejected_count == 0:
            decision = "accept_strict_repaired_slate"
        elif accepted_count > 0:
            decision = "accept_with_strict_repair_gaps"
        else:
            decision = "reject_all_repairs_keep_original_guarded_slate"

        summary = StrictRepairSummary(
            query=query,
            input_quality_slate_size=len(quality_slate_df),
            input_repair_rows=input_repair_count,
            strict_accepted_repair_rows=accepted_count,
            strict_rejected_repair_rows=rejected_count,
            final_strict_slate_size=len(strict_slate_df),
            accepted_repaired_sub_intents=json.dumps(accepted_sub_intents),
            rejected_repaired_sub_intents=json.dumps(rejected_sub_intents),
            final_strict_decision=decision,
            explanation=(
                f"Strict repair rules reviewed {input_repair_count} repaired rows for '{query}'. "
                f"Accepted {accepted_count}; rejected {rejected_count}. "
                f"Accepted sub-intents: {accepted_sub_intents if accepted_sub_intents else 'none'}. "
                f"Rejected sub-intents: {rejected_sub_intents if rejected_sub_intents else 'none'}."
            ),
        )

        if not strict_slate_df.empty:
            all_slates.append(strict_slate_df)

        all_summaries.append(pd.DataFrame([asdict(summary)]))

        if not strict_rejections_df.empty:
            all_rejections.append(strict_rejections_df)

        if not strict_candidates_df.empty:
            all_candidates.append(strict_candidates_df)

    final_slate_df = (
        pd.concat(all_slates, ignore_index=True)
        if all_slates
        else pd.DataFrame()
    )
    final_summary_df = pd.concat(all_summaries, ignore_index=True)
    final_rejections_df = (
        pd.concat(all_rejections, ignore_index=True)
        if all_rejections
        else pd.DataFrame()
    )
    final_candidates_df = (
        pd.concat(all_candidates, ignore_index=True)
        if all_candidates
        else pd.DataFrame()
    )

    final_slate_df.to_csv(STRICT_REPAIR_SLATE_OUTPUT, index=False)
    final_summary_df.to_csv(STRICT_REPAIR_SUMMARY_OUTPUT, index=False)
    final_rejections_df.to_csv(STRICT_REPAIR_REJECTIONS_OUTPUT, index=False)
    final_candidates_df.to_csv(QUALITY_CANDIDATE_TEMP_PATH(), index=False)

    print()
    print("MVP 15.9 Strict Compound-Intent Repair Rules Demo")
    print("=" * 100)

    print()
    print("Strict Repair Summary")
    print("-" * 100)
    print(final_summary_df.to_string(index=False))

    if not final_slate_df.empty:
        print()
        print("Strict Repaired Slate Sample")
        print("-" * 100)
        cols = [
            "query",
            "final_strict_rank",
            "slate_source",
            "sub_intent",
            "sub_intent_role",
            "product_title",
            "strict_repair_reason",
        ]
        print(final_slate_df[cols].head(40).to_string(index=False))

    if not final_rejections_df.empty:
        print()
        print("Strict Repair Rejections")
        print("-" * 100)
        cols = [
            "query",
            "slate_source",
            "sub_intent",
            "sub_intent_role",
            "product_title",
            "strict_repair_reason",
        ]
        print(final_rejections_df[cols].head(40).to_string(index=False))

    print()
    print("Files written")
    print("-" * 100)
    print(f"- strict repaired slate: {STRICT_REPAIR_SLATE_OUTPUT}")
    print(f"- strict summary: {STRICT_REPAIR_SUMMARY_OUTPUT}")
    print(f"- strict rejections: {STRICT_REPAIR_REJECTIONS_OUTPUT}")
    print(f"- strict candidates: {QUALITY_CANDIDATE_TEMP_PATH()}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Mission query to strict-check after repair.",
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
        slate, summary, rejections = build_strict_repair_slate(
            query=args.query,
            slate_size=args.slate_size,
            max_repairs=args.max_repairs,
        )

        print()
        print("Strict Repair Summary")
        print("-" * 100)
        print(summary.to_string(index=False))

        if slate.empty:
            print()
            print("No strict repaired slate generated.")
        else:
            print()
            print("Strict Repaired Slate")
            print("-" * 100)
            print(
                slate[
                    [
                        "final_strict_rank",
                        "slate_source",
                        "sub_intent",
                        "sub_intent_role",
                        "product_title",
                        "strict_repair_reason",
                    ]
                ].to_string(index=False)
            )

        if not rejections.empty:
            print()
            print("Strict Repair Rejections")
            print("-" * 100)
            print(
                rejections[
                    [
                        "slate_source",
                        "sub_intent",
                        "sub_intent_role",
                        "product_title",
                        "strict_repair_reason",
                    ]
                ].to_string(index=False)
            )
    else:
        run_demo()