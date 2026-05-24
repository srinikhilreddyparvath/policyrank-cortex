"""
MVP 15.1: Mission-Based Shopping Agent

Purpose:
--------
Detect whether a query is a normal single-product query or a mission-based shopping query.

Examples:
---------
Single-product:
- adidas soccer cleats
- iphone 15 case
- coffee maker

Mission-based:
- world cup watch party
- camping trip essentials
- beach vacation packing list
- birthday party for 5 year old
- new apartment kitchen setup

Outputs:
--------
A structured mission analysis object containing:
- is_mission
- mission_type
- mission_name
- confidence
- sub_intents
- reasoning
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional

import pandas as pd


OUTPUT_DIR = "outputs"
MISSION_ANALYSIS_OUTPUT = os.path.join(OUTPUT_DIR, "mission_agent_analysis.csv")


MISSION_KEYWORDS = {
    "event": [
        "party",
        "birthday",
        "wedding",
        "baby shower",
        "graduation",
        "watch party",
        "super bowl",
        "world cup",
        "game night",
        "movie night",
        "bbq",
        "barbecue",
        "tailgate",
        "thanksgiving",
        "christmas",
        "halloween",
        "easter",
    ],
    "trip": [
        "trip",
        "travel",
        "vacation",
        "beach",
        "camping",
        "hiking",
        "road trip",
        "flight",
        "packing",
        "luggage",
        "backpacking",
    ],
    "setup": [
        "setup",
        "starter kit",
        "essentials",
        "new apartment",
        "dorm",
        "home office",
        "kitchen setup",
        "gym setup",
        "gaming setup",
        "baby registry",
    ],
    "occasion": [
        "date night",
        "interview",
        "first day",
        "school",
        "back to school",
        "workout",
        "festival",
        "concert",
    ],
    "bundle": [
        "bundle",
        "kit",
        "pack",
        "everything i need",
        "must haves",
        "essentials",
        "supplies",
    ],
}


MISSION_TEMPLATES = {
    "world cup watch party": [
        {"name": "snacks", "priority": 0.95, "role": "food"},
        {"name": "drinks", "priority": 0.90, "role": "beverage"},
        {"name": "party plates", "priority": 0.80, "role": "serveware"},
        {"name": "party cups", "priority": 0.75, "role": "serveware"},
        {"name": "soccer decorations", "priority": 0.72, "role": "decor"},
        {"name": "soccer jersey", "priority": 0.65, "role": "apparel"},
        {"name": "cooler", "priority": 0.55, "role": "utility"},
        {"name": "portable speaker", "priority": 0.50, "role": "entertainment"},
    ],
    "camping trip essentials": [
        {"name": "tent", "priority": 0.95, "role": "shelter"},
        {"name": "sleeping bag", "priority": 0.92, "role": "sleep"},
        {"name": "camping lantern", "priority": 0.85, "role": "lighting"},
        {"name": "cooler", "priority": 0.80, "role": "food storage"},
        {"name": "camping chair", "priority": 0.75, "role": "comfort"},
        {"name": "first aid kit", "priority": 0.72, "role": "safety"},
        {"name": "bug spray", "priority": 0.65, "role": "protection"},
        {"name": "water bottle", "priority": 0.60, "role": "hydration"},
    ],
    "beach vacation packing list": [
        {"name": "sunscreen", "priority": 0.95, "role": "protection"},
        {"name": "beach towel", "priority": 0.90, "role": "comfort"},
        {"name": "swimsuit", "priority": 0.88, "role": "apparel"},
        {"name": "sunglasses", "priority": 0.80, "role": "accessory"},
        {"name": "beach bag", "priority": 0.78, "role": "storage"},
        {"name": "sandals", "priority": 0.72, "role": "footwear"},
        {"name": "water bottle", "priority": 0.68, "role": "hydration"},
        {"name": "portable cooler", "priority": 0.60, "role": "food storage"},
    ],
    "birthday party": [
        {"name": "birthday decorations", "priority": 0.95, "role": "decor"},
        {"name": "party plates", "priority": 0.85, "role": "serveware"},
        {"name": "party cups", "priority": 0.82, "role": "serveware"},
        {"name": "balloons", "priority": 0.80, "role": "decor"},
        {"name": "birthday candles", "priority": 0.76, "role": "cake"},
        {"name": "party favors", "priority": 0.70, "role": "gifts"},
        {"name": "snacks", "priority": 0.65, "role": "food"},
        {"name": "gift wrap", "priority": 0.55, "role": "packaging"},
    ],
    "new apartment kitchen setup": [
        {"name": "dinnerware set", "priority": 0.95, "role": "serveware"},
        {"name": "cookware set", "priority": 0.92, "role": "cooking"},
        {"name": "knife set", "priority": 0.88, "role": "prep"},
        {"name": "cutting board", "priority": 0.82, "role": "prep"},
        {"name": "food storage containers", "priority": 0.75, "role": "storage"},
        {"name": "dish drying rack", "priority": 0.68, "role": "cleaning"},
        {"name": "trash can", "priority": 0.62, "role": "utility"},
        {"name": "kitchen towels", "priority": 0.58, "role": "cleaning"},
    ],
}


GENERIC_MISSION_SUB_INTENTS = {
    "event": [
        {"name": "decorations", "priority": 0.85, "role": "decor"},
        {"name": "plates and cups", "priority": 0.78, "role": "serveware"},
        {"name": "snacks", "priority": 0.72, "role": "food"},
        {"name": "drinks", "priority": 0.70, "role": "beverage"},
        {"name": "party favors", "priority": 0.55, "role": "extras"},
    ],
    "trip": [
        {"name": "bag or luggage", "priority": 0.85, "role": "storage"},
        {"name": "travel accessories", "priority": 0.78, "role": "utility"},
        {"name": "personal care", "priority": 0.72, "role": "care"},
        {"name": "weather protection", "priority": 0.65, "role": "protection"},
        {"name": "snacks and hydration", "priority": 0.58, "role": "food"},
    ],
    "setup": [
        {"name": "starter essentials", "priority": 0.88, "role": "core"},
        {"name": "storage", "priority": 0.72, "role": "organization"},
        {"name": "cleaning supplies", "priority": 0.65, "role": "maintenance"},
        {"name": "comfort items", "priority": 0.58, "role": "comfort"},
        {"name": "accessories", "priority": 0.50, "role": "extras"},
    ],
    "occasion": [
        {"name": "outfit", "priority": 0.82, "role": "apparel"},
        {"name": "accessories", "priority": 0.70, "role": "accessory"},
        {"name": "personal care", "priority": 0.62, "role": "care"},
        {"name": "bag", "priority": 0.52, "role": "utility"},
    ],
    "bundle": [
        {"name": "main item", "priority": 0.90, "role": "core"},
        {"name": "supporting accessories", "priority": 0.72, "role": "accessory"},
        {"name": "replacement supplies", "priority": 0.60, "role": "supplies"},
        {"name": "storage", "priority": 0.50, "role": "organization"},
    ],
}


@dataclass
class MissionAnalysis:
    query: str
    is_mission: bool
    mission_type: str
    mission_name: str
    confidence: float
    sub_intents: List[Dict]
    reasoning: str


def normalize_query(query: str) -> str:
    return " ".join(str(query).lower().strip().split())


def find_keyword_matches(query: str) -> Dict[str, List[str]]:
    normalized = normalize_query(query)
    matches: Dict[str, List[str]] = {}

    for mission_type, keywords in MISSION_KEYWORDS.items():
        mission_matches = []

        for keyword in keywords:
            if keyword in normalized:
                mission_matches.append(keyword)

        if mission_matches:
            matches[mission_type] = mission_matches

    return matches


def infer_mission_type(keyword_matches: Dict[str, List[str]]) -> str:
    if not keyword_matches:
        return "single_product"

    ranked_types = sorted(
        keyword_matches.items(),
        key=lambda item: len(item[1]),
        reverse=True,
    )

    return ranked_types[0][0]


def find_template_sub_intents(query: str) -> Optional[List[Dict]]:
    normalized = normalize_query(query)

    for template_name, sub_intents in MISSION_TEMPLATES.items():
        if template_name in normalized:
            return sub_intents

    if "world cup" in normalized or "watch party" in normalized:
        return MISSION_TEMPLATES["world cup watch party"]

    if "camping" in normalized:
        return MISSION_TEMPLATES["camping trip essentials"]

    if "beach" in normalized and (
        "vacation" in normalized or "trip" in normalized or "packing" in normalized
    ):
        return MISSION_TEMPLATES["beach vacation packing list"]

    if "birthday" in normalized and "party" in normalized:
        return MISSION_TEMPLATES["birthday party"]

    if "new apartment" in normalized and "kitchen" in normalized:
        return MISSION_TEMPLATES["new apartment kitchen setup"]

    return None


def calculate_mission_confidence(
    query: str,
    keyword_matches: Dict[str, List[str]],
    template_match_found: bool,
) -> float:
    normalized = normalize_query(query)
    token_count = len(normalized.split())
    total_keyword_matches = sum(len(matches) for matches in keyword_matches.values())

    confidence = 0.10

    if total_keyword_matches > 0:
        confidence += min(0.45, total_keyword_matches * 0.15)

    if template_match_found:
        confidence += 0.35

    if token_count >= 3:
        confidence += 0.10

    mission_phrases = [
        "essentials",
        "setup",
        "packing list",
        "everything i need",
        "must haves",
    ]

    if any(phrase in normalized for phrase in mission_phrases):
        confidence += 0.15

    return round(min(confidence, 0.98), 4)


def analyze_mission_query(query: str) -> MissionAnalysis:
    normalized = normalize_query(query)
    keyword_matches = find_keyword_matches(normalized)
    mission_type = infer_mission_type(keyword_matches)

    template_sub_intents = find_template_sub_intents(normalized)
    template_match_found = template_sub_intents is not None

    confidence = calculate_mission_confidence(
        query=normalized,
        keyword_matches=keyword_matches,
        template_match_found=template_match_found,
    )

    is_mission = confidence >= 0.50 and mission_type != "single_product"

    if is_mission:
        if template_sub_intents is not None:
            sub_intents = template_sub_intents
            reasoning = (
                "Query matched mission keywords and a known mission template. "
                "Using template-based sub-intent decomposition."
            )
        else:
            sub_intents = GENERIC_MISSION_SUB_INTENTS.get(
                mission_type,
                GENERIC_MISSION_SUB_INTENTS["bundle"],
            )
            reasoning = (
                "Query matched mission keywords but no exact template. "
                "Using generic mission decomposition for the inferred mission type."
            )

        mission_name = normalized
    else:
        sub_intents = []
        mission_type = "single_product"
        mission_name = normalized
        confidence = max(0.0, min(confidence, 0.49))
        reasoning = (
            "Query appears to be a single-product search rather than a "
            "mission-based shopping query."
        )

    return MissionAnalysis(
        query=query,
        is_mission=is_mission,
        mission_type=mission_type,
        mission_name=mission_name,
        confidence=confidence,
        sub_intents=sub_intents,
        reasoning=reasoning,
    )


def analyze_queries(queries: List[str]) -> pd.DataFrame:
    rows = []

    for query in queries:
        analysis = analyze_mission_query(query)
        row = asdict(analysis)
        row["sub_intents_json"] = json.dumps(row.pop("sub_intents"))
        row["sub_intent_count"] = len(json.loads(row["sub_intents_json"]))
        rows.append(row)

    return pd.DataFrame(rows)


def run_demo() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    demo_queries = [
        "adidas soccer cleats",
        "iphone 15 case",
        "world cup watch party",
        "camping trip essentials",
        "birthday party for 5 year old",
        "new apartment kitchen setup",
        "beach vacation packing list",
        "coffee maker with grinder",
        "red dress for wedding guest",
        "back to school supplies",
    ]

    df = analyze_queries(demo_queries)
    df.to_csv(MISSION_ANALYSIS_OUTPUT, index=False)

    print()
    print("MVP 15.1 Mission Agent Demo")
    print("=" * 100)
    print(
        df[
            [
                "query",
                "is_mission",
                "mission_type",
                "confidence",
                "sub_intent_count",
                "reasoning",
            ]
        ].to_string(index=False)
    )

    print()
    print("Files written")
    print("-" * 100)
    print(f"- mission analysis: {MISSION_ANALYSIS_OUTPUT}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Optional single query to analyze.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.query:
        result = analyze_mission_query(args.query)
        print(json.dumps(asdict(result), indent=2))
    else:
        run_demo()