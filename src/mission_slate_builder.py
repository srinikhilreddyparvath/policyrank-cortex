"""
MVP 15.2: Mission Slate Builder

Purpose:
--------
Build a mission-aware product slate from a mission-based shopping query.

Input:
------
A shopping query such as:
- world cup watch party
- camping trip essentials
- beach vacation packing list
- new apartment kitchen setup

What it does:
-------------
1. Uses src.mission_agent to detect/decompose mission queries.
2. Loads the local product/sample dataset.
3. Retrieves candidate products for each mission sub-intent.
4. Selects a diverse slate across sub-intents.
5. Produces coverage and quality summaries.

Outputs:
--------
outputs/mission_slate_builder_demo.csv
outputs/mission_slate_builder_summary.csv
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Tuple

import pandas as pd

from src.mission_agent import analyze_mission_query


OUTPUT_DIR = "outputs"
MISSION_SLATE_OUTPUT = os.path.join(OUTPUT_DIR, "mission_slate_builder_demo.csv")
MISSION_SUMMARY_OUTPUT = os.path.join(OUTPUT_DIR, "mission_slate_builder_summary.csv")


CANDIDATE_DATA_PATHS = [
    "data/esci_balanced_sample.csv",
    "data/esci_sample_products.csv",
    "data/sample_products.csv",
]


TEXT_COLUMNS_CANDIDATES = [
    "product_title",
    "title",
    "product_description",
    "description",
    "product_bullet_point",
    "bullet_point",
    "product_brand",
    "brand",
    "product_color",
    "color",
    "product_locale",
    "product_type",
    "category",
]


@dataclass
class MissionSlateRow:
    query: str
    mission_type: str
    mission_name: str
    mission_confidence: float
    sub_intent: str
    sub_intent_priority: float
    sub_intent_role: str
    selected_rank: int
    product_id: str
    product_title: str
    product_brand: str
    product_text_score: float
    product_role_score: float
    final_mission_score: float
    retrieval_reason: str


@dataclass
class MissionSlateSummary:
    query: str
    is_mission: bool
    mission_type: str
    mission_name: str
    mission_confidence: float
    requested_sub_intent_count: int
    covered_sub_intent_count: int
    missing_sub_intent_count: int
    slate_size: int
    mission_coverage_score: float
    avg_final_mission_score: float
    missing_sub_intents: str
    reasoning: str


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
    if not normalized:
        return []

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

    return [token for token in normalized.split() if token not in stopwords and len(token) >= 2]


def load_product_data() -> Tuple[pd.DataFrame, str]:
    for path in CANDIDATE_DATA_PATHS:
        if os.path.exists(path):
            df = pd.read_csv(path)
            if len(df) > 0:
                return df, path

    raise FileNotFoundError(
        "No product dataset found. Expected one of: "
        + ", ".join(CANDIDATE_DATA_PATHS)
    )


def pick_column(df: pd.DataFrame, candidates: List[str], fallback: Optional[str] = None) -> Optional[str]:
    for col in candidates:
        if col in df.columns:
            return col

    return fallback


def prepare_product_dataframe(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = raw_df.copy()

    product_id_col = pick_column(
        df,
        ["product_id", "item_id", "id", "asin"],
    )
    title_col = pick_column(
        df,
        ["product_title", "title", "item_title", "name"],
    )
    brand_col = pick_column(
        df,
        ["product_brand", "brand", "manufacturer"],
    )

    if product_id_col is None:
        df["__product_id"] = [f"row_{i}" for i in range(len(df))]
        product_id_col = "__product_id"

    if title_col is None:
        raise ValueError(
            "Could not find a product title column. Expected one of: "
            "product_title, title, item_title, name"
        )

    if brand_col is None:
        df["__brand"] = ""
        brand_col = "__brand"

    available_text_cols = [col for col in TEXT_COLUMNS_CANDIDATES if col in df.columns]

    if title_col not in available_text_cols:
        available_text_cols.insert(0, title_col)

    def build_search_text(row: pd.Series) -> str:
        parts = []
        for col in available_text_cols:
            value = row.get(col, "")
            if not pd.isna(value):
                parts.append(str(value))
        return normalize_text(" ".join(parts))

    out = pd.DataFrame()
    out["product_id"] = df[product_id_col].astype(str)
    out["product_title"] = df[title_col].astype(str)
    out["product_brand"] = df[brand_col].fillna("").astype(str)
    out["search_text"] = df.apply(build_search_text, axis=1)

    out = out.drop_duplicates(subset=["product_id", "product_title"]).reset_index(drop=True)

    return out


def calculate_text_score(query: str, product_text: str) -> float:
    query_tokens = tokenize(query)
    product_tokens = set(tokenize(product_text))

    if not query_tokens or not product_tokens:
        return 0.0

    exact_matches = sum(1 for token in query_tokens if token in product_tokens)
    partial_matches = 0

    for token in query_tokens:
        if token in product_tokens:
            continue

        for product_token in product_tokens:
            if token in product_token or product_token in token:
                partial_matches += 0.5
                break

    raw_score = exact_matches + partial_matches
    normalized = raw_score / max(len(query_tokens), 1)

    return round(min(normalized, 1.0), 4)


def role_boost(role: str, product_text: str) -> float:
    role = normalize_text(role)
    product_text = normalize_text(product_text)

    role_keywords = {
        "food": ["snack", "chips", "popcorn", "cookies", "candy", "food"],
        "beverage": ["drink", "soda", "water", "juice", "gatorade", "beverage"],
        "serveware": ["plate", "plates", "cup", "cups", "napkin", "fork", "spoon", "serveware"],
        "decor": ["decor", "decoration", "banner", "balloon", "party", "soccer"],
        "apparel": ["shirt", "jersey", "dress", "wear", "apparel", "clothing"],
        "utility": ["cooler", "storage", "bag", "container", "utility"],
        "entertainment": ["speaker", "bluetooth", "audio", "music", "game"],
        "shelter": ["tent", "canopy", "shelter"],
        "sleep": ["sleeping", "blanket", "pillow", "air mattress"],
        "lighting": ["lantern", "light", "flashlight"],
        "safety": ["first aid", "bandage", "safety", "medical"],
        "protection": ["sunscreen", "bug", "spray", "repellent", "protection"],
        "hydration": ["water", "bottle", "hydration"],
        "comfort": ["chair", "towel", "blanket", "comfort"],
        "storage": ["bag", "storage", "container", "luggage"],
        "footwear": ["shoe", "shoes", "sandals", "boots"],
        "cooking": ["cookware", "pan", "pot", "skillet", "cooking"],
        "prep": ["knife", "cutting", "board", "prep"],
        "cleaning": ["clean", "towel", "rack", "dish", "trash"],
    }

    keywords = role_keywords.get(role, [])
    if not keywords:
        return 0.0

    hits = sum(1 for keyword in keywords if keyword in product_text)

    return round(min(hits * 0.15, 0.45), 4)


def retrieve_for_sub_intent(
    products_df: pd.DataFrame,
    sub_intent_name: str,
    sub_intent_role: str,
    top_n: int = 20,
) -> pd.DataFrame:
    rows = []

    for _, product in products_df.iterrows():
        text_score = calculate_text_score(sub_intent_name, product["search_text"])
        product_role_boost = role_boost(sub_intent_role, product["search_text"])
        final_score = round((0.75 * text_score) + product_role_boost, 4)

        if final_score <= 0:
            continue

        rows.append(
            {
                "product_id": product["product_id"],
                "product_title": product["product_title"],
                "product_brand": product["product_brand"],
                "search_text": product["search_text"],
                "product_text_score": text_score,
                "product_role_score": product_role_boost,
                "candidate_score": final_score,
            }
        )

    if not rows:
        return pd.DataFrame()

    candidate_df = pd.DataFrame(rows)

    candidate_df = candidate_df.sort_values(
        by=["candidate_score", "product_text_score", "product_role_score"],
        ascending=[False, False, False],
    ).head(top_n)

    return candidate_df.reset_index(drop=True)


def build_mission_slate(
    query: str,
    slate_size: int = 10,
    max_per_sub_intent: int = 2,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    ensure_output_dir()

    mission = analyze_mission_query(query)
    raw_products, data_path = load_product_data()
    products = prepare_product_dataframe(raw_products)

    slate_rows: List[MissionSlateRow] = []
    used_product_ids = set()
    covered_sub_intents = set()
    missing_sub_intents = []

    if not mission.is_mission:
        summary = MissionSlateSummary(
            query=query,
            is_mission=False,
            mission_type=mission.mission_type,
            mission_name=mission.mission_name,
            mission_confidence=mission.confidence,
            requested_sub_intent_count=0,
            covered_sub_intent_count=0,
            missing_sub_intent_count=0,
            slate_size=0,
            mission_coverage_score=0.0,
            avg_final_mission_score=0.0,
            missing_sub_intents="",
            reasoning=mission.reasoning,
        )

        return pd.DataFrame(), pd.DataFrame([asdict(summary)])

    sub_intents = sorted(
        mission.sub_intents,
        key=lambda item: float(item.get("priority", 0.0)),
        reverse=True,
    )

    selected_rank = 1

    for sub_intent in sub_intents:
        if selected_rank > slate_size:
            break

        sub_intent_name = str(sub_intent.get("name", ""))
        sub_intent_priority = float(sub_intent.get("priority", 0.0))
        sub_intent_role = str(sub_intent.get("role", ""))

        candidate_df = retrieve_for_sub_intent(
            products_df=products,
            sub_intent_name=sub_intent_name,
            sub_intent_role=sub_intent_role,
            top_n=25,
        )

        if candidate_df.empty:
            missing_sub_intents.append(sub_intent_name)
            continue

        picks_for_this_intent = 0

        for _, candidate in candidate_df.iterrows():
            if selected_rank > slate_size:
                break

            product_id = str(candidate["product_id"])

            if product_id in used_product_ids:
                continue

            used_product_ids.add(product_id)
            covered_sub_intents.add(sub_intent_name)
            picks_for_this_intent += 1

            final_mission_score = round(
                (0.70 * float(candidate["candidate_score"]))
                + (0.30 * sub_intent_priority),
                4,
            )

            slate_rows.append(
                MissionSlateRow(
                    query=query,
                    mission_type=mission.mission_type,
                    mission_name=mission.mission_name,
                    mission_confidence=mission.confidence,
                    sub_intent=sub_intent_name,
                    sub_intent_priority=sub_intent_priority,
                    sub_intent_role=sub_intent_role,
                    selected_rank=selected_rank,
                    product_id=product_id,
                    product_title=str(candidate["product_title"]),
                    product_brand=str(candidate["product_brand"]),
                    product_text_score=float(candidate["product_text_score"]),
                    product_role_score=float(candidate["product_role_score"]),
                    final_mission_score=final_mission_score,
                    retrieval_reason=(
                        f"Selected for sub-intent '{sub_intent_name}' "
                        f"with role '{sub_intent_role}' using dataset {data_path}."
                    ),
                )
            )

            selected_rank += 1

            if picks_for_this_intent >= max_per_sub_intent:
                break

        if picks_for_this_intent == 0:
            missing_sub_intents.append(sub_intent_name)

    slate_df = pd.DataFrame([asdict(row) for row in slate_rows])

    requested_count = len(sub_intents)
    covered_count = len(covered_sub_intents)
    missing_count = max(requested_count - covered_count, 0)
    coverage_score = round(covered_count / requested_count, 4) if requested_count else 0.0
    avg_score = (
        round(float(slate_df["final_mission_score"].mean()), 4)
        if not slate_df.empty
        else 0.0
    )

    missing = [
        str(item.get("name", ""))
        for item in sub_intents
        if str(item.get("name", "")) not in covered_sub_intents
    ]

    summary = MissionSlateSummary(
        query=query,
        is_mission=mission.is_mission,
        mission_type=mission.mission_type,
        mission_name=mission.mission_name,
        mission_confidence=mission.confidence,
        requested_sub_intent_count=requested_count,
        covered_sub_intent_count=covered_count,
        missing_sub_intent_count=missing_count,
        slate_size=len(slate_df),
        mission_coverage_score=coverage_score,
        avg_final_mission_score=avg_score,
        missing_sub_intents=json.dumps(missing),
        reasoning=mission.reasoning,
    )

    summary_df = pd.DataFrame([asdict(summary)])

    return slate_df, summary_df


def run_demo() -> None:
    ensure_output_dir()

    demo_queries = [
        "world cup watch party",
        "camping trip essentials",
        "new apartment kitchen setup",
        "beach vacation packing list",
    ]

    all_slate_rows = []
    all_summary_rows = []

    for query in demo_queries:
        slate_df, summary_df = build_mission_slate(query=query, slate_size=10)

        if not slate_df.empty:
            all_slate_rows.append(slate_df)

        all_summary_rows.append(summary_df)

    final_slate_df = (
        pd.concat(all_slate_rows, ignore_index=True)
        if all_slate_rows
        else pd.DataFrame()
    )
    final_summary_df = pd.concat(all_summary_rows, ignore_index=True)

    final_slate_df.to_csv(MISSION_SLATE_OUTPUT, index=False)
    final_summary_df.to_csv(MISSION_SUMMARY_OUTPUT, index=False)

    print()
    print("MVP 15.2 Mission Slate Builder Demo")
    print("=" * 100)

    print()
    print("Mission Summary")
    print("-" * 100)
    print(final_summary_df.to_string(index=False))

    if not final_slate_df.empty:
        print()
        print("Mission Slate Sample")
        print("-" * 100)
        print(
            final_slate_df[
                [
                    "query",
                    "selected_rank",
                    "sub_intent",
                    "sub_intent_role",
                    "product_title",
                    "final_mission_score",
                ]
            ]
            .head(30)
            .to_string(index=False)
        )

    print()
    print("Files written")
    print("-" * 100)
    print(f"- mission slate: {MISSION_SLATE_OUTPUT}")
    print(f"- mission summary: {MISSION_SUMMARY_OUTPUT}")


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--query",
        type=str,
        default=None,
        help="Mission query to build a slate for.",
    )

    parser.add_argument(
        "--slate-size",
        type=int,
        default=10,
        help="Maximum slate size.",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    if args.query:
        slate, summary = build_mission_slate(
            query=args.query,
            slate_size=args.slate_size,
        )

        print()
        print("Mission Summary")
        print("-" * 100)
        print(summary.to_string(index=False))

        if slate.empty:
            print()
            print("No mission slate generated.")
        else:
            print()
            print("Mission Slate")
            print("-" * 100)
            print(
                slate[
                    [
                        "selected_rank",
                        "sub_intent",
                        "sub_intent_role",
                        "product_title",
                        "final_mission_score",
                    ]
                ].to_string(index=False)
            )
    else:
        run_demo()