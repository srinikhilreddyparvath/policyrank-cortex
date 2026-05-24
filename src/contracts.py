from typing import Dict, List


def infer_intent_from_query(query: str) -> Dict:
    query_lower = query.lower()

    intent = {
        "raw_query": query,
        "detected_category": "general",
        "price_sensitivity": "medium",
        "quality_preference": "medium",
        "important_terms": [],
        "possible_constraints": []
    }

    electronics_terms = [
        "headphones",
        "earbuds",
        "charger",
        "mouse",
        "speaker",
        "bluetooth",
        "wireless",
        "laptop"
    ]

    shoe_terms = [
        "shoes",
        "running",
        "sneakers"
    ]

    budget_terms = [
        "cheap",
        "budget",
        "affordable",
        "low price",
        "lowest price"
    ]

    premium_terms = [
        "best",
        "premium",
        "top rated",
        "high quality",
        "noise cancelling"
    ]

    for term in electronics_terms:
        if term in query_lower:
            intent["detected_category"] = "electronics"
            intent["important_terms"].append(term)

    for term in shoe_terms:
        if term in query_lower:
            intent["detected_category"] = "shoes"
            intent["important_terms"].append(term)

    for term in budget_terms:
        if term in query_lower:
            intent["price_sensitivity"] = "high"
            intent["possible_constraints"].append("prefer_lower_price")

    for term in premium_terms:
        if term in query_lower:
            intent["quality_preference"] = "high"
            intent["possible_constraints"].append("prefer_high_rating")

    return intent


def generate_search_contract(query: str, user_preference: str) -> Dict:
    intent = infer_intent_from_query(query)

    contract = {
        "query": query,
        "intent": intent,
        "ranking_objective": user_preference,
        "ranking_rules": [],
        "reward_signal": "esci_label_based_simulated_reward",
        "safety_guardrails": [
            "do_not_rank_irrelevant_items_high",
            "prefer_items_matching_detected_category",
            "avoid_extreme_price_mismatch"
        ]
    }

    if user_preference == "Most relevant":
        contract["ranking_rules"] = [
            "boost_exact_esci_matches",
            "boost_query_title_matches",
            "penalize_irrelevant_items"
        ]

    elif user_preference == "Best rating":
        contract["ranking_rules"] = [
            "boost_high_rating_items",
            "keep_minimum_relevance_threshold",
            "penalize_irrelevant_items"
        ]

    elif user_preference == "Lowest price":
        contract["ranking_rules"] = [
            "boost_lower_price_items",
            "keep_minimum_relevance_threshold",
            "penalize_irrelevant_items"
        ]

    return contract