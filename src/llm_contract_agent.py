import json
import os
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from pydantic import BaseModel, Field, ValidationError


load_dotenv()


class IntentSchema(BaseModel):
    detected_category: str = Field(default="general")
    product_type: str = Field(default="unknown")
    price_sensitivity: str = Field(default="medium")
    quality_preference: str = Field(default="medium")
    ambiguity_level: str = Field(default="medium")
    important_terms: List[str] = Field(default_factory=list)
    excluded_terms: List[str] = Field(default_factory=list)


class DynamicFilterSchema(BaseModel):
    must_have_terms: List[str] = Field(default_factory=list)
    nice_to_have_terms: List[str] = Field(default_factory=list)
    blocked_terms: List[str] = Field(default_factory=list)
    allowed_substitutes: List[str] = Field(default_factory=list)
    allowed_complements: List[str] = Field(default_factory=list)
    filter_explanation: str = Field(default="")


class RankingWeightsSchema(BaseModel):
    lexical_weight: float = Field(default=0.20)
    semantic_weight: float = Field(default=0.30)
    esci_weight: float = Field(default=0.30)
    price_weight: float = Field(default=0.05)
    rating_weight: float = Field(default=0.10)
    diversity_weight: float = Field(default=0.05)
    exploration_weight: float = Field(default=0.00)


class SearchContractSchema(BaseModel):
    contract_source: str = Field(default="llm_agent")
    query: str
    intent: IntentSchema
    dynamic_filters: DynamicFilterSchema
    ranking_objective: str
    ranking_rules: List[str]
    guardrails: List[str]
    reward_signal: str
    ranking_weights: RankingWeightsSchema
    explanation: str
    fallback_used: bool = Field(default=False)


def _clip_weight(value: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except Exception:
        return 0.0


def normalize_contract_weights(contract: Dict[str, Any]) -> Dict[str, Any]:
    weights = contract.get("ranking_weights", {})

    expected_keys = [
        "lexical_weight",
        "semantic_weight",
        "esci_weight",
        "price_weight",
        "rating_weight",
        "diversity_weight",
        "exploration_weight",
    ]

    cleaned = {}
    for key in expected_keys:
        cleaned[key] = _clip_weight(weights.get(key, 0.0))

    total = sum(cleaned.values())

    if total <= 0:
        cleaned = {
            "lexical_weight": 0.20,
            "semantic_weight": 0.30,
            "esci_weight": 0.30,
            "price_weight": 0.05,
            "rating_weight": 0.10,
            "diversity_weight": 0.05,
            "exploration_weight": 0.00,
        }
    else:
        cleaned = {key: value / total for key, value in cleaned.items()}

    contract["ranking_weights"] = cleaned
    return contract


def infer_simple_product_type(query: str) -> Dict[str, Any]:
    lowered = query.lower()

    if "headphone" in lowered or "earbud" in lowered or "headset" in lowered:
        return {
            "detected_category": "electronics",
            "product_type": "headphones",
            "must_have_terms": ["headphone", "headphones", "earbud", "earbuds", "headset", "earphone"],
            "nice_to_have_terms": ["bluetooth", "wireless", "noise cancelling", "microphone"],
            "blocked_terms": ["controller", "shirt", "toy", "fidget", "massager", "hearing aid"],
            "allowed_substitutes": ["earbuds", "headset"],
            "allowed_complements": ["headphone case", "earbud case"],
        }

    if "fan" in lowered:
        return {
            "detected_category": "home improvement",
            "product_type": "fan",
            "must_have_terms": ["fan", "ventilation", "exhaust", "blower"],
            "nice_to_have_terms": ["quiet", "bathroom", "ceiling", "cfm", "led"],
            "blocked_terms": ["headphone", "earbud", "shirt", "toy", "controller"],
            "allowed_substitutes": ["ventilation fan", "exhaust fan"],
            "allowed_complements": ["fan cover", "ventilation grille"],
        }

    if "shoe" in lowered or "sneaker" in lowered:
        return {
            "detected_category": "footwear",
            "product_type": "shoes",
            "must_have_terms": ["shoe", "shoes", "sneaker", "sneakers", "trainer", "footwear"],
            "nice_to_have_terms": ["running", "walking", "trail", "cushion", "lightweight"],
            "blocked_terms": ["shoe rack", "shoe cleaner", "insole only", "headphone", "fan"],
            "allowed_substitutes": ["sneakers", "trainers"],
            "allowed_complements": ["shoe inserts", "shoe laces"],
        }

    if "canine" in lowered or "dog" in lowered or "pet" in lowered:
        return {
            "detected_category": "pet supplies",
            "product_type": "dog supplies",
            "must_have_terms": ["dog", "canine", "pet"],
            "nice_to_have_terms": ["waste bag", "poop bag", "leash", "dispenser", "treat", "carrier"],
            "blocked_terms": ["handbag", "laptop bag", "grocery bag", "sandwich bag", "tea bag"],
            "allowed_substitutes": ["pet waste bags", "dog poop bags", "dog treat bag", "dog carrier"],
            "allowed_complements": ["leash dispenser", "poop bag holder"],
        }

    return {
        "detected_category": "general",
        "product_type": "unknown",
        "must_have_terms": [term for term in query.split() if len(term) >= 3],
        "nice_to_have_terms": [],
        "blocked_terms": [],
        "allowed_substitutes": [],
        "allowed_complements": [],
    }


def fallback_llm_contract(query: str, reason: str = "No API key or LLM call failed") -> Dict[str, Any]:
    lowered = query.lower()

    price_sensitivity = "high" if any(
        term in lowered for term in ["cheap", "budget", "affordable", "low price", "discount"]
    ) else "medium"

    quality_preference = "high" if any(
        term in lowered for term in ["best", "premium", "top rated", "quiet", "durable"]
    ) else "medium"

    ambiguity_level = "low" if len(query.split()) >= 3 else "medium"

    inferred = infer_simple_product_type(query)

    contract = {
        "contract_source": "fallback_agent",
        "query": query,
        "intent": {
            "detected_category": inferred["detected_category"],
            "product_type": inferred["product_type"],
            "price_sensitivity": price_sensitivity,
            "quality_preference": quality_preference,
            "ambiguity_level": ambiguity_level,
            "important_terms": query.split(),
            "excluded_terms": [],
        },
        "dynamic_filters": {
            "must_have_terms": inferred["must_have_terms"],
            "nice_to_have_terms": inferred["nice_to_have_terms"],
            "blocked_terms": inferred["blocked_terms"],
            "allowed_substitutes": inferred["allowed_substitutes"],
            "allowed_complements": inferred["allowed_complements"],
            "filter_explanation": "Fallback dynamic filters inferred from the query.",
        },
        "ranking_objective": "Return products that best satisfy the user intent while avoiding irrelevant results.",
        "ranking_rules": [
            "Prioritize exact product-type matches.",
            "Use semantic relevance and lexical relevance together.",
            "Do not over-rank irrelevant products even if they satisfy a secondary preference.",
            "Apply price or quality preference only after relevance is satisfied.",
        ],
        "guardrails": [
            "Do not place irrelevant products in the top results when exact matches exist.",
            "Avoid ranking accessories above the main requested product type unless the query clearly asks for accessories.",
            "Maintain diversity among top results when multiple similar products appear.",
        ],
        "reward_signal": "High reward for exact matches, medium reward for substitutes, lower reward for complements, zero reward for irrelevant products.",
        "ranking_weights": {
            "lexical_weight": 0.20,
            "semantic_weight": 0.30,
            "esci_weight": 0.30,
            "price_weight": 0.15 if price_sensitivity == "high" else 0.05,
            "rating_weight": 0.20 if quality_preference == "high" else 0.10,
            "diversity_weight": 0.05,
            "exploration_weight": 0.00,
        },
        "explanation": f"Fallback contract used because: {reason}. The contract includes dynamic filters, relevance rules, and guardrails.",
        "fallback_used": True,
    }

    return normalize_contract_weights(contract)


def build_llm_prompt(query: str, sample_products: Optional[List[Dict[str, Any]]] = None) -> str:
    sample_products = sample_products or []

    product_context = ""
    if sample_products:
        compact_products = []
        for product in sample_products[:10]:
            compact_products.append(
                {
                    "product_title": product.get("product_title", ""),
                    "category": product.get("category", ""),
                    "price": product.get("price", None),
                    "rating": product.get("rating", None),
                    "esci_label": product.get("esci_label", None),
                }
            )
        product_context = json.dumps(compact_products, indent=2)

    return f"""
You are the CORTEX Contract Agent for an e-commerce search ranking system.

Your job:
Given a shopping search query and a small sample of retrieved products, create an executable search contract.

The contract must:
1. Extract user intent.
2. Identify likely product type and category.
3. Determine price sensitivity.
4. Determine quality preference.
5. Determine ambiguity level.
6. Generate dynamic filtering terms.
7. Create ranking rules and guardrails.
8. Define reward signal.
9. Suggest ranking weights.

Dynamic filtering terms are very important:
- must_have_terms: terms that should identify the main product type.
- nice_to_have_terms: useful modifiers or features.
- blocked_terms: terms that likely indicate wrong products.
- allowed_substitutes: acceptable alternative product forms.
- allowed_complements: related accessories that should usually rank lower than main products.

Rules:
- Do NOT make generic words like cheap, best, wireless, premium, durable the only must-have terms.
- For a query like “cheap wireless headphones”, must_have_terms should include headphone, headphones, earbud, earbuds, headset.
- For a query like “canine bags”, infer likely pet/dog-related bags and include dog, canine, pet, waste bag, poop bag if appropriate.
- blocked_terms should prevent confusing the product with unrelated items that share a word.
- Price and quality are secondary to relevance unless strongly indicated.
- Return only valid JSON. Do not include markdown.

Query:
{query}

Sample retrieved products:
{product_context}

Return JSON with this exact structure:
{{
  "contract_source": "llm_agent",
  "query": "{query}",
  "intent": {{
    "detected_category": "string",
    "product_type": "string",
    "price_sensitivity": "low|medium|high",
    "quality_preference": "low|medium|high",
    "ambiguity_level": "low|medium|high",
    "important_terms": ["string"],
    "excluded_terms": ["string"]
  }},
  "dynamic_filters": {{
    "must_have_terms": ["string"],
    "nice_to_have_terms": ["string"],
    "blocked_terms": ["string"],
    "allowed_substitutes": ["string"],
    "allowed_complements": ["string"],
    "filter_explanation": "string"
  }},
  "ranking_objective": "string",
  "ranking_rules": ["string"],
  "guardrails": ["string"],
  "reward_signal": "string",
  "ranking_weights": {{
    "lexical_weight": 0.0,
    "semantic_weight": 0.0,
    "esci_weight": 0.0,
    "price_weight": 0.0,
    "rating_weight": 0.0,
    "diversity_weight": 0.0,
    "exploration_weight": 0.0
  }},
  "explanation": "string",
  "fallback_used": false
}}
""".strip()


def generate_llm_search_contract(
    query: str,
    sample_products: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("CORTEX_LLM_MODEL", "gpt-4o-mini")

    if not api_key or api_key == "replace_this_with_your_key":
        return fallback_llm_contract(query, reason="OPENAI_API_KEY is missing")

    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)

        prompt = build_llm_prompt(query=query, sample_products=sample_products)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You generate strict JSON search contracts for an e-commerce ranking engine. "
                        "You never include markdown. You only output valid JSON."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            response_format={"type": "json_object"},
            temperature=0.15,
        )

        raw_content = response.choices[0].message.content
        parsed = json.loads(raw_content)

        if "dynamic_filters" not in parsed:
            parsed["dynamic_filters"] = fallback_llm_contract(query)["dynamic_filters"]

        parsed = normalize_contract_weights(parsed)
        validated = SearchContractSchema(**parsed)

        return validated.model_dump()

    except (json.JSONDecodeError, ValidationError) as error:
        return fallback_llm_contract(
            query,
            reason=f"LLM returned invalid contract: {str(error)}",
        )

    except Exception as error:
        return fallback_llm_contract(
            query,
            reason=f"LLM call failed: {str(error)}",
        )