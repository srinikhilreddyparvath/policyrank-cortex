"""
MVP 24: Ollama LLM Query Understanding Advisor

Local LLM advisory layer for query understanding. The LLM does not choose
products, does not mutate governance, and always falls back to deterministic
rule-based query understanding when unavailable or invalid.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from src.query_understanding_agent import (
    console_text,
    load_all_queries,
    select_queries,
    understand_query,
)
from src.query_normalization_agent import (
    RESULT_FIELDS as NORMALIZATION_RESULT_FIELDS,
    normalize_query as normalize_raw_query,
    write_outputs as write_normalization_outputs,
)


DEFAULT_OUTPUT_DIR = Path("outputs/ollama_query_understanding")
DEFAULT_CACHE_DIR = DEFAULT_OUTPUT_DIR / "cache"
DEFAULT_OLLAMA_URL = "http://localhost:11434/api/generate"
DEFAULT_MODEL = "qwen3:8b"
ADVISOR_VERSION = "mvp24.3"

ALLOWED_QUERY_TYPES = {
    "narrow_product",
    "numeric_model",
    "mission_query",
    "setup_or_kit",
    "gift_or_party",
    "compatibility_query",
    "negation_constraint",
    "noisy_query",
    "broad_discovery",
    "general_retail",
}

ALLOWED_ROUTES = {
    "BASELINE_ONLY",
    "MISSION_REPAIR",
    "STRICT_REPAIR",
    "BEHAVIOR_AWARE_RERANK",
    "REJECT_REPAIR_NARROW_QUERY",
    "CRITIC_REVIEW",
}

ALLOWED_BIASES = {
    "preserve_baseline",
    "allow_mission_repair",
    "allow_behavior_aware",
    "strict_guardrails",
    "reject_aggressive_repair",
    "send_to_critic",
}

RESULT_FIELDS = [
    "query",
    "raw_query",
    "model",
    "advisor_version",
    "cache_hit",
    "ollama_called",
    "llm_policy",
    "llm_policy_reason",
    "ollama_available",
    "llm_success",
    "llm_error",
    "llm_runtime_seconds",
    "cached_llm_runtime_seconds",
    "normalized_query",
    "correction_applied",
    "correction_count",
    "protected_tokens",
    "protected_token_count",
    "query_noise_score",
    "normalization_confidence",
    "normalization_risk",
    "normalization_status",
    "normalization_trace",
    "llm_query_type",
    "hard_constraints",
    "soft_preferences",
    "mission_sub_intents",
    "recommended_route",
    "recommended_governance_bias",
    "risk_level",
    "confidence",
    "reason",
    "raw_response_excerpt",
    "rule_query_type",
    "rule_recommended_governance_bias",
    "rule_recommended_route",
    "llm_rule_query_type_match",
    "llm_rule_bias_match",
    "llm_rule_route_conflict",
    "fallback_to_rule",
    "final_advisor_status",
]

REQUIRED_LLM_FIELDS = [
    "normalized_query",
    "llm_query_type",
    "recommended_route",
    "recommended_governance_bias",
]

BIAS_TO_ROUTE = {
    "preserve_baseline": "BASELINE_ONLY",
    "allow_mission_repair": "MISSION_REPAIR",
    "allow_behavior_aware": "BEHAVIOR_AWARE_RERANK",
    "strict_guardrails": "STRICT_REPAIR",
    "reject_aggressive_repair": "REJECT_REPAIR_NARROW_QUERY",
    "send_to_critic": "CRITIC_REVIEW",
}

ROUTE_STRICTNESS = {
    "BASELINE_ONLY": 0,
    "MISSION_REPAIR": 1,
    "BEHAVIOR_AWARE_RERANK": 2,
    "STRICT_REPAIR": 3,
    "REJECT_REPAIR_NARROW_QUERY": 3,
    "CRITIC_REVIEW": 4,
}

LLM_POLICY_QUERY_TYPES = {
    "noisy_query",
    "broad_discovery",
    "mission_query",
    "setup_or_kit",
    "gift_or_party",
    "compatibility_query",
    "negation_constraint",
}

LLM_POLICY_BIASES = {
    "allow_mission_repair",
    "allow_behavior_aware",
    "strict_guardrails",
    "send_to_critic",
}

LLM_AUDIT_HIGH_RISK_TYPES = {"compatibility_query", "negation_constraint", "noisy_query"}


def clean_text(value: object) -> str:
    return str(value or "").strip()


def safe_float(value: object, default: float = 0.0) -> float:
    try:
        if value is None or value == "":
            return default
        return float(value)
    except Exception:
        return default


def safe_int(value: object, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(value))
    except Exception:
        return default


def ensure_output_dir(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def list_text(value: object) -> str:
    if isinstance(value, list):
        return "|".join(clean_text(item) for item in value if clean_text(item))
    return clean_text(value)


def clamp_confidence(value: object) -> float:
    confidence = safe_float(value)
    if confidence > 1.0 and confidence <= 100.0:
        confidence = confidence / 100.0
    return round(max(0.0, min(confidence, 1.0)), 4)


def now_utc_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


def normalized_cache_query(query: str) -> str:
    return re.sub(r"\s+", " ", clean_text(query).lower())


def cache_key(query: str, model: str) -> str:
    key_text = f"{ADVISOR_VERSION}|{model}|{normalized_cache_query(query)}"
    return hashlib.sha256(key_text.encode("utf-8")).hexdigest()


def cache_path(cache_dir: Path, query: str, model: str) -> Path:
    return cache_dir / f"{cache_key(query, model)}.json"


def read_cache(cache_dir: Path, query: str, model: str) -> Dict[str, object]:
    path = cache_path(cache_dir, query, model)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    if payload.get("advisor_version") != ADVISOR_VERSION:
        return {}
    if clean_text(payload.get("model")) != model:
        return {}
    return payload if isinstance(payload, dict) else {}


def write_cache(cache_dir: Path, query: str, model: str, payload: Dict[str, object]) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_payload = {
        "query": query,
        "model": model,
        "advisor_version": ADVISOR_VERSION,
        "created_at": now_utc_iso(),
        "llm_success": safe_int(payload.get("llm_success")),
        "llm_error": clean_text(payload.get("llm_error")),
        "raw_response_excerpt": clean_text(payload.get("raw_response_excerpt")),
        "runtime_seconds": safe_float(payload.get("llm_runtime_seconds")),
        "ollama_available": safe_int(payload.get("ollama_available")),
        "normalized_query": clean_text(payload.get("normalized_query")),
        "llm_query_type": clean_text(payload.get("llm_query_type")),
        "hard_constraints": clean_text(payload.get("hard_constraints")),
        "soft_preferences": clean_text(payload.get("soft_preferences")),
        "mission_sub_intents": clean_text(payload.get("mission_sub_intents")),
        "recommended_route": clean_text(payload.get("recommended_route")),
        "recommended_governance_bias": clean_text(payload.get("recommended_governance_bias")),
        "risk_level": clean_text(payload.get("risk_level")),
        "confidence": safe_float(payload.get("confidence")),
        "reason": clean_text(payload.get("reason")),
    }
    cache_path(cache_dir, query, model).write_text(
        json.dumps(cache_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def llm_from_cache(payload: Dict[str, object]) -> Dict[str, object]:
    return {
        "normalized_query": clean_text(payload.get("normalized_query")),
        "llm_query_type": clean_text(payload.get("llm_query_type")),
        "hard_constraints": clean_text(payload.get("hard_constraints")),
        "soft_preferences": clean_text(payload.get("soft_preferences")),
        "mission_sub_intents": clean_text(payload.get("mission_sub_intents")),
        "recommended_route": clean_text(payload.get("recommended_route")),
        "recommended_governance_bias": clean_text(payload.get("recommended_governance_bias")),
        "risk_level": clean_text(payload.get("risk_level")),
        "confidence": safe_float(payload.get("confidence")),
        "reason": clean_text(payload.get("reason")),
    }


def build_prompt(query: str) -> str:
    return f"""
/no_think

You are a query understanding advisor for a retail search governance system.
Return exactly one compact JSON object.
Do not include markdown.
Do not include explanation.
Do not include reasoning.
Do not include thinking text.
Do not choose products. Do not rank products.

Allowed query_type values:
{sorted(ALLOWED_QUERY_TYPES)}

Allowed recommended_route values:
{sorted(ALLOWED_ROUTES)}

Allowed recommended_governance_bias values:
{sorted(ALLOWED_BIASES)}

JSON schema:
{{
  "normalized_query": "lowercase normalized query",
  "llm_query_type": "one allowed query_type",
  "hard_constraints": ["constraints that must be preserved; use [] if none"],
  "soft_preferences": ["preferences that may help but are not hard constraints; use [] if none"],
  "mission_sub_intents": ["sub-intents if this is a mission/setup/gift query; use [] if none"],
  "recommended_route": "one allowed recommended_route",
  "recommended_governance_bias": "one allowed recommended_governance_bias",
  "risk_level": "low|medium|high|unknown",
  "confidence": 0.0,
  "reason": "short plain English reason"
}}

Classify this query:
{query}
""".strip()


def extract_json_object(text: str) -> str:
    text = clean_text(text)
    if not text:
        raise ValueError("empty response")

    try:
        json.loads(text)
        return text
    except Exception:
        pass

    start = text.find("{")
    if start < 0:
        raise ValueError("no JSON object found")

    depth = 0
    in_string = False
    escape = False
    for index in range(start, len(text)):
        char = text[index]
        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]

    raise ValueError("unterminated JSON object")


def repair_json_text(text: str) -> str:
    repaired = text.strip()
    repaired = re.sub(r",\s*([}\]])", r"\1", repaired)
    if '"' not in repaired and "'" in repaired:
        repaired = repaired.replace("'", '"')
    return repaired


def parse_llm_json(raw_text: str) -> Dict[str, object]:
    json_text = extract_json_object(raw_text)
    try:
        parsed = json.loads(json_text)
    except Exception:
        parsed = json.loads(repair_json_text(json_text))
    if not isinstance(parsed, dict):
        raise ValueError("LLM JSON was not an object")
    return parsed


def validate_llm_payload(payload: Dict[str, object]) -> Tuple[Dict[str, object], str]:
    payload = dict(payload)
    if "llm_query_type" not in payload and "query_type" in payload:
        payload["llm_query_type"] = payload.get("query_type")
    if "recommended_governance_bias" not in payload and "governance_bias" in payload:
        payload["recommended_governance_bias"] = payload.get("governance_bias")
    if "recommended_route" not in payload and "route" in payload:
        payload["recommended_route"] = payload.get("route")
    payload.setdefault("hard_constraints", [])
    payload.setdefault("soft_preferences", [])
    payload.setdefault("mission_sub_intents", [])
    payload.setdefault("reason", "")
    payload.setdefault("confidence", 0.0)
    payload.setdefault("risk_level", "unknown")

    missing = [field for field in REQUIRED_LLM_FIELDS if field not in payload]
    if missing:
        return {}, f"missing required fields: {', '.join(missing)}"

    query_type = clean_text(payload.get("llm_query_type"))
    route = clean_text(payload.get("recommended_route")).upper()
    bias = clean_text(payload.get("recommended_governance_bias"))
    risk = clean_text(payload.get("risk_level")).lower()

    errors = []
    if query_type not in ALLOWED_QUERY_TYPES:
        errors.append(f"invalid query_type={query_type}")
    if route not in ALLOWED_ROUTES:
        errors.append(f"invalid recommended_route={route}")
    if bias not in ALLOWED_BIASES:
        errors.append(f"invalid recommended_governance_bias={bias}")
    if risk not in {"low", "medium", "high", "unknown"}:
        errors.append(f"invalid risk_level={risk}")
    if errors:
        return {}, "; ".join(errors)

    return {
        "normalized_query": clean_text(payload.get("normalized_query")),
        "llm_query_type": query_type,
        "hard_constraints": list_text(payload.get("hard_constraints")),
        "soft_preferences": list_text(payload.get("soft_preferences")),
        "mission_sub_intents": list_text(payload.get("mission_sub_intents")),
        "recommended_route": route,
        "recommended_governance_bias": bias,
        "risk_level": risk,
        "confidence": clamp_confidence(payload.get("confidence")),
        "reason": clean_text(payload.get("reason")),
    }, ""


def should_call_llm(
    query: str,
    rule_understanding: object,
    policy: str,
    selective_llm: bool,
    query_position: int = 0,
    audit_sample_rate: float = 0.1,
) -> Dict[str, object]:
    del query
    normalized_policy = "conservative" if policy == "default" else clean_text(policy).lower()
    rule_type = clean_text(getattr(rule_understanding, "query_type", ""))
    rule_bias = clean_text(getattr(rule_understanding, "recommended_governance_bias", ""))

    if not selective_llm:
        return {
            "should_call": True,
            "policy": normalized_policy,
            "reason": "selective LLM disabled; calling Ollama for all queries",
        }
    if normalized_policy == "all":
        return {
            "should_call": True,
            "policy": normalized_policy,
            "reason": "all policy calls Ollama for every query",
        }
    if normalized_policy == "audit":
        if rule_type in LLM_AUDIT_HIGH_RISK_TYPES:
            return {
                "should_call": True,
                "policy": normalized_policy,
                "reason": f"audit policy calls high-risk query_type={rule_type}",
            }
        rate = max(0.0, min(float(audit_sample_rate), 1.0))
        stride = max(int(round(1.0 / rate)), 1) if rate > 0 else 0
        sampled = bool(stride and query_position % stride == 0)
        return {
            "should_call": sampled,
            "policy": normalized_policy,
            "reason": (
                f"audit policy sampled every {stride} queries"
                if sampled
                else f"audit policy skipped non-high-risk query_type={rule_type}"
            ),
        }

    if rule_type in LLM_POLICY_QUERY_TYPES:
        return {
            "should_call": True,
            "policy": normalized_policy,
            "reason": f"conservative policy calls query_type={rule_type}",
        }
    if rule_bias in LLM_POLICY_BIASES:
        return {
            "should_call": True,
            "policy": normalized_policy,
            "reason": f"conservative policy calls governance_bias={rule_bias}",
        }
    return {
        "should_call": False,
        "policy": normalized_policy,
        "reason": f"selective policy skipped low-risk rule={rule_type}/{rule_bias}",
    }


def call_ollama(
    query: str,
    model: str,
    timeout: int,
    ollama_url: str,
    num_predict: int,
    temperature: float,
    think: bool,
) -> Tuple[bool, bool, str, Dict[str, object], str]:
    prompt = build_prompt(query)
    body = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "format": "json",
            "stream": False,
            "think": bool(think),
            "options": {
                "temperature": temperature,
                "top_p": 0.9,
                "num_predict": num_predict,
            },
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        ollama_url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            response_text = response.read().decode("utf-8", errors="replace")
    except urllib.error.URLError as exc:
        return False, False, f"ollama unavailable: {exc}", {}, ""
    except TimeoutError as exc:
        return True, False, f"ollama timeout: {exc}", {}, ""
    except Exception as exc:
        return False, False, f"ollama call failed: {exc}", {}, ""

    try:
        envelope = json.loads(response_text)
    except Exception as exc:
        return True, False, f"bad Ollama envelope JSON: {exc}", {}, response_text[:1000]

    raw_response = clean_text(envelope.get("response"))
    if not raw_response:
        return True, False, "empty Ollama response field", {}, response_text[:1000]

    try:
        parsed = parse_llm_json(raw_response)
        validated, error = validate_llm_payload(parsed)
        if error:
            return True, False, error, {}, raw_response[:1000]
        return True, True, "", validated, raw_response[:1000]
    except Exception as exc:
        return True, False, f"LLM JSON parse failed: {exc}", {}, raw_response[:1000]


def status_for_comparison(
    llm_success: bool,
    ollama_available: bool,
    rule_query_type: str,
    rule_bias: str,
    rule_route: str,
    llm_query_type: str,
    llm_bias: str,
    llm_route: str,
) -> str:
    if not ollama_available:
        return "llm_unavailable_rule_fallback"
    if not llm_success:
        return "llm_failed_rule_fallback"
    if llm_query_type == rule_query_type and llm_bias == rule_bias:
        return "llm_agrees_with_rule"

    rule_strictness = ROUTE_STRICTNESS.get(rule_route, 0)
    llm_strictness = ROUTE_STRICTNESS.get(llm_route, 0)

    if is_unsafe_strict_constraint_disagreement(
        rule_query_type=rule_query_type,
        rule_route=rule_route,
        llm_route=llm_route,
        llm_bias=llm_bias,
    ):
        return "llm_disagrees_needs_review"
    if rule_query_type in {"narrow_product", "numeric_model"} and llm_route in {
        "MISSION_REPAIR",
        "BEHAVIOR_AWARE_RERANK",
    }:
        return "llm_disagrees_needs_review"
    if llm_strictness >= rule_strictness:
        return "llm_disagrees_safe"
    return "llm_disagrees_needs_review"


def is_unsafe_strict_constraint_disagreement(
    rule_query_type: str,
    rule_route: str,
    llm_route: str,
    llm_bias: str,
) -> bool:
    if rule_query_type not in {"negation_constraint", "compatibility_query"}:
        return False
    if rule_route != "STRICT_REPAIR":
        return False
    if llm_route in {"BASELINE_ONLY", "MISSION_REPAIR", "BEHAVIOR_AWARE_RERANK"}:
        return True
    if llm_bias == "preserve_baseline":
        return True
    return False


def analyze_query(
    query: str,
    model: str,
    timeout: int,
    no_ollama: bool,
    ollama_url: str,
    num_predict: int,
    temperature: float,
    think: bool,
    use_cache: bool,
    refresh_cache: bool,
    cache_dir: Path,
    selective_llm: bool,
    llm_policy: str,
    audit_sample_rate: float,
    normalize_query_enabled: bool = False,
    query_position: int = 0,
) -> Dict[str, object]:
    raw_query = query
    if normalize_query_enabled:
        normalization = normalize_raw_query(raw_query)
        effective_query = normalization.normalized_query or raw_query
    else:
        normalization = None
        effective_query = raw_query

    rule = understand_query(effective_query)
    rule_route = BIAS_TO_ROUTE.get(rule.recommended_governance_bias, "BASELINE_ONLY")
    policy_result = should_call_llm(
        query=effective_query,
        rule_understanding=rule,
        policy=llm_policy,
        selective_llm=selective_llm,
        query_position=query_position,
        audit_sample_rate=audit_sample_rate,
    )

    if no_ollama:
        ollama_available = False
        llm_success = False
        llm_error = "--no-ollama enabled"
        llm = {}
        raw_excerpt = ""
        cache_hit = 0
        ollama_called = 0
        llm_runtime_seconds = 0.0
        cached_llm_runtime_seconds = 0.0
    elif not policy_result["should_call"]:
        ollama_available = True
        llm_success = False
        llm_error = "selective policy skipped LLM; rule understanding used"
        llm = {}
        raw_excerpt = ""
        cache_hit = 0
        ollama_called = 0
        llm_runtime_seconds = 0.0
        cached_llm_runtime_seconds = 0.0
    else:
        cache_payload = {}
        if use_cache and not refresh_cache:
            cache_payload = read_cache(cache_dir=cache_dir, query=effective_query, model=model)
        if cache_payload:
            ollama_available = bool(safe_int(cache_payload.get("ollama_available")))
            llm_success = bool(safe_int(cache_payload.get("llm_success")))
            llm_error = clean_text(cache_payload.get("llm_error"))
            llm = llm_from_cache(cache_payload)
            raw_excerpt = clean_text(cache_payload.get("raw_response_excerpt"))
            cache_hit = 1
            ollama_called = 0
            llm_runtime_seconds = 0.0
            cached_llm_runtime_seconds = safe_float(cache_payload.get("runtime_seconds"))
        else:
            call_start = time.perf_counter()
            ollama_available, llm_success, llm_error, llm, raw_excerpt = call_ollama(
                query=effective_query,
                model=model,
                timeout=timeout,
                ollama_url=ollama_url,
                num_predict=num_predict,
                temperature=temperature,
                think=think,
            )
            llm_runtime_seconds = time.perf_counter() - call_start
            cached_llm_runtime_seconds = 0.0
            cache_hit = 0
            ollama_called = 1
            if use_cache:
                write_cache(
                    cache_dir=cache_dir,
                    query=effective_query,
                    model=model,
                    payload={
                        "ollama_available": int(ollama_available),
                        "llm_success": int(llm_success),
                        "llm_error": llm_error,
                        "raw_response_excerpt": raw_excerpt,
                        "llm_runtime_seconds": llm_runtime_seconds,
                        **llm,
                    },
                )

    llm_query_type = clean_text(llm.get("llm_query_type"))
    llm_bias = clean_text(llm.get("recommended_governance_bias"))
    llm_route = clean_text(llm.get("recommended_route"))

    status = status_for_comparison(
        llm_success=llm_success,
        ollama_available=ollama_available,
        rule_query_type=rule.query_type,
        rule_bias=rule.recommended_governance_bias,
        rule_route=rule_route,
        llm_query_type=llm_query_type,
        llm_bias=llm_bias,
        llm_route=llm_route,
    )
    if (not no_ollama) and (not policy_result["should_call"]):
        status = "llm_skipped_rule_confident"
    unsafe_strict_disagreement = llm_success and is_unsafe_strict_constraint_disagreement(
        rule_query_type=rule.query_type,
        rule_route=rule_route,
        llm_route=llm_route,
        llm_bias=llm_bias,
    )
    route_conflict = int(llm_success and (llm_route != rule_route or unsafe_strict_disagreement))
    fallback_to_rule = int(
        (not llm_success)
        or (not ollama_available)
        or unsafe_strict_disagreement
        or status in {
            "llm_failed_rule_fallback",
            "llm_unavailable_rule_fallback",
            "llm_skipped_rule_confident",
        }
    )
    normalization_fields = {
        "correction_applied": "",
        "correction_count": "",
        "protected_tokens": "",
        "protected_token_count": "",
        "query_noise_score": "",
        "normalization_confidence": "",
        "normalization_risk": "",
        "normalization_status": "",
        "normalization_trace": "",
    }
    if normalization is not None:
        normalization_fields = {
            "correction_applied": int(normalization.correction_applied),
            "correction_count": normalization.correction_count,
            "protected_tokens": normalization.protected_tokens,
            "protected_token_count": normalization.protected_token_count,
            "query_noise_score": normalization.query_noise_score,
            "normalization_confidence": normalization.normalization_confidence,
            "normalization_risk": normalization.normalization_risk,
            "normalization_status": normalization.normalization_status,
            "normalization_trace": normalization.normalization_trace,
        }

    return {
        "query": effective_query,
        "raw_query": raw_query,
        "model": model,
        "advisor_version": ADVISOR_VERSION,
        "cache_hit": cache_hit,
        "ollama_called": ollama_called,
        "llm_policy": policy_result["policy"],
        "llm_policy_reason": policy_result["reason"],
        "ollama_available": int(ollama_available),
        "llm_success": int(llm_success),
        "llm_error": llm_error,
        "llm_runtime_seconds": round(llm_runtime_seconds, 4),
        "cached_llm_runtime_seconds": round(cached_llm_runtime_seconds, 4),
        "normalized_query": clean_text(llm.get("normalized_query")) if llm_success else rule.normalized_query,
        **normalization_fields,
        "llm_query_type": llm_query_type,
        "hard_constraints": clean_text(llm.get("hard_constraints")),
        "soft_preferences": clean_text(llm.get("soft_preferences")),
        "mission_sub_intents": clean_text(llm.get("mission_sub_intents")),
        "recommended_route": llm_route,
        "recommended_governance_bias": llm_bias,
        "risk_level": clean_text(llm.get("risk_level")),
        "confidence": safe_float(llm.get("confidence")),
        "reason": clean_text(llm.get("reason")),
        "raw_response_excerpt": raw_excerpt,
        "rule_query_type": rule.query_type,
        "rule_recommended_governance_bias": rule.recommended_governance_bias,
        "rule_recommended_route": rule_route,
        "llm_rule_query_type_match": int(llm_success and llm_query_type == rule.query_type),
        "llm_rule_bias_match": int(llm_success and llm_bias == rule.recommended_governance_bias),
        "llm_rule_route_conflict": route_conflict,
        "fallback_to_rule": fallback_to_rule,
        "final_advisor_status": status,
    }


def summarize(rows: List[Dict[str, object]], runtime_seconds: float) -> Dict[str, object]:
    total = len(rows)
    llm_success_count = sum(safe_int(row.get("llm_success")) for row in rows)
    unavailable_count = sum(1 for row in rows if safe_int(row.get("ollama_available")) == 0)
    cache_hit_count = sum(safe_int(row.get("cache_hit")) for row in rows)
    ollama_called_count = sum(safe_int(row.get("ollama_called")) for row in rows)
    skipped_count = sum(1 for row in rows if clean_text(row.get("final_advisor_status")) == "llm_skipped_rule_confident")
    query_type_matches = sum(safe_int(row.get("llm_rule_query_type_match")) for row in rows)
    bias_matches = sum(safe_int(row.get("llm_rule_bias_match")) for row in rows)
    needs_review_count = sum(1 for row in rows if clean_text(row.get("final_advisor_status")) == "llm_disagrees_needs_review")
    safe_disagreement_count = sum(1 for row in rows if clean_text(row.get("final_advisor_status")) == "llm_disagrees_safe")
    fallback_count = sum(safe_int(row.get("fallback_to_rule")) for row in rows)
    success_rows = [row for row in rows if safe_int(row.get("llm_success")) == 1]
    avg_conf = sum(safe_float(row.get("confidence")) for row in success_rows) / max(len(success_rows), 1)
    actual_llm_runtime = sum(safe_float(row.get("llm_runtime_seconds")) for row in rows)
    cached_runtime = sum(safe_float(row.get("cached_llm_runtime_seconds")) for row in rows)
    estimated_uncached_runtime = actual_llm_runtime + cached_runtime

    return {
        "total_queries": total,
        "llm_success_count": llm_success_count,
        "llm_failure_count": total - llm_success_count,
        "ollama_unavailable_count": unavailable_count,
        "llm_success_rate": round(llm_success_count / max(total, 1), 6),
        "cache_hit_count": cache_hit_count,
        "cache_hit_rate": round(cache_hit_count / max(total, 1), 6),
        "ollama_called_count": ollama_called_count,
        "ollama_call_rate": round(ollama_called_count / max(total, 1), 6),
        "llm_skipped_count": skipped_count,
        "llm_skipped_rate": round(skipped_count / max(total, 1), 6),
        "avg_llm_runtime_seconds": round(actual_llm_runtime / max(ollama_called_count, 1), 6),
        "total_runtime_seconds": round(runtime_seconds, 4),
        "estimated_uncached_runtime_seconds": round(estimated_uncached_runtime, 4),
        "estimated_cache_savings_seconds": round(cached_runtime, 4),
        "rule_match_query_type_rate": round(query_type_matches / max(llm_success_count, 1), 6),
        "rule_match_bias_rate": round(bias_matches / max(llm_success_count, 1), 6),
        "needs_review_count": needs_review_count,
        "safe_disagreement_count": safe_disagreement_count,
        "fallback_to_rule_count": fallback_count,
        "top_llm_query_type": top_value(success_rows, "llm_query_type"),
        "top_llm_recommended_route": top_value(success_rows, "recommended_route"),
        "top_final_advisor_status": top_value(rows, "final_advisor_status"),
        "avg_confidence": round(avg_conf, 6),
        "runtime_seconds": round(runtime_seconds, 4),
    }


def top_value(rows: List[Dict[str, object]], field: str) -> str:
    counts = Counter(clean_text(row.get(field)) for row in rows if clean_text(row.get(field)))
    return counts.most_common(1)[0][0] if counts else ""


def conflict_rows(rows: List[Dict[str, object]]) -> List[Dict[str, object]]:
    statuses = {"llm_disagrees_needs_review", "llm_disagrees_safe"}
    return [
        row
        for row in rows
        if clean_text(row.get("final_advisor_status")) in statuses
        or safe_int(row.get("llm_rule_route_conflict")) == 1
    ]


def normalization_row(row: Dict[str, object]) -> Dict[str, object]:
    return {
        "raw_query": row.get("raw_query", row.get("query", "")),
        "normalized_query": row.get("query", ""),
        "correction_applied": bool(safe_int(row.get("correction_applied"))),
        "correction_count": row.get("correction_count", 0),
        "protected_tokens": row.get("protected_tokens", ""),
        "protected_token_count": row.get("protected_token_count", 0),
        "normalized_tokens": "|".join(re.findall(r"[a-z0-9]+(?:-[a-z0-9]+)?", clean_text(row.get("query")).lower())),
        "query_noise_score": row.get("query_noise_score", 0.0),
        "normalization_confidence": row.get("normalization_confidence", 0.0),
        "normalization_risk": row.get("normalization_risk", ""),
        "normalization_status": row.get("normalization_status", ""),
        "normalization_trace": row.get("normalization_trace", ""),
    }


def report_markdown(rows: List[Dict[str, object]], summary: Dict[str, object]) -> str:
    lines = [
        "# MVP 24 Ollama LLM Query Understanding Advisor",
        "",
        "## Summary",
    ]
    for key, value in summary.items():
        lines.append(f"- {key}: {value}")

    lines.extend(["", "## Status Distribution"])
    for status, count in Counter(clean_text(row.get("final_advisor_status")) for row in rows).most_common():
        lines.append(f"- {status}: {count}")

    lines.extend(["", "## Conflicts And Review Items"])
    for row in conflict_rows(rows)[:20]:
        lines.append(
            f"- {row['query']}: rule={row['rule_query_type']}/{row['rule_recommended_governance_bias']} "
            f"llm={row['llm_query_type']}/{row['recommended_governance_bias']} "
            f"status={row['final_advisor_status']}"
        )
    return "\n".join(lines) + "\n"


def write_outputs(rows: List[Dict[str, object]], output_dir: Path, runtime_seconds: float) -> Dict[str, object]:
    ensure_output_dir(output_dir)
    summary = summarize(rows, runtime_seconds=runtime_seconds)
    conflicts = conflict_rows(rows)

    write_csv(output_dir / "ollama_query_understanding_results.csv", rows, RESULT_FIELDS)
    write_csv(output_dir / "ollama_query_understanding_summary.csv", [summary], list(summary.keys()))
    write_csv(output_dir / "ollama_query_understanding_conflicts.csv", conflicts, RESULT_FIELDS)
    (output_dir / "ollama_query_understanding_report.md").write_text(
        report_markdown(rows, summary),
        encoding="utf-8",
    )
    return summary


def print_single(row: Dict[str, object], summary: Dict[str, object]) -> None:
    print("\nMVP 24 Ollama LLM Query Understanding Advisor")
    print("-" * 100)
    if clean_text(row.get("normalization_status")):
        print("Query normalization")
        print(f"raw_query: {console_text(row['raw_query'])}")
        print(f"normalized_query: {console_text(row['query'])}")
        print(f"correction_applied: {row['correction_applied']}")
        print(f"correction_count: {row['correction_count']}")
        print(f"protected_tokens: {console_text(row['protected_tokens'])}")
        print(f"query_noise_score: {row['query_noise_score']}")
        print(f"normalization_confidence: {row['normalization_confidence']}")
        print(f"normalization_risk: {row['normalization_risk']}")
        print(f"normalization_status: {row['normalization_status']}")
        print(f"normalization_trace: {console_text(row['normalization_trace'])}")
        print("")
    print("Rule understanding")
    print(f"query_type: {row['rule_query_type']}")
    print(f"recommended_governance_bias: {row['rule_recommended_governance_bias']}")
    print(f"recommended_route: {row['rule_recommended_route']}")

    print("\nLLM understanding")
    print(f"advisor_version: {row['advisor_version']}")
    print(f"llm_policy: {row['llm_policy']}")
    print(f"llm_policy_reason: {row['llm_policy_reason']}")
    print(f"cache_hit: {row['cache_hit']}")
    print(f"ollama_called: {row['ollama_called']}")
    print(f"ollama_available: {row['ollama_available']}")
    print(f"llm_success: {row['llm_success']}")
    print(f"llm_query_type: {row['llm_query_type']}")
    print(f"recommended_governance_bias: {row['recommended_governance_bias']}")
    print(f"recommended_route: {row['recommended_route']}")
    print(f"confidence: {row['confidence']}")
    print(f"reason: {console_text(row['reason'])}")
    if row["llm_error"]:
        print(f"llm_error: {console_text(row['llm_error'])}")

    print("\nComparison")
    print(f"llm_rule_query_type_match: {row['llm_rule_query_type_match']}")
    print(f"llm_rule_bias_match: {row['llm_rule_bias_match']}")
    print(f"llm_rule_route_conflict: {row['llm_rule_route_conflict']}")
    print(f"final_advisor_status: {row['final_advisor_status']}")

    print("\nRaw response excerpt")
    print("-" * 100)
    print(console_text(row["raw_response_excerpt"]))

    print("\nSummary")
    print("-" * 100)
    for key, value in summary.items():
        print(f"{key}: {value}")


def print_batch_summary(summary: Dict[str, object], output_dir: Path) -> None:
    print("\nSummary")
    print("-" * 100)
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("\nOutput files")
    print("-" * 100)
    print(output_dir / "ollama_query_understanding_results.csv")
    print(output_dir / "ollama_query_understanding_summary.csv")
    print(output_dir / "ollama_query_understanding_conflicts.csv")
    print(output_dir / "ollama_query_understanding_report.md")


def run_single(args: argparse.Namespace) -> None:
    start = time.perf_counter()
    row = analyze_query(
        query=args.query,
        model=args.model,
        timeout=args.timeout,
        no_ollama=args.no_ollama,
        ollama_url=args.ollama_url,
        num_predict=args.num_predict,
        temperature=args.temperature,
        think=args.think,
        use_cache=args.use_cache,
        refresh_cache=args.refresh_cache,
        cache_dir=Path(args.cache_dir),
        selective_llm=args.selective_llm,
        llm_policy=args.llm_policy,
        audit_sample_rate=args.audit_sample_rate,
        normalize_query_enabled=args.normalize_query,
        query_position=0,
    )
    runtime = time.perf_counter() - start
    summary = write_outputs([row], Path(args.output_dir), runtime_seconds=runtime)
    if args.normalization_output_dir and args.normalize_query:
        write_normalization_outputs(
            [normalization_row(row)],
            Path(args.normalization_output_dir),
            runtime_seconds=runtime,
        )
    print_single(row, summary)


def run_batch(args: argparse.Namespace) -> None:
    all_queries, source = load_all_queries(args.query_mode)
    selected = select_queries(
        queries=all_queries,
        query_mode=args.query_mode,
        sample_size=args.sample_size,
        start_index=args.start_index,
    )

    print("\nMVP 24 Ollama LLM Query Understanding Advisor")
    print("-" * 100)
    print(f"query_mode: {args.query_mode}")
    print(f"source: {source}")
    print(f"selected_query_count: {len(selected)}")
    print(f"model: {args.model}")
    print(f"no_ollama: {args.no_ollama}")
    print(f"use_cache: {args.use_cache}")
    print(f"refresh_cache: {args.refresh_cache}")
    print(f"cache_dir: {args.cache_dir}")
    print(f"selective_llm: {args.selective_llm}")
    print(f"llm_policy: {args.llm_policy}")
    print(f"normalize_query: {args.normalize_query}")

    rows = []
    normalization_rows = []
    start = time.perf_counter()
    for index, (_query_index, query) in enumerate(selected, start=1):
        row = analyze_query(
            query=query,
            model=args.model,
            timeout=args.timeout,
            no_ollama=args.no_ollama,
            ollama_url=args.ollama_url,
            num_predict=args.num_predict,
            temperature=args.temperature,
            think=args.think,
            use_cache=args.use_cache,
            refresh_cache=args.refresh_cache,
            cache_dir=Path(args.cache_dir),
            selective_llm=args.selective_llm,
            llm_policy=args.llm_policy,
            audit_sample_rate=args.audit_sample_rate,
            normalize_query_enabled=args.normalize_query,
            query_position=index - 1,
        )
        rows.append(row)
        if args.normalize_query:
            normalization_rows.append(normalization_row(row))
        if index <= 10 or index % 100 == 0 or index == len(selected):
            print(
                f"[{index}/{len(selected)}] {console_text(row['raw_query'])} -> {console_text(row['query'])} "
                f"rule={row['rule_query_type']}/{row['rule_recommended_governance_bias']} "
                f"llm={row['llm_query_type'] or 'fallback'} status={row['final_advisor_status']} "
                f"cache_hit={row['cache_hit']} ollama_called={row['ollama_called']}"
            )

    runtime = time.perf_counter() - start
    output_dir = Path(args.output_dir)
    summary = write_outputs(rows, output_dir, runtime_seconds=runtime)
    if args.normalization_output_dir and normalization_rows:
        write_normalization_outputs(
            normalization_rows,
            Path(args.normalization_output_dir),
            runtime_seconds=runtime,
        )
    print_batch_summary(summary, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MVP 24 Ollama LLM Query Understanding Advisor")
    parser.add_argument("--query", default="", help="Single query to advise.")
    parser.add_argument("--sample-size", type=int, default=100, help="Batch sample size.")
    parser.add_argument(
        "--query-mode",
        choices=["smoke", "esci", "stratified_esci"],
        default="smoke",
        help="Batch query source mode.",
    )
    parser.add_argument("--start-index", type=int, default=0, help="Batch start offset.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Ollama model.")
    parser.add_argument("--timeout", type=int, default=20, help="Ollama timeout seconds.")
    parser.add_argument("--num-predict", type=int, default=256, help="Ollama num_predict generation limit.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Ollama temperature.")
    parser.add_argument("--think", action="store_true", help="Allow model thinking if the local model supports it.")
    parser.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL, help="Ollama generate API URL.")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory.")
    parser.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR), help="LLM cache directory.")
    parser.add_argument(
        "--use-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the local LLM result cache.",
    )
    parser.add_argument("--refresh-cache", action="store_true", help="Refresh cache entries by calling Ollama.")
    parser.add_argument("--selective-llm", action="store_true", help="Only call LLM when the policy selects a query.")
    parser.add_argument(
        "--llm-policy",
        choices=["conservative", "default", "all", "audit"],
        default="conservative",
        help="Selective LLM invocation policy.",
    )
    parser.add_argument("--audit-sample-rate", type=float, default=0.1, help="Audit policy random-equivalent sample rate.")
    parser.add_argument(
        "--normalize-query",
        action="store_true",
        help="Normalize the raw query before rule understanding and LLM advisory.",
    )
    parser.add_argument(
        "--normalization-output-dir",
        default="",
        help="Optional directory for standalone normalization outputs.",
    )
    parser.add_argument("--no-ollama", action="store_true", help="Do not call Ollama; force rule fallback.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.query:
        run_single(args)
        return
    if args.sample_size <= 0:
        raise ValueError("--sample-size must be positive")
    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative")
    run_batch(args)


if __name__ == "__main__":
    main()
