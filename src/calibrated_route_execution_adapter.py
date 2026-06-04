"""
MVP 23D: Calibrated Route Execution Adapter

Executes calibrated experimental routes through the best available local CORTEX
pipeline without changing live governance behavior or governed runner outputs.
"""

from __future__ import annotations

import inspect
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd


ADAPTER_OUTPUT_DIR = Path("outputs/calibrated_route_execution_adapter")

ROUTE_BASELINE_ONLY = "BASELINE_ONLY"
ROUTE_MISSION_REPAIR = "MISSION_REPAIR"
ROUTE_STRICT_REPAIR = "STRICT_REPAIR"
ROUTE_BEHAVIOR_AWARE_RERANK = "BEHAVIOR_AWARE_RERANK"
ROUTE_CRITIC_REVIEW = "CRITIC_REVIEW"
ROUTE_REJECT_REPAIR_NARROW_QUERY = "REJECT_REPAIR_NARROW_QUERY"


def clean_text(value: object) -> str:
    return str(value or "").strip()


def lower_text(value: object) -> str:
    return clean_text(value).lower()


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


def ensure_adapter_output_dir() -> None:
    ADAPTER_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def patch_module_output_dir(module: object, output_dir: Path) -> None:
    if hasattr(module, "OUTPUT_DIR"):
        current = getattr(module, "OUTPUT_DIR")
        setattr(module, "OUTPUT_DIR", str(output_dir) if isinstance(current, str) else output_dir)

    replacements = {
        "MISSION_SLATE_OUTPUT": "mission_slate_builder_demo.csv",
        "MISSION_SUMMARY_OUTPUT": "mission_slate_builder_summary.csv",
        "GUARDED_SLATE_OUTPUT": "mission_slate_guarded_demo.csv",
        "GUARDED_SUMMARY_OUTPUT": "mission_slate_guarded_summary.csv",
        "GUARDED_REJECTIONS_OUTPUT": "mission_slate_guarded_rejections.csv",
        "MISSION_CRITIC_REPORT_OUTPUT": "mission_critic_report.csv",
        "MISSION_CRITIC_SUMMARY_OUTPUT": "mission_critic_summary.csv",
        "MISSION_CRITIC_REPAIR_ACTIONS_OUTPUT": "mission_critic_repair_actions.csv",
        "REPAIRED_SLATE_OUTPUT": "mission_repair_loop_repaired_slate.csv",
        "REPAIR_SUMMARY_OUTPUT": "mission_repair_loop_summary.csv",
        "REPAIR_CANDIDATES_OUTPUT": "mission_repair_loop_candidates.csv",
        "QUALITY_GUARDED_SLATE_OUTPUT": "mission_repair_quality_guarded_slate.csv",
        "QUALITY_SUMMARY_OUTPUT": "mission_repair_quality_summary.csv",
        "QUALITY_CANDIDATES_OUTPUT": "mission_repair_quality_candidates.csv",
        "STRICT_REPAIR_SLATE_OUTPUT": "mission_strict_repair_slate.csv",
        "STRICT_REPAIR_SUMMARY_OUTPUT": "mission_strict_repair_summary.csv",
        "STRICT_REPAIR_REJECTIONS_OUTPUT": "mission_strict_repair_rejections.csv",
    }

    for attr, filename in replacements.items():
        if hasattr(module, attr):
            current = getattr(module, attr)
            replacement = output_dir / filename
            setattr(module, attr, str(replacement) if isinstance(current, str) else replacement)

    if hasattr(module, "CANDIDATE_DATA_PATHS"):
        setattr(
            module,
            "CANDIDATE_DATA_PATHS",
            [
                "data/esci_sample_products.csv",
                "data/sample_products.csv",
                "data/esci_balanced_sample.csv",
            ],
        )


def configure_adapter_scratch() -> None:
    ensure_adapter_output_dir()
    modules = [
        "src.mission_slate_builder",
        "src.mission_slate_guardrails",
        "src.mission_coverage_analyzer",
        "src.mission_critic_agent",
        "src.mission_repair_loop",
        "src.mission_repair_quality_guardrails",
        "src.mission_strict_repair_rules",
        "src.behavior_aware_cortex",
    ]
    for module_name in modules:
        try:
            module = __import__(module_name, fromlist=["*"])
            patch_module_output_dir(module, ADAPTER_OUTPUT_DIR)
        except Exception:
            continue


def call_with_supported_kwargs(func: object, **kwargs: object) -> object:
    signature = inspect.signature(func)
    supported = {
        key: value
        for key, value in kwargs.items()
        if key in signature.parameters
    }
    return func(**supported)


def object_to_dict(value: object) -> Dict[str, object]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return dict(value)
    if is_dataclass(value):
        return asdict(value)
    if hasattr(value, "to_dict"):
        try:
            return dict(value.to_dict())
        except Exception:
            pass
    if isinstance(value, pd.Series):
        return value.to_dict()
    if isinstance(value, str):
        return {"product_title": value}
    return {"product_title": str(value)}


def iterable_rows(value: object) -> List[Dict[str, object]]:
    if value is None:
        return []
    if isinstance(value, pd.DataFrame):
        return [row.to_dict() for _, row in value.iterrows()]
    if isinstance(value, tuple):
        return iterable_rows(value[0] if value else [])
    if isinstance(value, list):
        return [object_to_dict(item) for item in value]
    if isinstance(value, dict):
        return [dict(value)]
    return [object_to_dict(value)]


def normalize_slate_rows(
    query: str,
    rows: Iterable[Dict[str, object]],
    execution_source: str,
    max_items: int,
) -> List[Dict[str, object]]:
    normalized: List[Dict[str, object]] = []
    seen_titles = set()

    for index, row in enumerate(rows, start=1):
        if len(normalized) >= max_items:
            break

        title = clean_text(
            row.get("product_title")
            or row.get("title")
            or row.get("item_title")
            or row.get("name")
            or row.get("product_name")
        )
        if not title:
            title = clean_text(row.get("sub_intent") or row.get("query") or "Adapter fallback row")

        dedupe_key = title.lower()
        if dedupe_key in seen_titles:
            continue
        seen_titles.add(dedupe_key)

        rank = (
            row.get("behavior_rank")
            or row.get("final_strict_rank")
            or row.get("final_quality_rank")
            or row.get("final_rank")
            or row.get("guarded_rank")
            or row.get("selected_rank")
            or index
        )

        normalized.append(
            {
                "query": clean_text(row.get("query") or query),
                "adapter_rank": safe_int(rank, len(normalized) + 1),
                "execution_source": execution_source,
                "product_id": clean_text(row.get("product_id") or row.get("asin") or row.get("item_id")),
                "product_title": title,
                "product_brand": clean_text(row.get("product_brand") or row.get("brand")),
                "sub_intent": clean_text(row.get("sub_intent") or row.get("intent") or row.get("repair_query")),
                "sub_intent_role": clean_text(row.get("sub_intent_role") or row.get("role")),
                "mission_stage": clean_text(row.get("mission_stage")),
                "slate_source": clean_text(row.get("slate_source") or execution_source),
                "score": safe_float(
                    row.get("final_policy_score")
                    or row.get("final_mission_score")
                    or row.get("candidate_score")
                    or row.get("retrieval_score")
                    or row.get("baseline_score")
                ),
                "policy_reason": clean_text(
                    row.get("policy_reason")
                    or row.get("strict_repair_reason")
                    or row.get("repair_quality_reason")
                    or row.get("guardrail_reason")
                    or row.get("retrieval_reason")
                ),
                "cold_start_proxy": clean_text(row.get("cold_start_proxy")),
                "raw_row": {key: clean_text(value) for key, value in row.items()},
            }
        )

    for index, row in enumerate(normalized, start=1):
        row["adapter_rank"] = index

    return normalized


def build_fallback_result(
    query: str,
    calibrated_route: str,
    execution_source: str,
    fallback_reason: str,
    trace: List[str],
    max_items: int,
) -> Dict[str, object]:
    row = {
        "query": query,
        "product_title": fallback_reason,
        "sub_intent": "baseline_preservation" if calibrated_route != ROUTE_CRITIC_REVIEW else "critic_review",
        "policy_reason": fallback_reason,
    }
    slate = normalize_slate_rows(query, [row], execution_source, max_items=max_items)
    return build_result(
        query=query,
        calibrated_route=calibrated_route,
        execution_source=execution_source,
        final_slate=slate,
        fallback_used=True,
        fallback_reason=fallback_reason,
        trace=trace,
    )


def build_result(
    query: str,
    calibrated_route: str,
    execution_source: str,
    final_slate: List[Dict[str, object]],
    fallback_used: bool,
    fallback_reason: str,
    trace: List[str],
) -> Dict[str, object]:
    unique_sub_intents = {
        lower_text(row.get("sub_intent"))
        for row in final_slate
        if lower_text(row.get("sub_intent"))
    }
    return {
        "query": query,
        "calibrated_route": calibrated_route,
        "execution_source": execution_source,
        "final_slate": final_slate,
        "final_slate_size": len(final_slate),
        "unique_sub_intents": len(unique_sub_intents),
        "fallback_used": bool(fallback_used),
        "fallback_reason": fallback_reason,
        "adapter_trace": " | ".join(trace),
    }


def execute_baseline(query: str, calibrated_route: str, max_items: int, trace: List[str]) -> Dict[str, object]:
    try:
        from src.mission_slate_builder import load_product_data, prepare_product_dataframe
        from src.retrieval import tfidf_search

        raw_df, source = load_product_data()
        products = prepare_product_dataframe(raw_df)
        baseline_df = products.rename(columns={"product_brand": "brand"}).copy()
        baseline_df["category"] = "unknown"
        baseline_df["rating"] = 0.0
        if "search_text" in baseline_df.columns:
            baseline_df["product_title"] = baseline_df["product_title"].astype(str)
        result_df = tfidf_search(query, baseline_df, top_k=max_items)
        rows = normalize_slate_rows(query, iterable_rows(result_df), "baseline_retrieval", max_items)
        if rows:
            trace.append(f"baseline_retrieval:{source}:{len(rows)}")
            return build_result(query, calibrated_route, "baseline_retrieval", rows, False, "", trace)
    except Exception as exc:
        trace.append(f"baseline_retrieval_failed:{exc}")

    return build_fallback_result(
        query=query,
        calibrated_route=calibrated_route,
        execution_source="baseline_safe_fallback",
        fallback_reason="Baseline retrieval unavailable; preserved baseline-safe fallback.",
        trace=trace,
        max_items=max_items,
    )


def execute_mission_repair(query: str, calibrated_route: str, max_items: int, trace: List[str]) -> Dict[str, object]:
    try:
        from src.mission_repair_loop import build_repaired_mission_slate

        slate_df, _summary_df, _candidates_df = call_with_supported_kwargs(
            build_repaired_mission_slate,
            query=query,
            slate_size=max_items,
            max_repairs=max(3, max_items // 2),
        )
        rows = normalize_slate_rows(query, iterable_rows(slate_df), "mission_repair_loop", max_items)
        if rows:
            trace.append(f"mission_repair_loop:{len(rows)}")
            return build_result(query, calibrated_route, "mission_repair_loop", rows, False, "", trace)
    except Exception as exc:
        trace.append(f"mission_repair_loop_failed:{exc}")

    try:
        from src.mission_slate_guardrails import build_guarded_mission_slate

        slate_df, _summary_df, _rejected_df = call_with_supported_kwargs(
            build_guarded_mission_slate,
            query=query,
            slate_size=max_items,
        )
        rows = normalize_slate_rows(query, iterable_rows(slate_df), "guarded_mission_slate", max_items)
        if rows:
            trace.append(f"guarded_mission_slate:{len(rows)}")
            return build_result(query, calibrated_route, "guarded_mission_slate", rows, False, "", trace)
    except Exception as exc:
        trace.append(f"guarded_mission_slate_failed:{exc}")

    result = execute_baseline(query, calibrated_route, max_items, trace + ["mission_repair_fallback_to_baseline"])
    result["fallback_used"] = True
    result["fallback_reason"] = "Mission repair path produced no slate; used baseline retrieval fallback."
    return result


def execute_strict_repair(query: str, calibrated_route: str, max_items: int, trace: List[str]) -> Dict[str, object]:
    try:
        from src.mission_strict_repair_rules import build_strict_repair_slate

        slate_df, _summary_df, _rejections_df = call_with_supported_kwargs(
            build_strict_repair_slate,
            query=query,
            slate_size=max_items,
            max_repairs=max(3, max_items // 2),
        )
        rows = normalize_slate_rows(query, iterable_rows(slate_df), "strict_repair_rules", max_items)
        if rows:
            trace.append(f"strict_repair_rules:{len(rows)}")
            return build_result(query, calibrated_route, "strict_repair_rules", rows, False, "", trace)
    except Exception as exc:
        trace.append(f"strict_repair_rules_failed:{exc}")

    result = execute_mission_repair(query, calibrated_route, max_items, trace + ["strict_fallback_to_mission_repair"])
    result["fallback_used"] = True
    if not result["fallback_reason"]:
        result["fallback_reason"] = "Strict repair unavailable; used mission repair fallback."
    return result


def execute_behavior_aware(query: str, calibrated_route: str, max_items: int, trace: List[str]) -> Dict[str, object]:
    try:
        from src.behavior_aware_cortex import behavior_aware_rank
        from src.mission_strict_repair_rules import build_strict_repair_slate

        strict_df, _summary_df, _rejections_df = call_with_supported_kwargs(
            build_strict_repair_slate,
            query=query,
            slate_size=max_items,
            max_repairs=max(3, max_items // 2),
        )
        strict_rows = iterable_rows(strict_df)
        behavior_rows, _behavior_summary, _reasons = behavior_aware_rank(query, strict_rows)
        rows = normalize_slate_rows(query, iterable_rows(behavior_rows), "behavior_aware_cortex", max_items)
        if rows:
            trace.append(f"behavior_aware_cortex:{len(rows)}")
            return build_result(query, calibrated_route, "behavior_aware_cortex", rows, False, "", trace)
        trace.append("behavior_aware_cortex_empty")
    except Exception as exc:
        trace.append(f"behavior_aware_cortex_failed:{exc}")

    result = execute_strict_repair(query, calibrated_route, max_items, trace + ["behavior_fallback_to_strict"])
    result["fallback_used"] = True
    if not result["fallback_reason"]:
        result["fallback_reason"] = "Behavior-aware rerank unavailable; used strict/mission fallback."
    return result


def execute_critic_review(query: str, calibrated_route: str, max_items: int, trace: List[str]) -> Dict[str, object]:
    try:
        from src.mission_critic_agent import critique_mission_query

        critic_report, _summary, repair_actions = call_with_supported_kwargs(
            critique_mission_query,
            query=query,
            slate_size=max_items,
        )
        trace.append(f"critic_review:{len(iterable_rows(critic_report))}:actions={len(iterable_rows(repair_actions))}")
    except Exception as exc:
        trace.append(f"critic_review_failed:{exc}")

    baseline = execute_baseline(query, calibrated_route, max_items, trace)
    baseline["execution_source"] = "critic_review_baseline_fallback"
    baseline["fallback_used"] = True
    baseline["fallback_reason"] = "Critic review route inspected critic path, then preserved baseline-safe slate."
    return baseline


def execute_calibrated_route(
    query: str,
    calibrated_route: str,
    query_understanding: dict | None = None,
    max_items: int = 12,
) -> dict:
    route = clean_text(calibrated_route)
    trace = [f"route={route or 'unknown'}"]
    if query_understanding:
        trace.append(f"query_type={clean_text(query_understanding.get('query_type'))}")
        trace.append(f"bias={clean_text(query_understanding.get('recommended_governance_bias'))}")

    try:
        configure_adapter_scratch()

        if route in {ROUTE_BASELINE_ONLY, ROUTE_REJECT_REPAIR_NARROW_QUERY}:
            return execute_baseline(query, route, max_items, trace)
        if route == ROUTE_MISSION_REPAIR:
            return execute_mission_repair(query, route, max_items, trace)
        if route == ROUTE_STRICT_REPAIR:
            return execute_strict_repair(query, route, max_items, trace)
        if route == ROUTE_BEHAVIOR_AWARE_RERANK:
            return execute_behavior_aware(query, route, max_items, trace)
        if route == ROUTE_CRITIC_REVIEW:
            return execute_critic_review(query, route, max_items, trace)

        return build_fallback_result(
            query=query,
            calibrated_route=route,
            execution_source="unknown_route_fallback",
            fallback_reason=f"Unknown calibrated route '{route}'.",
            trace=trace,
            max_items=max_items,
        )
    except Exception as exc:
        trace.append(f"adapter_failed:{exc}")
        return build_fallback_result(
            query=query,
            calibrated_route=route,
            execution_source="adapter_exception_fallback",
            fallback_reason=f"Adapter failed safely: {exc}",
            trace=trace,
            max_items=max_items,
        )
