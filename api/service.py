from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List

from api.schemas import ProductResult, SearchRequest, SearchResponse


RUNTIME_DIR = Path("outputs/cortex_search_api_runtime")


def clean_text(value: object) -> str:
    return str(value or "").strip()


def safe_int(value: object, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(str(value)))
    except Exception:
        return default


def safe_float_or_none(value: object) -> float | None:
    try:
        if value is None or value == "":
            return None
        return float(str(value))
    except Exception:
        return None


def badge_list(row: Dict[str, Any], route: str) -> List[str]:
    badges = [route]
    for key in ("sub_intent", "retrieval_source", "esci_label"):
        value = clean_text(row.get(key))
        if value:
            badges.append(value)
    return badges


def safety_badges(row: Dict[str, Any]) -> List[str]:
    badges: List[str] = []
    if safe_int(row.get("strict_constraint_applied")):
        badges.append("constraint-aware")
    severity = clean_text(row.get("strict_violation_severity"))
    if severity and severity != "none":
        badges.append(f"{severity} constraint signal")
    elif clean_text(row.get("strict_constraint_type")):
        badges.append("clean constraint pass")
    if safe_int(row.get("critic_review_flag")):
        badges.append("review-ready")
    if safe_int(row.get("strict_boost_applied_count")):
        badges.append("scale-aware rerank")
    return badges


def normalize_result(row: Dict[str, Any], index: int, route: str) -> ProductResult:
    title = clean_text(row.get("product_title") or row.get("title"))
    return ProductResult(
        rank=safe_int(row.get("rank"), index + 1),
        product_id=clean_text(row.get("product_id") or row.get("item_id")),
        title=title,
        brand=clean_text(row.get("product_brand")),
        color=clean_text(row.get("product_color")),
        score=safe_float_or_none(row.get("score")),
        esci_label=clean_text(row.get("esci_label")),
        retrieval_source=clean_text(row.get("retrieval_source")),
        execution_source=clean_text(row.get("execution_source")),
        route_badges=badge_list(row, route),
        safety_badges=safety_badges(row),
        metadata={
            "sub_intent": clean_text(row.get("sub_intent")),
            "strict_constraint_type": clean_text(row.get("strict_constraint_type")),
            "strict_negative_terms": clean_text(row.get("strict_negative_terms")),
            "strict_filter_mode": clean_text(row.get("strict_filter_mode")),
            "strict_constraint_violation": safe_int(row.get("strict_constraint_violation")),
            "strict_violation_severity": clean_text(row.get("strict_violation_severity")),
            "scale_aware_rerank_mode": clean_text(row.get("scale_aware_rerank_mode")),
        },
    )


def max_field(rows: List[Dict[str, Any]], field: str) -> int:
    values = [safe_int(row.get(field)) for row in rows]
    return max(values) if values else 0


def run_cortex_search(request: SearchRequest, diagnostics: bool = False) -> SearchResponse:
    start = time.perf_counter()
    query = clean_text(request.query)
    if not query:
        return SearchResponse(
            query="",
            route="",
            execution_source="",
            final_slate_size=0,
            fallback_used=True,
            fallback_reason="Empty query.",
            runtime_seconds=0.0,
            status="error",
            error_message="Query must not be empty.",
        )

    try:
        from src.calibrated_route_execution_adapter import execute_calibrated_route
        from src.governance_calibration_dry_run import calibrate_query

        RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
        calibration = calibrate_query(query=query, output_dir=RUNTIME_DIR)
        route = clean_text(calibration.get("calibrated_governance_route")) or "BASELINE_ONLY"
        result = execute_calibrated_route(
            query=query,
            calibrated_route=route,
            query_understanding={
                "query_type": clean_text(calibration.get("query_type")),
                "recommended_governance_bias": clean_text(calibration.get("recommended_governance_bias")),
                "confidence_score": calibration.get("query_understanding_confidence"),
                "risk_score": calibration.get("query_understanding_risk"),
            },
            max_items=request.top_k,
            retrieval_mode=request.retrieval_mode,
            retrieval_backend=request.retrieval_backend,
            strict_filter_mode=request.strict_filter_mode,
            scale_aware_rerank_mode=request.scale_aware_rerank_mode,
            index_dir=request.index_dir,
            top_k=request.top_k,
        )
        slate = [dict(row) for row in result.get("final_slate", [])]
        route = clean_text(result.get("calibrated_route")) or route
        diagnostics_payload: Dict[str, Any] = {
            "query_type": clean_text(calibration.get("query_type")),
            "recommended_governance_bias": clean_text(calibration.get("recommended_governance_bias")),
            "current_governance_route": clean_text(calibration.get("current_governance_route")),
            "current_governance_decision": clean_text(calibration.get("current_governance_decision")),
            "calibrated_governance_route": route,
            "calibrated_governance_decision": clean_text(calibration.get("calibrated_governance_decision")),
            "calibration_action": clean_text(calibration.get("calibration_action")),
            "execution_source": clean_text(result.get("execution_source")),
            "retrieval_backend": request.retrieval_backend,
            "strict_filter_mode": request.strict_filter_mode,
            "scale_aware_rerank_mode": request.scale_aware_rerank_mode,
            "strict_clean_count": max_field(slate, "strict_clean_count"),
            "strict_removed_count": max_field(slate, "strict_removed_count"),
            "strict_demoted_count": max_field(slate, "strict_demoted_count"),
            "strict_boost_applied_count": max_field(slate, "strict_boost_applied_count"),
            "reranked_candidate_count": max_field(slate, "reranked_candidate_count"),
            "adapter_trace": clean_text(result.get("adapter_trace")),
        }
        if diagnostics:
            diagnostics_payload["raw_calibration"] = calibration
            diagnostics_payload["final_slate_metadata"] = [
                {
                    "rank": index + 1,
                    "product_id": clean_text(row.get("product_id") or row.get("item_id")),
                    "score": row.get("score", ""),
                    "strict_violation_severity": clean_text(row.get("strict_violation_severity")),
                    "sub_intent": clean_text(row.get("sub_intent")),
                    "esci_label": clean_text(row.get("esci_label")),
                }
                for index, row in enumerate(slate)
            ]

        elapsed = round(time.perf_counter() - start, 6)
        return SearchResponse(
            query=query,
            route=route,
            execution_source=clean_text(result.get("execution_source")),
            final_slate_size=safe_int(result.get("final_slate_size")),
            fallback_used=bool(result.get("fallback_used")),
            fallback_reason=clean_text(result.get("fallback_reason")),
            top_results=[normalize_result(row, index, route) for index, row in enumerate(slate)],
            diagnostics=diagnostics_payload if request.include_diagnostics or diagnostics else {},
            runtime_seconds=elapsed,
            status="ok",
        )
    except Exception as exc:
        elapsed = round(time.perf_counter() - start, 6)
        return SearchResponse(
            query=query,
            route="",
            execution_source="",
            final_slate_size=0,
            fallback_used=True,
            fallback_reason="CORTEX backend returned a structured error.",
            top_results=[],
            diagnostics={"error_type": type(exc).__name__} if request.include_diagnostics or diagnostics else {},
            runtime_seconds=elapsed,
            status="error",
            error_message=str(exc)[:1000],
        )
