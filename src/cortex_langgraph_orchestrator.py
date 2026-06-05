"""
MVP 24.4: LangGraph CORTEX Agent Orchestration Prototype

Optional orchestration layer that connects existing CORTEX agents as graph
nodes without replacing the current direct Python runners.
"""

from __future__ import annotations

import argparse
import csv
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, TypedDict

try:
    from langgraph.graph import END, START, StateGraph

    LANGGRAPH_AVAILABLE = True
except Exception:
    END = "__end__"
    START = "__start__"
    StateGraph = None
    LANGGRAPH_AVAILABLE = False

from src.calibrated_route_execution_adapter import execute_calibrated_route
from src.governance_calibration_dry_run import apply_calibration
from src.ollama_query_understanding_advisor import analyze_query as analyze_llm_advisor
from src.query_normalization_agent import normalize_query
from src.query_understanding_agent import console_text, load_all_queries, select_queries, understand_query


DEFAULT_OUTPUT_DIR = Path("outputs/langgraph_orchestration")

BIAS_TO_ROUTE = {
    "preserve_baseline": "BASELINE_ONLY",
    "allow_mission_repair": "MISSION_REPAIR",
    "allow_behavior_aware": "BEHAVIOR_AWARE_RERANK",
    "strict_guardrails": "STRICT_REPAIR",
    "reject_aggressive_repair": "REJECT_REPAIR_NARROW_QUERY",
    "send_to_critic": "CRITIC_REVIEW",
}

ROUTE_TO_DECISION = {
    "BASELINE_ONLY": "PRESERVE_BASELINE",
    "MISSION_REPAIR": "RUN_MISSION_REPAIR",
    "BEHAVIOR_AWARE_RERANK": "RUN_MISSION_REPAIR_AND_BEHAVIOR_AWARE",
    "STRICT_REPAIR": "RUN_STRICT_REPAIR",
    "REJECT_REPAIR_NARROW_QUERY": "REJECT_REPAIR_NARROW_QUERY",
    "CRITIC_REVIEW": "RUN_CRITIC_REVIEW",
}

RESULT_FIELDS = [
    "raw_query",
    "normalized_query",
    "normalization_enabled",
    "normalization_status",
    "normalization_risk",
    "normalization_trace",
    "protected_tokens",
    "rule_query_type",
    "rule_recommended_governance_bias",
    "rule_recommended_route",
    "llm_success",
    "llm_query_type",
    "llm_recommended_governance_bias",
    "llm_recommended_route",
    "final_advisor_status",
    "governance_route",
    "governance_decision",
    "calibrated_route",
    "calibrated_decision",
    "calibration_action",
    "execution_source",
    "final_slate_size",
    "unique_sub_intents",
    "fallback_used",
    "fallback_reason",
    "quality_label",
    "agent_trace",
    "errors",
]


class CortexGraphState(TypedDict, total=False):
    raw_query: str
    normalized_query: str
    normalization_enabled: bool
    normalization_status: str
    normalization_risk: str
    normalization_trace: str
    protected_tokens: str
    rule_query_type: str
    rule_recommended_governance_bias: str
    rule_recommended_route: str
    llm_success: int
    llm_query_type: str
    llm_recommended_governance_bias: str
    llm_recommended_route: str
    final_advisor_status: str
    governance_route: str
    governance_decision: str
    calibrated_route: str
    calibrated_decision: str
    calibration_action: str
    execution_source: str
    final_slate_size: int
    unique_sub_intents: int
    fallback_used: bool
    fallback_reason: str
    quality_label: str
    agent_trace: str
    errors: str
    args: argparse.Namespace


def clean_text(value: object) -> str:
    return str(value or "").strip()


def safe_int(value: object, default: int = 0) -> int:
    try:
        if value is None or value == "":
            return default
        return int(float(str(value)))
    except Exception:
        return default


def append_trace(state: CortexGraphState, message: str) -> None:
    existing = clean_text(state.get("agent_trace"))
    state["agent_trace"] = f"{existing} | {message}" if existing else message


def append_error(state: CortexGraphState, message: str) -> None:
    existing = clean_text(state.get("errors"))
    state["errors"] = f"{existing} | {message}" if existing else message


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def normalize_query_node(state: CortexGraphState) -> CortexGraphState:
    args = state["args"]
    raw_query = clean_text(state.get("raw_query"))
    state["normalization_enabled"] = bool(args.normalize_query)

    if not args.normalize_query:
        state["normalized_query"] = raw_query
        state["normalization_status"] = "normalization_disabled"
        state["normalization_risk"] = ""
        state["normalization_trace"] = ""
        state["protected_tokens"] = ""
        append_trace(state, "normalize_query_node:disabled")
        return state

    try:
        result = normalize_query(raw_query)
        state["normalized_query"] = result.normalized_query or raw_query
        state["normalization_status"] = result.normalization_status
        state["normalization_risk"] = result.normalization_risk
        state["normalization_trace"] = result.normalization_trace
        state["protected_tokens"] = result.protected_tokens
        append_trace(state, f"normalize_query_node:{result.normalization_status}")
    except Exception as exc:
        state["normalized_query"] = raw_query
        state["normalization_status"] = "normalizer_not_available"
        state["normalization_risk"] = ""
        state["normalization_trace"] = str(exc)
        state["protected_tokens"] = ""
        append_error(state, f"normalize_query_node:{type(exc).__name__}")
    return state


def rule_understanding_node(state: CortexGraphState) -> CortexGraphState:
    query = clean_text(state.get("normalized_query")) or clean_text(state.get("raw_query"))
    try:
        result = understand_query(query)
        route = BIAS_TO_ROUTE.get(result.recommended_governance_bias, "BASELINE_ONLY")
        state["rule_query_type"] = result.query_type
        state["rule_recommended_governance_bias"] = result.recommended_governance_bias
        state["rule_recommended_route"] = route
        state["rule_confidence_score"] = result.confidence_score  # type: ignore[typeddict-unknown-key]
        state["rule_risk_score"] = result.risk_score  # type: ignore[typeddict-unknown-key]
        append_trace(state, f"rule_understanding_node:{result.query_type}/{route}")
    except Exception as exc:
        state["rule_query_type"] = ""
        state["rule_recommended_governance_bias"] = ""
        state["rule_recommended_route"] = "BASELINE_ONLY"
        state["rule_confidence_score"] = 0.0  # type: ignore[typeddict-unknown-key]
        state["rule_risk_score"] = 1.0  # type: ignore[typeddict-unknown-key]
        append_error(state, f"rule_understanding_node:{type(exc).__name__}")
    return state


def llm_advisor_node(state: CortexGraphState) -> CortexGraphState:
    args = state["args"]
    query = clean_text(state.get("normalized_query")) or clean_text(state.get("raw_query"))
    try:
        row = analyze_llm_advisor(
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
            normalize_query_enabled=False,
            query_position=safe_int(state.get("query_position")),
        )
        state["llm_success"] = safe_int(row.get("llm_success"))
        state["llm_query_type"] = clean_text(row.get("llm_query_type"))
        state["llm_recommended_governance_bias"] = clean_text(row.get("recommended_governance_bias"))
        state["llm_recommended_route"] = clean_text(row.get("recommended_route"))
        state["final_advisor_status"] = clean_text(row.get("final_advisor_status"))
        append_trace(state, f"llm_advisor_node:{state['final_advisor_status']}")
    except Exception as exc:
        state["llm_success"] = 0
        state["llm_query_type"] = ""
        state["llm_recommended_governance_bias"] = ""
        state["llm_recommended_route"] = ""
        state["final_advisor_status"] = "llm_failed_rule_fallback"
        append_error(state, f"llm_advisor_node:{type(exc).__name__}")
    return state


def governance_node(state: CortexGraphState) -> CortexGraphState:
    query = clean_text(state.get("normalized_query")) or clean_text(state.get("raw_query"))
    try:
        current_route = "BASELINE_ONLY"
        current_decision = "USE_BASELINE_ONLY"
        bias = clean_text(state.get("rule_recommended_governance_bias"))
        query_type = clean_text(state.get("rule_query_type"))
        rule_recommended_route = clean_text(state.get("rule_recommended_route"))
        confidence = float(state.get("rule_confidence_score", 0.0))  # type: ignore[typeddict-item]
        risk = float(state.get("rule_risk_score", 1.0))  # type: ignore[typeddict-item]
        calibration = apply_calibration(
            query=query,
            query_type=query_type,
            bias=bias,
            confidence=confidence,
            risk=risk,
            current_route=current_route,
            current_decision=current_decision,
        )
        calibrated_route = clean_text(calibration.get("calibrated_governance_route")) or "BASELINE_ONLY"
        calibrated_decision = clean_text(calibration.get("calibrated_governance_decision")) or ROUTE_TO_DECISION.get(
            calibrated_route,
            current_decision,
        )
        action = clean_text(calibration.get("calibration_action")) or "keep_current_route"
        override_used = False

        if calibrated_route == "BASELINE_ONLY" and rule_recommended_route and rule_recommended_route != "BASELINE_ONLY":
            calibrated_route = rule_recommended_route
            calibrated_decision = ROUTE_TO_DECISION.get(rule_recommended_route, calibrated_decision)
            action = "langgraph_rule_alignment_override"
            override_used = True

        state["governance_route"] = current_route
        state["governance_decision"] = current_decision
        state["calibrated_route"] = calibrated_route
        state["calibrated_decision"] = calibrated_decision
        state["calibration_action"] = action
        trace_suffix = " via rule_alignment_override" if override_used else ""
        append_trace(state, f"governance_node:{state['governance_route']}=>{state['calibrated_route']}{trace_suffix}")
    except Exception as exc:
        state["governance_route"] = clean_text(state.get("rule_recommended_route"))
        state["governance_decision"] = "FALLBACK_TO_RULE_ROUTE"
        state["calibrated_route"] = clean_text(state.get("rule_recommended_route")) or "BASELINE_ONLY"
        state["calibrated_decision"] = "FALLBACK_TO_RULE_ROUTE"
        state["calibration_action"] = "calibration_unavailable"
        append_error(state, f"governance_node:{type(exc).__name__}")
    return state


def execution_node(state: CortexGraphState) -> CortexGraphState:
    query = clean_text(state.get("normalized_query")) or clean_text(state.get("raw_query"))
    route = clean_text(state.get("calibrated_route")) or clean_text(state.get("rule_recommended_route")) or "BASELINE_ONLY"
    try:
        result = execute_calibrated_route(
            query=query,
            calibrated_route=route,
            query_understanding={
                "query_type": clean_text(state.get("rule_query_type")),
                "recommended_governance_bias": clean_text(state.get("rule_recommended_governance_bias")),
                "recommended_route": clean_text(state.get("rule_recommended_route")),
            },
            max_items=safe_int(getattr(state["args"], "top_k", 12), 12),
            retrieval_mode=getattr(state["args"], "retrieval_mode", "sample"),
            retrieval_backend=getattr(state["args"], "retrieval_backend", "lexical"),
            strict_filter_mode=getattr(state["args"], "strict_filter_mode", "hybrid"),
            strict_min_clean_results=safe_int(getattr(state["args"], "strict_min_clean_results", 8), 8),
            strict_candidate_multiplier=safe_int(getattr(state["args"], "strict_candidate_multiplier", 8), 8),
            scale_aware_rerank_mode=getattr(state["args"], "scale_aware_rerank_mode", "none"),
            index_dir=getattr(state["args"], "index_dir", "data/esci_index"),
            top_k=safe_int(getattr(state["args"], "top_k", 12), 12),
        )
        state["execution_source"] = clean_text(result.get("execution_source"))
        state["final_slate_size"] = safe_int(result.get("final_slate_size"))
        state["unique_sub_intents"] = safe_int(result.get("unique_sub_intents"))
        state["fallback_used"] = bool(result.get("fallback_used"))
        state["fallback_reason"] = clean_text(result.get("fallback_reason"))
        append_trace(state, f"execution_node:{state['execution_source']}/slate={state['final_slate_size']}")
    except Exception as exc:
        state["execution_source"] = ""
        state["final_slate_size"] = 0
        state["unique_sub_intents"] = 0
        state["fallback_used"] = True
        state["fallback_reason"] = f"execution failed: {exc}"
        append_error(state, f"execution_node:{type(exc).__name__}")
    return state


def quality_node(state: CortexGraphState) -> CortexGraphState:
    final_slate_size = safe_int(state.get("final_slate_size"))
    unique_sub_intents = safe_int(state.get("unique_sub_intents"))
    fallback_used = bool(state.get("fallback_used"))

    if clean_text(state.get("errors")):
        label = "error"
    elif fallback_used:
        label = "fallback_execution"
    elif final_slate_size >= 10 and unique_sub_intents >= 5:
        label = "strong_diverse_execution"
    elif final_slate_size >= 10:
        label = "strong_execution"
    elif final_slate_size >= 3:
        label = "acceptable_execution"
    else:
        label = "weak_slate"

    state["quality_label"] = label
    append_trace(state, f"quality_node:{label}")
    return state


def build_graph():
    if not LANGGRAPH_AVAILABLE or StateGraph is None:
        return None

    graph = StateGraph(CortexGraphState)
    graph.add_node("normalize_query_node", normalize_query_node)
    graph.add_node("rule_understanding_node", rule_understanding_node)
    graph.add_node("llm_advisor_node", llm_advisor_node)
    graph.add_node("governance_node", governance_node)
    graph.add_node("execution_node", execution_node)
    graph.add_node("quality_node", quality_node)
    graph.add_edge(START, "normalize_query_node")
    graph.add_edge("normalize_query_node", "rule_understanding_node")
    graph.add_edge("rule_understanding_node", "llm_advisor_node")
    graph.add_edge("llm_advisor_node", "governance_node")
    graph.add_edge("governance_node", "execution_node")
    graph.add_edge("execution_node", "quality_node")
    graph.add_edge("quality_node", END)
    return graph.compile()


def run_direct_graph(state: CortexGraphState) -> CortexGraphState:
    for node in (
        normalize_query_node,
        rule_understanding_node,
        llm_advisor_node,
        governance_node,
        execution_node,
        quality_node,
    ):
        state = node(state)
    return state


def run_orchestration(query: str, args: argparse.Namespace, query_position: int = 0, graph=None) -> Dict[str, object]:
    state: CortexGraphState = {
        "raw_query": query,
        "normalized_query": query,
        "normalization_enabled": bool(args.normalize_query),
        "agent_trace": "",
        "errors": "",
        "args": args,
        "query_position": query_position,  # type: ignore[typeddict-unknown-key]
    }
    if graph is not None:
        state = graph.invoke(state)
    else:
        state = run_direct_graph(state)
    return {field: state.get(field, "") for field in RESULT_FIELDS}


def summarize(rows: List[Dict[str, object]], runtime_seconds: float) -> Dict[str, object]:
    total = len(rows)
    success_rows = [row for row in rows if not clean_text(row.get("errors"))]
    fallback_execution_count = sum(1 for row in rows if str(row.get("fallback_used")).lower() in {"true", "1"})
    avg_slate = sum(safe_int(row.get("final_slate_size")) for row in rows) / max(total, 1)
    avg_intents = sum(safe_int(row.get("unique_sub_intents")) for row in rows) / max(total, 1)
    return {
        "total_queries": total,
        "success_count": len(success_rows),
        "failure_count": total - len(success_rows),
        "normalization_enabled_count": sum(1 for row in rows if str(row.get("normalization_enabled")).lower() in {"true", "1"}),
        "llm_success_count": sum(safe_int(row.get("llm_success")) for row in rows),
        "llm_fallback_count": sum(1 for row in rows if "fallback" in clean_text(row.get("final_advisor_status"))),
        "fallback_execution_count": fallback_execution_count,
        "avg_final_slate_size": round(avg_slate, 6),
        "avg_unique_sub_intents": round(avg_intents, 6),
        "top_calibrated_route": top_value(rows, "calibrated_route"),
        "top_execution_source": top_value(rows, "execution_source"),
        "top_quality_label": top_value(rows, "quality_label"),
        "runtime_seconds": round(runtime_seconds, 4),
    }


def top_value(rows: List[Dict[str, object]], field: str) -> str:
    counts = Counter(clean_text(row.get(field)) for row in rows if clean_text(row.get(field)))
    return counts.most_common(1)[0][0] if counts else ""


def report_markdown(rows: List[Dict[str, object]], summary: Dict[str, object]) -> str:
    lines = ["# MVP 24.4 LangGraph CORTEX Orchestrator", "", "## Summary"]
    for key, value in summary.items():
        lines.append(f"- {key}: {value}")
    lines.extend(["", "## Runs"])
    for row in rows[:25]:
        lines.append(
            f"- {row['raw_query']} -> route={row['calibrated_route']} "
            f"source={row['execution_source']} slate={row['final_slate_size']} "
            f"quality={row['quality_label']}"
        )
    return "\n".join(lines) + "\n"


def write_outputs(rows: List[Dict[str, object]], output_dir: Path, runtime_seconds: float) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = summarize(rows, runtime_seconds=runtime_seconds)
    write_csv(output_dir / "langgraph_orchestration_results.csv", rows, RESULT_FIELDS)
    write_csv(output_dir / "langgraph_orchestration_summary.csv", [summary], list(summary.keys()))
    (output_dir / "langgraph_orchestration_report.md").write_text(report_markdown(rows, summary), encoding="utf-8")
    return summary


def print_single(row: Dict[str, object], summary: Dict[str, object]) -> None:
    print("\nMVP 24.4 LangGraph CORTEX Orchestrator")
    print("-" * 100)
    for field in RESULT_FIELDS:
        value = row.get(field)
        print(f"{field}: {value if not isinstance(value, str) else console_text(value)}")
    print("\nSummary")
    print("-" * 100)
    for key, value in summary.items():
        print(f"{key}: {value}")


def print_summary(summary: Dict[str, object], output_dir: Path) -> None:
    print("\nSummary")
    print("-" * 100)
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("\nOutput files")
    print("-" * 100)
    print(output_dir / "langgraph_orchestration_results.csv")
    print(output_dir / "langgraph_orchestration_summary.csv")
    print(output_dir / "langgraph_orchestration_report.md")


def langgraph_message(graph) -> None:
    if graph is None:
        print("LangGraph is not installed. Run: python -m pip install -U langgraph")
        print("Using direct fallback orchestration for this prototype run.")


def run_single(args: argparse.Namespace, graph) -> None:
    start = time.perf_counter()
    row = run_orchestration(args.query, args=args, graph=graph)
    runtime = time.perf_counter() - start
    output_dir = Path(args.output_dir)
    summary = write_outputs([row], output_dir=output_dir, runtime_seconds=runtime)
    print_single(row, summary)


def run_batch(args: argparse.Namespace, graph) -> None:
    all_queries, source = load_all_queries(args.query_mode)
    selected = select_queries(
        queries=all_queries,
        query_mode=args.query_mode,
        sample_size=args.sample_size,
        start_index=args.start_index,
    )
    print("\nMVP 24.4 LangGraph CORTEX Orchestrator")
    print("-" * 100)
    print(f"query_mode: {args.query_mode}")
    print(f"source: {source}")
    print(f"selected_query_count: {len(selected)}")

    rows = []
    start = time.perf_counter()
    for position, (_query_index, query) in enumerate(selected):
        row = run_orchestration(query, args=args, query_position=position, graph=graph)
        rows.append(row)
        index = position + 1
        if index <= 10 or index % 100 == 0 or index == len(selected):
            print(
                f"[{index}/{len(selected)}] {console_text(query)} -> "
                f"normalized={console_text(row['normalized_query'])} "
                f"route={row['calibrated_route']} source={row['execution_source']} "
                f"slate={row['final_slate_size']} quality={row['quality_label']}"
            )

    runtime = time.perf_counter() - start
    output_dir = Path(args.output_dir)
    summary = write_outputs(rows, output_dir=output_dir, runtime_seconds=runtime)
    print_summary(summary, output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MVP 24.4 LangGraph CORTEX Orchestrator")
    parser.add_argument("--query", default="", help="Single query to orchestrate.")
    parser.add_argument("--sample-size", type=int, default=10, help="Batch sample size.")
    parser.add_argument(
        "--query-mode",
        choices=["smoke", "esci", "stratified_esci"],
        default="smoke",
        help="Batch query source mode.",
    )
    parser.add_argument("--start-index", type=int, default=0, help="Batch start offset.")
    parser.add_argument("--normalize-query", action="store_true", help="Run normalization node.")
    parser.add_argument("--no-ollama", action="store_true", help="Disable Ollama in LLM advisor node.")
    parser.add_argument("--model", default="qwen3:8b", help="Ollama model.")
    parser.add_argument("--timeout", type=int, default=20, help="Ollama timeout seconds.")
    parser.add_argument("--num-predict", type=int, default=256, help="Ollama num_predict limit.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Ollama temperature.")
    parser.add_argument("--think", action="store_true", help="Allow model thinking if supported.")
    parser.add_argument("--ollama-url", default="http://localhost:11434/api/generate", help="Ollama API URL.")
    parser.add_argument("--selective-llm", action="store_true", help="Use advisor selective LLM policy.")
    parser.add_argument(
        "--llm-policy",
        choices=["conservative", "default", "all", "audit"],
        default="conservative",
        help="LLM invocation policy.",
    )
    parser.add_argument("--audit-sample-rate", type=float, default=0.1, help="Audit LLM sample rate.")
    parser.add_argument(
        "--use-cache",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use advisor LLM cache.",
    )
    parser.add_argument("--refresh-cache", action="store_true", help="Refresh advisor LLM cache.")
    parser.add_argument(
        "--cache-dir",
        default="outputs/ollama_query_understanding/cache",
        help="Advisor LLM cache directory.",
    )
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR), help="Output directory.")
    parser.add_argument(
        "--retrieval-mode",
        choices=["sample", "full_esci"],
        default="sample",
        help="Route execution retrieval mode.",
    )
    parser.add_argument("--index-dir", default="data/esci_index", help="Full ESCI index directory.")
    parser.add_argument("--top-k", type=int, default=12, help="Maximum slate size / retrieval top-k.")
    parser.add_argument(
        "--retrieval-backend",
        choices=["lexical", "fts"],
        default="lexical",
        help="Full ESCI retrieval backend.",
    )
    parser.add_argument("--strict-filter-mode", choices=["remove", "demote", "hybrid"], default="hybrid")
    parser.add_argument("--strict-min-clean-results", type=int, default=8)
    parser.add_argument("--strict-candidate-multiplier", type=int, default=8)
    parser.add_argument("--scale-aware-rerank-mode", choices=["none", "strict_boost"], default="none")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.sample_size <= 0:
        raise ValueError("--sample-size must be positive")
    if args.start_index < 0:
        raise ValueError("--start-index must be non-negative")
    if args.top_k <= 0:
        raise ValueError("--top-k must be positive")

    graph = build_graph()
    langgraph_message(graph)

    if args.query:
        run_single(args, graph=graph)
        return
    run_batch(args, graph=graph)


if __name__ == "__main__":
    main()
