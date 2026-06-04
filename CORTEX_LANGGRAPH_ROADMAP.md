# CORTEX LangGraph Roadmap

## Why LangGraph Is Being Added

CORTEX now has several deterministic and optional advisory agents: query normalization, rule query understanding, Ollama query understanding, governance calibration, route execution, and route quality analysis. LangGraph gives us a clean way to describe that flow as explicit nodes with traceable state between each step.

The goal is orchestration clarity. LangGraph makes it easier to see which agent ran, what it produced, and where fallback behavior occurred.

## Why It Is Optional Now

The current direct Python runners remain the source of truth. They are stable, scriptable from PowerShell, and already produce the benchmark artifacts used by the project.

MVP 24.4 keeps LangGraph optional so the repo still works when `langgraph` is not installed. The prototype can run in direct fallback mode, and installing LangGraph only changes the orchestration mechanism, not governance or route behavior.

## Current Source Of Truth

Direct Python modules remain authoritative:

- `src/query_normalization_agent.py`
- `src/query_understanding_agent.py`
- `src/ollama_query_understanding_advisor.py`
- `src/governance_calibration_dry_run.py`
- `src/calibrated_route_execution_adapter.py`
- `src/route_specific_slate_quality_analyzer.py`

LangGraph is an orchestration layer, not a replacement for governance.

## Current Graph Nodes

- Spell Check / Normalization
- Rule Query Understanding
- Ollama Advisor
- Governance Calibration
- Calibrated Route Execution
- Quality Labeling

## Future Graph Nodes

- Spell Check / Normalization
- Rule Query Understanding
- Ollama Advisor
- Governance Agent
- Strict Repair
- Mission Repair
- Behavior-Aware Rerank
- LLM Slate Critic
- Critic-Guided Repair Loop
- Human Review
- Benchmark Logger

## Roadmap

- MVP 24.4 optional orchestrator prototype
- MVP 25 LLM mission sub-intent expansion
- MVP 26 full ESCI product index
- MVP 27 reward evaluator
- MVP 28 critic-guided repair
- MVP 28.2 full LangGraph orchestration
