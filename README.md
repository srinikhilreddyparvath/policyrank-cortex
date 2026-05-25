# CORTEX / PolicyRank-RL

**An agentic AI search-ranking prototype for governed, explainable product discovery.**

CORTEX / PolicyRank-RL explores how an AI-assisted search system can make controlled ranking decisions instead of applying the same reranking strategy to every query. It combines semantic retrieval, intent contracts, reinforcement-learning-inspired slate ranking, mission-aware repair, governance decisions, offline evaluation, and a polished Streamlit demonstration experience.

CORTEX decides when to preserve baseline ranking, when to apply mission repair, when to use behavior-aware reranking, and when to block aggressive repair for narrow product queries.

This repository is designed as a research and product prototype: it demonstrates system architecture, decision traceability, offline analysis, and a stakeholder-friendly demo rather than claiming a deployed production ranking service.

## Why It Matters

Product search queries behave differently. A focused request such as `adidas soccer cleats` should not be expanded into an unrelated shopping mission, while `beach vacation packing list` may benefit from a useful multi-item slate. CORTEX treats that distinction as a governed routing decision, then exposes the reasoning in both simple and technical views.

## Key Capabilities

| Capability | What it provides |
|---|---|
| Semantic retrieval | Finds candidates based on query and product meaning. |
| Search contract generation | Turns a query into structured ranking intent and constraints. |
| Contract-aware filtering | Filters and scores candidates against intent requirements. |
| Slate Q-learning | Selects ranking behavior at the result-slate level. |
| Multi-agent diversification | Balances relevance, substitutes, complements, and coverage. |
| Mission-aware shopping repair | Identifies and repairs missing needs in multi-intent shopping tasks. |
| Strict compound-intent repair rules | Prevents weak or overly broad repairs for narrow queries. |
| Behavior-aware CORTEX | Uses behavior confidence, mission stage, and policy rationale. |
| Governance Agent | Chooses a controlled route based on opportunity and risk. |
| End-to-end Governed CORTEX Runner | Executes a governed decision and emits reviewable output artifacts. |
| Scalable governed evaluation | Runs the governed workflow across sampled ESCI queries. |
| Cost vs value governance analyzer | Models infrastructure cost against scenario-based business value. |
| Dark dual-mode Streamlit UX | Provides simple stakeholder explanations and technical diagnostics. |

## Architecture

```text
Query -> Retrieval -> Contract Agent -> Policy Ranking -> Mission/Repair Stack
      -> Governance Agent -> Governed Runner -> Cost/Value Analysis -> Streamlit Demo
```

At a high level:

| Stage | Purpose |
|---|---|
| Retrieval | Generate candidate products using semantic or TF-IDF retrieval. |
| Contract Agent | Infer intent, constraints, and ranking priorities. |
| Policy Ranking | Rank a slate with policy and contract signals. |
| Mission/Repair Stack | Expand or repair broad missions only when justified. |
| Governance Agent | Preserve, rerank, repair, or block intervention based on signals. |
| Governed Runner | Execute the selected route and produce traceable outputs. |
| Cost/Value Analysis | Estimate cost and scenario value at production-scale volumes. |
| Streamlit Demo | Communicate the decision in Simple Mode or expose internals in Technical Mode. |

## Run The Demo

From PowerShell:

```powershell
cd C:\Users\tonic\GitHubProjects\policyrank-cortex-clean
.\.venv\Scripts\streamlit.exe run app.py
```

### Current Demo Path

1. Open Streamlit.
2. `Live Search Console` opens first.
3. Type any query.
4. Run in `Simple Mode` for a clear governed result explanation.
5. Toggle `Technical Mode` for contracts, signals, policy outputs, and traceability.
6. Open the `Cost vs Value` tab to review production-volume scenario economics.

### Example Queries

```text
beach vacation packing list
new apartment kitchen setup
adidas soccer cleats
office desk setup
baby shower decorations
```

## Backend Evaluation Commands

Run these from the repository root:

```powershell
python -m src.governed_cortex_runner --query "beach vacation packing list" --skip-refresh
python -m src.scalable_governed_evaluator --sample-size 100 --query-mode esci
python -m src.cost_value_governance_analyzer --daily-query-volume 1000000 --scenario base
```

The analyzers write CSV output artifacts under `outputs/`, which the Streamlit app reads for governed decisions, scalable evaluation summaries, and cost/value presentation.

## MVP Status

| Stage | Outcome | Status |
|---|---|---|
| MVP 13.x | Baseline gates, governance foundations, critic/repair simulation, learned router | Complete |
| MVP 14.x | Saved router, scalable router evaluation, Streamlit dry run, logging and analysis | Complete |
| MVP 15.x | Mission detection, slate construction, repair, quality guardrails, strict repair rules | Complete |
| MVP 16.x | Behavior-aware CORTEX and interactive product-oriented UX | Complete |
| MVP 17 | Governance Agent | Complete |
| MVP 18 | End-to-end Governed CORTEX Runner | Complete |
| MVP 19 | Scalable governed evaluation, dark dual-mode UI, cost/value analysis and dashboard | Complete |
| MVP 20 | Final README and demo report polish | Complete |

## Outputs Reviewers Can Inspect

| Output | Description |
|---|---|
| Governed result summary and slate | Selected governance route and final ranked output. |
| Governance decision trace | Decision signals and human-readable reasoning. |
| Scalable governed evaluation | Aggregate behavior across a query sample. |
| Cost/value summary | Chosen scenario economics and recommendation. |
| Route-level cost/value data | Which governance routes drive modeled cost and value. |
| Scenario comparison data | Conservative, base, and optimistic modeled outcomes. |

## Scope And Limitations

> This is a research prototype using ESCI/sample data. Some arbitrary queries may have limited product coverage depending on the local candidate pool.

> Cost/value outputs are scenario-based assumptions for business-case analysis, not measured production revenue.

Offline reward metrics and modeled cost/value scenarios support experimentation and system evaluation. They do not represent verified production click-through-rate, engagement, conversion, or revenue lift.

## Documentation

- [PROJECT_STATUS.md](PROJECT_STATUS.md) summarizes the current milestone and remaining optional enhancements.
- [DEMO_REPORT.md](DEMO_REPORT.md) provides an executive overview and a concise demonstration script.
