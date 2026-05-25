# CORTEX / PolicyRank-RL - Demo Report

## Executive Summary

CORTEX is an agentic AI search-ranking prototype that demonstrates governed decisions for product search. Rather than always applying an aggressive reranker, it can preserve an appropriate baseline result, execute mission-aware or behavior-aware intervention, or block repair that is too broad for a narrow query. The demo pairs stakeholder-readable explanations with inspectable technical traces and scenario-based cost/value analysis.

## Problem

Search intent is not uniform. A broad shopping mission may need coverage across several related needs, while a precise branded product request may be harmed by unnecessary expansion. A useful intelligent ranking system therefore needs more than relevance scoring: it needs controlled routing, repair guardrails, and explanations of what it changed and why.

## What CORTEX Does

- Retrieves product candidates using semantic or TF-IDF matching.
- Generates a search contract describing intent and constraints.
- Applies policy ranking, diversification, and mission/repair logic.
- Uses governance signals to choose a responsible execution route.
- Produces a final ranked slate with plain-English reasoning and technical evidence.
- Evaluates governed execution across sampled ESCI queries.
- Models monthly cost and possible value under explicit scenarios.

## Architecture

```text
Query -> Retrieval -> Contract Agent -> Policy Ranking -> Mission/Repair Stack
      -> Governance Agent -> Governed Runner -> Cost/Value Analysis -> Streamlit Demo
```

The Streamlit interface communicates this workflow in two layers:

| Mode | Audience | View |
|---|---|---|
| Simple Mode | Recruiters, product leaders, managers | Decision, rationale, business meaning, and ranked slate. |
| Technical Mode | Engineers and research reviewers | Contracts, filters, route signals, policy traces, and output tables. |

## Demo Script

1. Start the app with `.\.venv\Scripts\streamlit.exe run app.py`.
2. Begin in `Live Search Console`, which is the first/default tab.
3. Run `beach vacation packing list` in Simple Mode to demonstrate a broad shopping mission.
4. Review the ranked slate and the explanation of the governed route.
5. Toggle Technical Mode to show contract, ranking, enforcement, and trace details.
6. Run `adidas soccer cleats` to discuss why narrow queries need conservative repair behavior.
7. Open `Cost vs Value` and run the `base` scenario at `1,000,000` daily queries.
8. Explain monthly estimated cost, modeled value, break-even value per query, route contribution, and the scenario caveat.

Additional demo queries:

```text
new apartment kitchen setup
office desk setup
baby shower decorations
```

## Key Outputs

| Output family | What it demonstrates |
|---|---|
| Governed summary and final slate | The selected route and delivered result list for a query. |
| Governance decisions and trace | Why intervention was selected, preserved, or constrained. |
| Scalable governed evaluation | Behavior of the governed pipeline over a sampled workload. |
| Cost/value CSV outputs | Scenario economics, by-route contributions, and comparison across assumptions. |

## Cost/Value Scenario Summary

The Cost vs Value dashboard accepts a daily query volume and one of three modeled scenarios: `conservative`, `base`, or `optimistic`. It presents estimated monthly AI cost, estimated scenario value, net modeled value, value-to-cost ratio, break-even value per query, and a recommendation under the selected assumptions.

These figures are useful for business-case exploration and infrastructure discussion. **They are scenario-based assumptions, not measured production revenue, and they do not establish real CTR, conversion, engagement, or revenue lift.**

## Limitations

- The project uses ESCI/sample data and offline artifacts rather than a live commerce catalog.
- The local candidate pool may not cover every arbitrary query well.
- Offline routing and reward behavior should not be treated as production performance.
- Scenario value analysis depends on assumptions and must be validated with real operating data before business decisions.

## Future Work

- Add polished product screenshots and a hosted demo.
- Support real product imagery and richer catalog metadata.
- Run broader full-ESCI evaluation experiments.
- Evaluate online outcome models only if legitimate production labels and measurement controls become available.
