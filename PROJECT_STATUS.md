# CORTEX / PolicyRank-RL - Project Status

## Current Stage

**MVP 20: Final README + Demo Report Polish - Complete**

CORTEX is now presented as a demo-ready agentic AI search-ranking research prototype. The repository includes an interactive governed-search experience, traceable backend runners and evaluation artifacts, and a cost/value dashboard that is clearly framed as scenario analysis.

## Completed MVPs

| MVP | Scope | Completed outcome |
|---|---|---|
| MVP 13.x | Foundation, router, and gates | Established baseline preservation, governance signals, critic/repair analysis, and learned routing foundations. |
| MVP 14.x | Router scalable evaluation and logging | Saved and evaluated the router, added dry-run integration, logging, and analysis outputs. |
| MVP 15.x | Mission stack | Added mission detection, slate building, coverage review, repair, guardrails, and strict compound-intent rules. |
| MVP 16.x | Behavior-aware CORTEX | Added behavior-aware governed ranking signals and interactive presentation. |
| MVP 17 | Governance Agent | Consolidated routing decisions that determine when intervention is warranted or risky. |
| MVP 18 | Governed runner | Added an end-to-end runner that executes the governance decision and emits inspectable outputs. |
| MVP 19 | Scalable evaluation, dark UI, and cost/value | Added scalable governed evaluation, a dark dual-mode Streamlit workflow, and scenario-based cost/value analysis. |
| MVP 20 | Final polish | Refined repository documentation and added a concise demo report for technical and non-technical reviewers. |

## What Is Working Now

- Dark Streamlit experience with `Live Search Console` as the default entry point.
- Simple Mode for stakeholder-readable result summaries and Technical Mode for diagnostics.
- `Cost vs Value` dashboard for scenario comparison, break-even framing, route-level analysis, and raw output inspection.
- Semantic and TF-IDF retrieval paths with contract-aware candidate handling.
- Slate ranking, diversification, mission repair, strict repair constraints, and behavior-aware execution.
- Governance routing that can preserve baseline handling, execute intervention, or reject overly aggressive repair.
- End-to-end governed runner for a selected query.
- Scalable governed evaluator for sampled ESCI query runs.
- Cost/value analyzer producing summary, by-route, and scenario CSV artifacts.
- Offline artifacts suitable for review in the app and through generated CSV outputs.

## Demonstration Entry Points

```powershell
.\.venv\Scripts\streamlit.exe run app.py
python -m src.governed_cortex_runner --query "beach vacation packing list" --skip-refresh
python -m src.scalable_governed_evaluator --sample-size 100 --query-mode esci
python -m src.cost_value_governance_analyzer --daily-query-volume 1000000 --scenario base
```

## Evidence Boundaries

- This repository is a research prototype based on ESCI/sample data and offline workflow artifacts.
- Candidate coverage varies with the local sample, so arbitrary user queries may produce limited result coverage.
- Offline evaluation results are not evidence of deployed production behavior.
- Cost/value outputs are scenario-based assumptions for business-case analysis, not measured production revenue.
- No claim is made here about real production CTR, engagement, conversion, or revenue lift.

## Optional Future Enhancements

- Capture polished screenshots for the README and demo report.
- Host a deployable demonstration environment.
- Add real product image support where licensed product assets are available.
- Run a larger full ESCI evaluation for broader coverage analysis.
- Add a true production CTR or engagement model only if valid real labels and appropriate evaluation controls exist.
