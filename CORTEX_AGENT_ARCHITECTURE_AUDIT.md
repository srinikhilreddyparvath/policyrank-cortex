# CORTEX Agent Architecture Audit

## Scope

This audit classifies the major CORTEX / PolicyRank-RL modules into the requested architecture categories:

1. Agent
2. Guardrail
3. Analyzer
4. Evaluator
5. Runner
6. UI/Dashboard
7. Storage/Utility

The current system is best understood as a governed ranking pipeline with specialized agents and guardrails. Some files are named "agent" because they make decisions or critiques, while others are analyzers, evaluators, or runners even when they participate in agentic workflows.

## Architecture Classification

| Module/file | Category | Purpose | Inputs | Outputs | When it runs | Overlap with other modules | Keep / Rename / Merge recommendation |
|---|---|---|---|---|---|---|---|
| `src/baseline_preservation_gate.py` | Guardrail | Prevents CORTEX from over-reranking when the semantic baseline is already strong. Computes baseline confidence, contract alignment, and selects `preserve_baseline`, `light_rerank`, or `full_cortex`. | Baseline candidate dataframe, CORTEX candidate dataframe, optional contract dict, `top_k`. | Final selected slate plus `BaselineGateDecision` metadata; flat logging dict via `summarize_gate_for_logging`. | During baseline-vs-CORTEX slate selection in evaluators or app flows. | Overlaps with governance baseline preservation, but at a lower level: this is score/contract based, while governance is route based. | Keep. Rename later to `baseline_preservation_guardrail.py` for category clarity. |
| `src/critic_verifier_agent.py` | Agent | Offline critic that diagnoses where CORTEX underperformed baseline/full CORTEX, classifies failure modes, risk, priority, and suggested repair actions. | `outputs/scalable_eval_mvp13_3_1_resilient_1000.csv`; optional `outputs/agent_governance_routes.csv`. | `critic_verifier_report.csv`, `critic_verifier_summary.csv`, `critic_verifier_failure_modes.csv`, `critic_verifier_repair_actions.csv`, `critic_verifier_high_risk_queries.csv`. | Offline after scalable evaluation logs exist. | Overlaps conceptually with `mission_critic_agent.py`, but this critic is global ranking/evaluation focused rather than mission-slate focused. | Keep. Rename to `offline_critic_verifier_agent.py` if an online critic is added. |
| `src/critic_guided_repair_simulator.py` | Analyzer | Simulates what would happen if critic repair actions selected baseline, gated CORTEX, full CORTEX, or rerun-required repairs. | `outputs/critic_verifier_report.csv`. | `critic_guided_repair_simulation.csv`, summary, by-action summary, high-impact queries. | Offline after critic verifier output exists. | Feeds learned router training and overlaps with router evaluation, but it does not learn or route online. | Keep. Rename to `repair_policy_simulator.py`. |
| `src/learned_repair_router.py` | Analyzer | Trains offline models to predict best logged policy: baseline, gated CORTEX, or full CORTEX. | `outputs/critic_guided_repair_simulation.csv`. | Training data, predictions, feature importance, summary, policy simulation CSVs. | Offline model development/training. | Overlaps with `repair_router_inference.py` and `train_and_save_repair_router.py`; this file trains/evaluates but does not appear to be the production inference path. | Keep, but rename to `learned_repair_router_trainer.py` if persisted model training is separated. |
| `src/repair_router_inference.py` | Runner | Loads persisted repair router artifacts and scores rows as a dry-run inference smoke test. | `models/learned_repair_router.pkl`, feature/metadata JSON, `outputs/critic_guided_repair_simulation.csv`. | `repair_router_inference_smoke_test.csv`, `repair_router_inference_summary.csv`. | Offline smoke test after model artifacts exist. | Overlaps with learned router training and app dry-run helpers. | Keep as `repair_router_inference_smoke_test.py` or merge into a router tooling package later. |
| `src/mission_agent.py` | Agent | Detects whether a query is mission-based and decomposes it into mission type, mission name, confidence, and sub-intents. | Query string or demo query list. | `MissionAnalysis`; optional `mission_agent_analysis.csv` from demo. | First stage of mission pipeline, or standalone demo. | Its keyword/template logic is partly echoed in governance query-type scoring. | Keep. Consider extracting shared mission keyword config to utility later. |
| `src/mission_slate_builder.py` | Runner | Builds a mission-aware product slate by retrieving candidates per sub-intent and selecting diverse rows. | Query, product dataset from `data/esci_balanced_sample.csv`, `data/esci_sample_products.csv`, or `data/sample_products.csv`. | Slate dataframe and summary dataframe; demo CSVs. | After `mission_agent` identifies/decomposes a mission query. | Overlaps retrieval/scoring with repair loop, which reuses its product loading and sub-intent retrieval helpers. | Keep. Rename to `mission_slate_runner.py` only if stricter category naming is desired. |
| `src/mission_slate_guardrails.py` | Guardrail | Filters weak/noisy mission slate rows using title relevance, role keywords, bad-match penalties, and max-per-sub-intent limits. | Raw mission slate from `build_mission_slate`. | Guarded slate, guarded summary, rejected candidates. | Immediately after mission slate building. | Overlaps with repair quality guardrails and strict repair rules; this one protects initial mission slate rows. | Keep. Rename to `mission_slate_relevance_guardrails.py`. |
| `src/mission_coverage_analyzer.py` | Analyzer | Measures sub-intent coverage, missing critical/important/optional needs, weighted coverage, readiness, and next action. | Query; guarded mission slate from `build_guarded_mission_slate`. | Coverage detail dataframe and coverage summary; demo CSVs. | After guarded mission slate is available. | Overlaps with mission critic because both discuss missing needs; analyzer computes facts, critic turns them into decisions. | Keep. |
| `src/mission_critic_agent.py` | Agent | Critiques mission coverage and decides whether to accept, retry missing needs, or reject/expand mission. Builds repair actions. | Query; coverage detail and summary from `analyze_mission_coverage`. | Mission critic report, summary, repair actions. | After mission coverage analyzer. | Overlaps with global `critic_verifier_agent.py`, but this one is mission-specific and online-ish in the mission pipeline. | Keep. Rename to `mission_slate_critic_agent.py` for precision. |
| `src/mission_repair_loop.py` | Runner | Executes the mission critic's repair actions by retrieving candidates for missing sub-intents, applying guardrails, and appending accepted repairs. | Query; mission critic repair actions; guarded slate; product dataset. | Repaired slate, repair summary, repair candidates. | After mission critic identifies missing needs. | Overlaps with slate builder retrieval and repair quality guardrails. | Keep. Rename to `mission_repair_runner.py`. |
| `src/mission_repair_quality_guardrails.py` | Guardrail | Applies stricter quality checks to repaired products before accepting them into the repaired slate. | Repaired slate and repair candidates from `mission_repair_loop`. | Quality-guarded repaired slate, quality summary, enriched candidates. | After mission repair loop. | Overlaps with initial mission slate guardrails and strict rules; this protects repaired rows generally. | Keep, but consider merging with `mission_strict_repair_rules.py` into layered `mission_repair_guardrails.py` later. |
| `src/mission_strict_repair_rules.py` | Guardrail | Adds sub-intent-specific hard rules for compound repairs such as cooler, soccer jersey, water bottle, kitchen towels, trash can, dish rack, first aid, bug spray, sandals. | Query; quality-guarded repair slate/candidates from `build_quality_guarded_repair_slate`. | Strict repair slate, strict summary, strict rejections, strict candidates. | Final mission repair validation stage before behavior-aware ranking. | Overlaps with repair quality guardrails, but it is the hard-rule layer for known risky sub-intents. | Keep. Rename to `mission_strict_repair_guardrails.py`. Merge later only after preserving separate soft vs hard rule semantics. |
| `src/behavior_aware_cortex.py` | Runner | Reranks strict mission slate using behavior proxy, confidence, mission stage, coverage contribution, cold-start rescue, exploration flags, concentration penalty, and policy reason codes. | Query; strict repair slate CSV. Optionally refreshes strict repair first. | Behavior-aware slate, summary, policy reasons. | After strict repair, or as a command that refreshes strict repair first. | Overlaps with governance in behavior rescue scoring; governance reads this module's outputs. | Keep. Rename to `behavior_aware_cortex_runner.py` if module category clarity matters. |
| `src/cortex_governance_agent.py` | Agent | Top-level route decision layer. Decides baseline-only, mission repair, strict repair, behavior-aware rerank, critic review, or repair rejection. Produces allowed/blocked modules and plain-English route reasoning. | Query; refreshed or existing strict repair and behavior-aware outputs; heuristic query signals. | Governance decisions, summary, trace. | Before governed execution; can refresh behavior/strict dependencies unless `--skip-refresh` is used. | Overlaps with baseline gate and mission agent query classification, but operates at route-policy level. | Keep. Rename to `cortex_route_governance_agent.py` if the system later adds other governance agents. |
| `src/governed_cortex_runner.py` | Runner | Executes the governed path selected by the governance agent and emits one final governed slate. Uses behavior-aware slate when available, strict slate as fallback, or baseline fallback row for baseline/preserved routes. | Query; governance decision/summary; strict slate; behavior-aware slate. | `governed_cortex_final_slate.csv`, `governed_cortex_summary.csv`, `governed_cortex_trace.csv`. | End-to-end per-query execution. | Overlaps with governance because it invokes governance; it should remain the execution wrapper, not the route policy owner. | Keep. Name is accurate. |
| `src/scalable_governed_evaluator.py` | Evaluator | Runs governed CORTEX across many queries and measures route distribution, final slate success, baseline preservation, coverage status, behavior-aware execution, and errors. | Query list from smoke, mission seed, ESCI data, or custom queries; governed runner. | Scalable governed eval, summary, by-route, by-execution-source, errors CSVs. | Offline batch evaluation. | Overlaps with Streamlit scalable dashboard, which visualizes its outputs. | Keep. Name is accurate. |
| `src/cost_value_governance_analyzer.py` | Analyzer | Estimates cost vs scenario value for governed routing at production volume using evaluation outputs and configurable assumptions. | Scalable governed eval outputs; daily query volume; scenario. | Cost/value summary, by-route, scenarios CSVs. | After scalable governed evaluation. | Overlaps with dashboard rendering only; no runtime ranking role. | Keep. Rename to `governance_cost_value_analyzer.py` for noun consistency. |
| `app.py` | UI/Dashboard | Main Streamlit app. Presents CORTEX story, demos, governed CORTEX run controls, router dry-run helpers, cost/value dashboard, and reads/writes selected UI-triggered outputs by calling backend modules. | User-selected query/settings; backend CSVs; model artifacts; subprocess calls to backend modules. | Streamlit UI; may refresh backend output CSVs when user clicks run controls. | Interactive app runtime. | Overlaps with pages under `pages/`, especially governed runner, governance, behavior-aware, scalable eval, and cost/value views. | Keep for consolidated demo. Consider moving backend helper logic out of app later, but do not change now. |
| `pages/16_1_Behavior_Aware_CORTEX.py` | UI/Dashboard | Dedicated Streamlit page for behavior-aware CORTEX outputs and charts. | Behavior-aware CSVs; query presets; subprocess call to `src.behavior_aware_cortex`. | Interactive UI; refreshed behavior-aware outputs when run. | Streamlit page runtime. | Overlaps with `app.py` behavior-aware sections. | Keep if multi-page demo remains. Otherwise merge into `app.py` navigation. |
| `pages/17_1_CORTEX_Governance_Agent.py` | UI/Dashboard | Dedicated Streamlit page for governance route decisions, signals, allowed/blocked modules, and governance output CSVs. | Governance CSVs; subprocess call to `src.cortex_governance_agent`. | Interactive UI; refreshed governance outputs when run. | Streamlit page runtime. | Overlaps with `app.py` governed demo and governance explanation. | Keep if multi-page demo remains. |
| `pages/18_1_Governed_CORTEX_Runner.py` | UI/Dashboard | Dedicated Streamlit page for governed runner execution, final slate, summary, trace, and download views. | Governed, governance, strict, behavior output CSVs; subprocess call to `src.governed_cortex_runner`. | Interactive UI; refreshed governed outputs when run. | Streamlit page runtime. | Overlaps with `app.py` governed CORTEX demo. | Keep if multi-page demo remains. |
| `pages/19_1_Scalable_Governed_Evaluation.py` | UI/Dashboard | Dedicated Streamlit page for scalable governed evaluation summaries, route charts, coverage diagnostics, errors, and run controls. | Scalable governed evaluation CSVs; subprocess call to `src.scalable_governed_evaluator`. | Interactive UI; refreshed scalable eval outputs when run. | Streamlit page runtime. | Overlaps with cost/value analyzer inputs and app's technical tabs. | Keep. |
| `storage/*.csv`, `storage/*.json`, `outputs/*.csv`, `models/*.pkl/json` | Storage/Utility | Persist intermediate state, logs, generated evaluation outputs, learned router artifacts, and dashboard data. | Written by backend modules and Streamlit UI actions. | CSV/JSON/model artifacts consumed by later modules. | Throughout offline training, demo, evaluation, and UI execution. | Storage files create loose coupling between stages but also duplicate state across runs. | Keep. Later define an artifact manifest to reduce ambiguity. |

## Specific Audit Notes

### Baseline Preservation

`baseline_preservation_gate.py` is a guardrail, not an agent. It makes a local slate-selection decision from baseline quality and contract alignment signals. The route-level governance layer also preserves baseline-style handling, but at a different abstraction. The baseline gate protects ranking quality within a CORTEX-vs-baseline choice; the governance agent blocks whole mission/repair branches when a query should not be expanded.

Recommendation: keep both, but rename the gate to make it clearly a guardrail.

### Critic Verifier

`critic_verifier_agent.py` is an offline critic agent. It reads historical evaluation logs, diagnoses failure modes, recommends repair actions, and prioritizes risky queries. It does not execute live repairs.

Recommendation: keep as the global offline critic. If an online critic is added, rename this to `offline_critic_verifier_agent.py`.

### Repair Simulator / Router

`critic_guided_repair_simulator.py` is an analyzer that turns critic recommendations into conservative and oracle-style simulations over logged policies. `learned_repair_router.py` trains a learned router from those simulations. `repair_router_inference.py` is a smoke-test runner for persisted router artifacts.

Recommendation: keep the stages separate, but rename them around lifecycle:

- `repair_policy_simulator.py`
- `learned_repair_router_trainer.py`
- `repair_router_inference_smoke_test.py`

### Learned Repair Router

The learned router is not currently the top-level governed CORTEX route owner. It is an offline model that predicts best logged policy among baseline, gated CORTEX, and full CORTEX. Governance now routes among broader mission/repair/behavior paths.

Recommendation: keep, but document it as a lower-level learned policy router, not the current governance agent.

### Mission Agent

`mission_agent.py` is correctly named. It converts a query into mission intent, confidence, mission type, and sub-intents. It is the planning/decomposition agent for mission shopping.

Recommendation: keep.

### Mission Slate Builder

The builder is a runner/builder rather than an agent. It operationalizes the mission decomposition by retrieving and selecting products for sub-intents.

Recommendation: keep. Rename only if the team wants category-aligned names.

### Mission Guardrails

`mission_slate_guardrails.py`, `mission_repair_quality_guardrails.py`, and `mission_strict_repair_rules.py` are guardrail layers. They are not redundant if treated as staged filters:

- Initial slate relevance guardrails protect raw mission slate rows.
- Repair quality guardrails protect generated repair rows with soft/general repair-quality checks.
- Strict repair rules protect known risky sub-intents with hard product-type rules.

Recommendation: keep, but rename strict rules to strict guardrails for consistency.

### Mission Coverage Analyzer

`mission_coverage_analyzer.py` computes coverage facts and readiness labels. It should remain an analyzer, not an agent, because it does not decide policy beyond recommended next actions.

Recommendation: keep.

### Mission Critic Agent

`mission_critic_agent.py` turns coverage facts into a critique, priority, risk score, decision, and repair actions. It is a mission-specific critic agent.

Recommendation: keep. Rename to `mission_slate_critic_agent.py` if the global critic remains in the architecture.

### Mission Repair Loop

`mission_repair_loop.py` is a runner. It executes missing-need repair actions from the mission critic.

Recommendation: keep. Rename to `mission_repair_runner.py` for category clarity.

### Strict Repair Rules

`mission_strict_repair_rules.py` is a hard guardrail layer. It is intentionally more specific than the quality guardrails and should not be merged prematurely unless the soft/hard rule distinction is preserved.

Recommendation: keep. Rename to `mission_strict_repair_guardrails.py`.

### Behavior-Aware CORTEX

`behavior_aware_cortex.py` is a ranking runner layered after strict repair. It is not a learned behavior model; it applies explainable proxy behavior, mission stage, coverage contribution, cold-start rescue, exploration, and policy reason codes.

Recommendation: keep. Rename to `behavior_aware_cortex_runner.py` if category clarity matters.

### Governance Agent

`cortex_governance_agent.py` is the top-level route policy agent. It should be presented as the coordinator that prevents the repeated mission/repair/critic modules from looking redundant. It decides which path should be trusted, which modules are allowed, and which modules are blocked.

Recommendation: keep. It is the clearest actual "agent" in the current architecture.

### Governed Runner

`governed_cortex_runner.py` is the execution wrapper. It should not absorb governance policy logic. Its job is to call governance, choose the best available governed slate artifact, and publish final governed outputs.

Recommendation: keep.

### Scalable Governed Evaluator

`scalable_governed_evaluator.py` is a batch evaluator. It validates that governed execution works across query sets and reports success, fallback, coverage, and route distribution.

Recommendation: keep.

### Cost Value Analyzer

`cost_value_governance_analyzer.py` is an analyzer that uses scenario assumptions, not measured revenue. It belongs after scalable evaluation.

Recommendation: keep. The module is not part of ranking execution.

### Streamlit App

`app.py` is the main dashboard and demo shell. It reads backend artifacts and triggers backend modules. The `pages/` files are dedicated dashboards for behavior-aware, governance, governed runner, and scalable governed evaluation.

Recommendation: keep `app.py` as the consolidated demo until the UI is reorganized. Later, extract subprocess/helper logic into a UI service utility to reduce duplication between `app.py` and `pages/`.

## Redundancy And Naming Risks

The system can look redundant because several modules contain words like critic, repair, guardrail, and router. The clean separation is:

| Concern | Current modules | Distinction |
|---|---|---|
| Global ranking critique | `critic_verifier_agent.py` | Offline diagnosis from historical eval logs. |
| Mission slate critique | `mission_critic_agent.py` | Per-query mission coverage critique and repair actions. |
| Repair simulation | `critic_guided_repair_simulator.py` | Offline "what if" analysis over logged policies. |
| Learned repair router | `learned_repair_router.py`, `repair_router_inference.py` | Lower-level learned policy selection among logged baseline/gated/full CORTEX. |
| Mission repair execution | `mission_repair_loop.py` | Actually retrieves replacements for missing mission needs. |
| Initial slate guardrails | `mission_slate_guardrails.py` | Filters the first mission slate. |
| Repair quality guardrails | `mission_repair_quality_guardrails.py` | Filters repaired rows with general quality logic. |
| Strict repair guardrails | `mission_strict_repair_rules.py` | Applies hard product-type rules for known risky sub-intents. |
| Route governance | `cortex_governance_agent.py` | Decides which modules should be trusted for a query. |
| Governed execution | `governed_cortex_runner.py` | Produces the final governed slate from route decision and artifacts. |

## Clean Final Architecture Story

CORTEX should be described as a governed agentic ranking system with five layers:

1. **Baseline Safety Layer**
   - Standard retrieval/ranking produces a baseline.
   - The baseline preservation guardrail decides whether CORTEX should preserve, lightly rerank, or fully rerank.

2. **Mission Understanding Layer**
   - The mission agent detects compound shopping missions and decomposes them into sub-intents.
   - The mission slate builder retrieves candidate products for those sub-intents.

3. **Quality And Coverage Layer**
   - Mission slate guardrails reject weak initial matches.
   - The coverage analyzer identifies missing critical, important, and optional needs.
   - The mission critic converts coverage gaps into accept/retry/reject decisions and repair actions.

4. **Repair And Behavior Layer**
   - The mission repair runner retrieves missing-needs candidates.
   - Repair quality guardrails and strict repair guardrails prevent bad repairs.
   - Behavior-aware CORTEX reranks the strict slate with explainable behavior proxy, mission-stage balance, cold-start rescue, exploration, and policy reasons.

5. **Governance And Evaluation Layer**
   - The CORTEX governance agent decides which path is appropriate: preserve baseline, run mission repair, run strict repair, run behavior-aware rerank, send to critic review, or reject repair for narrow queries.
   - The governed runner executes the selected path and emits one final governed slate.
   - Scalable evaluation and cost/value analysis measure whether the governed system works and whether route selectivity controls cost.

In this story, repeated agents are not redundant. They operate at different scopes:

- The **global critic verifier** audits historical CORTEX failures.
- The **mission critic** critiques one mission slate's missing needs.
- The **governance agent** chooses which route should run.
- The **learned repair router** is a lower-level offline policy-selection model, not the top-level route owner.

The naming cleanup should make these scopes explicit so the architecture reads as layered control, not duplicated agents.
