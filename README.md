# CORTEX Engine — Agentic RL Search Ranking System

**CORTEX Engine** is an experimental search ranking and recommendation research prototype that explores how **Agentic AI**, **Reinforcement Learning**, **contract-aware ranking**, and **learned routing** can improve search result quality beyond a single static reranker.

The project started as a **PolicyRank-RL** prototype and evolved into a modular agentic ranking system with **semantic retrieval**, **LLM-based search contracts**, **candidate filtering**, **slate-level reinforcement learning**, **critic verification**, **repair simulation**, and a **learned repair router**.

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![RL](https://img.shields.io/badge/RL-Slate_Q--Learning-purple)
![Agentic AI](https://img.shields.io/badge/Agentic_AI-CORTEX-green)
![Status](https://img.shields.io/badge/Status-Research_Prototype-orange)

---

## What is CORTEX?

**CORTEX Engine** is a research prototype that asks a simple question:

> Instead of applying the same reranker to every query, can a search system learn when to trust baseline retrieval, when to apply gated reranking, and when to use full agentic reranking?

Most traditional search systems follow a fixed ranking flow:

```text
query → retrieval → reranker → final ranked list
```

CORTEX follows a more adaptive agentic flow:

```text
query → retrieve → reason → filter → rank → critique → route → repair
```

The system does **not** force full reranking on every query. Instead, it learns to choose among:

- **Baseline retrieval**
- **Gated CORTEX**
- **Full CORTEX**

This helps reduce **over-reranking**, preserve strong baseline results when appropriate, and improve slate-level search quality.

---

## Key Experimental Result

The strongest offline evaluation so far used **1000 queries**.

| Strategy | Average Reward@5 |
|---|---:|
| Baseline | 0.7528 |
| Gated CORTEX | 0.8704 |
| Full CORTEX | 0.8669 |
| **Router-Integrated CORTEX** | **0.9101** |
| Oracle Upper Bound | 0.9162 |

The learned router achieved:

- **+0.1573 lift vs baseline**
- **+0.0397 lift vs gated CORTEX**
- **+0.0433 lift vs full CORTEX**
- **0.0060 regret vs oracle**
- **90.0% prediction match rate**
- **92.4% near-oracle rate**

This suggests that the router is successfully learning when to trust **baseline**, **gated CORTEX**, or **full CORTEX**.

---

## High-Level Architecture

```text
User Query
   ↓
Semantic / TF-IDF Retrieval
   ↓
LLM Search Contract Agent
   ↓
Contract-Aware Candidate Filter
   ↓
Slate Q-Learning Policy
   ↓
Multi-Agent Diversification
   ↓
Final Slate Enforcement
   ↓
Critic / Verifier Agent
   ↓
Repair Simulation
   ↓
Learned Repair Router
   ↓
Router-Integrated CORTEX Policy
```

---

## Core Components

| Component | Purpose |
|---|---|
| **Retrieval Agent** | Retrieves product candidates using TF-IDF or semantic search |
| **Search Contract Agent** | Converts raw queries into structured intent contracts |
| **Contract-Aware Filter** | Scores and filters candidates against query intent |
| **Slate Q-Learning Agent** | Learns ranking actions based on query and slate state |
| **Multi-Agent Diversifier** | Improves diversity, substitutes, complements, and slate coverage |
| **Final Slate Enforcer** | Applies final contract guardrails before output |
| **Critic / Verifier Agent** | Detects ranking failure modes and risk signals |
| **Repair Simulator** | Simulates repair actions using logged outcomes |
| **Learned Repair Router** | Chooses between baseline, gated CORTEX, and full CORTEX |
| **Router-Integrated Evaluator** | Compares router policy against baseline, gated, full, and oracle policies |

---

## Retrieval Layer

The project supports two retrieval modes:

- **TF-IDF retrieval**
- **Semantic retrieval**

Semantic retrieval is the preferred default because it captures query-product meaning beyond exact token overlap.

Relevant modules:

```text
src/retrieval.py
src/semantic_retrieval.py
```

---

## Search Contract Agent

The contract agent converts a raw query into a structured search contract.

The contract can include:

- **Product type**
- **Required terms**
- **Preferred terms**
- **Blocked terms**
- **Brand preference**
- **Quality preference**
- **Price sensitivity**
- **Ranking weights**
- **Dynamic filters**
- **Explanation**

Two contract modes are supported:

```text
Rule-based contract
LLM Agent contract
```

Relevant modules:

```text
src/contracts.py
src/llm_contract_agent.py
```

---

## Contract-Aware Candidate Filtering

After retrieval, CORTEX applies a contract-aware filter to check whether products align with the generated search contract.

The filter can reason about:

- **Required query terms**
- **Preferred terms**
- **Product type alignment**
- **Brand preference**
- **Blocked terms**
- **Contract match score**

Relevant module:

```text
src/contract_filters.py
```

---

## Final Slate Enforcement

Even after reranking, the final slate can still violate user intent. The final slate enforcer applies additional guardrails before showing the result list.

It tracks:

- **Positive contract rows**
- **Blocked rows**
- **Low coverage**
- **Brand preference**
- **Final contract score**
- **Enforcement status**
- **Enforcement explanation**

Relevant module:

```text
src/final_slate_enforcer.py
```

---

## Slate Q-Learning

CORTEX includes a slate-level reinforcement learning loop. Instead of only ranking individual items, it learns which slate-level action is useful under different contract states.

The Q-learning state can depend on:

- **Contract quality**
- **Retrieval confidence**
- **Query/candidate condition**

The action can change ranking weights across:

- **Relevance**
- **Rating**
- **Price**
- **Diversity**
- **Contract priority**

Relevant modules:

```text
src/slate_q_learning.py
src/policy_compiler.py
src/slate_reward.py
```

---

## Multi-Agent Diversification

CORTEX includes a multi-agent slate diversification layer that selects positions using multiple agent-like objectives.

The diversifier can reason across:

- **Exact intent matches**
- **Substitutes**
- **Complements**
- **Diversity**
- **Redundancy**
- **Position-level tradeoffs**

Relevant module:

```text
src/multi_agent_diversifier.py
```

---

## Baseline Preservation Gate

One key lesson from the experiments is that full reranking is not always better. Sometimes the baseline is already strong.

The baseline preservation gate is designed to prevent **over-reranking**. It allows CORTEX to preserve the baseline, lightly rerank, or fully rerank depending on query and slate conditions.

Relevant module:

```text
src/baseline_preservation_gate.py
```

---

## Critic / Verifier Agent

The critic agent identifies possible failure modes in the ranking process.

It can flag issues such as:

- **Baseline stronger than CORTEX**
- **Full CORTEX over-reranking**
- **Low contract alignment**
- **Retrieval uncertainty**
- **Possible label noise**
- **Over-diversification**
- **Contract over-filtering**
- **Mission query requiring decomposition**

Relevant modules:

```text
src/critic_verifier_agent.py
src/critic_guided_repair_simulator.py
```

---

## Learned Repair Router

The learned repair router is one of the most important pieces of the project.

It learns to choose among:

```text
baseline
gated_cortex
full_cortex
```

using features such as:

- **Critic risk score**
- **Critic priority**
- **Governance route**
- **Gate confidence**
- **Contract alignment**
- **Top-k retrieval confidence**
- **Label quality**
- **Exclusion violation rate**
- **Blocked rows**
- **Positive contract rows**

Relevant modules:

```text
src/learned_repair_router.py
src/repair_router_oos_validator.py
src/train_and_save_repair_router.py
src/repair_router_inference.py
```

---

## MVP Progress

| MVP | Component | Status |
|---|---|---|
| MVP 12.x | Experiment logging | Complete |
| MVP 13.3 | Baseline preservation gate | Complete |
| MVP 13.5 | Agent governance controller | Complete |
| MVP 13.6 | Critic / verifier agent | Complete |
| MVP 13.7 | Critic-guided repair simulator | Complete |
| MVP 13.8 | Learned repair router | Complete |
| MVP 13.9 | Out-of-sample validation | Complete |
| MVP 14.1 | Saved router model | Complete |
| MVP 14.2 | Router inference smoke test | Complete |
| MVP 14.3 | Router-integrated scalable evaluator | Complete |
| MVP 14.4 | Streamlit router dry-run toggle | Complete |
| MVP 14.5 | Router decision logging | Complete |
| MVP 14.6 | Router dry-run analyzer | Complete |

---

## Streamlit App

The project includes a Streamlit app for interactive experimentation.

The app supports:

- **Live search queries**
- **TF-IDF and semantic retrieval**
- **Rule-based and LLM-generated contracts**
- **Manual policy ranking**
- **Bandit auto-selection**
- **Slate Q-learning**
- **Multi-agent diversification**
- **Router dry-run mode**
- **Router decision logging**
- **Router dashboard**
- **Learning dashboard**
- **Architecture overview**

Main file:

```text
app.py
```

Run the app with:

```powershell
.venv\Scripts\streamlit.exe run app.py
```

or:

```powershell
.venv\Scripts\python.exe -m streamlit run app.py
```

---

## Main Commands

Run the Streamlit app:

```powershell
.venv\Scripts\streamlit.exe run app.py
```

Run the scalable evaluator:

```powershell
.venv\Scripts\python.exe -m src.scalable_evaluator --sample-size 1000
```

Run the router-integrated scalable evaluator:

```powershell
.venv\Scripts\python.exe -m src.router_integrated_scalable_evaluator
```

Run the router dry-run analyzer:

```powershell
.venv\Scripts\python.exe -m src.router_dry_run_analyzer
```

Check Git status:

```powershell
git status
```

---

## Important Output Files

### Scalable Evaluation

```text
outputs/scalable_eval_mvp13_3_1_resilient_1000.csv
outputs/scalable_eval_summary_mvp13_3_1_resilient_1000.csv
outputs/scalable_eval_by_query_mvp13_3_1_resilient_1000.csv
outputs/scalable_eval_lift_chart_mvp13_3_1_resilient_1000.png
```

### Router Evaluation

```text
outputs/router_integrated_scalable_eval.csv
outputs/router_integrated_scalable_eval_summary.csv
outputs/router_integrated_scalable_eval_by_policy.csv
outputs/router_integrated_scalable_eval_by_route.csv
outputs/router_integrated_scalable_eval_high_impact.csv
```

### Router Dry-Run Logs

```text
storage/router_dry_run_log.csv
outputs/router_dry_run_log_summary.csv
outputs/router_dry_run_log_by_policy.csv
outputs/router_dry_run_log_by_route.csv
outputs/router_dry_run_log_policy_probability_audit.csv
```

### Saved Router Model

```text
models/learned_repair_router.pkl
models/learned_repair_router_features.json
models/learned_repair_router_metadata.json
```

---

## Repository Structure

```text
policyrank-rl/
│
├── app.py
├── README.md
├── PROJECT_STATUS.md
├── requirements.txt
│
├── data/
├── models/
├── outputs/
├── storage/
└── src/
```

Key source files:

```text
src/semantic_retrieval.py
src/llm_contract_agent.py
src/contract_filters.py
src/final_slate_enforcer.py
src/slate_q_learning.py
src/policy_compiler.py
src/slate_reward.py
src/multi_agent_diversifier.py
src/baseline_preservation_gate.py
src/agent_governance_controller.py
src/critic_verifier_agent.py
src/critic_guided_repair_simulator.py
src/learned_repair_router.py
src/router_integrated_scalable_evaluator.py
src/repair_router_inference.py
src/router_dry_run_analyzer.py
```

---

## Setup

Clone the repo:

```powershell
git clone https://github.com/srinikhilreddyparvath/policyrank-cortex.git
cd policyrank-cortex
git lfs pull
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Create a local environment file if needed:

```powershell
notepad .env
```

Do **not** commit `.env`.

---

## Current Limitations

This is a research prototype, not a production ranking system.

Current limitations:

- Uses **Amazon ESCI-style data**
- Some reward signals are **simulated**
- Router dry-run features are approximated from **session state**
- No live online experimentation yet
- Mission-based shopping is planned but not complete
- Behavior and multimodal signals are planned but not complete

---

## Roadmap

Next planned stages:

- **MVP 15:** Mission-Based Shopping Agent
- **MVP 16:** Behavior-Aware CORTEX
- **MVP 17:** Multimodal CORTEX
- **MVP 18:** Online Learning Loop
- **Refactor:** Modularize Streamlit and router utilities

---

## Disclaimer

This repository is an independent research and prototype project for experimentation and demonstration of agentic ranking concepts.

It is **not production-ready** and should not be treated as a production ranking system without additional validation, testing, monitoring, and governance.
