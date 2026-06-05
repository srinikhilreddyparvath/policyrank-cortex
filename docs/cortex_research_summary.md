# CORTEX Research Summary

## Project Thesis

CORTEX is a governed route-aware ranking system for commerce retrieval. The central thesis is that product retrieval quality depends not only on larger candidate indexes, but on explicit query understanding, governance routes, constraint preservation, and route-specific evaluation.

The system shows that naive scale and simple penalty-based reranking are insufficient. Larger lexical candidate spaces can introduce noisy near matches, especially for strict negation and critic-review queries. This motivates governed route-aware retrieval optimization.

## Problem Statement

Commerce queries mix narrow product lookup, setup missions, gift planning, compatibility constraints, and hard negation constraints. A single baseline retrieval path does not reliably handle all of these cases. In particular:

- Narrow product queries should preserve baseline precision.
- Mission queries need diversified sub-intent coverage.
- Compatibility and negation queries need strict constraint tracking.
- Noisy or ambiguous queries need critic-review handling.
- Scaling an FTS index can increase recall while also increasing near-match noise.

## System Overview

CORTEX separates query interpretation, governance, execution, and evaluation:

- Query normalization cleans commerce typos while protecting model and constraint tokens.
- Query understanding classifies query type and recommends governance bias.
- Governance calibration maps conservative baseline behavior into experimental calibrated routes.
- Route execution materializes route-specific slates.
- SQLite FTS retrieves real ESCI products from local indexes.
- Evaluators report route-level quality, strict violation rates, fallback rates, and ESCI label hit rates.

## Architecture Summary

Major backend components:

- `src/query_normalization_agent.py`: deterministic query cleanup.
- `src/query_understanding_agent.py`: rule-based query type and governance bias classification.
- `src/ollama_query_understanding_advisor.py`: optional local LLM advisory layer, not final ranking.
- `src/governance_calibration_dry_run.py`: calibrated route dry run.
- `src/calibrated_route_execution_adapter.py`: route-specific slate execution.
- `src/full_esci_product_index_builder.py`: ESCI product/query index builder.
- `src/full_esci_fts_index_builder.py`: SQLite FTS product index builder.
- `src/full_esci_route_retrieval_evaluator.py`: route-aware retrieval evaluation.
- `src/full_esci_index_scale_comparison.py`: deterministic scale comparison.
- `src/full_esci_paper_benchmark_report.py`: paper benchmark report.
- `src/full_esci_1m_scale_validation_report.py`: 1M validation report.

## Governance Routes

CORTEX evaluates the following governance routes:

- `BASELINE_ONLY`: preserve direct lexical retrieval.
- `REJECT_REPAIR_NARROW_QUERY`: prevent aggressive expansion for narrow queries.
- `MISSION_REPAIR`: materialize mission and sub-intent slates.
- `STRICT_REPAIR`: preserve strict negation, compatibility, and model constraints.
- `BEHAVIOR_AWARE_RERANK`: experimental behavior-aware route, currently fallback-backed in full ESCI mode.
- `CRITIC_REVIEW`: materialize retrieval-ready slates for future critic or human review.

## Retrieval Backend

The paper-ready Track A backend uses local SQLite FTS5 over ESCI product text. It supports 100k, 500k, and 1M index scales without cloud services, ANN, or external APIs.

Observed scale:

| Index | Products | Examples |
| --- | ---: | ---: |
| 100k | 84,302 | 100,000 |
| 500k | 401,358 | 500,000 |
| 1M | 760,149 | 1,000,000 |

## Strict Filtering and Reranking Experiments

MVP 26.6 and 26.7 added strict constraint filtering and tuning. The strict route tracks hard and soft violations for negation constraints and preserves diagnostic fields.

MVP 26.10 added `strict_boost`, a scale-aware penalty reranker. It was active, but did not recover benchmark quality:

- 500k raw vs 100k exact delta: `-0.03`
- 500k raw vs 100k exact_or_sub delta: `-0.05`
- 500k strict_boost vs 100k exact delta: `-0.05`
- 500k strict_boost vs 100k exact_or_sub delta: `-0.11`

## Main Benchmark Findings

The 1M validation run produced:

- success_count: `100`
- top_k_has_exact_rate: `0.68`
- top_k_has_exact_or_substitute_rate: `0.73`
- fallback_rate: `0.03`
- strict_any_violation_rate: `0.03599`
- runtime_seconds: `25.1703`

## Negative Result

The strict_boost experiment is a useful negative result. It confirms that applying stronger forbidden-token penalties can reduce some violation signals, but can also demote or displace labeled exact/substitute products. This supports the paper contribution: route-aware retrieval needs evaluator-guided or learned optimization, not only larger indexes or simple penalty rules.

## What Track A Excludes

Track A intentionally excludes:

- UI cleanup
- FastAPI or React
- CTR prediction
- personalization
- cloud deployment
- ANN/FAISS
- new agents unrelated to governed route-aware ranking

## Track B Product-Showcase Items

Track B can later add:

- a concise paper demo dashboard
- route trace visualization
- before/after slate inspection
- human-readable constraint violation examples
- product-facing UX polish
