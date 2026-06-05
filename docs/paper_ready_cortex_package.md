# Paper-Ready CORTEX Package

## Final Claim Statements

1. CORTEX implements a governed route-aware retrieval architecture for commerce queries.
2. The system separates query understanding, governance, route execution, strict guardrails, and route-level evaluation.
3. Local SQLite FTS scales ESCI retrieval from 100k to 1M examples without cloud infrastructure.
4. Larger lexical indexes are not automatically better: 500k raw retrieval underperformed 100k on the 100-query scale comparison.
5. Simple penalty-based strict reranking was active but did not recover quality, motivating evaluator-guided route-aware optimization.
6. The 1M validation confirms the pipeline remains reproducible and stable at larger local scale.

## Paper Abstract Draft

CORTEX is a governed route-aware ranking system for commerce retrieval. Unlike single-path lexical retrieval, CORTEX classifies query intent, routes queries through calibrated governance decisions, materializes route-specific slates, and evaluates retrieval quality by route, query type, and strict constraint preservation. Using the Amazon ESCI dataset, we build local SQLite FTS indexes at 100k, 500k, and 1M example scales. Experiments show that larger lexical candidate spaces do not automatically improve top-k ESCI relevance, and that a simple strict penalty reranker fails to recover quality despite being active. These results motivate governed route-aware optimization as a distinct research direction for reliable commerce ranking under mission, compatibility, negation, and critic-review constraints.

## Experiment Table

| Experiment | Index Scale | Products | Exact Rate | Exact/Sub Rate | Fallback Rate | Strict Any Violation | Runtime Seconds |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 100k baseline | 100k | 84,302 | reference | reference | reference | reference | reference |
| 500k raw vs 100k | 500k | 401,358 | -0.03 delta | -0.05 delta | 0.0 delta | +0.035833 delta | +21.4577 delta |
| 500k strict_boost vs 100k | 500k | 401,358 | -0.05 delta | -0.11 delta | 0.0 delta | -0.011667 delta | +30.9616 delta |
| 1M validation | 1M | 760,149 | 0.68 | 0.73 | 0.03 | 0.03599 | 25.1703 |

## Limitations

- The current reranking experiments are deterministic and heuristic.
- The optional LLM advisor is not integrated into final route execution.
- Behavior-aware reranking is represented as a route but is not yet a full learned reranker.
- The 1M validation is a 100-query evaluation slice, not a full ESCI benchmark sweep.
- SQLite FTS is a strong reproducible baseline but not an ANN or semantic retrieval backend.
- Strict filtering tracks violations, but perfect semantic constraint satisfaction remains future work.

## Future Work

- Evaluator-guided route optimization.
- Learned candidate blending by route and query type.
- Critic-guided repair loop over retrieval-ready slates.
- Full behavior-aware reranking.
- Larger deterministic benchmark slices.
- Product-showcase dashboard for route traces and slate inspection.

## Suggested Paper Sections

1. Introduction
2. Related Work
3. Problem Formulation
4. CORTEX Architecture
5. Query Understanding and Governance Routes
6. Route-Specific Retrieval Execution
7. Strict Constraint Filtering
8. ESCI Index Construction and SQLite FTS Backend
9. Experimental Setup
10. Results
11. Negative Result: Strict Boost Reranking
12. Discussion
13. Limitations
14. Conclusion

## What Is Excluded From Track A

Track A intentionally excludes UI polish, FastAPI, React, cloud deployment, CTR prediction, personalization, ANN/FAISS, and unrelated new agents. The package is focused on a reproducible research backend and paper-ready evidence.

## Next Track B Product Showcase Items

- Minimal demo dashboard for route traces.
- Query example gallery.
- Before/after slate comparison.
- Constraint violation visualization.
- README polish for non-research viewers.
