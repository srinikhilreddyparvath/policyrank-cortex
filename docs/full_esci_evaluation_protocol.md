# Full ESCI evaluation protocol

## Research checkpoint after MVP 29.6

MVP 29.6 found no evidence that SearchContractV1 improves route-utility prediction over generic query/ranking features in the current official ESCI setting. This negative ablation is retained without reinterpretation. Routing is now secondary analysis rather than the primary CORTEX paper contribution unless future independent evidence changes that conclusion.

The later CF-ESCI feasibility study narrows the candidate primary question: do modern product rankers respond correctly and locally to a minimal change in exactly one explicit shopping requirement, and can contract enforcement improve that sensitivity while preserving standard relevance? ContractESCI's typed requirements and compliance evidence remain the foundation. ContractESCI and CF-ESCI augment, but do not replace, official ESCI relevance evaluation.

## Primary protocol: official Task-1 candidate-set reranking

The paper-comparable primary benchmark follows the upstream Amazon ESCI Task-1 definition:

- `small_version == 1`
- `split == "test"`
- candidates are only the judged query-product rows supplied for each query
- methods may reorder or filter that supplied candidate set but may not introduce catalog products

The upstream reference is `esci-data/ranking/prepare_trec_eval_files.py`. It maps E=4, C=3, S=2, I=1 and invokes TREC nDCG with `1=0,2=0.01,3=0.1,4=1`, equivalent to gains I=0, S=.01, C=.1, E=1. `official_esci_ndcg` uses those exact gains over the complete supplied candidate ranking. Its ideal denominator always uses the full query qrels, including when a method filters candidates. `official_esci_ndcg_at_k` uses the same gains at a cutoff.

Separately, `ndcg_5` and `ndcg_10` are CORTEX research diagnostics using linear gains E=1, S=.7, C=.4, I=0, matching the existing CORTEX reward map in `src/ranking.py` and `src/slate_reward.py`. `mrr_10` treats the first Exact item as relevant. Exact@K and ExactOrSubstitute@K are binary query-level presence indicators. Irrelevant@K is a binary presence indicator.

## Secondary protocol: open-corpus retrieval

`--evaluation-mode open_corpus` is scaffolded but deliberately not executable in MVP 29.1. It will retrieve against a complete product index. ESCI qrels are incomplete outside each supplied candidate pool, so arbitrary retrieved products are predominantly unjudged. Open-corpus results must report judged and unjudged counts, require an explicit unjudged-item policy, and must not be described as official Task-1 nDCG.

## Data, split, and selection

The canonical root defaults to `esci-data/shopping_queries_dataset`. Required files are verified and SHA-256 checksums are stored in `dataset_manifest.json`. Observed row, query, product, split, locale, label, and version counts are discovered rather than hardcoded.

Selection operates on sorted unique query IDs with a deterministic seeded shuffle followed by a stable sorted selected list. All judgments for every selected query are preserved. `selected_query_ids.txt` is reused for every method. The default paper mode is small/test; other combinations receive a warning and are not primary paper-comparable runs.

After the MVP 29.1 infrastructure-only smoke, development uses only `small_version == 1, split == train`. Query IDs are deterministically partitioned 80/20 into `train_fit` and `validation`, persisted as `train_fit_query_ids.txt` and `validation_query_ids.txt`. Test queries are frozen for final evaluation. The dataset manifest records the split seed, fraction, counts, and overlap count.

## Method parity

MVP 29.1 enables only adapters whose existing behavior can be invoked on the official candidate set:

- `fts_baseline`: existing `FullEsciRetrievalEngine.score_product` lexical scorer
- `strict_filter`: existing hybrid `apply_strict_constraint_filter` after the lexical scorer
- `strict_boost`: existing `apply_scale_aware_strict_boost` after the lexical scorer

`current_reranker`, `always_cortex`, and `gated_cortex` remain disabled because official-candidate-set parity is not established. `semantic_baseline` remains disabled pending model identity, provenance, corpus coverage, and reproducibility.

### MVP 29.2 adapter audit

The legacy CORTEX path is baseline candidates → contract construction → contract filtering → Q-action selection → policy compilation → final-slate enforcement → multi-agent diversification → optional baseline-preservation gate. Candidate retrieval previously happened before this chain. The fixed-candidate adapter replaces only that retrieval boundary.

The adapter receives product metadata without `esci_label`; evaluator judgments are joined back only after ordered IDs return. It rejects judgment columns at entry and fails explicitly if output IDs are not a unique subset of input IDs.

`always_cortex` is enabled through the existing filter/compiler/enforcer/diversifier chain. Because the legacy compiler structurally requires an `esci_label` column, the adapter supplies an empty internal compatibility column—never judgment values—so its ESCI feature is neutral. Q-action selection reads the existing Q table without updating it.

`current_reranker` remains disabled: its policy compiler directly treats ESCI labels as ranking features, so faithful historical parity conflicts with leakage-safe evaluation. `gated_cortex` is enabled by applying the existing baseline-preservation gate to the label-free lexical baseline and the fixed-candidate always-CORTEX slate. Its thresholds are unchanged. Those thresholds were historically tuned before the leakage-safe protocol and are therefore contamination-risk configuration for final claims; they must be re-established using `train_fit` only before frozen evaluation.

The legacy gate selects a top-five slate. Official Task-1 requires a complete ranking, so the adapter preserves that selected top five exactly and appends all unselected candidates in untouched baseline order. This is a protocol completion rule, not a gate or ranking-formula change.

The versioned route family is `cortex_candidate_routes_v1`: preserve, strict_filter, strict_boost, contract_rerank, and fallback. Oracle regret remains null unless all required routes have outcomes for the same query.

## Missing and unjudged data

Official reranking candidates are judged by construction. Missing labels are retained as unjudged and reduce label coverage; they are never silently mapped to I. Constraint fields remain null for methods that do not expose them. Oracle reward/regret remains null unless the complete required route family succeeds, with an explicit unavailable reason.

## Statistics

Bootstrap mean confidence intervals and paired bootstrap method differences use deterministic NumPy generators. Paired comparisons align methods by query ID. MVP 29.1 smoke outputs use 500 bootstrap iterations only; larger paper runs should predeclare a larger count.

## Failure handling and source of truth

`query_level_results.parquet` contains one row per selected query×enabled method. Exceptions produce explicit failure rows with stage, type, and bounded message. No query is silently dropped. Every CSV and report is generated by re-reading this Parquet artifact.

## Reproducibility

Each run stores dataset checksums, current Git revision when available, full CLI configuration, method versions, disabled-method reasons, selected query IDs, timestamp, latency, and success/failure counts. The runner is single-worker in MVP 29.1; parallel execution is rejected until deterministic process/SQLite parity is tested.
## MVP 29.3 leakage-safe contract and routing boundary

`SearchContractV1` is a deterministic, non-LLM parser. It receives query text and may receive only label-free product metadata. Its versioned output records normalized text, conservative product-type and term signals, explicit exclusions, limited brand/price signals, constraint strength, ambiguity, status, and parser provenance. Unknown signals remain null. `resolved`, `partial`, and `unresolved` describe extraction completeness; `fallback_used` is reserved for an actual parser exception and is not the default contract representation.

`clean_contract_router_v1` has two paper-safe routes: `PRESERVE` and `STRICT_FILTER`. An explicit hard exclusion selects the existing strict-filter adapter; every other contract preserves the lexical baseline. These rules were defined semantically and were not tuned on validation or test outcomes. The current contract-reranking path is excluded because its historical Q-table was trained using ESCI-label-derived rewards.

The historical Q-table and legacy baseline-preservation gate are marked `legacy_contaminated_for_paper_eval`. The evaluator rejects them by default. `--allow-unsafe-contaminated-state` exists only for debugging and must never be used for paper claims. The gate is also excluded because its threshold-calibration provenance cannot be established as train_fit-only.

The Task-1 train queries are first split into disjoint `train_fit` and `validation` query IDs. `train_fit` is then split into disjoint `policy_train` and `calibration` IDs. All four ID lists are persisted; overlap is asserted to be zero. Validation and test outcomes may not select thresholds or router rules.

Intervention accounting distinguishes `route_selected`, `ranking_changed`, `explicit_preserve`, `intervention_attempted`, and `intervention_effective`. An active route can be attempted without changing the final permutation; only a changed permutation is effective.

On the deterministic 100-query calibration diagnostic, `strict_filter` and `strict_boost` produced 99 identical complete permutations and one difference below rank five (99% identical; query ID 97100). Their NDCG@10, MRR@10, and Exact@1 aggregates were identical. `strict_boost` is therefore marked `redundant_for_official_rerank`; it remains available for open-corpus experiments.

## MVP 29.4 contract reranking behavior

`contract_rerank_v1` is deterministic and receives only `SearchContractV1`, label-free product metadata, and the existing lexical score. Baseline scores are divided by the maximum non-negative score within the query; when no positive score exists, every candidate receives the neutral value `0.5`. The fixed score is:

```text
contract_score =
    0.65 * normalized_baseline_score
  + 0.20 * positive_term_coverage
  + 0.10 * product_type_title_coverage
  + 0.03 * brand_match
  + 0.02 * price_signal_match
```

The weights sum to one and were fixed a priori to keep baseline relevance dominant while allowing explicit compatibility to break close lexical rankings. They were not selected from ESCI outcomes. The immutable configuration provenance is `a_priori_interpretable_v1; no outcome-based tuning`, and the complete configuration is included in the run fingerprint and evaluation configuration.

Unknown product type, brand, or price signals contribute zero compatibility and therefore do not penalize a candidate. The current ESCI metadata has no structured price column, so price matching is limited to an explicit query signal appearing in available text and is otherwise neutral. `must_have` remains empty unless the contract parser can derive it reliably.

Hard exclusions are not penalized or filtered inside `contract_rerank_v1`. They are recorded as diagnostics only. `STRICT_FILTER` exclusively owns hard-exclusion enforcement, preventing hidden duplication between route behaviors.

`clean_contract_router_v2` uses three deterministic rules, defined before validation:

1. An explicit hard exclusion routes to `STRICT_FILTER`.
2. A non-unresolved contract with a brand signal, price signal, or at least three positive terms routes to `CONTRACT_RERANK`.
3. Simple or unresolved contracts route to `PRESERVE`.

The deterministic 100-query calibration sanity sample exercised all routes. Always-rerank changed 41 rankings and changed top-1 twice, with mean absolute candidate movement `0.44681`. Router V2 selected `CONTRACT_RERANK` 58 times, `STRICT_FILTER` 5 times, and `PRESERVE` 37 times. Contract reranking changed 35 of its 58 routed rankings. No weights or rules were changed after this analysis or after the frozen validation diagnostic.
