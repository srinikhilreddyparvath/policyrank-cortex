# ContractESCI Benchmark Design v1

## Motivation and research question

Official ESCI labels measure product-ranking relevance, not whether a result obeys explicit shopping constraints. ContractESCI asks: **Can executable shopping-intent contracts reduce explicit constraint violations while preserving product-ranking relevance?**

ContractESCI is an augmentation for constraint-sensitive evaluation. It does **not** replace official ESCI relevance evaluation. Relevance and constraint satisfaction remain separate dimensions and should be reported together.

For the CF-ESCI feasibility extension, the candidate primary question is: **Do modern product rankers respond correctly and locally to minimal changes in explicit shopping requirements, and can CORTEX enforce this counterfactual requirement sensitivity while preserving standard ESCI relevance?** This MVP audits whether defensible examples can be constructed; it does not evaluate rankers or implement a final model.

The unit is an original query, a counterfactual query changing exactly one typed requirement atom, same-pool candidate evidence for old/new values, one or more semantically expected direction pairs, and an unaffected candidate set for locality analysis. Every machine proposal requires human review.

## Pilot scope and sampling

The 250-query methodology pilot draws only from frozen policy-train and calibration partitions of the ESCI Task-1-small train split. Validation and ESCI test are excluded.

Sampling uses normalized query text only. Queries enter the first matching mutually exclusive stratum and are ordered within strata by SHA-256 of `seed:query_id`, seed 29. Quotas are: explicit negation 30; price/value 25; conservative named-brand pattern 25; must-have attribute 35; multi-attribute 30; soft preference 25; ambiguous language 20; long query 25; negative control 35. Negative controls are ordinary product queries with no detected explicit contract beyond product type.

The sampler never reads CORTEX outputs, method wins/losses, routes, oracle data, route regret, reranker improvement, or method-specific failures. Text rules and quotas are versioned and tested. Selection strata are sampling aids, not human labels.

## Annotation schema

Layer A records query ID/text, minimally normalized product type, typed requirements, must-have/must-not-have references, brand and price requirements, attribute requirements, hard constraints, soft preferences, negation, comparative/superlative intent, ambiguity, annotatability, and notes.

Each requirement has a logical type (inclusion, exclusion, comparison, preference), conservative attribute, operator, explicit value/unit, hard/soft strength, exact evidence, and optional normalization note.

Layer B assesses every requirement for every candidate as satisfied, violated, unknown, or not applicable. Unknown remains distinct from violated. Definitions and edge cases are in `contract_esci_annotation_guidelines.md`; `annotation_schema.json` is machine-readable.

## Candidate-compliance methodology

The pilot exposes five candidates per query (1,250 pairs). A fixed label-free lexical scorer selects the top five from the original judged candidate set. Candidate selection does not invoke SearchContractV1, CORTEX, route decisions, or performance. ESCI labels and scorer values are removed before annotation files are written.

Five candidates keeps the methodology pilot feasible while focusing on products a relevance system could plausibly surface. It under-samples deep-rank behavior. A full benchmark may add predeclared deterministic label-stratified candidates without reference to CORTEX; that extension must be separately versioned and sensitivity-tested.

## Metrics

All compliance metrics require frozen human annotations. Unknown is reported separately, never silently counted as violation or satisfaction.

- **Hard Constraint Violation Rate @K:** violated hard assessments among assessable hard assessments in top K.
- **Hard Constraint Satisfaction @K:** candidates satisfying every applicable hard requirement among fully assessable candidates.
- **Query-Level Contract Failure Rate:** queries with at least one top-K hard violation; alternate definitions must be separately named.
- **Must-Have Satisfaction @K:** satisfied hard inclusions among assessable must-have assessments.
- **Must-Not-Have Violation Rate @K:** violated hard exclusions among assessable must-not-have assessments.
- **Soft Preference Satisfaction @K:** satisfied soft assessments among assessable soft assessments under a frozen rule.
- **Constraint Coverage:** requirements with sufficient metadata for a satisfied/violated decision.
- **Unknown-Compliance Rate:** unknown assessments divided by applicable assessments.
- **Contract Satisfaction @K:** predeclared aggregate of hard safety and soft satisfaction, with both components also reported.

Retain official ESCI nDCG, MRR, and Exact@1. Future analysis should show a relevance-versus-safety frontier instead of collapsing both dimensions into one opaque score.

## SearchContractV1 evaluation

Human query annotations and schema must freeze before running frozen SearchContractV1. Planned comparisons include product-type accuracy; requirement extraction precision/recall/F1; must-have and must-not-have precision/recall/F1; negation detection; brand and price-intent accuracy; and hard/soft classification. Matching, normalization, and partial-credit policies freeze before evaluation. Human annotations must not be revised after parser errors are seen.

## Planned baselines

Future evaluation should include a lexical relevance baseline, strong reproducible neural reranker, simple hard-constraint filter, contract-aware reranker, reproducible structured/LLM extraction baseline, and CORTEX contract extraction plus enforcement and verification. MVP 29.7 implements none of these models.

## Agreement and adjudication

Use at least two independent annotators for all 250 query contracts if feasible and a minimum stratified 20% (50 queries and associated candidate pairs) for candidate compliance. Select overlap by the same seeded hash ordering within every stratum.

Report Cohen’s kappa for categorical fields when prevalence permits, plus raw agreement and prevalence; exact agreement for normalized fields; micro/macro precision, recall, and F1 for requirement matching under a frozen matcher; evidence-span overlap; and state-specific compliance agreement, especially unknown versus violated. Do not compute agreement from one source. Adjudicate only after independent passes freeze and preserve both originals.

## Bias risks and limitations

Text rules can miss implicit or multilingual expressions; conservative brand lists can distort coverage. Top-five lexical candidates favor surface relevance. Product metadata can be incomplete, producing high unknown rates. ESCI’s distribution limits external validity. Constraint categories may be sparse. Report stratum-level uncertainty and do not present the pilot as a finished benchmark.

## Relationship to official ESCI and routing

Official ESCI remains the relevance authority. ContractESCI adds human constraint annotations on development queries so relevance preservation and explicit violation reduction can be studied together.

MVP 29.6 found **no evidence that SearchContractV1 improves route-utility prediction over generic features in the current official ESCI setting**. That negative result is preserved. Routing is a secondary analysis/negative ablation unless future independent evidence changes the conclusion. ContractESCI tests constraint safety, not a retrospective reinterpretation of routing performance.
