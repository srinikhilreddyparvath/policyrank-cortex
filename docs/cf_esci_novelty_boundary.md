# CF-ESCI Novelty Boundary

## Proposed gap, not a priority claim

CF-ESCI investigates whether a controlled semantic intervention to one explicit shopping requirement induces the expected localized change in ranking behavior. The proposed gap is narrow:

> Existing work primarily studies requirement extraction, fixed-query requirement satisfaction, exclusion handling, structured intent ranking, or counterfactual explanations. CF-ESCI instead evaluates whether controlled semantic interventions to individual shopping requirements induce the expected localized change in ranking behavior.

This document does not claim CF-ESCI is first, and it does not establish novelty. It records the current boundary for external review and should be revised if closer work is found.

## Neighboring work

- **REAlign (2026), Requirement–Evidence Alignment for Compositional E-Commerce Queries** aligns typed requirements with visible product evidence, distinguishes satisfaction/violation/unsupported evidence, and optimizes fixed-pool reranking. It is the closest fixed-query requirement-aware neighbor. CF-ESCI changes the query requirement and evaluates directional response plus locality rather than only satisfaction for a fixed query. Source: https://arxiv.org/abs/2608.02500
- **Hint-Augmented Re-ranking (2025)** decomposes superlative product queries into structured hints and transfers those interpretations into efficient rerankers. CF-ESCI is an evaluation design for minimal requirement changes, not a superlative-hint ranking method. Source: https://arxiv.org/abs/2511.13994
- **ExcluIR (AAAI 2025)** supplies exclusionary queries and training/evaluation resources for retrieval systems. CF-ESCI includes exclusion as one possible intervention family but additionally requires an original/counterfactual pair, same-pool evidence direction, and unaffected-set locality. Source: https://doi.org/10.1609/aaai.v39i12.33451
- **Learning to Rewrite Negation Queries in Product Search (COLING 2025)** trains query rewrites to improve negation-query product search. CF-ESCI does not train a rewriter; it tests response to human-validated atomic requirement edits. Source: https://aclanthology.org/2025.coling-industry.49/
- **Multi-Aspect Dense Retrieval (KDD 2022)** explicitly represents aspects such as category, brand, and color to improve product retrieval. CF-ESCI does not prescribe an aspect representation; it uses typed aspects to construct controlled behavioral tests. Source: https://research.google/pubs/multi-aspect-dense-retrieval/
- **Beyond Semantic Similarity / explicit-intent commerce ranking** describes structured or explicit intent signals for query-product matching. This overlaps CF-ESCI’s motivation but generally evaluates matching on fixed queries; CF-ESCI’s object is a paired semantic intervention and localized ranking change. Public bibliographic details for the exact cited item should be independently confirmed before publication.
- **CFE2 (2023), Counterfactual Editing for Search Result Explanation** creates pairwise counterfactual queries as explanations for why a lower-ranked document could outrank another. CF-ESCI is not an explanation generator: its counterfactual is a catalog-grounded shopping-requirement intervention used to test ranker behavior and unaffected-candidate stability. Source: https://arxiv.org/abs/2301.10389
- **A Counterfactual Explanation Framework for Retrieval Models (ACL Findings 2026)** asks what document terms would need to be added to improve a document’s rank. It edits/explains the document side of an existing model decision; CF-ESCI edits one query requirement and evaluates resulting behavioral sensitivity. Source: https://aclanthology.org/2026.findings-acl.1917/

## Boundary conditions

CF-ESCI’s claim weakens substantially if prior work already provides all of: shopping-domain original/counterfactual query pairs; exactly one grounded requirement change; evidence-derived expected product-pair direction independent of rankings/labels; and explicit unaffected-candidate locality evaluation. A broader systematic review and expert review are required before any novelty language appears in a paper.
