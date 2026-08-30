# CF-ESCI Benchmark Design

CF-ESCI asks whether a minimal change to one explicit shopping requirement produces the expected localized ranking response while preserving standard relevance. It is an augmentation to official ESCI, not a replacement: CDA, RFC, UCS, and Counterfactual Selectivity remain separate from official nDCG, MRR@10, and Exact@1.

Version 1 combines generated grounded proposals and naturally occurring minimal pairs. The current human-validation package tests whether humans agree that intent is preserved, exactly one requirement changes, and catalog evidence supports a direction. Machine proposals are not ground truth, and no ranker may be evaluated until independent annotation and adjudication are complete.

The candidate-pool pilot takes the union of original ESCI candidate pools, prioritizes already catalog-grounded support/contrast candidates, then uses deterministic SHA-256 order, capped at 30. Generated pairs use their single original pool under the same rule. Candidate provenance is retained. This is a pilot design, not a claim that the final pool protocol is settled.

The metric definitions from MVP 29.8/29.9 are frozen. They must not be changed after observing human labels or model performance except to correct a genuine mathematical bug. The main risks are sparse directional evidence, metadata insufficiency, prevalence-sensitive agreement, ambiguity in attribute roles, natural-pair pool mismatch, and over-representation of rare strata caused by deliberate sampling.
