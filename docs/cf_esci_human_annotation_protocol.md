# CF-ESCI Human Annotation Protocol

## Status and scope

CF-ESCI is a candidate research benchmark. Machine-generated and naturally mined proposals are `NOT_HUMAN_GROUND_TRUTH`; human validation is required before any ranker evaluation. The packet uses only policy-train and calibration queries and contains no ESCI labels or ranking outcomes.

## Sampling and calibration

The frozen sample contains 120 pairs: all 60 natural pairs and a deterministic, seed-30 sample of 60 generated pairs (15 each from color, brand, numeric specification, and explicit attribute). This deliberately oversamples rare natural ADD, REMOVE, and exclusion cases. These proportions are not corpus prevalence estimates.

Twenty examples (10 per source) form a guideline-development calibration round. Their answers are not included in final agreement. Wording changes discovered during calibration must be versioned before the 100-example independent round; calibration examples may be reused only in a future protocol that explicitly freezes unchanged guidelines.

## Independent query judgments

Annotators A and B work separately and label core-intent preservation, single-requirement change, correctness of the changed requirement, requirement family, operation, query naturalness, semantic plausibility, confidence, and normalized old/new values. Annotate only linguistically supportable meaning. `UNCLEAR` is valid; do not infer hidden preferences.

## Candidate-direction judgments

Candidate comparisons expose only query text, catalog metadata, and visible evidence. Choose `A_SHOULD_GAIN`, `B_SHOULD_GAIN`, `NO_REQUIRED_DIRECTION`, or `INSUFFICIENT_EVIDENCE`, plus confidence. Missing metadata is insufficient evidence, not a violation. Machine directions are never shown.

## Agreement interpretation

Report sample size, class prevalence, raw agreement, and Cohen's kappa for core intent, single-change status, family, operation, naturalness, and candidate direction. Report exact match for normalized old/new values and separate agreement for uncertain/insufficient states. No arbitrary kappa cutoff determines validity: interpretation considers prevalence, field difficulty, direction coverage, sample size, and family-specific disagreement.

## Acceptance rule (predeclared)

A pair enters CF-ESCI only after adjudication confirms all of: core intent `YES`, single requirement `YES`, changed requirement correct `YES`, naturalness `YES`, and semantic plausibility not `NO`. CDA/RFC additionally require a direction other than `NO_REQUIRED_DIRECTION` or `INSUFFICIENT_EVIDENCE`. Query-valid pairs without direction remain auxiliary only.
