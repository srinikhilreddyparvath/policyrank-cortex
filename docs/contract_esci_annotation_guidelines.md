# ContractESCI Annotation Guidelines v1.0

## Purpose

ContractESCI records two separate judgments: what a shopping query explicitly requires or prefers, and whether a candidate product complies with those requirements. It does not ask annotators to judge general relevance, and it does not replace official ESCI relevance labels.

Annotate only interpretations supported by the query wording. Do not guess demographics, intended use, quality expectations, budget, or other hidden preferences.

## Layer A: query contract annotation

### Product type

Record the requested product head with minimal normalization: `running shoes` may normalize to `shoe`; `gift` has no supportable product type and should be null. If two types remain plausible, choose the most literal supported type and mark ambiguity medium/high; use null when neither is defensible.

### Explicit requirements and typed objects

Create one requirement object per independently testable condition. Each object needs a query-local ID, logical type, attribute, operator, normalized value, hard/soft strength, and exact evidence phrase.

- `inclusion`: the product must contain or equal a stated value, such as “red cotton shirt.”
- `exclusion`: a stated value must be absent, such as “shoes without laces.”
- `comparison`: an explicit threshold or comparison, such as “under $30.”
- `preference`: an explicitly directional but non-binding goal, such as “cheaper desk lamp.”

Use conservative attributes (`product_type`, `brand`, `price`, `color`, `material`, `size`, `compatibility`) when they fit. Use `other_explicit` with a note rather than inventing an unsupported ontology entry.

```json
{"requirement_id":"r1","type":"exclusion","attribute":"closure_type","operator":"not_equal","value":"laces","unit":null,"strength":"hard","evidence_text":"without laces","normalization_notes":null}
```

For “cheaper phone case,” use `type=preference`, `attribute=price`, `operator=lower_preferred`, `value=null`, `strength=soft`. The query does not establish a numeric ceiling.

### Hard versus soft

A hard constraint defines acceptability: violating it contradicts an explicit condition. “Without sugar,” “Nike shoes,” “case for iPhone 15,” and “under $50” are normally hard.

A soft preference gives direction without a defensible pass/fail boundary. “Affordable,” “better,” “lightweight,” and “stylish” are soft unless the query states a measurable boundary. Do not convert “best” into a hard requirement.

### Negation and exclusions

Mark negation when the query linguistically excludes or denies something (`no`, `not`, `without`, `excluding`, `free of`). Attach negation to the narrowest supported attribute. “Not expensive” is a soft price preference unless a threshold is given. Lexical accidents and model guesses are not negations.

### Comparatives and superlatives

Use `comparative` for wording such as cheaper, larger, or more durable. Use `superlative` for cheapest, largest, or best. A comparative needs a reference point to be objectively testable; when absent, annotate the preference but expect candidate compliance to be unknown or comparison-set dependent. Do not verify a superlative from one product record.

### Price language

Numeric ceilings/floors are hard only when explicitly stated (`under $25`, `at least $100`). Terms such as cheap, budget, value, and not expensive are soft directional preferences. A record without price metadata yields unknown, not violated.

### Brands

Annotate a brand requirement only when the query explicitly names the requested product brand. Brand names used solely for compatibility (“case for Samsung Galaxy”) belong to a compatibility requirement unless the case itself must be Samsung-branded.

### Attributes and compatibility

Annotate colors, sizes, materials, quantities, audiences, compatibility targets, and functional features only when linguistically present. Preserve stated units. “For kids” does not justify inferred safety or quality requirements.

### Ambiguity and annotatability

- `low`: one clear literal contract.
- `medium`: a plausible contract exists but one element has multiple readings.
- `high`: key product type or requirements cannot be resolved without guessing.
- `annotatable`: all material explicit content can be structured.
- `partially_annotatable`: some content can be structured but a material phrase remains unresolved.
- `not_annotatable`: no defensible shopping contract can be extracted.

A plain query such as “desk lamp” is a valid negative control: annotate the product type and an empty requirements list. Do not manufacture constraints.

## Layer B: candidate compliance

Assess every Layer-A requirement independently using only displayed product metadata.

- `satisfied`: metadata affirmatively supports compliance.
- `violated`: metadata affirmatively contradicts the requirement.
- `unknown`: metadata is missing, vague, or insufficient to decide.
- `not_applicable`: the requirement logically does not apply to this candidate. Use sparingly and explain.

Unknown is never equivalent to violated. Absence of “laces” from a short title does not prove a shoe is laceless. A title explicitly saying “lace-up” can support violation of “without laces.” Conflicting metadata should be unknown with a note unless one source is clearly authoritative under the protocol.

Overall hard-constraint state is violated if any hard requirement is violated; satisfied only if every applicable hard requirement is satisfied; unknown if none is violated and at least one is unknown; not applicable when there are no applicable hard requirements.

## Edge cases

- Misspellings: normalize only when intent is clear; retain the original evidence phrase.
- Bundles: do not assume every pictured or mentioned accessory is included.
- Compatibility: a shared brand name does not prove model compatibility.
- Marketing claims: note unverifiable superlatives as unknown.
- Negative controls: empty requirements are correct and important.
- Ordinary relevance: wrong product type is not automatically a constraint violation unless product type is an adjudicated requirement.

## Independent annotation and disagreements

Annotators work independently on the agreement subset and cannot see another annotator’s labels. After both passes freeze, disagreements are exported for adjudication. The adjudicator records the selected value and rationale without deleting originals. Schema questions are logged; schema changes require a new version and never silently rewrite completed labels.

## CF-ESCI counterfactual extension

CF-ESCI proposals are machine-generated review candidates, never human-valid examples by default. For each `q → q'` pair, independently answer:

1. Does the counterfactual preserve core product intent? `YES`, `NO`, or `UNCLEAR`.
2. Does exactly one meaningful shopping requirement change? `YES`, `NO`, or `UNCLEAR`.
3. Is the changed requirement correctly identified? `YES`, `NO`, or `UNCLEAR`.
4. Is the counterfactual linguistically plausible as a shopping query? `YES`, `NO`, or `UNCLEAR`.
5. Is the replacement or new value semantically plausible? `YES`, `NO`, or `UNCLEAR`.
6. For each blinded evidence-backed pair, select `NEW_PRODUCT_SHOULD_GAIN`, `OLD_PRODUCT_SHOULD_GAIN`, `NO_REQUIRED_DIRECTION`, or `INSUFFICIENT_EVIDENCE`.
7. Record confidence as `HIGH`, `MEDIUM`, or `LOW`.

Accept a counterfactual only when the product type is invariant, one requirement atom changes, no other dimension intentionally changes, the result is linguistically plausible, and both the value and expected candidate behavior are grounded by visible catalog evidence. Reject multi-dimensional edits such as “cheap red shoes” to “premium blue boots.” Do not see or infer ranker identity, score, route, ranking outcome, ESCI label, or which system benefits.

For replacement evidence, catalog occurrence is necessary but not sufficient: annotators must confirm that the value has the same semantic role. A brand in a compatibility phrase is not necessarily the product brand; a number with the same unit may describe a different attribute. If evidence does not resolve this, choose `UNCLEAR` or `INSUFFICIENT_EVIDENCE`.
