# CF-ESCI Adjudication Protocol

Adjudication begins only after independent annotation is frozen. Raw A and B files are immutable and retained separately. The adjudication table stores `annotator_A_raw`, `annotator_B_raw`, `adjudicated_label`, reason, and notes for every field.

Allowed reasons are `GUIDELINE_CLARIFICATION`, `ATTRIBUTE_ROLE_AMBIGUITY`, `PRODUCT_TYPE_DISAGREEMENT`, `INSUFFICIENT_METADATA`, `DIRECTION_AMBIGUITY`, `QUERY_UNNATURAL`, `MULTIPLE_REQUIREMENTS_CHANGED`, and `OTHER`. The adjudicator resolves against the frozen guidelines and visible evidence, never ranker behavior. Guideline changes prompted by systematic disagreement require a new version; they must not silently alter this round.

Every accepted record preserves source type (`NATURAL` or `GENERATED`), source query IDs, family, operation, human-validation status, annotator count, adjudication status, and candidate-pool provenance. Generated examples never lose generated provenance.
