import hashlib
import json
import unittest
from pathlib import Path

import pandas as pd

from src.paper_eval.cf_esci import (
    ADDITIONAL_QUOTAS, EVIDENCE_STATES, GROUND_TRUTH_STATUS, OPERATIONS,
    counterfactual_direction_accuracy, counterfactual_selectivity,
    NUMERIC, requirement_flip_consistency, select_additional_sources,
    unaffected_candidate_stability, validate_counterfactual, validate_evidence_state,
)


class CFESCITests(unittest.TestCase):
    def synthetic_sources(self):
        examples={"COLOR":"red shoes","BRAND":"nike shoes","NUMERIC_SPECIFICATION":"10 inch pan","EXPLICIT_ATTRIBUTE":"cotton shirt","NEGATION_EXCLUSION":"shoes without laces"}; rows=[]; qid=1
        for family,quota in ADDITIONAL_QUOTAS.items():
            for _ in range(quota+2):rows.append({"query_id":qid,"query_text":examples[family],"source_partition":"policy_train"}); qid+=1
        return pd.DataFrame(rows)

    def test_deterministic_feasibility_selection(self):
        frame=self.synthetic_sources(); a=select_additional_sources(frame,set()); b=select_additional_sources(frame.sample(frac=1,random_state=7),set())
        self.assertEqual(a.query_id.tolist(),b.query_id.tolist()); self.assertEqual(len(a),250); self.assertFalse(a.query_id.duplicated().any())

    def test_validation_partition_is_rejected(self):
        frame=self.synthetic_sources(); frame.loc[0,"source_partition"]="validation"
        with self.assertRaises(ValueError):select_additional_sources(frame,set())

    def test_counterfactual_schema_and_single_atom(self):
        proposal={"counterfactual_id":"cf1","source_query_id":1,"source_query_text":"red shoes","counterfactual_query_text":"blue shoes","invariant_product_type":"shoe","changed_requirement_id":"r1","changed_requirement_type":"COLOR","operation":"REPLACE","old_value":"red","new_value":"blue","unit":None,"requirement_strength":"hard","generation_provenance":{},"evidence_sufficiency":"PAIR_GROUNDED","counterfactual_status":"NEEDS_HUMAN_REVIEW","ground_truth_status":GROUND_TRUTH_STATUS,"changed_atom_count":1}
        validate_counterfactual(proposal)
        for operation in OPERATIONS:validate_counterfactual({**proposal,"operation":operation})
        with self.assertRaises(ValueError):validate_counterfactual({**proposal,"changed_atom_count":2})
        with self.assertRaises(ValueError):validate_counterfactual({**proposal,"ground_truth_status":"HUMAN_VALID"})

    def test_evidence_states_and_unknown_semantics(self):
        for state in EVIDENCE_STATES:validate_evidence_state(state)
        self.assertNotEqual("UNKNOWN","CONTRADICTS_NEW")

    def test_fraction_denominator_is_not_numeric_specification(self):
        self.assertIsNone(NUMERIC.search("1/2 inch curling iron")); self.assertEqual(NUMERIC.search("12 inch planter").group(1),"12")

    def test_synthetic_metrics(self):
        before={"new":3,"old":1,"n2":4,"u1":2,"u2":5}; after={"new":1,"old":3,"n2":5,"u1":4,"u2":5}
        self.assertEqual(counterfactual_direction_accuracy(before,after,[("new","old")]),1.0)
        self.assertEqual(requirement_flip_consistency(before,after,["new"],["old","n2"]),1.0)
        self.assertEqual(unaffected_candidate_stability({"a":1,"b":2,"c":3},{"a":2,"b":3,"c":4},["a","b","c"]),1.0)
        self.assertEqual(counterfactual_selectivity({"a":2,"b":1},{"a":1,"b":1},["a"],["b"]),1.0)

    def test_generated_artifacts_are_blind_unique_and_reproducible(self):
        root=Path("data/cf_esci/feasibility_v1")
        if not (root/"feasibility_manifest.json").exists(): self.skipTest("generated CF-ESCI feasibility artifacts are not committed")
        manifest=json.loads((root/"feasibility_manifest.json").read_text(encoding="utf-8")); sources=pd.read_parquet(root/"source_queries.parquet")
        self.assertEqual(len(sources),500); self.assertEqual(sources.query_id.nunique(),500); self.assertTrue(set(sources.source_partition).issubset({"policy_train","calibration"}))
        proposals=[json.loads(line) for line in (root/"counterfactual_proposals.jsonl").read_text(encoding="utf-8").splitlines()]; self.assertEqual(len({p["counterfactual_id"] for p in proposals}),len(proposals))
        for proposal in proposals:validate_counterfactual(proposal)
        evidence=[json.loads(line) for line in (root/"candidate_requirement_evidence.jsonl").read_text(encoding="utf-8").splitlines()]
        for row in evidence:validate_evidence_state(row["directional_state"]); self.assertNotIn("esci_label",row)
        annotations=[json.loads(line) for line in (root/"human_counterfactual_annotation_template.jsonl").read_text(encoding="utf-8").splitlines()]
        forbidden={"esci_label","method","ranker","score","route","oracle","ndcg","mrr","ranking_outcome"}
        for record in proposals+evidence+annotations:self.assertFalse(set(record)&forbidden); self.assertEqual(record["ground_truth_status"],GROUND_TRUTH_STATUS)
        self.assertFalse(manifest["source"]["validation_used"]); self.assertFalse(manifest["source"]["test_used"]); self.assertFalse(manifest["generation"]["esci_labels_used_for_proposals"])
        for name,digest in manifest["hashes"].items():self.assertEqual(hashlib.sha256((root/name).read_bytes()).hexdigest(),digest)

if __name__=="__main__":unittest.main()
