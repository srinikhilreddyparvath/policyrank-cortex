import hashlib
import json
import unittest
from pathlib import Path

import pandas as pd

from src.paper_eval.cf_esci import GROUND_TRUTH_STATUS
from src.paper_eval.natural_pairs import (
    EVIDENCE_STATES,NATURAL_NUMERIC,VALIDATION_STATUS,build_candidate_evidence,
    canonical_pair_id,extract_atoms,filter_valid_pairs,mine_raw_pairs,
    normalize_number,validate_natural_pair,
)


class NaturalPairTests(unittest.TestCase):
    def frame(self):
        return pd.DataFrame([
            {"query_id":1,"query_text":"red running shoes","source_partition":"policy_train","product_type":"shoe"},
            {"query_id":2,"query_text":"blue running shoes","source_partition":"calibration","product_type":"shoe"},
            {"query_id":3,"query_text":"running shoes","source_partition":"policy_train","product_type":"shoe"},
            {"query_id":4,"query_text":"running shoes without laces","source_partition":"policy_train","product_type":"shoe"},
        ])

    def test_mining_is_deterministic_and_queries_are_distinct(self):
        a,_=mine_raw_pairs(self.frame()); b,_=mine_raw_pairs(self.frame().sample(frac=1,random_state=3)); self.assertEqual(a,b)
        self.assertTrue(all(row["query_id_a"]!=row["query_id_b"] for row in a))

    def test_validation_partition_rejected(self):
        frame=self.frame(); frame.loc[0,"source_partition"]="validation"
        with self.assertRaises(ValueError):mine_raw_pairs(frame)

    def test_canonical_pair_deduplicates_reciprocal_direction(self):
        self.assertEqual(canonical_pair_id(1,2,"COLOR"),canonical_pair_id(2,1,"COLOR"))

    def test_product_type_and_single_delta_invariants(self):
        raw,_=mine_raw_pairs(self.frame()); valid,failures=filter_valid_pairs(raw); self.assertTrue(valid); self.assertFalse(any(row["changed_atom_count"]!=1 for row in valid))
        bad={**valid[0],"product_type_b":"boot"}
        with self.assertRaises(ValueError):validate_natural_pair(bad)

    def test_operation_and_negation_detection(self):
        raw,_=mine_raw_pairs(self.frame()); operations={(row["requirement_family"],row["operation"]) for row in raw}
        self.assertIn(("COLOR","REPLACE"),operations); self.assertTrue(any(family=="NEGATION_EXCLUSION" and operation in {"ADD","REMOVE"} for family,operation in operations))

    def test_numeric_fractions_units_and_roles(self):
        self.assertEqual(normalize_number("1/2"),"0.5"); match=NATURAL_NUMERIC.search("1/2 inch curling iron"); self.assertEqual(match.group(1),"1/2")
        atoms=extract_atoms("1/2 inch curling iron"); numeric=[atom for atom in atoms if atom.family=="NUMERIC_SPECIFICATION"][0]; self.assertEqual((numeric.normalized_value,numeric.unit,numeric.semantic_role),("0.5","inch","length"))

    def test_candidate_evidence_unknown_is_distinct(self):
        pair={"canonical_pair_id":"p","requirement_family":"COLOR","changed_attribute":"color","operation":"REPLACE","old_value":"red","new_value":"blue","unit":None,"semantic_role":None,"query_text_a":"red shoes","query_text_b":"blue shoes"}
        rows=pd.DataFrame([{"product_id":"a","product_title":"red shoe"},{"product_id":"b","product_title":"blue shoe"},{"product_id":"u","product_title":"shoe"}]); evidence,directions,_=build_candidate_evidence(pair,rows)
        self.assertEqual({row["directional_state"] for row in evidence},{"SUPPORTS_A","SUPPORTS_B","UNKNOWN"}); self.assertTrue(directions); self.assertNotEqual("UNKNOWN","CONTRADICTS_B"); self.assertTrue(set(row["directional_state"] for row in evidence)<=EVIDENCE_STATES)

    def test_generated_artifacts_blind_unique_and_hashed(self):
        root=Path("data/cf_esci/natural_pairs_v1")
        if not (root/"natural_pair_manifest.json").exists():self.skipTest("generated natural-pair artifacts are not committed")
        manifest=json.loads((root/"natural_pair_manifest.json").read_text(encoding="utf-8")); pairs=pd.read_parquet(root/"natural_query_pairs.parquet"); source_ids=[int(value) for value in (root/"source_query_ids.txt").read_text().split()]; self.assertEqual(len(source_ids),manifest["statistics"]["total_queries_searched"]); self.assertEqual(len(source_ids),len(set(source_ids))); self.assertFalse(pairs.canonical_pair_id.duplicated().any()); self.assertTrue((pairs.query_id_a!=pairs.query_id_b).all()); self.assertTrue((pairs.changed_atom_count==1).all()); self.assertTrue((pairs.product_type_a==pairs.product_type_b).all())
        records=[]
        for name in ("natural_pair_proposals.jsonl","candidate_direction_evidence.jsonl","candidate_direction_pairs.jsonl","human_natural_pair_review_template.jsonl"):records.extend(json.loads(line) for line in (root/name).read_text(encoding="utf-8").splitlines())
        forbidden={"esci_label","score","ranker","method","route","oracle","ndcg","mrr","ranking_outcome"}
        for record in records:self.assertFalse(set(record)&forbidden); self.assertEqual(record["ground_truth_status"],GROUND_TRUTH_STATUS)
        for record in json.loads((root/"natural_pair_proposals.jsonl").read_text(encoding="utf-8").splitlines()[0]),: self.assertEqual(record["validation_status"],VALIDATION_STATUS)
        self.assertFalse(manifest["source"]["validation_used"]); self.assertFalse(manifest["source"]["test_used"]); self.assertFalse(manifest["construction"]["esci_labels_used"]); self.assertFalse(manifest["construction"]["ranker_outputs_used"])
        for name,digest in manifest["hashes"].items():self.assertEqual(hashlib.sha256((root/name).read_bytes()).hexdigest(),digest)

if __name__=="__main__":unittest.main()
