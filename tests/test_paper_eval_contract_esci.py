import hashlib
import json
import unittest
from pathlib import Path

import pandas as pd

from src.paper_eval.contract_esci import (
    FORBIDDEN_ANNOTATION_FIELDS, PILOT_SIZE, STRATUM_QUOTAS, assert_blind_artifact, assert_no_esci_test_queries,
    classify_query, select_pilot_queries, stable_order_key, validate_compliance_state,
    validate_requirement,
)


class ContractESCITests(unittest.TestCase):
    def synthetic_queries(self):
        examples = {
            "explicit_negation": "shoes without laces",
            "price_or_value": "affordable desk lamp",
            "brand_constraint": "nike running shoes",
            "must_have_attribute": "jacket with waterproof shell",
            "multi_attribute": "red cotton large shirt",
            "soft_preference": "comfortable office chair",
            "ambiguous_natural_language": "gift",
            "long_query": "replacement outdoor patio chair cushion suitable for year round use",
            "negative_control": "desk lamp shade",
        }
        rows=[]; query_id=1
        for stratum,quota in STRATUM_QUOTAS.items():
            for _ in range(quota+2): rows.append({"query_id":query_id,"query_text":examples[stratum],"source_partition":"policy_train"}); query_id+=1
        return pd.DataFrame(rows)

    def test_selection_is_deterministic_unique_and_exact(self):
        frame=self.synthetic_queries(); first=select_pilot_queries(frame); second=select_pilot_queries(frame.sample(frac=1,random_state=4))
        self.assertEqual(first.query_id.tolist(),second.query_id.tolist()); self.assertEqual(len(first),PILOT_SIZE); self.assertFalse(first.query_id.duplicated().any())
        self.assertEqual(first.selection_stratum.value_counts().to_dict(),STRATUM_QUOTAS)

    def test_query_only_classification(self):
        self.assertEqual(classify_query("boots without laces"),"explicit_negation"); self.assertEqual(stable_order_key(3),stable_order_key(3))

    def test_only_development_partitions_allowed(self):
        frame=self.synthetic_queries(); frame.loc[0,"source_partition"]="test"
        with self.assertRaises(ValueError): select_pilot_queries(frame)

    def test_esci_test_query_overlap_is_rejected(self):
        examples=pd.DataFrame({"query_id":[1,2],"split":["train","test"]}); assert_no_esci_test_queries([1],examples)
        with self.assertRaises(ValueError): assert_no_esci_test_queries([2],examples)

    def test_requirement_object_validation(self):
        valid={"requirement_id":"r1","type":"exclusion","attribute":"closure_type","operator":"not_equal","value":"laces","strength":"hard","evidence_text":"without laces"}
        validate_requirement(valid)
        with self.assertRaises(ValueError): validate_requirement({**valid,"strength":"inferred"})

    def test_unknown_is_distinct_from_violated(self):
        validate_compliance_state("unknown"); validate_compliance_state("violated"); self.assertNotEqual("unknown","violated")

    def test_annotation_artifacts_reject_outcomes(self):
        assert_blind_artifact({"query_id":1,"requirements":[]})
        for field in FORBIDDEN_ANNOTATION_FIELDS:
            with self.assertRaises(ValueError): assert_blind_artifact({field:None})

    def test_generated_pilot_schema_blindness_and_hashes(self):
        root=Path("data/contract_esci/pilot_v1")
        if not (root/"pilot_manifest.json").exists(): self.skipTest("generated ContractESCI pilot artifacts are not committed")
        manifest=json.loads((root/"pilot_manifest.json").read_text(encoding="utf-8")); schema=json.loads((root/"annotation_schema.json").read_text(encoding="utf-8"))
        self.assertEqual(schema["$schema"],"https://json-schema.org/draft/2020-12/schema"); self.assertIn("query_annotation",schema); self.assertIn("candidate_annotation",schema)
        states=schema["definitions"]["compliance_assessment"]["properties"]["state"]["enum"]; self.assertEqual(set(states),{"satisfied","violated","unknown","not_applicable"})
        queries=pd.read_parquet(root/"pilot_queries.parquet"); self.assertEqual(len(queries),PILOT_SIZE); self.assertEqual(queries.query_id.nunique(),PILOT_SIZE)
        self.assertTrue(set(queries.source_partition).issubset({"policy_train","calibration"})); self.assertFalse(set(queries.columns)&FORBIDDEN_ANNOTATION_FIELDS)
        query_records=[json.loads(line) for line in (root/"query_annotation_template.jsonl").read_text(encoding="utf-8").splitlines()]
        candidate_records=[json.loads(line) for line in (root/"candidate_annotation_template.jsonl").read_text(encoding="utf-8").splitlines()]
        self.assertEqual(len(query_records),PILOT_SIZE); self.assertEqual(len(candidate_records),PILOT_SIZE*5)
        for record in query_records+candidate_records: assert_blind_artifact(record); self.assertEqual(record["annotation_status"],"unannotated")
        self.assertTrue(all(record["requirements"]==[] for record in query_records)); self.assertTrue(all(record["requirement_assessments"]==[] for record in candidate_records))
        files={"pilot_query_ids_sha256":"pilot_query_ids.txt","pilot_queries_sha256":"pilot_queries.parquet","query_annotation_template_sha256":"query_annotation_template.jsonl","candidate_annotation_template_sha256":"candidate_annotation_template.jsonl","annotation_schema_sha256":"annotation_schema.json"}
        for key,name in files.items(): self.assertEqual(hashlib.sha256((root/name).read_bytes()).hexdigest(),manifest["hashes"][key])
        self.assertFalse(manifest["source"]["esci_test_used"])

if __name__=="__main__": unittest.main()
