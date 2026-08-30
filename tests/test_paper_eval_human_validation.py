import hashlib, json, tempfile, unittest
from pathlib import Path

import pandas as pd

from src.paper_eval.human_validation import *


ROOT=Path(__file__).resolve().parents[1]; OUT=ROOT/"data/cf_esci/human_validation_v1"


class HumanValidationUnitTests(unittest.TestCase):
    def test_agreement_synthetic(self):
        self.assertEqual(raw_agreement(["YES","NO"],["YES","YES"]),.5)
        self.assertAlmostEqual(cohens_kappa(["YES","YES","NO","NO"],["YES","YES","NO","NO"]),1.0)
        self.assertEqual(prevalence(pd.DataFrame({"x":["YES","NO","YES"]}),"x","A")[1]["count"],2)

    def test_acceptance_and_direction(self):
        good={"core_intent_preserved":"YES","single_requirement_change":"YES","changed_requirement_correct":"YES","query_naturalness":"YES","semantic_plausibility":"UNCLEAR"}
        self.assertTrue(accepted_query_label(good)); self.assertFalse(accepted_query_label({**good,"query_naturalness":"NO"}))
        self.assertTrue(direction_eligible("A_SHOULD_GAIN")); self.assertFalse(direction_eligible("INSUFFICIENT_EVIDENCE"))

    def test_sampling_is_deterministic_and_stratified(self):
        generated=pd.DataFrame([{"example_id":f"G_{f}_{i}","source_type":"GENERATED","requirement_family":f} for f in GENERATED_FAMILY_QUOTAS for i in range(20)])
        natural=[]
        families=["COLOR"]*9+["BRAND"]*8+["NUMERIC_SPECIFICATION"]*10+["EXPLICIT_ATTRIBUTE"]*4+["NEGATION_EXCLUSION"]*29
        for i,f in enumerate(families):natural.append({"example_id":f"N_{i}","source_type":"NATURAL","requirement_family":f})
        natural=pd.DataFrame(natural); a,c,f=select_human_validation_sample(generated,natural); b,_,_=select_human_validation_sample(generated,natural)
        self.assertEqual(a.example_id.tolist(),b.example_id.tolist()); self.assertEqual((len(a),len(c),len(f)),(120,20,100)); self.assertEqual(dict(a.source_type.value_counts()),{"GENERATED":60,"NATURAL":60})

    def test_blindness(self):
        assert_annotation_blind(pd.DataFrame(columns=["query_a","candidate_direction"]))
        with self.assertRaises(ValueError):assert_annotation_blind(pd.DataFrame(columns=["esci_label"]))


@unittest.skipUnless((OUT/"human_validation_manifest.json").exists(),"generated human-validation package unavailable")
class HumanValidationArtifactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.manifest=json.loads((OUT/"human_validation_manifest.json").read_text()); cls.a=pd.read_csv(OUT/"annotator_A_query_template.csv",keep_default_na=False); cls.b=pd.read_csv(OUT/"annotator_B_query_template.csv",keep_default_na=False)

    def test_manifest_and_partition_boundary(self):
        self.assertEqual(self.manifest["counts"]["sample"],120); self.assertFalse(self.manifest["source"]["validation_used"]); self.assertFalse(self.manifest["source"]["test_used"]); self.assertFalse(self.manifest["source"]["esci_labels_used"])

    def test_ab_templates_identical_except_identity_and_blank(self):
        pd.testing.assert_frame_equal(self.a.drop(columns="annotator_id"),self.b.drop(columns="annotator_id")); self.assertEqual(set(self.a.annotator_id),{"A"}); self.assertEqual(set(self.b.annotator_id),{"B"})
        for field in QUERY_LABEL_FIELDS:self.assertTrue((self.a[field]=="").all())

    def test_direction_blindness_and_equality(self):
        a=pd.read_csv(OUT/"annotator_A_direction_template.csv",keep_default_na=False); b=pd.read_csv(OUT/"annotator_B_direction_template.csv",keep_default_na=False); assert_annotation_blind(a); pd.testing.assert_frame_equal(a.drop(columns="annotator_id"),b.drop(columns="annotator_id")); self.assertTrue((a.candidate_direction=="").all())

    def test_candidate_pool_provenance(self):
        rows=[json.loads(x) for x in (OUT/"candidate_pool_manifest.jsonl").read_text().splitlines()]; self.assertEqual(len(rows),120)
        self.assertTrue(all(r["candidate_count"]<=MAX_POOL_SIZE and all(c["provenance"] for c in r["candidates"]) for r in rows))

    def test_schema_and_hash_reproducibility(self):
        schema=json.loads((OUT/"annotation_schema.json").read_text()); self.assertIn("INSUFFICIENT_EVIDENCE",schema["direction_labels"]["candidate_direction"]); self.assertEqual(tuple(schema["adjudication_reasons"]),ADJUDICATION_REASONS)
        for name,digest in self.manifest["hashes"].items():self.assertEqual(hashlib.sha256((OUT/name).read_bytes()).hexdigest(),digest)

    def test_raw_files_preserved_by_analysis(self):
        a=self.a.copy(); b=self.b.copy()
        for frame in (a,b):
            for field in QUERY_LABEL_FIELDS:frame[field]="YES" if field not in {"requirement_family","operation","confidence","old_value_normalized","new_value_normalized"} else ({"requirement_family":"COLOR","operation":"REPLACE","confidence":"HIGH","old_value_normalized":"red","new_value_normalized":"blue"}[field])
        before=a.to_csv(index=False)
        analyze_independent_annotations(a,b)
        self.assertEqual(before,a.to_csv(index=False))

if __name__=="__main__":unittest.main()
