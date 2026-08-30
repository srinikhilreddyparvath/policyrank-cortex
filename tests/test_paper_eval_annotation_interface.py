import csv
import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from tools.cf_esci_annotation.server import (
    DATA_DIR, DIRECTION_ANSWERS, MODE_CONFIG, QUERY_ANSWERS,
    atomic_write_csv, load_mode, read_csv, save_mode,
)


class AnnotationInterfaceTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.completed = Path(self.temporary.name) / "completed"

    def tearDown(self): self.temporary.cleanup()

    def test_frozen_template_hashes(self):
        manifest=json.loads((DATA_DIR/"human_validation_manifest.json").read_text(encoding="utf-8"))
        for name,digest in manifest["hashes"].items():
            self.assertEqual(hashlib.sha256((DATA_DIR/name).read_bytes()).hexdigest(),digest)

    def test_exact_counts_and_stable_ab_order(self):
        expected={"ANNOTATOR_A_QUERY":100,"ANNOTATOR_B_QUERY":100,"ANNOTATOR_A_DIRECTION":120,"ANNOTATOR_B_DIRECTION":120,"CALIBRATION_A_QUERY":20,"CALIBRATION_A_DIRECTION":28}
        for mode,count in expected.items(): self.assertEqual(len(load_mode(mode,completed_dir=self.completed)["rows"]),count)
        for task,key in (("QUERY","example_id"),("DIRECTION","direction_item_id")):
            a=load_mode(f"ANNOTATOR_A_{task}",completed_dir=self.completed)["rows"]
            b=load_mode(f"ANNOTATOR_B_{task}",completed_dir=self.completed)["rows"]
            self.assertEqual([r[key] for r in a],[r[key] for r in b])

    def test_initial_answers_blank_and_schema_compatible(self):
        for mode in MODE_CONFIG:
            payload=load_mode(mode,completed_dir=self.completed)
            self.assertTrue(all(not row.get(field) for row in payload["rows"] for field in payload["answer_fields"]))
        self.assertEqual(QUERY_ANSWERS[-2:],("old_value_normalized","new_value_normalized"))
        self.assertEqual(DIRECTION_ANSWERS,("candidate_direction","direction_confidence"))

    def test_save_resume_separation_and_row_counts(self):
        mode="ANNOTATOR_A_QUERY"; payload=load_mode(mode,completed_dir=self.completed); key=payload["key"]
        submitted=[]
        for i,row in enumerate(payload["rows"]):
            answer={key:row[key],**{field:"" for field in payload["answer_fields"]}}
            if i==0: answer["core_intent_preserved"]="YES"
            submitted.append(answer)
        result=save_mode(mode,submitted,completed_dir=self.completed); self.assertEqual(result["row_count"],100)
        resumed=load_mode(mode,completed_dir=self.completed); self.assertEqual(resumed["rows"][0]["core_intent_preserved"],"YES")
        self.assertFalse((self.completed/"annotator_B_query_annotations.csv").exists())
        self.assertFalse((self.completed/"calibration_annotations_A.csv").exists())

    def test_immutable_metadata_and_reordering_rejected(self):
        payload=load_mode("ANNOTATOR_A_DIRECTION",completed_dir=self.completed); rows=payload["rows"]; key=payload["key"]
        submitted=[{key:r[key],**{f:"" for f in payload["answer_fields"]}} for r in rows]
        submitted[0]["candidate_a_product_title"]="changed"
        with self.assertRaisesRegex(ValueError,"non-answer"):save_mode("ANNOTATOR_A_DIRECTION",submitted,completed_dir=self.completed)
        submitted=[{key:r[key],**{f:"" for f in payload["answer_fields"]}} for r in reversed(rows)]
        with self.assertRaisesRegex(ValueError,"count and ordering"):save_mode("ANNOTATOR_A_DIRECTION",submitted,completed_dir=self.completed)

    def test_atomic_write_leaves_complete_csv(self):
        path=self.completed/"atomic.csv"; atomic_write_csv(path,["id","answer"],[{"id":"1","answer":"YES"}])
        fields,rows=read_csv(path); self.assertEqual(fields,["id","answer"]); self.assertEqual(rows,[{"id":"1","answer":"YES"}])
        self.assertEqual(list(self.completed.glob("*.tmp")),[])

    def test_annotator_facing_field_blindness(self):
        prohibited={"esci_label","ndcg","rank_score","model","cortex","route","oracle","winner","expected_direction_machine","machine_confidence"}
        for mode in MODE_CONFIG:
            payload=load_mode(mode,completed_dir=self.completed)
            fields=set(payload["rows"][0]) if payload["rows"] else set()
            self.assertFalse({f.lower() for f in fields}&prohibited)
        for name in ("index.html","app.js","styles.css"):
            text=(Path(__file__).parents[1]/"tools/cf_esci_annotation"/name).read_text(encoding="utf-8").lower()
            self.assertNotIn("expected_direction_machine",text); self.assertNotIn("machine_confidence",text)

    def test_source_files_remain_unchanged_after_save(self):
        before={name:hashlib.sha256((DATA_DIR/name).read_bytes()).hexdigest() for name, *_ in MODE_CONFIG.values()}
        payload=load_mode("CALIBRATION_A_QUERY",completed_dir=self.completed)
        submitted=[{payload["key"]:r[payload["key"]],**{f:"" for f in payload["answer_fields"]}} for r in payload["rows"]]
        save_mode("CALIBRATION_A_QUERY",submitted,completed_dir=self.completed)
        after={name:hashlib.sha256((DATA_DIR/name).read_bytes()).hexdigest() for name in before}; self.assertEqual(before,after)


if __name__ == "__main__": unittest.main()
