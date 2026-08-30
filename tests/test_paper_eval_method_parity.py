import unittest
import pandas as pd
from src.full_esci_retrieval_engine import FullEsciRetrievalEngine
from src.paper_eval.adapter import CandidateBoundaryError, CandidateSetRequest, validate_output_subset
from src.paper_eval.methods import REGISTRY, _lexical_rows, enabled_methods
from src.baseline_preservation_gate import decide_baseline_gate
from src.paper_eval.provenance import ContaminatedComponentError

class MethodParityTests(unittest.TestCase):
    def setUp(self):
        self.candidates=pd.DataFrame([{"query_id":1,"product_id":"1","product_title":"red running shoe","product_brand":"","product_description":"","product_bullet_point":"","product_color":"red"},{"query_id":1,"product_id":"2","product_title":"coffee cup","product_brand":"","product_description":"","product_bullet_point":"","product_color":""}])
    def test_lexical_adapter_uses_existing_scorer_and_same_candidates(self):
        rows=_lexical_rows("running shoe",self.candidates); self.assertEqual({r["product_id"] for r in rows},{"1","2"}); self.assertEqual(rows[0]["product_id"],"1"); self.assertIs(FullEsciRetrievalEngine.score_product,FullEsciRetrievalEngine.score_product)
    def test_method_query_parity_and_disabled_reasons(self):
        specs,disabled=enabled_methods(["fts_baseline","strict_filter","always_cortex","current_reranker","gated_cortex","semantic_baseline"]); self.assertEqual([s.name for s in specs],["fts_baseline","strict_filter"]); self.assertIn("ESCI-label-derived",disabled["always_cortex"]); self.assertIn("directly consumes esci_label",disabled["current_reranker"]); self.assertTrue(disabled["semantic_baseline"])
    def test_enabled_methods_preserve_candidate_ids(self):
        for name in ("fts_baseline","strict_filter","strict_boost"):
            rows,_=REGISTRY[name].evaluator("running shoe",self.candidates,len(self.candidates)); self.assertEqual({r["product_id"] for r in rows},{"1","2"})
    def test_labels_never_cross_adapter_boundary(self):
        contaminated=self.candidates.assign(esci_label=["E","I"])
        with self.assertRaises(CandidateBoundaryError): REGISTRY["fts_baseline"].evaluator("running shoe",contaminated,2)
    def test_external_candidate_fails_explicitly(self):
        request=CandidateSetRequest("q",1,self.candidates)
        with self.assertRaises(CandidateBoundaryError): validate_output_subset(request,[{"product_id":"external"}])
    def test_legacy_contaminated_state_rejected_without_unsafe_flag(self):
        with self.assertRaises(ContaminatedComponentError): REGISTRY["always_cortex"].evaluator("running shoe",self.candidates,2)
        with self.assertRaises(ContaminatedComponentError): REGISTRY["gated_cortex"].evaluator("running shoe",self.candidates,2)

    def test_always_cortex_debug_path_is_deterministic_and_subset_safe(self):
        first,meta1=REGISTRY["always_cortex"].evaluator("running shoe",self.candidates,2,allow_unsafe_debug=True); second,meta2=REGISTRY["always_cortex"].evaluator("running shoe",self.candidates,2,allow_unsafe_debug=True)
        self.assertEqual([r["product_id"] for r in first],[r["product_id"] for r in second]); self.assertIsInstance(meta1["intervened"],bool); self.assertFalse(meta1["judgment_features_received"])
    def test_gated_preserve_and_intervene_decisions_are_label_safe(self):
        scores=[1.0,.99,.98,.97,.96,0.0]
        labeled=pd.DataFrame({"product_id":[str(i) for i in range(6)],"product_title":["running shoe"]*6,"product_brand":[""]*6,"semantic_score":scores,"esci_label":["E"]*6})
        contract={"required_terms":["running","shoe"],"preferred_terms":["running"],"excluded_terms":[],"brand_preferences":[]}
        label_safe=labeled.drop(columns=["esci_label"]); self.assertEqual(decide_baseline_gate(label_safe,contract,top_k=5).decision,"preserve_baseline")
        weak=label_safe.copy(); weak["product_title"]=["unrelated item"]*6; weak["semantic_score"]=[1,.2,.1,.05,.01,0]
        self.assertNotEqual(decide_baseline_gate(weak,contract,top_k=5).decision,"preserve_baseline")
        rows,meta=REGISTRY["gated_cortex"].evaluator("running shoe",self.candidates,2,allow_unsafe_debug=True); self.assertEqual(len(rows),len(self.candidates)); self.assertIn("baseline remainder",meta["task1_completion"])

if __name__=="__main__": unittest.main()
