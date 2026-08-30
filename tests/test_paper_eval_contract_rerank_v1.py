import unittest
import pandas as pd

from src.paper_eval.adapter import CandidateBoundaryError
from src.paper_eval.clean_router import ALLOWED_ROUTES_V2, route_contract_v2
from src.paper_eval.contract_rerank_v1 import DEFAULT_CONFIG, contract_rerank_v1
from src.paper_eval.contracts_v1 import SearchContractV1, build_search_contract_v1
from src.paper_eval.methods import REGISTRY


class ContractRerankV1Tests(unittest.TestCase):
    def setUp(self):
        self.candidates=pd.DataFrame([
            {"query_id":1,"product_id":"a","product_title":"casual blue shoe","product_brand":"generic","product_description":"running","product_bullet_point":"","product_color":"blue"},
            {"query_id":1,"product_id":"b","product_title":"nike red running shoe","product_brand":"nike","product_description":"performance running shoe","product_bullet_point":"","product_color":"red"},
            {"query_id":1,"product_id":"c","product_title":"coffee mug","product_brand":"generic","product_description":"ceramic","product_bullet_point":"","product_color":"white"},
        ])

    def test_deterministic_reproducible_subset_rerank(self):
        baseline=[{"product_id":"a","product_title":"casual blue shoe","product_brand":"generic","product_description":"running","product_bullet_point":"","product_color":"blue","score":2.0},
                  {"product_id":"b","product_title":"nike red running shoe","product_brand":"nike","product_description":"performance running shoe","product_bullet_point":"","product_color":"red","score":1.9}]
        contract=build_search_contract_v1("nike red running shoe")
        first,meta1=contract_rerank_v1(baseline,contract); second,meta2=contract_rerank_v1(baseline,contract)
        self.assertEqual(first,second); self.assertEqual(meta1,meta2); self.assertEqual({r["product_id"] for r in first},{"a","b"}); self.assertEqual(first[0]["product_id"],"b")

    def test_no_labels_reach_reranker(self):
        with self.assertRaises(CandidateBoundaryError):
            contract_rerank_v1([{"product_id":"a","score":1.0,"esci_label":"E"}],build_search_contract_v1("shoe"))

    def test_hard_terms_are_diagnostic_not_hidden_filtering(self):
        rows=[{"product_id":"a","product_title":"shoe with laces","score":1.0},{"product_id":"b","product_title":"slip on shoe","score":0.9}]
        output,meta=contract_rerank_v1(rows,build_search_contract_v1("shoe without laces"))
        self.assertEqual(len(output),2); self.assertEqual(meta["hard_constraint_policy"],"diagnostic_only; STRICT_FILTER owns enforcement"); self.assertGreater(meta["must_not_have_violations"],0)

    def test_unknown_signals_are_neutral(self):
        base=build_search_contract_v1("shoe")
        unknown=SearchContractV1(**{**base.to_dict(),"brand_signal":None,"price_signal":None,"product_type":None,"positive_terms":()})
        rows=[{"product_id":"a","product_title":"x","score":2.0},{"product_id":"b","product_title":"y","score":1.0}]
        output,_=contract_rerank_v1(rows,unknown); self.assertEqual([r["product_id"] for r in output],["a","b"])

    def test_router_v2_all_routes_reachable_and_deterministic(self):
        cases={"shoe without laces":"STRICT_FILTER","nike red running shoe":"CONTRACT_RERANK","running shoe":"PRESERVE"}
        for query,expected in cases.items():
            contract=build_search_contract_v1(query); first=route_contract_v2(contract)
            self.assertEqual(first,route_contract_v2(contract)); self.assertEqual(first.route,expected); self.assertIn(first.route,ALLOWED_ROUTES_V2)

    def test_always_rerank_and_diagnostics(self):
        rows,meta=REGISTRY["always_contract_rerank_v1"].evaluator("nike red running shoe",self.candidates,3)
        self.assertEqual({r["product_id"] for r in rows},{"a","b","c"}); self.assertEqual(meta["route_selected"],"CONTRACT_RERANK")
        for key in ("soft_preference_matches","must_not_have_violations","brand_matches","price_signal_matches","product_type_matches","contract_score_range","rerank_changed","mean_candidate_movement","top_1_changed"):
            self.assertIn(key,meta)
        self.assertEqual(DEFAULT_CONFIG.provenance,"a_priori_interpretable_v1; no outcome-based tuning")

    def test_clean_router_v2_does_not_use_contaminated_state(self):
        _,meta=REGISTRY["clean_contract_router_v2"].evaluator("nike red running shoe",self.candidates,3)
        self.assertEqual(meta["route_selected"],"CONTRACT_RERANK"); self.assertNotIn("selected_rl_action",meta); self.assertNotIn("gate_decision",meta)

if __name__=="__main__": unittest.main()
