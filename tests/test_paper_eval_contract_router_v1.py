import unittest
import pandas as pd

from src.paper_eval.adapter import CandidateBoundaryError
from src.paper_eval.clean_router import ALLOWED_ROUTES, route_contract_v1
from src.paper_eval.contracts_v1 import build_search_contract_v1, exceptional_fallback_contract
from src.paper_eval.methods import REGISTRY


class ContractRouterV1Tests(unittest.TestCase):
    def setUp(self):
        self.candidates=pd.DataFrame([
            {"query_id":1,"product_id":"a","product_title":"red running shoes no laces","product_brand":"","product_description":"","product_bullet_point":"","product_color":"red"},
            {"query_id":1,"product_id":"b","product_title":"blue running shoes with laces","product_brand":"","product_description":"","product_bullet_point":"","product_color":"blue"},
        ])

    def test_contract_is_deterministic_and_label_safe(self):
        first=build_search_contract_v1("Running shoes without laces")
        self.assertEqual(first,build_search_contract_v1("Running shoes without laces"))
        with self.assertRaises(CandidateBoundaryError): build_search_contract_v1("shoes",{"esci_label":"E"})

    def test_contract_status_and_fallback_semantics(self):
        self.assertEqual(build_search_contract_v1("running shoes without laces").contract_status,"resolved")
        self.assertEqual(build_search_contract_v1("nike").contract_status,"partial")
        self.assertEqual(build_search_contract_v1("").contract_status,"unresolved")
        self.assertFalse(build_search_contract_v1("shoes").fallback_used)
        self.assertTrue(exceptional_fallback_contract("shoes","parser_exception").fallback_used)

    def test_router_is_deterministic_and_allowed(self):
        contract=build_search_contract_v1("running shoes without laces")
        first=route_contract_v1(contract); self.assertEqual(first,route_contract_v1(contract)); self.assertIn(first.route,ALLOWED_ROUTES); self.assertEqual(first.route,"STRICT_FILTER")
        self.assertEqual(route_contract_v1(build_search_contract_v1("running shoes")).route,"PRESERVE")

    def test_attempted_and_effective_are_distinct(self):
        _,meta=REGISTRY["clean_contract_router_v1"].evaluator("running shoes without unicorns",self.candidates,2)
        self.assertTrue(meta["intervention_attempted"]); self.assertIsInstance(meta["intervention_effective"],bool)
        _,preserve=REGISTRY["clean_contract_router_v1"].evaluator("running shoes",self.candidates,2)
        self.assertFalse(preserve["intervention_attempted"]); self.assertFalse(preserve["intervention_effective"]); self.assertTrue(preserve["explicit_preserve"])

    def test_strict_filter_boost_parity_is_measurable(self):
        filtered,_=REGISTRY["strict_filter"].evaluator("running shoes without laces",self.candidates,2)
        boosted,_=REGISTRY["strict_boost"].evaluator("running shoes without laces",self.candidates,2)
        self.assertEqual({row["product_id"] for row in filtered},{"a","b"})
        self.assertEqual({row["product_id"] for row in boosted},{"a","b"})

if __name__=="__main__": unittest.main()
