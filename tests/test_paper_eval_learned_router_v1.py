import unittest
import numpy as np
import pandas as pd

from src.paper_eval.adapter import CandidateBoundaryError
from src.paper_eval.contracts_v1 import build_search_contract_v1
from src.paper_eval.learned_router_v1 import CONTRACT_FEATURES, GENERIC_FEATURES, choose_routes, extract_router_features, fit_ridge


class LearnedRouterV1Tests(unittest.TestCase):
    def setUp(self):
        self.rows=[{"product_id":"a","product_title":"nike running shoe","product_brand":"nike","product_description":"","product_bullet_point":"","product_color":"red","score":2.0,"query_token_coverage":1.0},
                   {"product_id":"b","product_title":"walking shoe","product_brand":"","product_description":"","product_bullet_point":"","product_color":"blue","score":1.0,"query_token_coverage":.5}]

    def test_generic_is_strict_subset_of_contract_features(self):
        generic,contract=extract_router_features("nike running shoe",self.rows,build_search_contract_v1("nike running shoe"))
        self.assertEqual(set(generic),set(GENERIC_FEATURES)); self.assertEqual(set(contract),set(CONTRACT_FEATURES)); self.assertTrue(set(generic)<set(contract))

    def test_labels_are_rejected(self):
        contaminated=[{**self.rows[0],"esci_label":"E"}]
        with self.assertRaises(CandidateBoundaryError): extract_router_features("shoe",contaminated,build_search_contract_v1("shoe"))

    def test_ridge_and_route_choice_are_reproducible(self):
        frame=pd.DataFrame({"x":[0.,1.,2.,3.],"target":[0.,1.,2.,3.]}); a=fit_ridge(frame,("x",),"target",1.0); b=fit_ridge(frame,("x",),"target",1.0)
        self.assertEqual(a.to_dict(),b.to_dict()); np.testing.assert_array_equal(choose_routes(np.array([.1,-.1,0]),np.array([0,.2,0]),.01),np.array(["STRICT_FILTER","CONTRACT_RERANK","PRESERVE"],dtype=object))

if __name__=="__main__":unittest.main()
