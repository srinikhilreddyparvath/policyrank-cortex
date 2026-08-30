import unittest
import pandas as pd

from src.cortex_routing_headroom_evaluation import _add_oracle


class RoutingHeadroomTests(unittest.TestCase):
    def test_oracle_preserves_ties_and_router_regret(self):
        rows=[]
        for method,score in (("PRESERVE",.8),("STRICT_FILTER",.9),("CONTRACT_RERANK",.9),("ROUTER_V2",.8)):
            rows.append({"query_id":1,"method":method,"success":True,"official_esci_ndcg":score})
        result=_add_oracle(pd.DataFrame(rows)); router=result[result.method=="ROUTER_V2"].iloc[0]
        self.assertEqual(router.oracle_route_set,["CONTRACT_RERANK","STRICT_FILTER"])
        self.assertAlmostEqual(router.oracle_ndcg,.9); self.assertAlmostEqual(router.selected_route_reward,.8); self.assertAlmostEqual(router.route_regret_ndcg,.1)

    def test_missing_atomic_route_does_not_fabricate_oracle(self):
        result=_add_oracle(pd.DataFrame([{"query_id":1,"method":"PRESERVE","success":True,"official_esci_ndcg":.8}]))
        self.assertIsNone(result.iloc[0].oracle_ndcg)

if __name__=="__main__": unittest.main()
