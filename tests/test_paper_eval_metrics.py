import math, unittest
from src.paper_eval.metrics import OFFICIAL_GAINS, OFFICIAL_RELEVANCE_LEVELS, beneficial_intervention_rate, classify_intervention, cortex_ndcg_at_k, exact_at_k, exact_or_substitute_at_k, harmful_intervention_rate, official_esci_ndcg, official_esci_ndcg_at_k, reciprocal_rank_at_k, violation_rate

class MetricsTests(unittest.TestCase):
    def test_official_mapping(self):
        self.assertEqual(OFFICIAL_RELEVANCE_LEVELS,{"E":4,"C":3,"S":2,"I":1}); self.assertEqual(OFFICIAL_GAINS,{"E":1.0,"C":0.1,"S":0.01,"I":0.0})
    def test_official_ndcg(self):
        self.assertEqual(official_esci_ndcg(["E","C","S","I"]),1.0); self.assertLess(official_esci_ndcg(["I","S","C","E"]),1.0); self.assertEqual(official_esci_ndcg_at_k(["E","I"],1),1.0)
        self.assertLess(official_esci_ndcg(["S"],["E","S"]),official_esci_ndcg(["S"]))
    def test_research_ndcg_and_mrr(self):
        self.assertEqual(cortex_ndcg_at_k(["E","S","I"],10),1.0); self.assertAlmostEqual(reciprocal_rank_at_k(["S","E","I"],10),0.5)
    def test_exact_metrics(self):
        self.assertEqual(exact_at_k(["S","E"],1),0.0); self.assertEqual(exact_at_k(["S","E"],5),1.0); self.assertEqual(exact_or_substitute_at_k(["I","S"],5),1.0)
    def test_missing(self): self.assertIsNone(official_esci_ndcg([])); self.assertIsNone(reciprocal_rank_at_k([]))
    def test_governance_and_constraint_rates(self):
        self.assertEqual(beneficial_intervention_rate([True,True,False],[.2,-.1,1.0]),.5); self.assertEqual(harmful_intervention_rate([True,True],[.2,-.1]),.5); self.assertEqual(violation_rate(2,10),.2); self.assertIsNone(violation_rate(None,10))
        self.assertEqual(classify_intervention(True,2,1),"beneficial"); self.assertEqual(classify_intervention(True,1,2),"harmful"); self.assertEqual(classify_intervention(True,1,1),"neutral"); self.assertIsNone(classify_intervention(False,2,1))

if __name__=="__main__": unittest.main()
