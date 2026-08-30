import unittest
from src.paper_eval.statistics import bootstrap_mean_ci, paired_bootstrap_difference

class StatisticsTests(unittest.TestCase):
    def test_bootstrap_reproducible(self):
        self.assertEqual(bootstrap_mean_ci([1,2,3],seed=9,iterations=100),bootstrap_mean_ci([1,2,3],seed=9,iterations=100))
    def test_paired_behavior(self):
        result=paired_bootstrap_difference([2,3,4],[1,2,3],seed=9,iterations=100); self.assertEqual(result["mean_difference"],1.0); self.assertEqual(result["lower"],1.0)
    def test_missing(self): self.assertEqual(bootstrap_mean_ci([],iterations=10)["n"],0); self.assertIsNone(paired_bootstrap_difference([],[],iterations=10)["mean_difference"])

if __name__=="__main__": unittest.main()
