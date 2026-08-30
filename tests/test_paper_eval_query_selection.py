import unittest
import pandas as pd
from src.paper_eval.dataset import development_query_split, policy_calibration_split, preserve_selected_queries, select_query_ids, task_examples

class QuerySelectionTests(unittest.TestCase):
    def setUp(self):
        self.frame=pd.DataFrame({"query_id":[1,1,2,2,3,3],"query":["a","a","b","b","c","c"],"product_id":["a1","a2","b1","b2","c1","c2"],"product_locale":["us"]*6,"esci_label":["E","I","S","I","C","E"],"small_version":[1,1,1,1,0,0],"large_version":[1]*6,"split":["test"]*6})
    def test_task_filter_and_group_preservation(self):
        task=task_examples(self.frame); self.assertEqual(set(task.query_id),{1,2}); selected=preserve_selected_queries(task,[2]); self.assertEqual(selected.product_id.tolist(),["b1","b2"])
    def test_deterministic_query_selection(self):
        task=task_examples(self.frame); self.assertEqual(select_query_ids(task,max_queries=1,seed=29),select_query_ids(task,max_queries=1,seed=29))
    def test_development_split_is_deterministic_and_disjoint(self):
        frame=pd.DataFrame({"query_id":list(range(20))}); a=development_query_split(frame,seed=29); b=development_query_split(frame,seed=29)
        self.assertEqual(a,b); self.assertFalse(set(a[0])&set(a[1])); self.assertEqual(len(a[0]),16); self.assertEqual(len(a[1]),4)
    def test_nested_policy_calibration_validation_are_disjoint(self):
        train_fit,validation=development_query_split(pd.DataFrame({"query_id":list(range(100))}),seed=29)
        policy,calibration=policy_calibration_split(train_fit,seed=29)
        self.assertEqual((policy,calibration),policy_calibration_split(train_fit,seed=29))
        self.assertFalse(set(policy)&set(calibration)); self.assertFalse(set(policy)&set(validation)); self.assertFalse(set(calibration)&set(validation))

if __name__=="__main__": unittest.main()
