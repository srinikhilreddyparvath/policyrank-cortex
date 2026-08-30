import tempfile, unittest
from pathlib import Path
import pandas as pd
from src.paper_eval.schema import QueryResult
from src.cortex_full_esci_evaluation import failure_result

class SchemaTests(unittest.TestCase):
    def test_parquet_round_trip_and_failure_row(self):
        ok=QueryResult("run","fts_baseline","v",1,"query","us","test",final_product_ids=["p"],final_labels=["E"],final_scores=[1.0])
        failure=QueryResult("run","fts_baseline","v",2,"bad","us","test",success=False,failure_stage="ranking",error_type="ValueError",error_message="bad")
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"rows.parquet"; pd.DataFrame([ok.to_dict(),failure.to_dict()]).to_parquet(path,index=False); loaded=pd.read_parquet(path)
        self.assertEqual(len(loaded),2); self.assertEqual(int((loaded.success==False).sum()),1); self.assertEqual(loaded.iloc[0].final_product_ids.tolist(),["p"])
    def test_failure_row_accepts_query_named_column(self):
        frame=pd.DataFrame({"query_id":[1],"query":["running shoe"],"product_locale":["us"],"split":["test"]})
        row=failure_result("run","method",frame,"ranking",ValueError("bad"),1.0)
        self.assertEqual(row.query,"running shoe"); self.assertFalse(row.success)

if __name__=="__main__": unittest.main()
