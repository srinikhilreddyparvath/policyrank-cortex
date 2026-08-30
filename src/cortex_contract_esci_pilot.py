from __future__ import annotations

import argparse
import hashlib
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.paper_eval.contract_esci import PILOT_SIZE, PILOT_VERSION, SCHEMA_VERSION, SELECTION_SEED, STRATUM_QUOTAS, assert_blind_artifact, assert_no_esci_test_queries, select_pilot_queries, sha256_file
from src.paper_eval.dataset import development_query_split, load_examples, load_products_for_candidates, policy_calibration_split, task_examples
from src.paper_eval.methods import REGISTRY

CANDIDATES_PER_QUERY = 5


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(record, ensure_ascii=False, sort_keys=True)+"\n" for record in records), encoding="utf-8")


def query_template(row) -> dict:
    record={"schema_version":SCHEMA_VERSION,"query_id":int(row.query_id),"query_text":str(row.query_text),"product_type":None,"requirements":[],"must_have":[],"must_not_have":[],"brand_requirement":None,"price_requirement":None,"attribute_requirements":[],"hard_constraints":[],"soft_preferences":[],"negation_present":None,"comparative_or_superlative_intent":None,"ambiguity":None,"contract_annotatability":None,"annotation_notes":None,"annotator_id":None,"annotation_status":"unannotated"}
    assert_blind_artifact(record); return record


def candidate_records(task:pd.DataFrame,pilot:pd.DataFrame,dataset_path:Path)->list[dict]:
    selected=task[task.query_id.isin(pilot.query_id)].copy(); products=load_products_for_candidates(dataset_path,set(selected.product_id.astype(str)))
    selected=selected.drop(columns=["esci_label"],errors="ignore").merge(products,on=["product_locale","product_id"],how="left",validate="many_to_one")
    records=[]
    for query_id,group in selected.groupby("query_id",sort=True):
        query=str(group["query"].iloc[0]); rows,_=REGISTRY["fts_baseline"].evaluator(query,group,CANDIDATES_PER_QUERY)
        for position,row in enumerate(rows,1):
            metadata={key:(None if pd.isna(row.get(key)) else str(row.get(key))) for key in ("product_title","product_brand","product_color","product_bullet_point","product_description")}
            record={"schema_version":SCHEMA_VERSION,"query_id":int(query_id),"query_text":query,"product_id":str(row["product_id"]),"selection_position":position,"product_metadata":metadata,"requirement_assessments":[],"overall_hard_constraint_state":None,"annotation_notes":None,"annotator_id":None,"annotation_status":"unannotated"}
            assert_blind_artifact(record); records.append(record)
    if len(records)!=len(pilot)*CANDIDATES_PER_QUERY: raise AssertionError("Candidate template count mismatch")
    return records


def run(dataset_path:Path,out:Path)->dict:
    out.mkdir(parents=True,exist_ok=True); examples=load_examples(dataset_path); task=task_examples(examples,task_version="small",split="train")
    train_fit,validation=development_query_split(task,seed=SELECTION_SEED,train_fraction=.8); policy,calibration=policy_calibration_split(train_fit,seed=SELECTION_SEED,policy_fraction=.8)
    partition={qid:"policy_train" for qid in policy}|{qid:"calibration" for qid in calibration}
    queries=task[task.query_id.isin(partition)].drop_duplicates("query_id")[["query_id","query"]].rename(columns={"query":"query_text"}); queries["source_partition"]=queries.query_id.map(partition)
    pilot=select_pilot_queries(queries,seed=SELECTION_SEED); ids=pilot.query_id.astype(int).tolist(); assert_no_esci_test_queries(ids,examples)
    ids_path=out/"pilot_query_ids.txt"; ids_path.write_text("\n".join(map(str,ids))+"\n",encoding="utf-8"); pilot.to_parquet(out/"pilot_queries.parquet",index=False)
    write_jsonl(out/"query_annotation_template.jsonl",[query_template(row) for row in pilot.itertuples(index=False)])
    write_jsonl(out/"candidate_annotation_template.jsonl",candidate_records(task,pilot,dataset_path))
    schema_path=out/"annotation_schema.json"; schema=json.loads(schema_path.read_text(encoding="utf-8")); schema_hash=sha256_file(schema_path)
    manifest={"pilot_version":PILOT_VERSION,"schema_version":SCHEMA_VERSION,"created_at_utc":datetime.now(timezone.utc).isoformat(),"source":{"task":"ESCI Task 1 small","split":"train only","allowed_partitions":["policy_train","calibration"],"esci_test_used":False},"selection":{"seed":SELECTION_SEED,"query_count":PILOT_SIZE,"method":"Mutually exclusive query-text characteristic strata; SHA-256(seed:query_id) order within each stratum; no ranking or route outcomes consulted.","stratum_quotas":STRATUM_QUOTAS},"candidate_sampling":{"candidates_per_query":CANDIDATES_PER_QUERY,"method":"Top five candidates from a fixed label-free lexical scorer over the original judged candidate set. ESCI labels and scores are omitted from annotation artifacts; no CORTEX output is consulted."},"blindness":{"forbidden_inputs":["CORTEX outcomes","route outcomes","oracle winners","method metrics","method identities","which method wins"],"human_labels_prepopulated":False},"counts":{"policy_train_available":len(policy),"calibration_available":len(calibration),"held_out_validation":len(validation),"pilot_queries":len(pilot),"candidate_pairs":len(pilot)*CANDIDATES_PER_QUERY},"hashes":{"pilot_query_ids_sha256":sha256_file(ids_path),"pilot_queries_sha256":sha256_file(out/"pilot_queries.parquet"),"query_annotation_template_sha256":sha256_file(out/"query_annotation_template.jsonl"),"candidate_annotation_template_sha256":sha256_file(out/"candidate_annotation_template.jsonl"),"annotation_schema_sha256":schema_hash},"libraries":{"python":platform.python_version(),"pandas":pd.__version__}}
    (out/"pilot_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8"); return manifest


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--dataset-path",type=Path,default=Path("esci-data/shopping_queries_dataset")); parser.add_argument("--output-dir",type=Path,default=Path("data/contract_esci/pilot_v1")); args=parser.parse_args(); print(json.dumps(run(args.dataset_path,args.output_dir),indent=2))

if __name__=="__main__": main()
