from __future__ import annotations

import argparse, csv, hashlib, json, platform
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.paper_eval.contract_esci import SELECTION_SEED, assert_no_esci_test_queries, sha256_file
from src.paper_eval.dataset import development_query_split, load_examples, load_products_for_candidates, policy_calibration_split, task_examples
from src.paper_eval.human_validation import *


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row, sort_keys=True, ensure_ascii=False)+"\n" for row in rows), encoding="utf-8")


def write_csv(path: Path, frame: pd.DataFrame) -> None:
    frame.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")


def train_task(dataset: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    examples=load_examples(dataset); task=task_examples(examples, task_version="small", split="train")
    fit, validation=development_query_split(task, seed=SELECTION_SEED, train_fraction=.8); policy, calibration=policy_calibration_split(fit, seed=SELECTION_SEED, policy_fraction=.8)
    allowed=set(policy)|set(calibration); assert_no_esci_test_queries(allowed, examples)
    return task[task.query_id.isin(allowed)].copy(), examples


def product_metadata(products: pd.DataFrame) -> dict[str,dict]:
    fields=("product_title","product_brand","product_color","product_bullet_point","product_description")
    return {str(row.product_id): {f: "" if pd.isna(getattr(row,f)) else str(getattr(row,f)) for f in fields} for row in products.itertuples()}


def make_direction_rows(selected: pd.DataFrame, generated_pairs:list[dict], natural_pairs:list[dict], metadata:dict[str,dict]) -> pd.DataFrame:
    by_generated=defaultdict(list); by_natural=defaultdict(list)
    for row in generated_pairs: by_generated[row["counterfactual_id"]].append(row)
    for row in natural_pairs: by_natural[row["canonical_pair_id"]].append(row)
    rows=[]
    for ex in selected.to_dict("records"):
        candidates=by_generated[ex["source_record_id"]] if ex["source_type"]=="GENERATED" else by_natural[ex["source_record_id"]]
        for i,pair in enumerate(sorted(candidates,key=lambda r:json.dumps(r,sort_keys=True))[:2],1):
            if ex["source_type"]=="GENERATED": a,b=pair["p_old"],pair["p_new"]; ae,be=pair.get("p_old_evidence",""),pair.get("p_new_evidence","")
            else: a,b=pair["p_a"],pair["p_b"]; ae,be="",""
            row={"direction_item_id":f"{ex['example_id']}_D{i}","example_id":ex["example_id"],"query_a":ex["query_a"],"query_b":ex["query_b"],"candidate_a_id":a,"candidate_b_id":b,"candidate_a_visible_evidence":ae,"candidate_b_visible_evidence":be}
            for side,pid in (("a",a),("b",b)):
                for field,value in metadata.get(pid,{}).items(): row[f"candidate_{side}_{field}"]=value
            rows.append(row)
    return pd.DataFrame(rows)


def blank_direction_template(rows:pd.DataFrame, annotator_id:str)->pd.DataFrame:
    out=rows.copy(); out.insert(1,"annotator_id",annotator_id); out["candidate_direction"]=""; out["direction_confidence"]=""; assert_annotation_blind(out); return out


def candidate_pools(selected:pd.DataFrame, task:pd.DataFrame, direction_rows:pd.DataFrame)->list[dict]:
    query_products={int(q): sorted(set(g.product_id.astype(str))) for q,g in task.groupby("query_id")}; direction=defaultdict(set)
    for row in direction_rows.to_dict("records"): direction[row["example_id"]].update((row["candidate_a_id"],row["candidate_b_id"]))
    result=[]
    for ex in selected.to_dict("records"):
        a=set(query_products.get(int(ex["source_query_id_a"]),[])); b=set(query_products.get(int(ex["source_query_id_b"]),[])) if pd.notna(ex["source_query_id_b"]) else set(); union=a|b|direction[ex["example_id"]]
        ordered=sorted(union,key=lambda pid:(pid not in direction[ex["example_id"]], hashlib.sha256(f"30:{ex['example_id']}:{pid}".encode()).hexdigest()))[:MAX_POOL_SIZE]
        members=[]
        for pid in ordered:
            prov=[]
            if pid in a:prov.append("QUERY_A_ORIGINAL_POOL")
            if pid in b:prov.append("QUERY_B_ORIGINAL_POOL")
            if pid in direction[ex["example_id"]]:prov.append("CATALOG_EVIDENCE_PAIR")
            members.append({"product_id":pid,"provenance":prov})
        result.append({"example_id":ex["example_id"],"source_type":ex["source_type"],"maximum_pool_size":MAX_POOL_SIZE,"selection_rule":"evidence-pair candidates first, then deterministic SHA-256 order over the source-pool union","candidate_count":len(members),"candidates":members})
    return result


def build(dataset:Path, generated_dir:Path, natural_dir:Path, out:Path)->dict:
    out.mkdir(parents=True,exist_ok=True)
    generated_records=read_jsonl(generated_dir/"counterfactual_proposals.jsonl"); natural_records=read_jsonl(natural_dir/"natural_pair_proposals.jsonl")
    sample,calibration,final=select_human_validation_sample(normalize_generated(generated_records),normalize_natural(natural_records))
    sample.to_parquet(out/"human_validation_source_pairs.parquet",index=False)
    task,examples=train_task(dataset); ids=set(sample.source_query_id_a.astype(int))|set(sample.source_query_id_b.dropna().astype(int)); assert ids<=set(task.query_id.astype(int))
    products=load_products_for_candidates(dataset,set(task[task.query_id.isin(ids)].product_id.astype(str))); metadata=product_metadata(products)
    directions=make_direction_rows(sample,read_jsonl(generated_dir/"candidate_direction_pairs.jsonl"),read_jsonl(natural_dir/"candidate_direction_pairs.jsonl"),metadata)
    calibration_directions=directions[directions.example_id.isin(calibration.example_id)].reset_index(drop=True); final_directions=directions[directions.example_id.isin(final.example_id)].reset_index(drop=True)
    calibration_query=blank_query_template(calibration,"").drop(columns="annotator_id"); calibration_answers=blank_query_template(calibration,"CALIBRATION")
    write_csv(out/"calibration_query_pairs.csv",calibration_query); write_csv(out/"annotation_calibration_answer_template.csv",calibration_answers)
    write_csv(out/"calibration_candidate_pairs.csv",blank_direction_template(calibration_directions,"").drop(columns="annotator_id"))
    for annotator in ("A","B"):
        write_csv(out/f"annotator_{annotator}_query_template.csv",blank_query_template(final,annotator)); write_csv(out/f"annotator_{annotator}_direction_template.csv",blank_direction_template(final_directions,annotator))
    pools=candidate_pools(sample,task,directions); write_jsonl(out/"candidate_pool_manifest.jsonl",pools)
    sampling={"version":HUMAN_VALIDATION_VERSION,"seed":SAMPLE_SEED,"sample_size":len(sample),"calibration_size":len(calibration),"final_size":len(final),"generated_sampling":"15 deterministic proposals from each of COLOR, BRAND, NUMERIC_SPECIFICATION, EXPLICIT_ATTRIBUTE","natural_sampling":"all 60 conservative natural proposals","calibration_quotas":CALIBRATION_QUOTAS,"warning":"Stratified oversampling means sample proportions are not estimates of corpus prevalence.","sample_ids":sample.example_id.tolist(),"calibration_ids":calibration.example_id.tolist(),"final_ids":final.example_id.tolist()}; (out/"sampling_manifest.json").write_text(json.dumps(sampling,indent=2),encoding="utf-8")
    schema={"version":HUMAN_VALIDATION_VERSION,"query_labels":{"core_intent_preserved":["YES","NO","UNCLEAR"],"single_requirement_change":["YES","NO","UNCLEAR"],"changed_requirement_correct":["YES","NO","UNCLEAR"],"requirement_family":["COLOR","BRAND","NUMERIC_SPECIFICATION","EXPLICIT_ATTRIBUTE","NEGATION_EXCLUSION","QUANTITY_PACK_SIZE","OTHER","UNCLEAR"],"operation":["REPLACE","ADD","REMOVE","NEGATE","UNCLEAR"],"query_naturalness":["YES","NO","UNCLEAR"],"semantic_plausibility":["YES","NO","UNCLEAR"],"confidence":["HIGH","MEDIUM","LOW"],"old_value_normalized":"free text; exact-match agreement","new_value_normalized":"free text; exact-match agreement"},"direction_labels":{"candidate_direction":["A_SHOULD_GAIN","B_SHOULD_GAIN","NO_REQUIRED_DIRECTION","INSUFFICIENT_EVIDENCE"],"direction_confidence":["HIGH","MEDIUM","LOW"]},"adjudication_reasons":ADJUDICATION_REASONS}; (out/"annotation_schema.json").write_text(json.dumps(schema,indent=2),encoding="utf-8")
    files=[p.name for p in out.iterdir() if p.is_file() and p.name not in {"human_validation_manifest.json","README.md"}]
    manifest={"version":HUMAN_VALIDATION_VERSION,"created_at_utc":datetime.now(timezone.utc).isoformat(),"source":{"partitions":["policy_train","calibration"],"validation_used":False,"test_used":False,"esci_labels_used":False,"ranking_outputs_used":False},"counts":{"sample":len(sample),"calibration":len(calibration),"final":len(final),"source_type":dict(Counter(sample.source_type)),"family":dict(Counter(sample.requirement_family)),"operation":dict(Counter(sample.operation)),"final_direction_items":len(final_directions)},"candidate_pool":{"maximum_size":MAX_POOL_SIZE,"mean_size":sum(x["candidate_count"] for x in pools)/len(pools)},"ground_truth_status":"NOT_HUMAN_GROUND_TRUTH","libraries":{"python":platform.python_version(),"pandas":pd.__version__},"hashes":{name:sha256_file(out/name) for name in sorted(files)}}
    (out/"human_validation_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8"); return manifest


def analyze(query_a:Path,query_b:Path,direction_a:Path,direction_b:Path,out:Path)->None:
    out.mkdir(parents=True,exist_ok=True); before={p:sha256_file(p) for p in (query_a,query_b,direction_a,direction_b)}
    qs,qp,_,qa=analyze_independent_annotations(pd.read_csv(query_a,keep_default_na=False),pd.read_csv(query_b,keep_default_na=False)); ds,dp,da=analyze_direction_annotations(pd.read_csv(direction_a,keep_default_na=False),pd.read_csv(direction_b,keep_default_na=False))
    write_csv(out/"agreement_summary.csv",qs); write_csv(out/"field_prevalence.csv",pd.concat([qp,dp],ignore_index=True)); write_csv(out/"direction_agreement.csv",ds); write_csv(out/"adjudication_template.csv",pd.concat([qa,da],ignore_index=True))
    if before!={p:sha256_file(p) for p in before}: raise AssertionError("Raw annotation files changed")


def accept(sample_path:Path,query_adjudication:Path,direction_adjudication:Path,output_path:Path)->None:
    sample=pd.read_parquet(sample_path); query=pd.read_csv(query_adjudication,keep_default_na=False); direction=pd.read_csv(direction_adjudication,keep_default_na=False)
    accepted_benchmark_pairs(sample,query,direction).to_parquet(output_path,index=False)


def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="command",required=True); b=sub.add_parser("build"); b.add_argument("--dataset-path",type=Path,default=Path("esci-data/shopping_queries_dataset")); b.add_argument("--generated-dir",type=Path,default=Path("data/cf_esci/feasibility_v1")); b.add_argument("--natural-dir",type=Path,default=Path("data/cf_esci/natural_pairs_v1")); b.add_argument("--output-dir",type=Path,default=Path("data/cf_esci/human_validation_v1")); a=sub.add_parser("analyze"); a.add_argument("--query-a",type=Path,required=True); a.add_argument("--query-b",type=Path,required=True); a.add_argument("--direction-a",type=Path,required=True); a.add_argument("--direction-b",type=Path,required=True); a.add_argument("--output-dir",type=Path,required=True); c=sub.add_parser("accept"); c.add_argument("--sample",type=Path,required=True); c.add_argument("--query-adjudication",type=Path,required=True); c.add_argument("--direction-adjudication",type=Path,required=True); c.add_argument("--output",type=Path,required=True); args=p.parse_args()
    if args.command=="build": print(json.dumps(build(args.dataset_path,args.generated_dir,args.natural_dir,args.output_dir),indent=2))
    elif args.command=="analyze": analyze(args.query_a,args.query_b,args.direction_a,args.direction_b,args.output_dir)
    else: accept(args.sample,args.query_adjudication,args.direction_adjudication,args.output)

if __name__=="__main__": main()
