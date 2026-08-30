from __future__ import annotations

import argparse
import json
import platform
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path

import pandas as pd

from src.paper_eval.cf_esci import GROUND_TRUTH_STATUS
from src.paper_eval.contract_esci import SELECTION_SEED,assert_no_esci_test_queries,sha256_file
from src.paper_eval.contracts_v1 import build_search_contract_v1
from src.paper_eval.dataset import development_query_split,load_examples,load_products_for_candidates,policy_calibration_split,task_examples
from src.paper_eval.natural_pairs import EVIDENCE_STATES,FAMILIES,NATURAL_PAIR_VERSION,VALIDATION_STATUS,build_candidate_evidence,filter_valid_pairs,mine_raw_pairs


def write_jsonl(path:Path,records:list[dict])->None:path.write_text("".join(json.dumps(row,ensure_ascii=False,sort_keys=True)+"\n" for row in records),encoding="utf-8")


def source_queries(examples:pd.DataFrame)->tuple[pd.DataFrame,pd.DataFrame]:
    task=task_examples(examples,task_version="small",split="train"); train_fit,validation=development_query_split(task,seed=SELECTION_SEED,train_fraction=.8); policy,calibration=policy_calibration_split(train_fit,seed=SELECTION_SEED,policy_fraction=.8); partition={qid:"policy_train" for qid in policy}|{qid:"calibration" for qid in calibration}
    queries=task[task.query_id.isin(partition)].drop_duplicates("query_id")[["query_id","query"]].rename(columns={"query":"query_text"}); queries["source_partition"]=queries.query_id.map(partition); queries["product_type"]=queries.query_text.map(lambda query:build_search_contract_v1(str(query)).product_type); assert_no_esci_test_queries(queries.query_id.astype(int).tolist(),examples); return queries,task


def annotation_record(pair:dict,directions:list[dict],evidence_by_product:dict[str,dict])->dict:
    return {"canonical_pair_id":pair["canonical_pair_id"],"query_a":pair["query_text_a"],"query_b":pair["query_text_b"],"proposed_invariant_product_type":pair["product_type_a"],"proposed_requirement_family":pair["requirement_family"],"proposed_changed_attribute":pair["changed_attribute"],"proposed_operation":pair["operation"],"old_value":pair["old_value"],"new_value":pair["new_value"],"same_core_shopping_intent":None,"exactly_one_requirement_differs":None,"requirement_type_correct":None,"operation_correct":None,"meaningful_natural_shopping_contrast":None,"candidate_direction_reviews":[{"p_b":row["p_b"],"p_a":row["p_a"],"p_b_catalog_evidence":evidence_by_product.get(row["p_b"],{}).get("evidence_text"),"p_a_catalog_evidence":evidence_by_product.get(row["p_a"],{}).get("evidence_text"),"expected_direction":None} for row in directions],"confidence":None,"notes":None,"annotator_id":None,"annotation_status":"unannotated","allowed_answers":{"validity":["YES","NO","UNCLEAR"],"direction":["A_PRODUCT_SHOULD_GAIN","B_PRODUCT_SHOULD_GAIN","NO_REQUIRED_DIRECTION","INSUFFICIENT_EVIDENCE"],"confidence":["HIGH","MEDIUM","LOW"]},"ground_truth_status":GROUND_TRUTH_STATUS,"validation_status":VALIDATION_STATUS}


def decision(stats:dict)->str:
    diverse=sum(value>0 for value in stats["counts_by_family"].values()); operations=sum(value>0 for value in stats["counts_by_operation"].values())
    if stats["valid_pair_count"]>=500 and diverse>=4 and operations>=3 and stats["direction_pair_query_coverage"]>=.5:return "STRONG_GO"
    if stats["valid_pair_count"]>=100 and diverse>=3 and stats["direction_pair_query_coverage"]>=.25:return "GO"
    if stats["valid_pair_count"]>=25:return "CAUTION"
    return "NO_GO_FOR_NATURAL_SUBSET"


def write_report(path:Path,manifest:dict,strong:list[dict],rejected:list[dict])->None:
    s=manifest["statistics"]; generated=manifest["generated_comparison"]
    examples="\n".join(f"- `{row['query_text_a']}` → `{row['query_text_b']}` ({row['requirement_family']}, {row['operation']}: {row['old_value']} → {row['new_value']})" for row in strong[:8]) or "- None"
    failures="\n".join(f"- `{row.get('query_text_a')}` / `{row.get('query_text_b')}`: {row.get('reason')}" for row in rejected[:8]) or "- None"
    path.write_text(f"""# CF-ESCI Natural Minimal-Pair Feasibility

## Scope

The miner searched all {s['total_queries_searched']:,} policy-train and calibration queries. It used exact normalized lexical skeletons, typed values, and frozen product-type agreement. It did not use validation, test, ESCI labels, ranker scores, CORTEX outputs, routes, oracle data, or prior performance. Every surviving pair remains `NEEDS_HUMAN_REVIEW` and `NOT_HUMAN_GROUND_TRUTH`.

## Results

- Raw candidates: {s['raw_candidate_pair_count']:,}
- Conservative valid pairs: {s['valid_pair_count']:,}
- Families: {json.dumps(s['counts_by_family'],sort_keys=True)}
- Operations: {json.dumps(s['counts_by_operation'],sort_keys=True)}
- Pairs with catalog-grounded direction evidence: {s['pairs_with_direction_evidence']:,} ({s['direction_pair_query_coverage']:.2%})
- Direction instances: {s['direction_instance_count']:,}
- Pairs with locality sets: {s['pairs_with_locality_set']:,} ({s['locality_set_coverage']:.2%})
- Mean locality-set size: {s['mean_locality_set_size']:.3f}
- UNKNOWN evidence rate: {s['unknown_evidence_rate']:.2%}
- Candidate-pool overlap mean/median: {s['candidate_pool_overlap_rate_mean']:.2%} / {s['candidate_pool_overlap_rate_median']:.2%}

## Candidate-pool protocol options

Shared candidates are the cleanest source of paired behavior but coverage is limited and overlap varies. The union pool is feasible with available metadata and is appropriate for evidence construction, but future ranking evaluation must score every union candidate under both queries using one frozen retrieval/reranking protocol. A fixed externally constructed pool offers the strongest comparability and control but adds annotation and sampling cost. The feasibility evidence supports retaining union-pool construction now while predeclaring shared-pool sensitivity analysis and externally fixed pools as the preferred confirmatory design; sample count alone must not decide.

## Representative natural pairs

{examples}

## Representative automatic rejections

{failures}

## Natural versus generated MVP 29.8

Generated: {generated.get('proposal_count')} proposals, families {json.dumps(generated.get('counts_by_requirement_family',{}),sort_keys=True)}, operations {json.dumps(generated.get('counts_by_operation',{}),sort_keys=True)}, direction coverage {generated.get('direction_pair_query_coverage')}, locality coverage {generated.get('unaffected_set_query_coverage')}, UNKNOWN {generated.get('unknown_evidence_rate')}.

Natural: {s['valid_pair_count']} pairs, families {json.dumps(s['counts_by_family'],sort_keys=True)}, operations {json.dumps(s['counts_by_operation'],sort_keys=True)}, direction coverage {s['direction_pair_query_coverage']:.4f}, locality coverage {s['locality_set_coverage']:.4f}, UNKNOWN {s['unknown_evidence_rate']:.4f}.

Natural queries improve linguistic provenance but are not automatically semantically cleaner; all pairs require blinded review. Generated pairs provide controlled coverage when natural matches are sparse. Neither construction is declared superior.

## Decision

**{manifest['decision']}** under the predeclared thresholds. A weak natural subset does not invalidate CF-ESCI; it means generated, human-validated counterfactuals remain necessary.
""",encoding="utf-8")


def run(dataset_path:Path,out:Path,generated_manifest_path:Path,report_path:Path)->dict:
    out.mkdir(parents=True,exist_ok=True); examples=load_examples(dataset_path); queries,task=source_queries(examples); raw,duplicate_failures=mine_raw_pairs(queries); valid,invalid=filter_valid_pairs(raw)
    valid_ids={qid for row in valid for qid in (row["query_id_a"],row["query_id_b"])}; task_pairs=task[task.query_id.isin(valid_ids)]; products=load_products_for_candidates(dataset_path,set(task_pairs.product_id.astype(str))); catalog=task_pairs.drop(columns=["esci_label"],errors="ignore").merge(products,on=["product_locale","product_id"],how="left",validate="many_to_one")
    evidence=[]; directions=[]; locality=[]; summaries=[]; annotations=[]
    for pair in valid:
        ids_a=set(task_pairs.loc[task_pairs.query_id==pair["query_id_a"],"product_id"].astype(str)); ids_b=set(task_pairs.loc[task_pairs.query_id==pair["query_id_b"],"product_id"].astype(str)); union=ids_a|ids_b; overlap=ids_a&ids_b; rows=catalog[catalog.product_id.astype(str).isin(union)]; pair_evidence,pair_directions,unaffected=build_candidate_evidence(pair,rows); evidence.extend(pair_evidence); directions.extend(pair_directions); locality.append({"canonical_pair_id":pair["canonical_pair_id"],"product_ids":unaffected,"set_size":len(unaffected),"ground_truth_status":GROUND_TRUTH_STATUS,"validation_status":VALIDATION_STATUS})
        evidence_by={row["product_id"]:row for row in pair_evidence}; annotations.append(annotation_record(pair,pair_directions,evidence_by)); summaries.append({**pair,"candidate_pool_overlap_count":len(overlap),"candidate_pool_union_count":len(union),"candidate_pool_overlap_rate":len(overlap)/len(union) if union else 0.0,"direction_pair_count":len(pair_directions),"locality_set_size":len(unaffected)})
    pair_ids={row["canonical_pair_id"] for row in summaries}; failures=duplicate_failures+invalid
    source_ids=sorted(queries.query_id.astype(int).tolist()); (out/"source_query_ids.txt").write_text("\n".join(map(str,source_ids))+"\n",encoding="utf-8"); pd.DataFrame(summaries).to_parquet(out/"natural_query_pairs.parquet",index=False); write_jsonl(out/"natural_pair_proposals.jsonl",valid); write_jsonl(out/"candidate_direction_evidence.jsonl",evidence); write_jsonl(out/"candidate_direction_pairs.jsonl",directions); write_jsonl(out/"unaffected_candidate_sets.jsonl",locality); write_jsonl(out/"human_natural_pair_review_template.jsonl",annotations); write_jsonl(out/"natural_pair_failure_cases.jsonl",failures)
    family_counts=Counter(row["requirement_family"] for row in summaries); operation_counts=Counter(row["operation"] for row in summaries); unknown=sum(row["directional_state"]=="UNKNOWN" for row in evidence); overlap_rates=pd.Series([row["candidate_pool_overlap_rate"] for row in summaries],dtype=float); locality_sizes=[row["set_size"] for row in locality]
    stats={"total_queries_searched":len(queries),"raw_candidate_pair_count":len(raw)+len(duplicate_failures),"valid_pair_count":len(summaries),"counts_by_family":{family:family_counts.get(family,0) for family in FAMILIES},"counts_by_operation":{op:operation_counts.get(op,0) for op in ("REPLACE","ADD","REMOVE","NEGATE")},"pairs_with_direction_evidence":sum(row["direction_pair_count"]>0 for row in summaries),"direction_pair_query_coverage":sum(row["direction_pair_count"]>0 for row in summaries)/len(summaries) if summaries else 0.0,"direction_instance_count":len(directions),"pairs_with_locality_set":sum(size>0 for size in locality_sizes),"locality_set_coverage":sum(size>0 for size in locality_sizes)/len(summaries) if summaries else 0.0,"mean_locality_set_size":sum(locality_sizes)/len(locality_sizes) if locality_sizes else 0.0,"unknown_evidence_rate":unknown/len(evidence) if evidence else None,"candidate_pool_overlap_count_mean":sum(row["candidate_pool_overlap_count"] for row in summaries)/len(summaries) if summaries else 0.0,"candidate_pool_union_count_mean":sum(row["candidate_pool_union_count"] for row in summaries)/len(summaries) if summaries else 0.0,"candidate_pool_overlap_rate_mean":float(overlap_rates.mean()) if len(overlap_rates) else 0.0,"candidate_pool_overlap_rate_median":float(overlap_rates.median()) if len(overlap_rates) else 0.0,"failure_counts":dict(sorted(Counter(row.get("reason","UNKNOWN") for row in failures).items()))}
    generated=json.loads(generated_manifest_path.read_text(encoding="utf-8"))["statistics"] if generated_manifest_path.exists() else {}; artifact_names=["source_query_ids.txt","natural_query_pairs.parquet","natural_pair_proposals.jsonl","candidate_direction_evidence.jsonl","candidate_direction_pairs.jsonl","unaffected_candidate_sets.jsonl","human_natural_pair_review_template.jsonl","natural_pair_failure_cases.jsonl","README.md"]
    manifest={"version":NATURAL_PAIR_VERSION,"created_at_utc":datetime.now(timezone.utc).isoformat(),"source":{"task":"ESCI Task 1 small","split":"train only","partitions":["policy_train","calibration"],"validation_used":False,"test_used":False},"construction":{"seed":SELECTION_SEED,"fuzzy_similarity_used":False,"esci_labels_used":False,"ranker_outputs_used":False,"canonical_reciprocal_deduplication":True},"statistics":stats,"generated_comparison":generated,"decision":None,"hashes":{},"libraries":{"python":platform.python_version(),"pandas":pd.__version__}}; manifest["decision"]=decision(stats); write_report(report_path,manifest,sorted(summaries,key=lambda row:(-row["direction_pair_count"],-row["locality_set_size"])),failures); manifest["hashes"]={name:sha256_file(out/name) for name in artifact_names}; manifest["report_sha256"]=sha256_file(report_path); (out/"natural_pair_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8"); return manifest


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--dataset-path",type=Path,default=Path("esci-data/shopping_queries_dataset")); parser.add_argument("--output-dir",type=Path,default=Path("data/cf_esci/natural_pairs_v1")); parser.add_argument("--generated-manifest",type=Path,default=Path("data/cf_esci/feasibility_v1/feasibility_manifest.json")); parser.add_argument("--report",type=Path,default=Path("docs/cf_esci_natural_pair_feasibility.md")); args=parser.parse_args(); print(json.dumps(run(args.dataset_path,args.output_dir,args.generated_manifest,args.report),indent=2))

if __name__=="__main__":main()
