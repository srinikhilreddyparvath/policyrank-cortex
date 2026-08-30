from __future__ import annotations

import argparse
import json
import platform
from collections import Counter
from datetime import datetime,timezone
from pathlib import Path

import pandas as pd

from src.paper_eval.cf_esci import ADDITIONAL_QUOTAS, CF_ESCI_VERSION, FAMILIES, GROUND_TRUTH_STATUS, REJECTION_REASONS, family_hint, propose_counterfactual, select_additional_sources
from src.paper_eval.contract_esci import SELECTION_SEED, assert_no_esci_test_queries, sha256_file
from src.paper_eval.contracts_v1 import build_search_contract_v1
from src.paper_eval.dataset import development_query_split,load_examples,load_products_for_candidates,policy_calibration_split,task_examples


def write_jsonl(path:Path,records:list[dict])->None:
    path.write_text("".join(json.dumps(record,ensure_ascii=False,sort_keys=True)+"\n" for record in records),encoding="utf-8")


def build_sources(examples:pd.DataFrame,pilot_path:Path)->pd.DataFrame:
    task=task_examples(examples,task_version="small",split="train"); train_fit,validation=development_query_split(task,seed=SELECTION_SEED,train_fraction=.8); policy,calibration=policy_calibration_split(train_fit,seed=SELECTION_SEED,policy_fraction=.8)
    partitions={qid:"policy_train" for qid in policy}|{qid:"calibration" for qid in calibration}
    allowed=task[task.query_id.isin(partitions)].drop_duplicates("query_id")[["query_id","query"]].rename(columns={"query":"query_text"}); allowed["source_partition"]=allowed.query_id.map(partitions)
    pilot=pd.read_parquet(pilot_path)[["query_id","query_text","source_partition"]].copy(); pilot["source_provenance"]="contract_esci_pilot_v1"
    additional=select_additional_sources(allowed,set(pilot.query_id)); additional["source_provenance"]="cf_esci_additional_feasibility_pool"
    result=pd.concat([pilot,additional],ignore_index=True).sort_values("query_id").reset_index(drop=True); assert_no_esci_test_queries(result.query_id.tolist(),examples)
    if len(result)!=500 or result.query_id.duplicated().any():raise AssertionError("Expected 500 unique source queries")
    return result


def annotation_template(proposal:dict,pairs:list[dict])->dict:
    return {"counterfactual_id":proposal["counterfactual_id"],"source_query_id":proposal["source_query_id"],"source_query_text":proposal["source_query_text"],"counterfactual_query_text":proposal["counterfactual_query_text"],"changed_requirement_id":proposal["changed_requirement_id"],"core_product_intent_preserved":None,"exactly_one_requirement_changed":None,"changed_requirement_correctly_identified":None,"linguistically_plausible":None,"replacement_value_semantically_plausible":None,"pair_direction_annotations":[{"p_new":pair["p_new"],"p_old":pair["p_old"],"p_new_catalog_evidence":pair["p_new_evidence"],"p_old_catalog_evidence":pair["p_old_evidence"],"expected_direction":None} for pair in pairs],"confidence":None,"annotation_notes":None,"annotator_id":None,"annotation_status":"unannotated","allowed_answers":{"validity":["YES","NO","UNCLEAR"],"direction":["NEW_PRODUCT_SHOULD_GAIN","OLD_PRODUCT_SHOULD_GAIN","NO_REQUIRED_DIRECTION","INSUFFICIENT_EVIDENCE"],"confidence":["HIGH","MEDIUM","LOW"]},"ground_truth_status":GROUND_TRUTH_STATUS}


def exact_substitute_feasibility(task:pd.DataFrame,evidence:list[dict],proposal_ids:set[str])->dict:
    # Auxiliary only: ESCI labels are consulted after construction and never enter proposal/evidence/direction generation.
    evidence_frame=pd.DataFrame(evidence); query_states={}
    if len(evidence_frame):
        for qid,group in evidence_frame.groupby("source_query_id"):query_states[int(qid)]=set(group.directional_state)-{"UNKNOWN"}
    counts=Counter()
    for qid,group in task.groupby("query_id"):
        labels=set(group.esci_label.astype(str).str.upper())
        if not {"E","S"}.issubset(labels):continue
        if int(qid) not in query_states:counts["NO_EXPLICIT_REQUIREMENT_FOUND"]+=1
        elif {"SUPPORTS_NEW","SUPPORTS_OLD"}.issubset(query_states[int(qid)]):counts["ATTRIBUTABLE"]+=1
        elif query_states[int(qid)]:counts["AMBIGUOUS"]+=1
        else:counts["INSUFFICIENT_METADATA"]+=1
    classes=("ATTRIBUTABLE","AMBIGUOUS","INSUFFICIENT_METADATA","NO_EXPLICIT_REQUIREMENT_FOUND")
    return {"scope":"feasibility classification only; not human ground truth","classes":{name:counts.get(name,0) for name in classes},"labels_used_only_after_counterfactual_construction":True}


def run(dataset_path:Path,out:Path,contract_pilot:Path)->dict:
    out.mkdir(parents=True,exist_ok=True); examples=load_examples(dataset_path); sources=build_sources(examples,contract_pilot/"pilot_queries.parquet"); ids=sources.query_id.astype(int).tolist(); task=task_examples(examples,task_version="small",split="train"); selected=task[task.query_id.isin(ids)].copy()
    products=load_products_for_candidates(dataset_path,set(selected.product_id.astype(str))); catalog=selected.drop(columns=["esci_label"],errors="ignore").merge(products,on=["product_locale","product_id"],how="left",validate="many_to_one")
    proposals=[]; evidence=[]; pairs=[]; unaffected=[]; controls=[]
    for source in sources.itertuples(index=False):
        rows=catalog[catalog.query_id==source.query_id]; contract=build_search_contract_v1(str(source.query_text)); bundle=propose_counterfactual(int(source.query_id),str(source.query_text),rows,contract.product_type)
        if bundle:
            proposals.append(bundle.proposal); evidence.extend(bundle.evidence); pairs.extend(bundle.pairs); unaffected.append({"counterfactual_id":bundle.proposal["counterfactual_id"],"source_query_id":int(source.query_id),"product_ids":bundle.unaffected,"set_size":len(bundle.unaffected),"definition":"Candidates with explicit same-family evidence that contradicts both old and new values; their changed-requirement relationship is unchanged.","ground_truth_status":GROUND_TRUTH_STATUS})
        else:
            hint=family_hint(str(source.query_text)); reason="NO_ATOMIC_REQUIREMENT" if hint is None else "INSUFFICIENT_CANDIDATE_EVIDENCE" if hint=="NEGATION_EXCLUSION" else "NO_GROUNDED_REPLACEMENT"
            controls.append({"source_query_id":int(source.query_id),"source_query_text":str(source.query_text),"family_audited":hint,"rejection_reason":reason,"counterfactual_status":"REJECTED","ground_truth_status":GROUND_TRUTH_STATUS})
    pairs_by={};
    for pair in pairs:pairs_by.setdefault(pair["counterfactual_id"],[]).append(pair)
    annotations=[annotation_template(proposal,pairs_by.get(proposal["counterfactual_id"],[])) for proposal in proposals]
    ids_path=out/"source_query_ids.txt"; ids_path.write_text("\n".join(map(str,ids))+"\n",encoding="utf-8"); sources.to_parquet(out/"source_queries.parquet",index=False)
    write_jsonl(out/"counterfactual_proposals.jsonl",proposals); write_jsonl(out/"candidate_requirement_evidence.jsonl",evidence); write_jsonl(out/"candidate_direction_pairs.jsonl",pairs); write_jsonl(out/"unaffected_candidate_sets.jsonl",unaffected); write_jsonl(out/"no_edit_controls.jsonl",controls); write_jsonl(out/"human_counterfactual_annotation_template.jsonl",annotations)
    family_counts=Counter(p["changed_requirement_type"] for p in proposals); operation_counts=Counter(p["operation"] for p in proposals); rejection_counts=Counter(c["rejection_reason"] for c in controls); unknown=sum(row["directional_state"]=="UNKNOWN" for row in evidence)
    eligible_queries=len({p["source_query_id"] for p in proposals}); paired_queries=len({p["source_query_id"] for p in pairs}); unaffected_nonempty=sum(row["set_size"]>0 for row in unaffected)
    source_hints={int(row.query_id):family_hint(str(row.query_text)) for row in sources.itertuples(index=False)}; eligible_ids={p["source_query_id"] for p in proposals}
    stats={"total_source_queries_audited":len(sources),"queries_with_proposal":eligible_queries,"queries_without_proposal":len(controls),"eligibility_rate":eligible_queries/len(sources),"proposal_count":len(proposals),"proposals_per_source_query":len(proposals)/len(sources),"family_audit":{family:{"source_queries":sum(hint==family for hint in source_hints.values()),"eligible_queries":sum(qid in eligible_ids and hint==family for qid,hint in source_hints.items())} for family in FAMILIES},"counts_by_requirement_family":{family:family_counts.get(family,0) for family in FAMILIES},"counts_by_operation":{operation:operation_counts.get(operation,0) for operation in ("REPLACE","ADD","REMOVE","NEGATE")},"direction_pair_count":len(pairs),"direction_pair_query_coverage":paired_queries/eligible_queries if eligible_queries else 0.0,"mean_direction_pairs_per_eligible_query":len(pairs)/eligible_queries if eligible_queries else 0.0,"unaffected_set_query_coverage":unaffected_nonempty/eligible_queries if eligible_queries else 0.0,"mean_unaffected_set_size":sum(row["set_size"] for row in unaffected)/eligible_queries if eligible_queries else 0.0,"unknown_evidence_rate":unknown/len(evidence) if evidence else None,"rejection_counts":{reason:rejection_counts.get(reason,0) for reason in sorted(REJECTION_REASONS)},"suitable_for_human_validation":sum(bool(pairs_by.get(p["counterfactual_id"])) for p in proposals)}
    auxiliary=exact_substitute_feasibility(selected,evidence,{p["counterfactual_id"] for p in proposals})
    metric_spec=json.loads((out/"metric_specification.json").read_text(encoding="utf-8"))
    artifact_names=["source_query_ids.txt","source_queries.parquet","counterfactual_proposals.jsonl","candidate_requirement_evidence.jsonl","candidate_direction_pairs.jsonl","unaffected_candidate_sets.jsonl","no_edit_controls.jsonl","human_counterfactual_annotation_template.jsonl","metric_specification.json","README.md"]
    manifest={"version":CF_ESCI_VERSION,"created_at_utc":datetime.now(timezone.utc).isoformat(),"source":{"task":"ESCI Task 1 small","split":"train only","allowed_partitions":["policy_train","calibration"],"validation_used":False,"test_used":False,"contract_esci_pilot_queries":250,"additional_deterministic_queries":sum(ADDITIONAL_QUOTAS.values())},"selection":{"seed":SELECTION_SEED,"additional_family_quotas":ADDITIONAL_QUOTAS,"outcome_based_selection":False},"generation":{"one_proposal_maximum_per_query":True,"ground_truth_status":GROUND_TRUTH_STATUS,"price_counterfactuals_enabled":False,"esci_labels_used_for_proposals":False,"ranker_or_method_outputs_used":False},"statistics":stats,"exact_substitute_auxiliary":auxiliary,"hashes":{name:sha256_file(out/name) for name in artifact_names},"libraries":{"python":platform.python_version(),"pandas":pd.__version__}}
    (out/"feasibility_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8"); return manifest


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--dataset-path",type=Path,default=Path("esci-data/shopping_queries_dataset")); parser.add_argument("--output-dir",type=Path,default=Path("data/cf_esci/feasibility_v1")); parser.add_argument("--contract-pilot",type=Path,default=Path("data/contract_esci/pilot_v1")); args=parser.parse_args(); print(json.dumps(run(args.dataset_path,args.output_dir,args.contract_pilot),indent=2))

if __name__=="__main__":main()
