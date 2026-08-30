from __future__ import annotations
import argparse, json, time
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from src.paper_eval.dataset import build_dataset_manifest, development_query_split, policy_calibration_split, load_examples, load_products_for_candidates, preserve_selected_queries, select_query_ids, task_examples, write_json
from src.paper_eval.adapter import diagnostics_json
from src.paper_eval.fingerprint import configuration_fingerprint, verify_resume_fingerprint
from src.paper_eval.methods import METHOD_VERSION, REGISTRY, enabled_methods
from src.paper_eval.metrics import OFFICIAL_GAINS, beneficial_intervention_rate, cortex_ndcg_at_k, cortex_reward, exact_at_k, exact_or_substitute_at_k, fallback_rate, harmful_intervention_rate, intervention_rate, irrelevant_at_k, label_coverage_at_k, neutral_intervention_rate, official_esci_ndcg, preserve_rate, reciprocal_rank_at_k, reward_delta_vs_baseline, violation_rate
from src.paper_eval.route_family import ROUTE_FAMILY_VERSION, ORACLE_REQUIRED_ROUTES
from src.paper_eval.schema import QueryResult
from src.paper_eval.statistics import bootstrap_mean_ci, paired_bootstrap_difference
from src.paper_eval.clean_router import ROUTER_V2_VERSION as CLEAN_ROUTER_VERSION, ROUTE_FAMILY_V2_VERSION as CLEAN_ROUTE_FAMILY_VERSION
from src.paper_eval.contracts_v1 import PARSER_VERSION
from src.paper_eval.contract_rerank_v1 import DEFAULT_CONFIG as CONTRACT_RERANK_CONFIG, RERANKER_VERSION

DEFAULT_METHODS="fts_baseline,strict_filter,always_contract_rerank_v1,clean_contract_router_v2"
METRICS=["official_esci_ndcg","ndcg_10","mrr_10","exact_1","exact_or_substitute_10","total_latency_ms"]

def parser():
    p=argparse.ArgumentParser(description="Canonical CORTEX ESCI paper evaluation")
    p.add_argument("--dataset-path",type=Path,default=Path("esci-data/shopping_queries_dataset")); p.add_argument("--evaluation-mode",choices=["official_rerank","open_corpus"],default="official_rerank")
    p.add_argument("--task-version",choices=["small","large"],default="small"); p.add_argument("--split",choices=["train","test"],default="test"); p.add_argument("--locale",choices=["us","es","jp"],default=None)
    p.add_argument("--development-partition",choices=["none","train_fit","policy_train","calibration","validation"],default="none"); p.add_argument("--train-fraction",type=float,default=.8)
    p.add_argument("--policy-fraction",type=float,default=.8); p.add_argument("--selected-query-ids-path",type=Path,default=None)
    p.add_argument("--max-queries",type=int,default=100); p.add_argument("--seed",type=int,default=29); p.add_argument("--top-k",type=int,default=10); p.add_argument("--retrieval-backend",default="candidate_lexical")
    p.add_argument("--methods",default=DEFAULT_METHODS); p.add_argument("--output-dir",type=Path,default=Path("outputs/paper_eval")); p.add_argument("--run-id",default=None); p.add_argument("--resume",action="store_true"); p.add_argument("--workers",type=int,default=1)
    p.add_argument("--allow-unsafe-contaminated-state",action="store_true",help="Debug only; prohibited for paper claims")
    p.add_argument("--allow-frozen-test",action="store_true",help="Explicit final-evaluation authorization; never use during development")
    return p

def make_result(run_id, method, group, rows, meta, total_ms):
    labels=[str(row.get("esci_label","")).upper() for row in rows]; ids=[str(row.get("product_id","")) for row in rows]; scores=[float(row.get("score",0.0)) for row in rows]
    counts=group.esci_label.astype(str).str.upper().value_counts().to_dict(); hard=meta.get("strict_hard_violation_count"); soft=meta.get("strict_soft_violation_count")
    result=QueryResult(run_id,method,METHOD_VERSION,int(group["query_id"].iloc[0]),str(group["query"].iloc[0]),str(group["product_locale"].iloc[0]),str(group["split"].iloc[0]),candidate_count=len(group),judged_candidate_count=len(group),unjudged_candidate_count=0,E_count=int(counts.get("E",0)),S_count=int(counts.get("S",0)),C_count=int(counts.get("C",0)),I_count=int(counts.get("I",0)),final_product_ids=ids,final_labels=labels,final_scores=scores)
    qrel_labels=group["esci_label"].astype(str).str.upper().tolist()
    result.official_esci_ndcg=official_esci_ndcg(labels,qrel_labels); result.ndcg_5=cortex_ndcg_at_k(labels,5,qrel_labels); result.ndcg_10=cortex_ndcg_at_k(labels,10,qrel_labels); result.mrr_10=reciprocal_rank_at_k(labels,10)
    result.exact_1=exact_at_k(labels,1); result.exact_5=exact_at_k(labels,5); result.exact_10=exact_at_k(labels,10); result.exact_or_substitute_5=exact_or_substitute_at_k(labels,5); result.exact_or_substitute_10=exact_or_substitute_at_k(labels,10); result.irrelevant_5=irrelevant_at_k(labels,5); result.irrelevant_10=irrelevant_at_k(labels,10); result.label_coverage_5=label_coverage_at_k(labels,5); result.label_coverage_10=label_coverage_at_k(labels,10)
    result.route=str(meta.get("route")) if meta.get("route") is not None else None; result.intervened=meta.get("intervened"); result.preserved=meta.get("preserved"); result.fallback=meta.get("fallback"); result.method_reward=cortex_reward(labels)
    result.route_selected=meta.get("route_selected",result.route); result.ranking_changed=meta.get("ranking_changed",result.intervened); result.explicit_preserve=meta.get("explicit_preserve",result.preserved)
    result.intervention_attempted=meta.get("intervention_attempted",False); result.intervention_effective=meta.get("intervention_effective",result.ranking_changed)
    for field_name in ("contract_status","detected_constraint_count","negative_constraint_count","positive_constraint_count","hard_constraint_count","soft_constraint_count","contract_ambiguity","contract_parser_route","contract_parser_reason","hard_compatibility_changes","soft_preference_matches","must_have_matches","must_not_have_violations","brand_matches","price_signal_matches","product_type_matches","contract_score_range","rerank_changed","mean_candidate_movement","top_1_changed"):
        setattr(result,field_name,meta.get(field_name))
    result.gate_decision=meta.get("gate_decision"); result.gate_reason=meta.get("gate_reason"); result.diagnostics_json=diagnostics_json(meta)
    if hard is None: hard=meta.get("hard_violation_penalty_count")
    if soft is None: soft=meta.get("soft_violation_penalty_count")
    result.hard_violation_count=int(hard) if hard is not None else None; result.soft_violation_count=int(soft) if soft is not None else None; result.any_violation_count=(result.hard_violation_count+result.soft_violation_count) if hard is not None and soft is not None else None
    result.clean_count=int(meta["strict_clean_count"]) if "strict_clean_count" in meta else None; result.removed_count=int(meta["strict_removed_count"]) if "strict_removed_count" in meta else None; result.demoted_count=int(meta["strict_demoted_count"]) if "strict_demoted_count" in meta else None
    result.retrieval_latency_ms=float(meta.get("retrieval_latency_ms",0)); result.ranking_latency_ms=float(meta.get("ranking_latency_ms",0)); result.total_latency_ms=total_ms; result.route_regret_unavailable_reason="complete_route_family_not_evaluated"
    return result

def failure_result(run_id,method,group,stage,exc,total_ms):
    return QueryResult(run_id,method,METHOD_VERSION,int(group["query_id"].iloc[0]),str(group["query"].iloc[0]),str(group["product_locale"].iloc[0]),str(group["split"].iloc[0]),candidate_count=len(group),judged_candidate_count=len(group),success=False,failure_stage=stage,error_type=type(exc).__name__,error_message=str(exc)[:1000],total_latency_ms=total_ms,route_regret_unavailable_reason="method_failed")

def aggregate(run_dir:Path, seed:int):
    data=pd.read_parquet(run_dir/"query_level_results.parquet"); success=data[data.success==True].copy()
    relevance=success.groupby("method")[["official_esci_ndcg","mrr_10","exact_1","exact_5","exact_10","exact_or_substitute_5","exact_or_substitute_10","irrelevant_5","irrelevant_10"]].mean(numeric_only=True).reset_index(); relevance.to_csv(run_dir/"main_relevance_table.csv",index=False)
    success.groupby("method")[["ndcg_5","ndcg_10"]].mean(numeric_only=True).rename(columns={"ndcg_5":"research_diagnostic_only_ndcg_5","ndcg_10":"research_diagnostic_only_ndcg_10"}).reset_index().to_csv(run_dir/"research_diagnostic_metrics.csv",index=False)
    success.groupby(["method","route"],dropna=False).agg(query_count=("query_id","count"),mean_reward=("method_reward","mean"),mean_reward_delta=("reward_delta","mean"),mean_regret=("route_regret","mean")).reset_index().to_csv(run_dir/"route_performance_table.csv",index=False)
    intervention_rows=[]; constraint_rows=[]
    for method,group in success.groupby("method"):
        intervention_rows.append({"method":method,"query_count":len(group),"intervention_attempted_rate":intervention_rate(group.intervention_attempted),"intervention_effective_rate":intervention_rate(group.intervention_effective),"explicit_preserve_rate":preserve_rate(group.explicit_preserve),"beneficial_effective_intervention_rate":beneficial_intervention_rate(group.intervention_effective,group.reward_delta),"harmful_effective_intervention_rate":harmful_intervention_rate(group.intervention_effective,group.reward_delta),"neutral_effective_intervention_rate":neutral_intervention_rate(group.intervention_effective,group.reward_delta),"fallback_rate":fallback_rate(group.fallback),"mean_reward_delta":group.reward_delta.mean()})
        exposed=group[group.hard_violation_count.notna() & group.soft_violation_count.notna()]
        denominator=float(exposed.candidate_count.sum()) if len(exposed) else None
        constraint_rows.append({"method":method,"query_count":len(group),"queries_with_constraint_diagnostics":len(exposed),"hard_violation_rate":violation_rate(float(exposed.hard_violation_count.sum()),denominator) if len(exposed) else None,"soft_violation_rate":violation_rate(float(exposed.soft_violation_count.sum()),denominator) if len(exposed) else None,"any_violation_rate":violation_rate(float(exposed.any_violation_count.sum()),denominator) if len(exposed) else None,"clean_result_count":exposed.clean_count.mean() if len(exposed) else None,"removed_result_count":exposed.removed_count.mean() if len(exposed) else None,"demoted_result_count":exposed.demoted_count.mean() if len(exposed) else None})
    pd.DataFrame(intervention_rows).to_csv(run_dir/"intervention_table.csv",index=False); pd.DataFrame(constraint_rows).to_csv(run_dir/"constraint_safety_table.csv",index=False)
    contract_rows=success[success.contract_status.notna()]
    contract_rows.groupby(["method","contract_status"],dropna=False).size().reset_index(name="query_count").to_csv(run_dir/"contract_status_table.csv",index=False)
    rerank_rows=[]
    for (method,route),group in success.groupby(["method","route_selected"],dropna=False):
        rerank_rows.append({"method":method,"route_selected":route,"queries":len(group),"ranking_changed":int(group.ranking_changed.fillna(False).sum()),
                            "unchanged":int((~group.ranking_changed.fillna(False)).sum()),"effective_intervention_rate":group.intervention_effective.mean(),
                            "top_1_changed_rate":group.top_1_changed.mean(),"mean_candidate_movement":group.mean_candidate_movement.mean()})
    pd.DataFrame(rerank_rows).to_csv(run_dir/"contract_rerank_activity_table.csv",index=False)
    success.groupby("method").agg(successful_queries=("query_id","count"),retrieval_latency_ms=("retrieval_latency_ms","mean"),ranking_latency_ms=("ranking_latency_ms","mean"),total_latency_ms=("total_latency_ms","mean")).reset_index().to_csv(run_dir/"runtime_table.csv",index=False)
    ci=[]
    for method,group in success.groupby("method"):
        for metric in METRICS:
            values=group[metric].dropna().tolist(); stats=bootstrap_mean_ci(values,seed=seed,iterations=500); ci.append({"method":method,"metric":metric,**stats})
    pd.DataFrame(ci).to_csv(run_dir/"bootstrap_confidence_intervals.csv",index=False)
    paired=[]
    base=success[success.method=="fts_baseline"][["query_id"]+METRICS]
    for method in sorted(set(success.method)-{"fts_baseline"}):
        other=success[success.method==method][["query_id"]+METRICS]; merged=other.merge(base,on="query_id",suffixes=("_method","_baseline"))
        for metric in METRICS: paired.append({"method":method,"baseline":"fts_baseline","metric":metric,**paired_bootstrap_difference(merged[f"{metric}_method"],merged[f"{metric}_baseline"],seed=seed,iterations=500)})
    pd.DataFrame(paired).to_csv(run_dir/"paired_comparisons.csv",index=False)
    official_changes=[]
    base_official=success[success.method=="fts_baseline"][["query_id","official_esci_ndcg"]].rename(columns={"official_esci_ndcg":"baseline_official_esci_ndcg"})
    for method in sorted(set(success.method)-{"fts_baseline"}):
        merged=success[(success.method==method)&(success.intervention_effective==True)].merge(base_official,on="query_id",how="inner")
        delta=merged.official_esci_ndcg-merged.baseline_official_esci_ndcg
        official_changes.append({"method":method,"effective_queries":len(merged),"beneficial":int((delta>1e-12).sum()),"harmful":int((delta< -1e-12).sum()),"neutral":int((delta.abs()<=1e-12).sum())})
    pd.DataFrame(official_changes).to_csv(run_dir/"official_intervention_analysis.csv",index=False)
    comparable=success[success.method!="fts_baseline"].copy(); comparable.sort_values("reward_delta",ascending=False,na_position="last").head(25).to_csv(run_dir/"improvement_cases.csv",index=False); comparable.sort_values("reward_delta",ascending=True,na_position="last").head(25).to_csv(run_dir/"negative_result_table.csv",index=False); data[data.success==False].to_csv(run_dir/"failure_cases.csv",index=False)
    summary=[]
    for method,group in data.groupby("method"):
        ok=group[group.success==True]; summary.append({"method":method,"selected":len(group),"successful":len(ok),"failed":int((group.success==False).sum()),"official_esci_ndcg":ok.official_esci_ndcg.mean(),"ndcg_10":ok.ndcg_10.mean(),"mrr_10":ok.mrr_10.mean(),"exact_1":ok.exact_1.mean(),"exact_or_substitute_10":ok.exact_or_substitute_10.mean(),"runtime_per_query_ms":ok.total_latency_ms.mean()})
    columns=list(summary[0]) if summary else []
    markdown=""
    if columns:
        markdown="| "+" | ".join(columns)+" |\n| "+" | ".join("---" for _ in columns)+" |\n"
        markdown+="\n".join("| "+" | ".join("NA" if row[column] is None or pd.isna(row[column]) else str(row[column]) for column in columns)+" |" for row in summary)+"\n"
    (run_dir/"full_esci_evaluation_report.md").write_text("# Full ESCI evaluation report\n\nSmoke diagnostics only; not paper claims.\n\n"+markdown,encoding="utf-8")
    return summary

def run(args):
    if args.evaluation_mode=="open_corpus": raise NotImplementedError("open_corpus is scaffolded only in MVP 29.1; an explicit unjudged-item policy is required before execution")
    if args.split=="test" and not args.allow_frozen_test: raise ValueError("ESCI test is frozen; pass --allow-frozen-test only for an authorized final evaluation")
    if args.development_partition!="none" and args.split!="train": raise ValueError("development partitions require --split train")
    if args.development_partition=="none" and args.split=="test": print("WARNING: test is frozen after MVP 29.1; use --split train --development-partition validation for development")
    if args.workers!=1: raise ValueError("MVP 29.1 supports --workers 1 only until deterministic parallel parity is verified")
    run_id=args.run_id or datetime.now(timezone.utc).strftime("mvp29_1_smoke_%Y%m%dT%H%M%SZ"); run_dir=args.output_dir/run_id; run_dir.mkdir(parents=True,exist_ok=args.resume)
    examples=load_examples(args.dataset_path); selected_pool=task_examples(examples,task_version=args.task_version,split=args.split,locale=args.locale)
    development=None
    if args.development_partition!="none":
        train_fit_ids,validation_ids=development_query_split(selected_pool,seed=args.seed,train_fraction=args.train_fraction); policy_train_ids,calibration_ids=policy_calibration_split(train_fit_ids,seed=args.seed,policy_fraction=args.policy_fraction)
        overlaps={"policy_train_calibration":len(set(policy_train_ids)&set(calibration_ids)),"policy_train_validation":len(set(policy_train_ids)&set(validation_ids)),"calibration_validation":len(set(calibration_ids)&set(validation_ids))}
        if any(overlaps.values()): raise AssertionError(f"development partition overlap: {overlaps}")
        development={"seed":args.seed,"train_fraction":args.train_fraction,"policy_fraction":args.policy_fraction,"train_fit_query_count":len(train_fit_ids),"policy_train_query_count":len(policy_train_ids),"calibration_query_count":len(calibration_ids),"validation_query_count":len(validation_ids),"overlap_counts":overlaps}
        (run_dir/"train_fit_query_ids.txt").write_text("\n".join(map(str,train_fit_ids))+"\n",encoding="utf-8"); (run_dir/"validation_query_ids.txt").write_text("\n".join(map(str,validation_ids))+"\n",encoding="utf-8")
        (run_dir/"policy_train_query_ids.txt").write_text("\n".join(map(str,policy_train_ids))+"\n",encoding="utf-8"); (run_dir/"calibration_query_ids.txt").write_text("\n".join(map(str,calibration_ids))+"\n",encoding="utf-8")
        partition_ids={"train_fit":train_fit_ids,"policy_train":policy_train_ids,"calibration":calibration_ids,"validation":validation_ids}[args.development_partition]; selected_pool=preserve_selected_queries(selected_pool,partition_ids)
    if args.selected_query_ids_path:
        ids=[int(line.strip()) for line in args.selected_query_ids_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        missing=sorted(set(ids)-set(selected_pool.query_id.astype(int)))
        if missing: raise ValueError(f"Selected query IDs are outside requested partition: {missing[:10]}")
        if args.max_queries is not None and len(ids)!=args.max_queries: raise ValueError("Frozen selected-query file count differs from --max-queries")
    else: ids=select_query_ids(selected_pool,max_queries=args.max_queries,seed=args.seed)
    selected=preserve_selected_queries(selected_pool,ids)
    specs,disabled=enabled_methods([v.strip() for v in args.methods.split(",") if v.strip()],allow_unsafe_contaminated_state=args.allow_unsafe_contaminated_state); products=load_products_for_candidates(args.dataset_path,set(selected.product_id.astype(str))); selected=selected.merge(products,on=["product_locale","product_id"],how="left",validate="many_to_one")
    manifest=build_dataset_manifest(args.dataset_path,examples,development); write_json(run_dir/"dataset_manifest.json",manifest)
    fingerprint_payload={"dataset_checksums":{k:v["sha256"] for k,v in manifest["files"].items()},"query_ids":ids,"split":args.split,"development_partition":args.development_partition,"seed":args.seed,"methods":{s.name:METHOD_VERSION for s in specs},"top_k":args.top_k,"evaluation_mode":args.evaluation_mode,"official_gains":OFFICIAL_GAINS,"route_family_version":CLEAN_ROUTE_FAMILY_VERSION,"contract_parser_version":PARSER_VERSION,"clean_router_version":CLEAN_ROUTER_VERSION,"contract_reranker_version":RERANKER_VERSION,"contract_rerank_config":CONTRACT_RERANK_CONFIG.to_dict(),"allow_unsafe_contaminated_state":args.allow_unsafe_contaminated_state}
    fingerprint=configuration_fingerprint(fingerprint_payload)
    config_path=run_dir/"evaluation_config.json"
    if args.resume and config_path.exists(): verify_resume_fingerprint(json.loads(config_path.read_text(encoding="utf-8")),fingerprint)
    serializable_args={key:(str(value) if isinstance(value,Path) else value) for key,value in vars(args).items()}
    config={**serializable_args,"dataset_path":str(args.dataset_path.resolve()),"output_dir":str(args.output_dir.resolve()),"run_id":run_id,"selected_query_count":len(ids),"enabled_methods":[s.name for s in specs],"disabled_requested_methods":disabled,"method_registry":{name:{"enabled":spec.enabled,"reason":spec.reason,"version":METHOD_VERSION,"contaminated":spec.contaminated} for name,spec in REGISTRY.items()},"configuration_fingerprint":fingerprint,"fingerprint_payload":fingerprint_payload,"route_family_version":CLEAN_ROUTE_FAMILY_VERSION,"legacy_route_family_version":ROUTE_FAMILY_VERSION,"oracle_required_routes":ORACLE_REQUIRED_ROUTES,"contract_parser_version":PARSER_VERSION,"clean_router_version":CLEAN_ROUTER_VERSION,"created_at_utc":datetime.now(timezone.utc).isoformat(),"official_protocol":{"small_version":1,"source_split":args.split,"development_partition":args.development_partition,"candidate_source":"judged candidates only"}}
    write_json(config_path,config); (run_dir/"selected_query_ids.txt").write_text("\n".join(map(str,ids))+"\n",encoding="utf-8")
    results=[]
    result_path=run_dir/"query_level_results.parquet"
    if args.resume and result_path.exists(): results=pd.read_parquet(result_path).to_dict("records")
    completed={(int(row["query_id"]),str(row["method"])) for row in results}
    for _,group in selected.groupby("query_id",sort=True):
        query_id=int(group["query_id"].iloc[0]); baseline_reward=next((row.get("method_reward") for row in results if int(row["query_id"])==query_id and row["method"]=="fts_baseline"),None); query_results=[]
        for spec in specs:
            if (query_id,spec.name) in completed: continue
            start=time.perf_counter()
            try:
                ranker_candidates=group.drop(columns=["esci_label"],errors="ignore").copy()
                rows,meta=spec.evaluator(str(group["query"].iloc[0]),ranker_candidates,len(group))
                qrels={str(pid):str(label).upper() for pid,label in zip(group["product_id"],group["esci_label"])}
                for row in rows: row["esci_label"]=qrels.get(str(row.get("product_id")),"")
                result=make_result(run_id,spec.name,group,rows,meta,(time.perf_counter()-start)*1000)
                if spec.name=="fts_baseline": baseline_reward=result.method_reward
            except Exception as exc: result=failure_result(run_id,spec.name,group,"method_evaluation",exc,(time.perf_counter()-start)*1000)
            query_results.append(result)
        if baseline_reward is None:
            baseline=next((r.method_reward for r in query_results if r.method=="fts_baseline"),None); baseline_reward=baseline
        for result in query_results:
            result.baseline_reward=baseline_reward; result.reward_delta=reward_delta_vs_baseline(result.method_reward,baseline_reward); results.append(result.to_dict())
        if query_results:
            checkpoint=run_dir/"query_level_results.checkpoint.parquet"; pd.DataFrame(results).to_parquet(checkpoint,index=False); checkpoint.replace(result_path)
    if not result_path.exists(): pd.DataFrame(results).to_parquet(result_path,index=False)
    return run_dir,aggregate(run_dir,args.seed),disabled

def main():
    args=parser().parse_args(); run_dir,summary,disabled=run(args); print(json.dumps({"run_dir":str(run_dir),"summary":summary,"disabled_methods":disabled},indent=2,default=str))
if __name__=="__main__": main()
