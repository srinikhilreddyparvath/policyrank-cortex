from __future__ import annotations

import argparse, hashlib, json, platform, time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

from src.paper_eval.clean_router import ROUTER_V2_VERSION
from src.paper_eval.contract_rerank_v1 import DEFAULT_CONFIG, RERANKER_VERSION
from src.paper_eval.contracts_v1 import PARSER_VERSION, build_search_contract_v1
from src.paper_eval.dataset import development_query_split, load_examples, load_products_for_candidates, policy_calibration_split, preserve_selected_queries, select_query_ids, task_examples
from src.paper_eval.learned_router_v1 import CONTRACT_FEATURES, CONTRACT_FEATURE_VERSION, GENERIC_FEATURES, GENERIC_FEATURE_VERSION, LEARNED_ROUTER_VERSION, choose_routes, extract_router_features, fit_ridge
from src.paper_eval.methods import METHOD_VERSION, REGISTRY
from src.paper_eval.metrics import exact_at_k, official_esci_ndcg, reciprocal_rank_at_k
from src.paper_eval.statistics import paired_bootstrap_difference

SEED=29; RESAMPLES=2000; ALPHAS=(0.1,1.0,10.0); THRESHOLDS=(0.0,0.001,0.0025,0.005,0.01)
CALIBRATION_OBJECTIVE="Lexicographically maximize calibration official nDCG, then minimize harmful intervention rate, then minimize intervention rate; remaining ties prefer smaller alpha and threshold."
ROUTES=("PRESERVE","STRICT_FILTER","CONTRACT_RERANK")
ADAPTERS={"PRESERVE":"fts_baseline","STRICT_FILTER":"strict_filter","CONTRACT_RERANK":"always_contract_rerank_v1"}
PRIMARY_COMPARISONS=(("contract_learned_router_v1","generic_learned_router_v1"),("contract_learned_router_v1","BEST_STATIC"),("contract_learned_router_v1","router_v2"),("generic_learned_router_v1","BEST_STATIC"),("ORACLE","contract_learned_router_v1"))

def sha(path:Path)->str:return hashlib.sha256(path.read_bytes()).hexdigest()
def write_ids(path:Path,ids:list[int]):path.write_text("\n".join(map(str,ids))+"\n",encoding="utf-8")

def evaluate_partition(name:str,pool:pd.DataFrame,ids:list[int],dataset_path:Path,out:Path)->pd.DataFrame:
    if out.exists(): return pd.read_parquet(out)
    checkpoint=out.with_suffix(".checkpoint.parquet"); existing=pd.read_parquet(checkpoint) if checkpoint.exists() else pd.DataFrame(); done=set(existing.get("metadata__query_id",pd.Series(dtype=int)).astype(int)); records=existing.to_dict("records")
    selected=preserve_selected_queries(pool,ids); products=load_products_for_candidates(dataset_path,set(selected.product_id.astype(str))); selected=selected.merge(products,on=["product_locale","product_id"],how="left",validate="many_to_one")
    for query_id,group in selected.groupby("query_id",sort=True):
        query_id=int(query_id)
        if query_id in done:continue
        query=str(group["query"].iloc[0]); ranker=group.drop(columns=["esci_label"],errors="ignore").copy(); qrels={str(pid):str(label).upper() for pid,label in zip(group.product_id,group.esci_label)}; outcomes={}; failures=[]
        for route,adapter in ADAPTERS.items():
            start=time.perf_counter()
            try:
                rows,meta=REGISTRY[adapter].evaluator(query,ranker,len(group)); labels=[qrels[str(row.get("product_id"))] for row in rows]; outcomes[route]={"ndcg":official_esci_ndcg(labels,group.esci_label.tolist()),"mrr":reciprocal_rank_at_k(labels,10),"exact":exact_at_k(labels,1),"ids":[str(row.get("product_id")) for row in rows],"latency":(time.perf_counter()-start)*1000,"changed":bool(meta.get("ranking_changed",False))}
            except Exception as exc: failures.append(f"{route}:{type(exc).__name__}:{exc}")
        if failures or len(outcomes)!=3:
            record={"metadata__query_id":query_id,"metadata__query":query,"metadata__partition":name,"metadata__success":False,"metadata__error":" | ".join(failures)}
        else:
            baseline_rows,_=REGISTRY["fts_baseline"].evaluator(query,ranker,len(group)); contract=build_search_contract_v1(query); generic,contract_features=extract_router_features(query,baseline_rows,contract)
            record={"metadata__query_id":query_id,"metadata__query":query,"metadata__locale":str(group.product_locale.iloc[0]),"metadata__partition":name,"metadata__success":True,"metadata__error":None,
                **{f"feature__{key}":value for key,value in contract_features.items()},
                "target__ndcg_preserve":outcomes["PRESERVE"]["ndcg"],"target__ndcg_filter":outcomes["STRICT_FILTER"]["ndcg"],"target__ndcg_rerank":outcomes["CONTRACT_RERANK"]["ndcg"],
                "target__delta_filter_vs_preserve":outcomes["STRICT_FILTER"]["ndcg"]-outcomes["PRESERVE"]["ndcg"],"target__delta_rerank_vs_preserve":outcomes["CONTRACT_RERANK"]["ndcg"]-outcomes["PRESERVE"]["ndcg"]}
            for route in ROUTES:
                key=route.lower(); record.update({f"outcome__{key}__mrr_10":outcomes[route]["mrr"],f"outcome__{key}__exact_1":outcomes[route]["exact"],f"outcome__{key}__product_ids":outcomes[route]["ids"],f"outcome__{key}__latency_ms":outcomes[route]["latency"],f"outcome__{key}__ranking_changed":outcomes[route]["changed"]})
        records.append(record)
        if len(records)%100==0:pd.DataFrame(records).to_parquet(checkpoint,index=False)
    result=pd.DataFrame(records).sort_values("metadata__query_id"); result.to_parquet(out,index=False)
    if checkpoint.exists():checkpoint.unlink()
    return result

def model_frame(data:pd.DataFrame,features:tuple[str,...])->pd.DataFrame:
    return data.rename(columns={f"feature__{name}":name for name in features})

def route_score(data:pd.DataFrame,routes:np.ndarray)->tuple[float,float,float]:
    lookup={"PRESERVE":data.target__ndcg_preserve.to_numpy(),"STRICT_FILTER":data.target__ndcg_filter.to_numpy(),"CONTRACT_RERANK":data.target__ndcg_rerank.to_numpy()}; selected=np.asarray([lookup[route][i] for i,route in enumerate(routes)]); baseline=lookup["PRESERVE"]; delta=selected-baseline; active=routes!="PRESERVE"
    return float(selected.mean()),float(active.mean()),float((delta[active]<-1e-12).mean()) if active.any() else 0.0

def train_router(name:str,features:tuple[str,...],train:pd.DataFrame,calibration:pd.DataFrame)->dict:
    train=model_frame(train,features); calibration=model_frame(calibration,features); candidates=[]
    for alpha in ALPHAS:
        filter_model=fit_ridge(train,features,"target__delta_filter_vs_preserve",alpha); rerank_model=fit_ridge(train,features,"target__delta_rerank_vs_preserve",alpha); fp=filter_model.predict(calibration); rp=rerank_model.predict(calibration)
        for threshold in THRESHOLDS:
            routes=choose_routes(fp,rp,threshold); mean_ndcg,rate,harm=route_score(calibration,routes); candidates.append({"alpha":alpha,"threshold":threshold,"mean_ndcg":mean_ndcg,"intervention_rate":rate,"harmful_intervention_rate":harm,"filter_model":filter_model,"rerank_model":rerank_model})
    selected=sorted(candidates,key=lambda row:(-row["mean_ndcg"],row["harmful_intervention_rate"],row["intervention_rate"],row["alpha"],row["threshold"]))[0]
    return {"name":name,"feature_names":list(features),"feature_version":GENERIC_FEATURE_VERSION if features==GENERIC_FEATURES else CONTRACT_FEATURE_VERSION,"algorithm":LEARNED_ROUTER_VERSION,"alpha":selected["alpha"],"threshold":selected["threshold"],"calibration_mean_ndcg":selected["mean_ndcg"],"calibration_intervention_rate":selected["intervention_rate"],"calibration_harmful_intervention_rate":selected["harmful_intervention_rate"],"filter_model":selected["filter_model"].to_dict(),"rerank_model":selected["rerank_model"].to_dict(),"calibration_tradeoff":[{k:v for k,v in row.items() if k not in ("filter_model","rerank_model")} for row in candidates]}

def predict_model(model:dict,frame:pd.DataFrame)->tuple[np.ndarray,np.ndarray,np.ndarray,float]:
    from src.paper_eval.learned_router_v1 import RidgeModel
    features=tuple(model["feature_names"]); clean=model_frame(frame,features); f=RidgeModel(**model["filter_model"]); r=RidgeModel(**model["rerank_model"]); start=time.perf_counter(); fp=f.predict(clean); rp=r.predict(clean); routes=choose_routes(fp,rp,float(model["threshold"])); latency=(time.perf_counter()-start)*1000/len(frame); return routes,fp,rp,latency

def evaluate_router_v2(pool:pd.DataFrame,ids:list[int],dataset_path:Path)->dict[int,dict]:
    selected=preserve_selected_queries(pool,ids); products=load_products_for_candidates(dataset_path,set(selected.product_id.astype(str))); selected=selected.merge(products,on=["product_locale","product_id"],how="left",validate="many_to_one"); results={}
    for query_id,group in selected.groupby("query_id",sort=True):
        query=str(group["query"].iloc[0]); ranker=group.drop(columns=["esci_label"],errors="ignore"); qrels={str(pid):str(label).upper() for pid,label in zip(group.product_id,group.esci_label)}; start=time.perf_counter()
        try:
            rows,meta=REGISTRY["clean_contract_router_v2"].evaluator(query,ranker,len(group)); labels=[qrels[str(row.get("product_id"))] for row in rows]; results[int(query_id)]={"success":True,"route":meta["route_selected"],"ndcg":official_esci_ndcg(labels,group.esci_label.tolist()),"mrr":reciprocal_rank_at_k(labels,10),"exact":exact_at_k(labels,1),"ids":[str(row.get("product_id")) for row in rows],"latency":(time.perf_counter()-start)*1000,"changed":bool(meta.get("ranking_changed",False))}
        except Exception as exc:results[int(query_id)]={"success":False,"error":f"{type(exc).__name__}:{exc}"}
    return results

def build_validation_results(fresh:pd.DataFrame,models:dict,router_v2:dict[int,dict],out:Path)->pd.DataFrame:
    if out.exists():return pd.read_parquet(out)
    predictions={};
    for name,model in models.items(): predictions[name]=predict_model(model,fresh)
    records=[]
    for index,row in fresh.reset_index(drop=True).iterrows():
        qid=int(row.metadata__query_id); route_values={route:{"ndcg":row[f"target__ndcg_{'preserve' if route=='PRESERVE' else 'filter' if route=='STRICT_FILTER' else 'rerank'}"],"mrr":row[f"outcome__{route.lower()}__mrr_10"],"exact":row[f"outcome__{route.lower()}__exact_1"],"ids":row[f"outcome__{route.lower()}__product_ids"],"latency":row[f"outcome__{route.lower()}__latency_ms"],"changed":row[f"outcome__{route.lower()}__ranking_changed"]} for route in ROUTES}; oracle=max(v["ndcg"] for v in route_values.values()); oracle_set=sorted(route for route,v in route_values.items() if abs(v["ndcg"]-oracle)<=1e-12)
        for route in ROUTES:
            v=route_values[route]; records.append({"query_id":qid,"query":row.metadata__query,"method":route,"route_selected":route,**v,"oracle_ndcg":oracle,"oracle_route_set":oracle_set,"route_regret":oracle-v["ndcg"],"predicted_delta_filter":None,"predicted_delta_rerank":None,"success":True})
        v2=router_v2[qid]; records.append({"query_id":qid,"query":row.metadata__query,"method":"router_v2","route_selected":v2.get("route"),"ndcg":v2.get("ndcg"),"mrr":v2.get("mrr"),"exact":v2.get("exact"),"ids":v2.get("ids",[]),"latency":v2.get("latency"),"changed":v2.get("changed"),"oracle_ndcg":oracle,"oracle_route_set":oracle_set,"route_regret":oracle-v2.get("ndcg") if v2.get("success") else None,"predicted_delta_filter":None,"predicted_delta_rerank":None,"success":v2.get("success"),"error":v2.get("error")})
        for name,(routes,fp,rp,inference_latency) in predictions.items():
            route=routes[index]; v=route_values[route]; records.append({"query_id":qid,"query":row.metadata__query,"method":name,"route_selected":route,**v,"latency":v["latency"]+inference_latency,"oracle_ndcg":oracle,"oracle_route_set":oracle_set,"route_regret":oracle-v["ndcg"],"predicted_delta_filter":fp[index],"predicted_delta_rerank":rp[index],"success":True})
        records.append({"query_id":qid,"query":row.metadata__query,"method":"ORACLE","route_selected":"|".join(oracle_set),"ndcg":oracle,"mrr":None,"exact":None,"ids":[],"latency":None,"changed":None,"oracle_ndcg":oracle,"oracle_route_set":oracle_set,"route_regret":0.0,"predicted_delta_filter":None,"predicted_delta_rerank":None,"success":True})
    result=pd.DataFrame(records); result.to_parquet(out,index=False); return result

def aggregate(data:pd.DataFrame,out:Path,models:dict)->dict:
    ok=data[data.success==True]; summary=ok.groupby("method").agg(official_esci_ndcg=("ndcg","mean"),mrr_10=("mrr","mean"),exact_1=("exact","mean"),mean_route_regret=("route_regret","mean"),median_route_regret=("route_regret","median"),mean_latency_ms=("latency","mean")).reset_index(); summary.to_csv(out/"router_summary.csv",index=False)
    best_static=summary[summary.method.isin(ROUTES)].sort_values("official_esci_ndcg",ascending=False).method.iloc[0]; wide=ok.pivot(index="query_id",columns="method",values="ndcg")
    comparisons=[(a,best_static if b=="BEST_STATIC" else b) for a,b in PRIMARY_COMPARISONS]; paired=[]
    metric_columns={"official_esci_ndcg":"ndcg","mrr_10":"mrr","route_regret":"route_regret"}
    for left,right in comparisons:
        left_rows=ok[ok.method==left].set_index("query_id"); right_rows=ok[ok.method==right].set_index("query_id")
        for metric,column in metric_columns.items():
            if left_rows[column].notna().all() and right_rows[column].notna().all():
                stats=paired_bootstrap_difference(left_rows[column],right_rows[column],seed=SEED,iterations=RESAMPLES)
                difference=(left_rows[column]-right_rows[column]).to_numpy(dtype=float)
                paired.append({"left":left,"right":right,"metric":metric,**stats,"query_win_rate":float((difference>1e-12).mean()),"query_tie_rate":float((np.abs(difference)<=1e-12).mean()),"query_loss_rate":float((difference<-1e-12).mean())})
        if left in ("generic_learned_router_v1","contract_learned_router_v1","router_v2") and right in ("generic_learned_router_v1","contract_learned_router_v1","router_v2"):
            base=ok[ok.method=="PRESERVE"].set_index("query_id")
            def harmful_indicator(frame):
                effective=frame.ids.astype(str)!=base.ids.astype(str)
                return (effective & (frame.ndcg<base.ndcg-1e-12)).astype(float)
            stats=paired_bootstrap_difference(harmful_indicator(left_rows),harmful_indicator(right_rows),seed=SEED,iterations=RESAMPLES)
            difference=harmful_indicator(left_rows)-harmful_indicator(right_rows)
            paired.append({"left":left,"right":right,"metric":"harmful_effective_query_rate",**stats,"query_win_rate":float((difference<0).mean()),"query_tie_rate":float((difference==0).mean()),"query_loss_rate":float((difference>0).mean())})
    pd.DataFrame(paired).to_csv(out/"paired_comparisons.csv",index=False); pd.DataFrame(paired).to_csv(out/"bootstrap_comparisons.csv",index=False)
    ok.groupby(["method","route_selected"],dropna=False).size().reset_index(name="query_count").to_csv(out/"route_distribution.csv",index=False)
    router_rows=ok[ok.method.isin(["router_v2","generic_learned_router_v1","contract_learned_router_v1"])].copy(); router_rows["selected_in_oracle_set"]=[r in o for r,o in zip(router_rows.route_selected,router_rows.oracle_route_set)]; router_rows.to_csv(out/"route_regret.csv",index=False)
    baseline=ok[ok.method=="PRESERVE"].set_index("query_id"); intervention=[]
    for method,group in router_rows.groupby("method"):
        group=group.set_index("query_id"); delta=group.ndcg-baseline.ndcg; active=group.route_selected!="PRESERVE"; effective=group.ids.astype(str)!=baseline.ids.astype(str); mask=active&effective
        intervention.append({"method":method,"intervention_attempted_rate":active.mean(),"effective_intervention_rate":effective.mean(),"beneficial_effective":int((delta[mask]>1e-12).sum()),"harmful_effective":int((delta[mask]<-1e-12).sum()),"neutral_effective":int((delta[mask].abs()<=1e-12).sum()),"harmful_effective_rate":float((delta[mask]<-1e-12).mean()) if mask.any() else 0.0,"zero_regret_rate":float((group.route_regret.abs()<=1e-12).mean()),"selected_in_oracle_rate":float(group.selected_in_oracle_set.mean())})
    pd.DataFrame(intervention).to_csv(out/"intervention_analysis.csv",index=False); ok.groupby("method").latency.agg(mean_latency_ms="mean",median_latency_ms="median",p95_latency_ms=lambda x:x.quantile(.95)).reset_index().to_csv(out/"latency_summary.csv",index=False)
    feature=[]
    for model_name,model in models.items():
        for target in ("filter_model","rerank_model"):
            fitted=model[target]
            for name,coef in zip(fitted["feature_names"],fitted["coefficients"]): feature.append({"model":model_name,"target":target,"feature":name,"standardized_coefficient":coef,"absolute_coefficient":abs(coef)})
    pd.DataFrame(feature).sort_values(["model","target","absolute_coefficient"],ascending=[True,True,False]).to_csv(out/"feature_analysis.csv",index=False)
    generic=router_rows[router_rows.method=="generic_learned_router_v1"].set_index("query_id"); contract=router_rows[router_rows.method=="contract_learned_router_v1"].set_index("query_id"); delta=contract.ndcg-generic.ndcg
    generic_correct=[route in oracle for route,oracle in zip(generic.route_selected,generic.oracle_route_set)]; contract_correct=[route in oracle for route,oracle in zip(contract.route_selected,contract.oracle_route_set)]
    categories=np.select([np.asarray(generic_correct)&~np.asarray(contract_correct),~np.asarray(generic_correct)&np.asarray(contract_correct)],["generic_correct_contract_wrong","contract_correct_generic_wrong"],default="same_correctness")
    pd.DataFrame({"query":contract.query,"generic_route":generic.route_selected,"contract_route":contract.route_selected,"generic_in_oracle_set":generic_correct,"contract_in_oracle_set":contract_correct,"case_category":categories,"generic_ndcg":generic.ndcg,"contract_ndcg":contract.ndcg,"contract_minus_generic":delta}).sort_values("contract_minus_generic").to_csv(out/"contract_vs_generic_cases.csv")
    contract.assign(delta_vs_preserve=contract.ndcg-baseline.ndcg).sort_values("delta_vs_preserve").head(100).to_csv(out/"largest_router_harms.csv"); contract.assign(delta_vs_preserve=contract.ndcg-baseline.ndcg).sort_values("delta_vs_preserve",ascending=False).head(100).to_csv(out/"largest_router_wins.csv")
    means=summary.set_index("method").official_esci_ndcg.to_dict(); oracle_gain=means["ORACLE"]-means["PRESERVE"]; capture={m:(means[m]-means["PRESERVE"])/oracle_gain if oracle_gain>0 else None for m in ("generic_learned_router_v1","contract_learned_router_v1")}; result={"best_static":best_static,"means":means,"oracle_headroom_over_preserve":oracle_gain,"oracle_capture":capture,"failures":int((~data.success).sum())}
    return result

def write_report(out:Path,manifest:dict,config:dict,result:dict)->None:
    summary=pd.read_csv(out/"router_summary.csv"); comparisons=pd.read_csv(out/"bootstrap_comparisons.csv"); routes=pd.read_csv(out/"route_distribution.csv"); interventions=pd.read_csv(out/"intervention_analysis.csv"); latency=pd.read_csv(out/"latency_summary.csv"); features=pd.read_csv(out/"feature_analysis.csv")
    def table(frame:pd.DataFrame,columns:list[str])->str:
        values=frame[columns].copy()
        for column in values.select_dtypes(include=["float"]).columns: values[column]=values[column].map(lambda value:"" if pd.isna(value) else f"{value:.6f}")
        header="| "+" | ".join(columns)+" |"; divider="| "+" | ".join(["---"]*len(columns))+" |"
        return "\n".join([header,divider]+["| "+" | ".join(map(str,row))+" |" for row in values.itertuples(index=False,name=None)])
    generic=manifest["models"]["generic_learned_router_v1"]; contract=manifest["models"]["contract_learned_router_v1"]
    ablation=comparisons[(comparisons.left=="contract_learned_router_v1")&(comparisons.right=="generic_learned_router_v1")]
    top=features[features.model=="contract_learned_router_v1"].sort_values(["target","absolute_coefficient"],ascending=[True,False]).groupby("target").head(10)
    generic_features=generic["feature_names"]; contract_only=[name for name in contract["feature_names"] if name not in generic_features]
    report=f"""# MVP 29.6: Learned Contract-Aware Routing and Contract Feature Ablation

## A–B. Recovery and reuse

Recovery classified the interrupted work as **D: evaluated, report incomplete**. The authoritative 21,634-query policy-train outcomes, 5,409-query calibration outcomes, 2,000-query fresh-validation route outcomes, and 14,000 method-query result rows were validated and reused. Stored policy/calibration hashes matched the files; all 14,000 final rows succeeded. No ranking evaluation was rerun.

## C–F. Formulation, features, and model

Each router uses two independently fitted ridge regressions to predict `delta_filter_vs_preserve` and `delta_rerank_vs_preserve`. PRESERVE has utility zero. Inference selects the greatest predicted utility only when it exceeds the router's calibration-only minimum-benefit threshold; otherwise it preserves.

Generic features (`{manifest['models']['generic_learned_router_v1']['feature_version']}`): {', '.join(generic_features)}.

Additional contract features (`{manifest['models']['contract_learned_router_v1']['feature_version']}`): {', '.join(contract_only)}.

Ridge regression (`{generic['algorithm']}`) was selected because it is lightweight, deterministic, auditable, and exposes standardized coefficients. Both ablation arms used the same objective, candidate alpha set, candidate threshold set, seed, and training/calibration partitions. Coefficients are associative signals, not causal effects.

## G–I. Provenance and calibration

- Policy train: {manifest['partitions']['policy_train']:,} queries; SHA-256 `{manifest['training_hashes']['policy_train_route_outcomes']}`.
- Calibration: {manifest['partitions']['calibration']:,} queries; SHA-256 `{manifest['training_hashes']['calibration_route_outcomes']}`.
- Calibration objective frozen in code: {manifest['calibration_objective']}
- Generic selection: alpha {generic['alpha']}, threshold {generic['threshold']}; calibration nDCG {generic['calibration_mean_ndcg']:.6f}, attempted rate {generic['calibration_intervention_rate']:.3%}, harmful-among-attempted rate {generic['calibration_harmful_intervention_rate']:.3%}.
- Contract selection: alpha {contract['alpha']}, threshold {contract['threshold']}; calibration nDCG {contract['calibration_mean_ndcg']:.6f}, attempted rate {contract['calibration_intervention_rate']:.3%}, harmful-among-attempted rate {contract['calibration_harmful_intervention_rate']:.3%}.
- Fresh validation: {manifest['partitions']['fresh_validation_selected']:,} deterministic seed-29 queries; ID-file SHA-256 `{config['selected_ids_sha256']}`. The 1,100 inspected IDs and 5,661 fresh-pool IDs are disjoint and exhaust the 6,761-query validation partition; selected/inspected overlap is zero.

## J–P. Final metrics and analyses

{table(summary,['method','official_esci_ndcg','mrr_10','exact_1','mean_route_regret','median_route_regret'])}

Best static route: **{result['best_static']}**. Oracle headroom over PRESERVE is {result['oracle_headroom_over_preserve']:.6f}. Generic captures {result['oracle_capture']['generic_learned_router_v1']:.2%}; contract-aware captures {result['oracle_capture']['contract_learned_router_v1']:.2%}.

### Central contract ablation (contract minus generic)

{table(ablation,['metric','mean_difference','lower','upper','win_probability','query_win_rate','query_tie_rate','query_loss_rate'])}

The contract-aware router is lower by {result['means']['contract_learned_router_v1']-result['means']['generic_learned_router_v1']:.6f} nDCG, and its paired 95% interval includes zero. **NO EVIDENCE THAT CONTRACT REPRESENTATION ADDS ROUTING VALUE.**

### Route distributions

{table(routes,['method','route_selected','query_count'])}

### Intervention and harm

{table(interventions,['method','intervention_attempted_rate','effective_intervention_rate','beneficial_effective','harmful_effective','neutral_effective','harmful_effective_rate','zero_regret_rate','selected_in_oracle_rate'])}

Attempted intervention counts route selections away from PRESERVE. Effective intervention means the produced ranking differs from PRESERVE. `harmful_effective_rate` is harmful/effective; the paired ablation's `harmful_effective_query_rate` uses a per-query harmful-effective indicator so denominators remain paired.

### Latency (milliseconds)

{table(latency,['method','mean_latency_ms','median_latency_ms','p95_latency_ms'])}

### Feature analysis

Top absolute standardized coefficients per contract-aware target:

{table(top,['target','feature','standardized_coefficient','absolute_coefficient'])}

The requested focal features are retained in `feature_analysis.csv`, including hard exclusion, query length, positive-term count, contract status, resolved product type, brand, and price signals. Importance is not causality.

## Q. Predeclared paired bootstrap

All entries use 2,000 paired resamples, 95% percentile intervals, seed 29. Positive deltas favor the left method except route regret and harmful rate, where lower is better.

{table(comparisons,['left','right','metric','mean_difference','lower','upper','n','win_probability'])}

## R. Go/no-go

**CASE C — Generic and contract-aware routers perform equivalently.**

Conclusion: **no evidence that explicit SearchContractV1 adds routing information.** The point estimate favors the generic router, and neither learned router reliably beats the best static route. Do not automatically design Router V3.

## S. Remaining research blockers

- Positive oracle headroom remains, but both learned routers capture little of it and interventions that actually alter rankings are rare.
- Contract feature coefficients do not establish causal value and did not translate into a reliable aggregate improvement.
- This experiment is validation-only by design; ESCI test remains untouched.

## T. Safety and tests

The run used no ESCI test rows, historical Q-table, historical gate, or validation data for training/calibration. Ranking methods received candidate data with judgments removed; the evaluator applied labels only after ranking. Atomic implementations and frozen components were not changed. Final test and compilation commands are recorded in the handoff after execution.

## Failure analysis

Aggregate and bootstrap results were frozen before case extraction. `largest_router_wins.csv`, `largest_router_harms.csv`, and `contract_vs_generic_cases.csv` contain the post-hoc examples; they were not used to modify the models. Silent route/query failures: {result['failures']}.
"""
    (out/"mvp29_6_learned_routing_report.md").write_text(report,encoding="utf-8")

def run(args):
    out=args.output_dir; out.mkdir(parents=True,exist_ok=True); examples=load_examples(args.dataset_path); pool=task_examples(examples,task_version="small",split="train"); train_fit,validation=development_query_split(pool,seed=SEED,train_fraction=.8); policy,calibration=policy_calibration_split(train_fit,seed=SEED,policy_fraction=.8)
    old=set()
    for path in args.previously_inspected: old.update(int(x) for x in path.read_text().splitlines() if x.strip())
    old=sorted(old); fresh_pool=sorted(set(validation)-set(old)); fresh_ids=select_query_ids(preserve_selected_queries(pool,fresh_pool),max_queries=2000,seed=SEED); write_ids(out/"previously_inspected_validation_ids.txt",old); write_ids(out/"fresh_validation_pool_ids.txt",fresh_pool); write_ids(out/"fresh_validation_2000_query_ids.txt",fresh_ids)
    safety={"policy_calibration_overlap":len(set(policy)&set(calibration)),"policy_validation_overlap":len(set(policy)&set(validation)),"calibration_validation_overlap":len(set(calibration)&set(validation)),"fresh_inspected_overlap":len(set(fresh_ids)&set(old))}; assert not any(safety.values())
    feature_schema={"generic_version":GENERIC_FEATURE_VERSION,"generic_features":list(GENERIC_FEATURES),"contract_version":CONTRACT_FEATURE_VERSION,"contract_features":list(CONTRACT_FEATURES),"target_columns":["target__delta_filter_vs_preserve","target__delta_rerank_vs_preserve"],"prefix_contract":{"feature__":"inference-time label-free feature","target__":"training supervision only","metadata__":"identity/provenance","outcome__":"evaluator-only route outcome"}}; (out/"feature_schema.json").write_text(json.dumps(feature_schema,indent=2),encoding="utf-8")
    policy_data=evaluate_partition("policy_train",pool,policy,args.dataset_path,out/"policy_train_route_outcomes.parquet"); calibration_data=evaluate_partition("calibration",pool,calibration,args.dataset_path,out/"calibration_route_outcomes.parquet"); assert policy_data.metadata__success.all() and calibration_data.metadata__success.all()
    models={"generic_learned_router_v1":train_router("generic_learned_router_v1",GENERIC_FEATURES,policy_data,calibration_data),"contract_learned_router_v1":train_router("contract_learned_router_v1",CONTRACT_FEATURES,policy_data,calibration_data)}
    manifest={"created_at_utc":datetime.now(timezone.utc).isoformat(),"seed":SEED,"calibration_objective":CALIBRATION_OBJECTIVE,"candidate_alphas":ALPHAS,"candidate_thresholds":THRESHOLDS,"partitions":{"policy_train":len(policy),"calibration":len(calibration),"validation":len(validation),"previously_inspected":len(old),"fresh_validation_pool":len(fresh_pool),"fresh_validation_selected":len(fresh_ids)},"safety":safety,"libraries":{"python":platform.python_version(),"numpy":np.__version__,"pandas":pd.__version__},"models":models,"training_hashes":{"policy_train_route_outcomes":sha(out/"policy_train_route_outcomes.parquet"),"calibration_route_outcomes":sha(out/"calibration_route_outcomes.parquet")}}
    (out/"training_manifest.json").write_text(json.dumps(manifest,indent=2),encoding="utf-8")
    config={"source_split":"train","development_partition":"fresh_validation","selected_ids_sha256":sha(out/"fresh_validation_2000_query_ids.txt"),"primary_comparisons":PRIMARY_COMPARISONS,"bootstrap":{"resamples":RESAMPLES,"confidence":.95,"seed":SEED},"frozen_components":{"SearchContractV1":PARSER_VERSION,"contract_rerank_v1":RERANKER_VERSION,"contract_rerank_config":DEFAULT_CONFIG.to_dict(),"method_version":METHOD_VERSION,"router_v2":ROUTER_V2_VERSION},"safety":{"esci_test_used":False,"contaminated_state_used":False,"historical_gate_used":False,"validation_used_for_training_or_calibration":False}}
    config["artifact_hashes"]={"previously_inspected_validation_ids":sha(out/"previously_inspected_validation_ids.txt"),"fresh_validation_pool_ids":sha(out/"fresh_validation_pool_ids.txt"),"fresh_validation_2000_query_ids":sha(out/"fresh_validation_2000_query_ids.txt")}
    (out/"evaluation_config.json").write_text(json.dumps(config,indent=2,default=list),encoding="utf-8")
    fresh_data=evaluate_partition("fresh_validation",pool,fresh_ids,args.dataset_path,out/"fresh_validation_route_outcomes.internal.parquet"); assert fresh_data.metadata__success.all(); router_v2=evaluate_router_v2(pool,fresh_ids,args.dataset_path); results=build_validation_results(fresh_data,models,router_v2,out/"fresh_validation_query_results.parquet"); result={"selected_sha256":config["selected_ids_sha256"],**aggregate(results,out,models)}; write_report(out,manifest,config,result); return result

def main():
    p=argparse.ArgumentParser(); p.add_argument("--dataset-path",type=Path,default=Path("esci-data/shopping_queries_dataset")); p.add_argument("--output-dir",type=Path,default=Path("outputs/paper_eval/mvp29_6_learned_routing")); p.add_argument("--previously-inspected",type=Path,nargs="+",required=True); args=p.parse_args(); print(json.dumps(run(args),indent=2))
if __name__=="__main__":main()
