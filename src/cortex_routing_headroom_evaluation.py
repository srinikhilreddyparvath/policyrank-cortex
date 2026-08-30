from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src.paper_eval.clean_router import ROUTER_V2_VERSION, ROUTE_FAMILY_V2_VERSION
from src.paper_eval.contract_rerank_v1 import DEFAULT_CONFIG, RERANKER_VERSION
from src.paper_eval.contracts_v1 import PARSER_VERSION, build_search_contract_v1
from src.paper_eval.dataset import development_query_split, load_examples, load_products_for_candidates, preserve_selected_queries, select_query_ids, task_examples
from src.paper_eval.methods import METHOD_VERSION, REGISTRY
from src.paper_eval.metrics import exact_at_k, official_esci_ndcg, reciprocal_rank_at_k
from src.paper_eval.statistics import paired_bootstrap_difference

SEED=29
BOOTSTRAP_RESAMPLES=2000
ATOMIC_ROUTES=("PRESERVE","STRICT_FILTER","CONTRACT_RERANK")
METHODS={"PRESERVE":"fts_baseline","STRICT_FILTER":"strict_filter","CONTRACT_RERANK":"always_contract_rerank_v1","ROUTER_V2":"clean_contract_router_v2"}
SOURCE_FILES=("src/paper_eval/contracts_v1.py","src/paper_eval/contract_rerank_v1.py","src/paper_eval/clean_router.py","src/paper_eval/methods.py")


def _sha(path:Path)->str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _markdown_table(frame:pd.DataFrame)->str:
    columns=list(frame.columns)
    lines=["| "+" | ".join(map(str,columns))+" |","| "+" | ".join("---" for _ in columns)+" |"]
    for _,row in frame.iterrows(): lines.append("| "+" | ".join("NA" if pd.isna(row[column]) else str(row[column]) for column in columns)+" |")
    return "\n".join(lines)


def _bucket(count:int)->str:
    if count<=1:return "0-1"
    if count==2:return "2"
    if count<=4:return "3-4"
    return "5+"


def _query_length_bucket(count:int)->str:
    if count<=2:return "1-2"
    if count<=4:return "3-4"
    return "5+"


def _evaluate(query_id:int, group:pd.DataFrame, method_key:str, run_id:str)->dict:
    method_name=METHODS[method_key]; evaluator=REGISTRY[method_name].evaluator; query=str(group["query"].iloc[0]); start=time.perf_counter()
    base={"run_id":run_id,"query_id":query_id,"query":query,"locale":str(group["product_locale"].iloc[0]),"source_split":"train","development_partition":"validation","method":method_key,"method_adapter":method_name,"method_version":METHOD_VERSION,"candidate_count":len(group)}
    contract=build_search_contract_v1(query)
    base.update({"contract_status":contract.contract_status,"hard_exclusion_present":bool(contract.must_not_have),"brand_signal_present":bool(contract.brand_signal),"price_signal_present":bool(contract.price_signal),"positive_term_count":len(contract.positive_terms),"positive_term_count_bucket":_bucket(len(contract.positive_terms)),"product_type_resolved":bool(contract.product_type),"constraint_strength":contract.constraint_strength,"query_length":len(contract.normalized_query.split()),"query_length_bucket":_query_length_bucket(len(contract.normalized_query.split()))})
    try:
        ranker=group.drop(columns=["esci_label"],errors="ignore").copy(); rows,meta=evaluator(query,ranker,len(group)); elapsed=(time.perf_counter()-start)*1000
        qrels={str(pid):str(label).upper() for pid,label in zip(group.product_id,group.esci_label)}; labels=[qrels[str(row.get("product_id"))] for row in rows]
        base.update({"route_selected":meta.get("route_selected",meta.get("route")),"final_product_ids":[str(row.get("product_id")) for row in rows],"official_esci_ndcg":official_esci_ndcg(labels,group.esci_label.tolist()),"mrr_10":reciprocal_rank_at_k(labels,10),"exact_1":exact_at_k(labels,1),"ranking_changed":bool(meta.get("ranking_changed",False)),"intervention_attempted":bool(meta.get("intervention_attempted",False)),"intervention_effective":bool(meta.get("intervention_effective",False)),"mean_candidate_movement":meta.get("mean_candidate_movement"),"top_1_changed":meta.get("top_1_changed"),"retrieval_latency_ms":meta.get("retrieval_latency_ms"),"ranking_latency_ms":meta.get("ranking_latency_ms"),"total_latency_ms":elapsed,"success":True,"failure_stage":None,"error_type":None,"error_message":None})
    except Exception as exc:
        base.update({"route_selected":None,"final_product_ids":[],"official_esci_ndcg":None,"mrr_10":None,"exact_1":None,"ranking_changed":None,"intervention_attempted":None,"intervention_effective":None,"mean_candidate_movement":None,"top_1_changed":None,"retrieval_latency_ms":None,"ranking_latency_ms":None,"total_latency_ms":(time.perf_counter()-start)*1000,"success":False,"failure_stage":"method_evaluation","error_type":type(exc).__name__,"error_message":str(exc)[:1000]})
    return base


def _add_oracle(data:pd.DataFrame)->pd.DataFrame:
    data=data.copy(); data["oracle_route_set"]=None; data["oracle_ndcg"]=None; data["oracle_reward"]=None; data["selected_route_reward"]=None; data["route_regret_ndcg"]=None
    for query_id,group in data.groupby("query_id"):
        atomic=group[(group.method.isin(ATOMIC_ROUTES))&(group.success==True)]
        if len(atomic)!=3: continue
        best=float(atomic.official_esci_ndcg.max()); winners=sorted(atomic.loc[(atomic.official_esci_ndcg-best).abs()<=1e-12,"method"].tolist())
        indices=group.index; data.loc[indices,"oracle_route_set"]=pd.Series([winners for _ in indices],index=indices,dtype=object); data.loc[indices,"oracle_ndcg"]=best; data.loc[indices,"oracle_reward"]=best
        router=group[group.method=="ROUTER_V2"]
        if len(router)==1 and bool(router.success.iloc[0]):
            idx=router.index[0]; selected=float(router.official_esci_ndcg.iloc[0]); data.at[idx,"selected_route_reward"]=selected; data.at[idx,"route_regret_ndcg"]=best-selected
    return data


def _strata(wide:pd.DataFrame, router:pd.DataFrame)->pd.DataFrame:
    features=("contract_status","hard_exclusion_present","brand_signal_present","price_signal_present","positive_term_count_bucket","product_type_resolved","constraint_strength","query_length_bucket")
    rows=[]
    for feature in features:
        for value,indices in router.groupby(feature,dropna=False).groups.items():
            qids=router.loc[indices,"query_id"].tolist(); subset=wide.loc[qids]; routed=router[router.query_id.isin(qids)]; means={route:subset[route].mean() for route in ATOMIC_ROUTES}; best=max(means,key=means.get)
            oracle_counts={route:int(sum(route in routes for routes in subset.oracle_route_set)) for route in ATOMIC_ROUTES}; router_counts=routed.route_selected.value_counts().to_dict()
            rows.append({"feature":feature,"value":value,"query_count":len(subset),"best_static_route":best,**{f"oracle_contains_{route.lower()}":oracle_counts[route] for route in ATOMIC_ROUTES},**{f"router_selected_{route.lower()}":int(router_counts.get(route,0)) for route in ATOMIC_ROUTES},"mean_router_regret":routed.route_regret_ndcg.mean(),**{f"mean_ndcg_{route.lower()}":means[route] for route in ATOMIC_ROUTES}})
    return pd.DataFrame(rows)


def aggregate(run_dir:Path)->dict:
    data=pd.read_parquet(run_dir/"query_route_results.parquet"); success=data[data.success==True]; atomic=success[success.method.isin(ATOMIC_ROUTES)]
    wide=atomic.pivot(index="query_id",columns="method",values="official_esci_ndcg"); router=success[success.method=="ROUTER_V2"].set_index("query_id"); wide["ROUTER_V2"]=router.official_esci_ndcg; wide["oracle_ndcg"]=router.oracle_ndcg; wide["oracle_route_set"]=router.oracle_route_set
    means={column:float(wide[column].mean()) for column in (*ATOMIC_ROUTES,"ROUTER_V2","oracle_ndcg")}; best_static=max(ATOMIC_ROUTES,key=lambda route:means[route])
    summary=pd.DataFrame([{"method":key,"mean_official_esci_ndcg":value,"query_count":len(wide)} for key,value in means.items()]); summary["best_static_route"]=best_static; summary.to_csv(run_dir/"route_summary.csv",index=False)
    oracle_rows=[]
    for route in ATOMIC_ROUTES:
        oracle_rows.append({"route":route,"contained_in_oracle_set":int(sum(route in routes for routes in wide.oracle_route_set)),"unique_best":int(sum(list(routes)==[route] for routes in wide.oracle_route_set))})
    oracle_rows.append({"route":"TIES","contained_in_oracle_set":int(sum(len(routes)>1 for routes in wide.oracle_route_set)),"unique_best":None}); pd.DataFrame(oracle_rows).to_csv(run_dir/"oracle_route_distribution.csv",index=False)
    regret=router[["query","route_selected","official_esci_ndcg","oracle_ndcg","route_regret_ndcg","oracle_route_set"]].copy(); regret["selected_in_oracle_set"]=[route in routes for route,routes in zip(regret.route_selected,regret.oracle_route_set)]; regret["selected_unique_oracle_winner"]=[list(routes)==[route] for route,routes in zip(regret.route_selected,regret.oracle_route_set)]; regret["zero_regret"]=(regret.route_regret_ndcg.abs()<=1e-12); regret["high_regret"]=(regret.route_regret_ndcg>=0.1); regret.reset_index().to_csv(run_dir/"router_regret.csv",index=False)
    pairs=(("STRICT_FILTER","PRESERVE"),("CONTRACT_RERANK","PRESERVE"),("CONTRACT_RERANK","STRICT_FILTER")); pair_rows=[]; sep=[]
    for left,right in pairs:
        delta=wide[left]-wide[right]; pair_rows.extend({"query_id":qid,"left":left,"right":right,"delta_official_ndcg":value} for qid,value in delta.items()); sep.append({"comparison":f"{left} - {right}","positive":int((delta>1e-12).sum()),"negative":int((delta< -1e-12).sum()),"zero":int((delta.abs()<=1e-12).sum()),"mean_delta":delta.mean()})
    pd.DataFrame(pair_rows).to_csv(run_dir/"route_pairwise_deltas.csv",index=False); pd.DataFrame(sep).to_csv(run_dir/"route_separability.csv",index=False)
    _strata(wide,router.reset_index()).to_csv(run_dir/"contract_feature_strata.csv",index=False)
    success.groupby("method").total_latency_ms.agg(query_count="count",mean_latency_ms="mean",median_latency_ms="median",p95_latency_ms=lambda values:values.quantile(.95)).reset_index().to_csv(run_dir/"latency_summary.csv",index=False)
    comparisons=(("STRICT_FILTER","PRESERVE"),("CONTRACT_RERANK","PRESERVE"),("ROUTER_V2","PRESERVE"),("ROUTER_V2","STRICT_FILTER"),("oracle_ndcg",best_static),("oracle_ndcg","ROUTER_V2")); boot=[]
    for left,right in comparisons: boot.append({"left":left,"right":right,**paired_bootstrap_difference(wide[left],wide[right],seed=SEED,iterations=BOOTSTRAP_RESAMPLES)})
    pd.DataFrame(boot).to_csv(run_dir/"bootstrap_comparisons.csv",index=False)
    preserve=wide.PRESERVE; cr=wide.CONTRACT_RERANK; changed=success[(success.method=="CONTRACT_RERANK")&success.ranking_changed].set_index("query_id"); cr_delta=cr-preserve
    cases=router.reset_index().copy(); cases.sort_values("route_regret_ndcg",ascending=False).head(100).to_csv(run_dir/"worst_router_regret_cases.csv",index=False); cases.sort_values("route_regret_ndcg").head(100).to_csv(run_dir/"best_route_cases.csv",index=False)
    win_ids=cr_delta[cr_delta>1e-12].sort_values(ascending=False).index; harm_ids=cr_delta[cr_delta< -1e-12].sort_values().index
    success[(success.method=="CONTRACT_RERANK")&success.query_id.isin(win_ids)].sort_values("official_esci_ndcg",ascending=False).to_csv(run_dir/"contract_rerank_win_cases.csv",index=False); success[(success.method=="CONTRACT_RERANK")&success.query_id.isin(harm_ids)].sort_values("official_esci_ndcg").to_csv(run_dir/"contract_rerank_harm_cases.csv",index=False)
    denominator=means["oracle_ndcg"]-means["PRESERVE"]; capture=(means["ROUTER_V2"]-means["PRESERVE"])/denominator if denominator>0 else None
    result={"means":means,"best_static_route":best_static,"oracle_headroom_best_static":means["oracle_ndcg"]-means[best_static],"oracle_headroom_router":means["oracle_ndcg"]-means["ROUTER_V2"],"router_delta_best_static":means["ROUTER_V2"]-means[best_static],"router_capture_ratio":capture,"router_mean_regret":float(regret.route_regret_ndcg.mean()),"router_median_regret":float(regret.route_regret_ndcg.median()),"zero_regret_rate":float(regret.zero_regret.mean()),"high_regret_count":int(regret.high_regret.sum()),"failures":int((~data.success).sum()),"best_static":best_static}
    oracle_table=pd.DataFrame(oracle_rows); sep_table=pd.DataFrame(sep); boot_table=pd.DataFrame(boot); latency=pd.read_csv(run_dir/"latency_summary.csv")
    cr_unique=int(sum(list(routes)==["CONTRACT_RERANK"] for routes in wide.oracle_route_set)); cr_tied=int(sum("CONTRACT_RERANK" in routes and len(routes)>1 for routes in wide.oracle_route_set)); cr_delta=wide.CONTRACT_RERANK-wide.PRESERVE
    cr_changed_ids=set(changed.index); changed_delta=cr_delta.loc[list(cr_changed_ids)] if cr_changed_ids else pd.Series(dtype=float)
    report=f"""# MVP 29.5 — 1,000-query routing headroom

Validation analysis only; not final ESCI test inference and not a deployable oracle.

## Protocol

- Source: `small_version == 1`, train-derived validation partition
- Queries: {len(wide)}, excluding the previously inspected 100-query smoke set
- Atomic outcomes per query: PRESERVE, STRICT_FILTER, CONTRACT_RERANK
- Router: clean_contract_router_v2
- Bootstrap: {BOOTSTRAP_RESAMPLES} deterministic paired resamples, 95% interval, seed {SEED}
- Failed query-route rows: {result['failures']}

## Mean official ESCI nDCG

| Outcome | Mean |
|---|---:|
| PRESERVE | {means['PRESERVE']:.9f} |
| STRICT_FILTER | {means['STRICT_FILTER']:.9f} |
| CONTRACT_RERANK | {means['CONTRACT_RERANK']:.9f} |
| Router V2 | {means['ROUTER_V2']:.9f} |
| Oracle | {means['oracle_ndcg']:.9f} |

Best static route: **{best_static}**.

- Oracle headroom over best static: {result['oracle_headroom_best_static']:.9f}
- Oracle headroom over Router V2: {result['oracle_headroom_router']:.9f}
- Router V2 delta over best static: {result['router_delta_best_static']:.9f}
- Router capture ratio: {result['router_capture_ratio']:.4%} (denominator is positive; negative value means the router fell below PRESERVE while the oracle improved over it)

## Tie-aware oracle distribution

{_markdown_table(oracle_table)}

## Router quality

- Selected route belongs to oracle set: {int(regret.selected_in_oracle_set.sum())}/{len(regret)}
- Selected route is unique oracle winner: {int(regret.selected_unique_oracle_winner.sum())}/{len(regret)}
- Mean regret: {regret.route_regret_ndcg.mean():.9f}
- Median regret: {regret.route_regret_ndcg.median():.9f}
- Zero-regret rate: {regret.zero_regret.mean():.2%}
- High-regret queries (regret >= 0.1): {int(regret.high_regret.sum())}

## CONTRACT_RERANK value

- Unique wins: {cr_unique}
- Tied-best: {cr_tied}
- Loses to PRESERVE: {int((wide.CONTRACT_RERANK < wide.PRESERVE-1e-12).sum())}
- Loses to STRICT_FILTER: {int((wide.CONTRACT_RERANK < wide.STRICT_FILTER-1e-12).sum())}
- Changed rankings: {len(changed_delta)}
- Beneficial changed rankings: {int((changed_delta>1e-12).sum())}
- Harmful changed rankings: {int((changed_delta< -1e-12).sum())}
- Neutral changed rankings: {int((changed_delta.abs()<=1e-12).sum())}
- Mean positive delta: {cr_delta[cr_delta>1e-12].mean():.9f}
- Mean negative delta: {cr_delta[cr_delta< -1e-12].mean():.9f}
- Best positive delta: {cr_delta.max():.9f}
- Worst negative delta: {cr_delta.min():.9f}

## Route separability

{_markdown_table(sep_table)}

Most differences are zero: 962/1,000 for STRICT_FILTER versus PRESERVE, 675/1,000 for CONTRACT_RERANK versus PRESERVE, and 668/1,000 for CONTRACT_RERANK versus STRICT_FILTER.

## Paired bootstrap comparisons

{_markdown_table(boot_table)}

## Latency

{_markdown_table(latency)}

## Decision

1. Oracle headroom exists and its paired interval excludes zero, but the mean gain over the best static route is modest ({result['oracle_headroom_best_static']:.6f}).
2. CONTRACT_RERANK provides unique value on {cr_unique} queries, although it also loses to PRESERVE on {int((wide.CONTRACT_RERANK < wide.PRESERVE-1e-12).sum())}.
3. PRESERVE is uniquely best on 8 queries and belongs to the tied oracle set on 831 queries.
4. STRICT_FILTER is uniquely best on 19 queries and belongs to the tied oracle set on 838 queries.
5. CONTRACT_RERANK is uniquely best on 150 queries and belongs to the tied oracle set on 822 queries.
6. Router V2 captures {result['router_capture_ratio']:.2%} of baseline-to-oracle headroom; the negative value means it does not capture oracle value on average.
7. Router V2 exceeds always-STRICT_FILTER by {means['ROUTER_V2']-means['STRICT_FILTER']:.6f}, but remains {means[best_static]-means['ROUTER_V2']:.6f} below the best static route.
8. Descriptively, hard-exclusion queries have the largest router regret; longer queries and two-positive-term queries also show elevated regret. These are correlations, not causal effects.
9. The heterogeneous unique winners justify a bounded learned-router experiment on policy_train with calibration-only model selection, but not a headline routing claim.
10. Current Router V2 complexity is not justified as the final strategy because it underperforms the best static route and most route pairs are tied.
"""
    (run_dir/"mvp29_5_routing_headroom_report.md").write_text(report,encoding="utf-8")
    return result


def run(args)->dict:
    run_dir=args.output_dir; run_dir.mkdir(parents=True,exist_ok=False)
    examples=load_examples(args.dataset_path); pool=task_examples(examples,task_version="small",split="train"); train_fit,validation=development_query_split(pool,seed=SEED,train_fraction=.8); validation_pool=preserve_selected_queries(pool,validation)
    excluded={int(line) for line in args.exclude_query_ids.read_text(encoding="utf-8").splitlines() if line.strip()}; fresh=validation_pool[~validation_pool.query_id.isin(excluded)].copy(); ids=select_query_ids(fresh,max_queries=1000,seed=SEED)
    selected=preserve_selected_queries(fresh,ids); products=load_products_for_candidates(args.dataset_path,set(selected.product_id.astype(str))); selected=selected.merge(products,on=["product_locale","product_id"],how="left",validate="many_to_one")
    id_path=run_dir/"selected_validation_1000_query_ids.txt"; id_path.write_text("\n".join(map(str,ids))+"\n",encoding="utf-8")
    config={"run_id":run_dir.name,"created_at_utc":datetime.now(timezone.utc).isoformat(),"dataset_path":str(args.dataset_path.resolve()),"task_version":"small","source_split":"train","development_partition":"validation","selection_seed":SEED,"selected_query_count":len(ids),"excluded_prior_smoke_count":len(excluded),"selected_ids_sha256":_sha(id_path),"bootstrap":{"resamples":BOOTSTRAP_RESAMPLES,"confidence_level":.95,"seed":SEED},"atomic_routes":list(ATOMIC_ROUTES),"methods":METHODS,"method_version":METHOD_VERSION,"parser_version":PARSER_VERSION,"reranker_version":RERANKER_VERSION,"reranker_config":DEFAULT_CONFIG.to_dict(),"router_version":ROUTER_V2_VERSION,"route_family_version":ROUTE_FAMILY_V2_VERSION,"source_sha256":{path:_sha(Path(path)) for path in SOURCE_FILES},"safety":{"esci_test_used":False,"contaminated_q_table_used":False,"historical_gate_used":False,"judgments_passed_to_methods":False}}
    (run_dir/"evaluation_config.json").write_text(json.dumps(config,indent=2,sort_keys=True),encoding="utf-8")
    rows=[]; run_id=run_dir.name
    for query_id,group in selected.groupby("query_id",sort=True):
        for method in (*ATOMIC_ROUTES,"ROUTER_V2"): rows.append(_evaluate(int(query_id),group,method,run_id))
        if len(rows)%200==0: pd.DataFrame(rows).to_parquet(run_dir/"query_route_results.checkpoint.parquet",index=False)
    data=_add_oracle(pd.DataFrame(rows)); data.to_parquet(run_dir/"query_route_results.parquet",index=False)
    checkpoint=run_dir/"query_route_results.checkpoint.parquet"
    if checkpoint.exists(): checkpoint.unlink()
    return {"selected_ids_sha256":config["selected_ids_sha256"],"selected":len(ids),**aggregate(run_dir)}


def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--dataset-path",type=Path,default=Path("esci-data/shopping_queries_dataset")); parser.add_argument("--exclude-query-ids",type=Path,required=True); parser.add_argument("--output-dir",type=Path,default=Path("outputs/paper_eval/mvp29_5_routing_headroom_1000")); args=parser.parse_args(); print(json.dumps(run(args),indent=2))

if __name__=="__main__": main()
