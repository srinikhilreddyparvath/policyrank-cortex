from __future__ import annotations
import json, math, time
from pathlib import Path
from collections import Counter
from dataclasses import dataclass
from typing import Callable
import pandas as pd
from src.full_esci_retrieval_engine import FullEsciRetrievalEngine, clean_text, tokenize
from src.calibrated_route_execution_adapter import apply_scale_aware_strict_boost, apply_strict_constraint_filter, detect_strict_constraint_type
from src.contract_filters import apply_contract_candidate_filter
from src.final_slate_enforcer import enforce_final_slate_contract
from src.multi_agent_diversifier import multi_agent_diversify_slate
from src.policy_compiler import compile_and_apply_policy
from src.slate_q_learning import ACTIONS, get_contract_state
from src.paper_eval.adapter import CandidateSetRequest, validate_output_subset
from src.baseline_preservation_gate import select_slate_with_baseline_preservation_gate, summarize_gate_for_logging
from src.paper_eval.clean_router import route_contract_v1, route_contract_v2
from src.paper_eval.contracts_v1 import build_search_contract_v1
from src.paper_eval.contract_rerank_v1 import contract_rerank_v1
from src.paper_eval.provenance import LEGACY_GATE_THRESHOLDS, LEGACY_Q_TABLE, require_safe

METHOD_VERSION="mvp29.4"
@dataclass(frozen=True)
class MethodSpec:
    name:str; enabled:bool; reason:str; evaluator:Callable|None; contaminated:bool=False

def _lexical_rows(query: str, candidates: pd.DataFrame) -> list[dict]:
    engine=FullEsciRetrievalEngine.__new__(FullEsciRetrievalEngine)
    products=[]; doc_freq=Counter()
    for _,row in candidates.iterrows():
        product=row.to_dict(); title=clean_text(product.get("product_title")); brand=clean_text(product.get("product_brand"))
        search_text=" ".join(clean_text(product.get(key)) for key in ("product_title","product_brand","product_description","product_bullet_point","product_color"))
        product.update({"search_text":search_text,"title_tokens":set(tokenize(title)),"text_tokens":set(tokenize(search_text)),"brand_tokens":set(tokenize(brand))})
        product["all_tokens"]=product["title_tokens"]|product["text_tokens"]|product["brand_tokens"]; doc_freq.update(product["all_tokens"]); products.append(product)
    engine.idf={token:math.log((1+max(len(products),1))/(1+freq))+1 for token,freq in doc_freq.items()}
    scored=[]
    for product in products:
        info=engine.score_product(query, product)
        scored.append({**product,"score":float(info["score"]),"matched_tokens":"|".join(info["matched_tokens"]),"query_token_coverage":info["coverage"]})
    return sorted(scored,key=lambda row:(-float(row["score"]),clean_text(row.get("product_title")),clean_text(row.get("product_id"))))

def _request(query:str,candidates:pd.DataFrame)->CandidateSetRequest:
    query_id=int(candidates["query_id"].iloc[0]) if "query_id" in candidates else -1
    return CandidateSetRequest(query,query_id,candidates)

def fts_baseline(query: str, candidates: pd.DataFrame, top_k: int):
    request=_request(query,candidates); start=time.perf_counter(); rows=_lexical_rows(query,request.ranker_frame()); elapsed=(time.perf_counter()-start)*1000; validate_output_subset(request,rows[:top_k])
    return rows[:top_k], {"route":"FTS_BASELINE","route_selected":"FTS_BASELINE","ranking_changed":False,"explicit_preserve":True,"intervention_attempted":False,"intervention_effective":False,"intervened":False,"preserved":True,"fallback":False,"retrieval_latency_ms":elapsed,"ranking_latency_ms":0.0}

def strict_filter(query: str, candidates: pd.DataFrame, top_k: int):
    request=_request(query,candidates); start=time.perf_counter(); rows=_lexical_rows(query,request.ranker_frame()); retrieval=(time.perf_counter()-start)*1000; rank_start=time.perf_counter()
    constraint=detect_strict_constraint_type(query); filtered,meta=apply_strict_constraint_filter(query,rows,constraint,target_count=top_k,filter_mode="hybrid")
    rank=(time.perf_counter()-rank_start)*1000
    output=filtered[:top_k]; changed=[str(r.get("product_id")) for r in output]!=[str(r.get("product_id")) for r in rows[:top_k]]
    validate_output_subset(request,output); return output, {"route":"STRICT_FILTER","route_selected":"STRICT_FILTER","ranking_changed":changed,"explicit_preserve":False,"intervention_attempted":True,"intervention_effective":changed,"intervened":changed,"preserved":False,"fallback":False,"retrieval_latency_ms":retrieval,"ranking_latency_ms":rank,**meta}

def strict_boost(query: str, candidates: pd.DataFrame, top_k: int):
    request=_request(query,candidates); start=time.perf_counter(); rows=_lexical_rows(query,request.ranker_frame()); retrieval=(time.perf_counter()-start)*1000; rank_start=time.perf_counter()
    constraint=detect_strict_constraint_type(query); boosted,meta=apply_scale_aware_strict_boost(query,rows,constraint,rerank_mode="strict_boost")
    rank=(time.perf_counter()-rank_start)*1000
    output=boosted[:top_k]; changed=[str(r.get("product_id")) for r in output]!=[str(r.get("product_id")) for r in rows[:top_k]]
    validate_output_subset(request,output); return output, {"route":"STRICT_BOOST","route_selected":"STRICT_BOOST","ranking_changed":changed,"explicit_preserve":False,"intervention_attempted":True,"intervention_effective":changed,"intervened":changed,"preserved":False,"fallback":False,"retrieval_latency_ms":retrieval,"ranking_latency_ms":rank,**meta}

def clean_contract_router_v1(query: str, candidates: pd.DataFrame, top_k: int):
    request=_request(query,candidates); start=time.perf_counter(); baseline_rows=_lexical_rows(query,request.ranker_frame()); retrieval=(time.perf_counter()-start)*1000
    rank_start=time.perf_counter(); contract=build_search_contract_v1(query); decision=route_contract_v1(contract)
    if decision.route=="STRICT_FILTER":
        constraint=detect_strict_constraint_type(query)
        routed_rows,route_meta=apply_strict_constraint_filter(query,baseline_rows,constraint,target_count=top_k,filter_mode="hybrid")
        output=routed_rows[:top_k]
    else:
        output=baseline_rows[:top_k]; route_meta={}
    baseline_ids=[str(row.get("product_id")) for row in baseline_rows[:top_k]]; output_ids=[str(row.get("product_id")) for row in output]
    changed=output_ids!=baseline_ids; attempted=decision.route!="PRESERVE"
    validate_output_subset(request,output)
    diagnostics={
        "route":decision.route,"route_selected":decision.route,"router_version":decision.router_version,"router_reason":decision.reason_code,
        "ranking_changed":changed,"explicit_preserve":decision.route=="PRESERVE","intervention_attempted":attempted,"intervention_effective":changed,
        "intervened":changed,"preserved":decision.route=="PRESERVE","fallback":contract.fallback_used,
        "contract":contract.to_dict(),"contract_status":contract.contract_status,
        "detected_constraint_count":len(contract.must_have)+len(contract.must_not_have),"negative_constraint_count":len(contract.must_not_have),
        "positive_constraint_count":len(contract.must_have),"hard_constraint_count":len(contract.must_not_have) if contract.constraint_strength=="hard" else 0,
        "soft_constraint_count":len(contract.must_have) if contract.constraint_strength=="soft" else 0,"contract_ambiguity":contract.ambiguity,
        "contract_parser_route":contract.parser_route,"contract_parser_reason":contract.parser_reason,
        "retrieval_latency_ms":retrieval,"ranking_latency_ms":(time.perf_counter()-rank_start)*1000,
        **route_meta,
    }
    return output,diagnostics

def _contract_diagnostics(contract):
    return {"contract":contract.to_dict(),"contract_status":contract.contract_status,
        "detected_constraint_count":len(contract.positive_terms)+len(contract.must_not_have),"negative_constraint_count":len(contract.must_not_have),
        "positive_constraint_count":len(contract.positive_terms),"hard_constraint_count":len(contract.must_not_have) if contract.constraint_strength=="hard" else 0,
        "soft_constraint_count":len(contract.positive_terms) if contract.constraint_strength!="hard" else 0,"contract_ambiguity":contract.ambiguity,
        "contract_parser_route":contract.parser_route,"contract_parser_reason":contract.parser_reason}

def always_contract_rerank_v1(query: str, candidates: pd.DataFrame, top_k: int):
    request=_request(query,candidates); start=time.perf_counter(); baseline_rows=_lexical_rows(query,request.ranker_frame()); retrieval=(time.perf_counter()-start)*1000
    rank_start=time.perf_counter(); contract=build_search_contract_v1(query); reranked,rerank_meta=contract_rerank_v1(baseline_rows,contract); output=reranked[:top_k]
    baseline_ids=[str(row.get("product_id")) for row in baseline_rows[:top_k]]; output_ids=[str(row.get("product_id")) for row in output]; changed=output_ids!=baseline_ids
    validate_output_subset(request,output)
    return output,{"route":"CONTRACT_RERANK","route_selected":"CONTRACT_RERANK","ranking_changed":changed,"explicit_preserve":False,
        "intervention_attempted":True,"intervention_effective":changed,"intervened":changed,"preserved":False,"fallback":contract.fallback_used,
        "retrieval_latency_ms":retrieval,"ranking_latency_ms":(time.perf_counter()-rank_start)*1000,**_contract_diagnostics(contract),**rerank_meta}

def clean_contract_router_v2(query: str, candidates: pd.DataFrame, top_k: int):
    request=_request(query,candidates); start=time.perf_counter(); baseline_rows=_lexical_rows(query,request.ranker_frame()); retrieval=(time.perf_counter()-start)*1000
    rank_start=time.perf_counter(); contract=build_search_contract_v1(query); decision=route_contract_v2(contract); route_meta={}
    if decision.route=="STRICT_FILTER":
        constraint=detect_strict_constraint_type(query); routed,route_meta=apply_strict_constraint_filter(query,baseline_rows,constraint,target_count=top_k,filter_mode="hybrid"); output=routed[:top_k]
    elif decision.route=="CONTRACT_RERANK":
        routed,route_meta=contract_rerank_v1(baseline_rows,contract); output=routed[:top_k]
    else: output=baseline_rows[:top_k]
    baseline_ids=[str(row.get("product_id")) for row in baseline_rows[:top_k]]; output_ids=[str(row.get("product_id")) for row in output]; changed=output_ids!=baseline_ids; attempted=decision.route!="PRESERVE"
    validate_output_subset(request,output)
    return output,{"route":decision.route,"route_selected":decision.route,"router_version":decision.router_version,"router_reason":decision.reason_code,
        "ranking_changed":changed,"explicit_preserve":decision.route=="PRESERVE","intervention_attempted":attempted,"intervention_effective":changed,
        "intervened":changed,"preserved":decision.route=="PRESERVE","fallback":contract.fallback_used,"retrieval_latency_ms":retrieval,
        "ranking_latency_ms":(time.perf_counter()-rank_start)*1000,**_contract_diagnostics(contract),**route_meta}

def _read_only_action(state:str):
    try: table=json.loads(Path("storage/slate_q_table.json").read_text(encoding="utf-8"))
    except Exception: table={}
    values=table.get(state,{})
    action=max(ACTIONS,key=lambda name:float(values.get(name,0.0)))
    return action,ACTIONS[action]

def _local_contract(query:str, rows:list[dict])->dict:
    normalized=" ".join(str(query).lower().split()); terms=[]
    for token in normalized.replace("-"," ").split():
        cleaned="".join(char for char in token if char.isalnum())
        if cleaned and len(cleaned)>1 and cleaned not in terms: terms.append(cleaned)
    selected=terms[:6]
    return {"contract_source":"local_fallback_contract","fallback_used":True,"fallback_reason":"deterministic_label_safe_candidate_adapter","query":query,"intent":{"detected_category":"general","product_type":"general","price_sensitivity":"unknown","quality_preference":"unknown","ambiguity_level":"medium"},"dynamic_filters":{"must_have_terms":selected,"blocked_terms":[],"preferred_terms":selected},"required_terms":selected,"preferred_terms":selected,"excluded_terms":[],"brand_preferences":[],"notes":{"source":"Label-safe adapter equivalent of the legacy local fallback schema.","sample_titles":[str(row.get("product_title","")) for row in rows[:5]]}}

def always_cortex(query:str,candidates:pd.DataFrame,top_k:int, *, allow_unsafe_debug:bool=False):
    require_safe(LEGACY_Q_TABLE,allow_unsafe_debug=allow_unsafe_debug)
    request=_request(query,candidates); start=time.perf_counter(); baseline_rows=_lexical_rows(query,request.ranker_frame()); retrieval=(time.perf_counter()-start)*1000; rank_start=time.perf_counter()
    baseline=pd.DataFrame(baseline_rows); baseline["baseline_score"]=baseline["score"]; baseline["semantic_score"]=baseline["score"]; baseline["retrieval_score"]=baseline["score"]; baseline["rating"]=0.0
    # The legacy compiler structurally requires this column. It is deliberately blank:
    # no evaluator judgment value crosses the adapter boundary.
    baseline["esci_label"]=""
    contract=_local_contract(query,baseline.head(8).to_dict("records"))
    filtered=apply_contract_candidate_filter(baseline,contract,min_match_score=.5,strict_top_k=len(baseline)); ranking_input=filtered if len(filtered) else baseline
    state=get_contract_state(contract,float(baseline.baseline_score.max()) if len(baseline) else 0.0); action,weights=_read_only_action(state)
    policy=compile_and_apply_policy(ranking_input,weights,"CandidateSet"); enforced,enforcement=enforce_final_slate_contract(policy,contract,query,top_k=len(baseline)); diversified,trace=multi_agent_diversify_slate(enforced,top_k=len(enforced))
    rows=diversified.to_dict("records"); validate_output_subset(request,rows); rank=(time.perf_counter()-rank_start)*1000
    changed=[str(row.get("product_id")) for row in rows]!=[str(row.get("product_id")) for row in baseline_rows[:len(rows)]] or len(rows)!=len(baseline_rows)
    meta={"route":"CONTRACT_RERANK","intervened":changed,"preserved":False,"fallback":bool(contract.get("fallback_used")),"retrieval_latency_ms":retrieval,"ranking_latency_ms":rank,"selected_rl_action":action,"contract_state":state,"enforcement_status":enforcement.get("enforcement_status"),"blocked_rows":enforcement.get("blocked_rows"),"agent_trace_rows":len(trace),"judgment_features_received":False}
    return rows,meta

def gated_cortex(query:str,candidates:pd.DataFrame,top_k:int, *, allow_unsafe_debug:bool=False):
    require_safe(LEGACY_GATE_THRESHOLDS,allow_unsafe_debug=allow_unsafe_debug)
    request=_request(query,candidates); start=time.perf_counter(); baseline_rows=_lexical_rows(query,request.ranker_frame()); retrieval=(time.perf_counter()-start)*1000
    cortex_rows,cortex_meta=always_cortex(query,candidates,top_k,allow_unsafe_debug=allow_unsafe_debug); rank_start=time.perf_counter()
    baseline=pd.DataFrame(baseline_rows); baseline["baseline_score"]=baseline["score"]; baseline["semantic_score"]=baseline["score"]
    cortex=pd.DataFrame(cortex_rows).drop(columns=["esci_label"],errors="ignore")
    contract=_local_contract(query,baseline.head(8).to_dict("records")); gated,decision=select_slate_with_baseline_preservation_gate(baseline,cortex,contract=contract,top_k=min(5,len(baseline)))
    rows=gated.to_dict("records")
    used={str(row.get("product_id")) for row in rows}; rows.extend(dict(row) for row in baseline_rows if str(row.get("product_id")) not in used)
    validate_output_subset(request,rows); baseline_ids=[str(r.get("product_id")) for r in baseline_rows]; output_ids=[str(r.get("product_id")) for r in rows]
    gate=summarize_gate_for_logging(decision); preserved=decision.decision=="preserve_baseline"; changed=output_ids!=baseline_ids
    return rows,{"route":"PRESERVE" if preserved else "CONTRACT_RERANK","intervened":changed and not preserved,"preserved":preserved,"fallback":bool(cortex_meta.get("fallback")),"retrieval_latency_ms":retrieval,"ranking_latency_ms":float(cortex_meta.get("ranking_latency_ms",0))+(time.perf_counter()-rank_start)*1000,"gate_decision":decision.decision,"gate_reason":decision.reason,"gate_inputs":gate,"task1_completion":"gate-selected top-5 followed by untouched baseline remainder","judgment_features_received":False}

REGISTRY={
 "fts_baseline":MethodSpec("fts_baseline",True,"Parity: wraps FullEsciRetrievalEngine.score_product on the official supplied candidate set.",fts_baseline),
 "strict_filter":MethodSpec("strict_filter",True,"Parity: applies existing hybrid apply_strict_constraint_filter after the lexical candidate scorer.",strict_filter),
 "strict_boost":MethodSpec("strict_boost",True,"redundant_for_official_rerank: 99/100 complete permutations identical to strict_filter on the seeded calibration sample; retained for open_corpus.",strict_boost),
 "clean_contract_router_v1":MethodSpec("clean_contract_router_v1",True,"Deterministic SearchContractV1 router limited to PRESERVE and STRICT_FILTER.",clean_contract_router_v1),
 "always_contract_rerank_v1":MethodSpec("always_contract_rerank_v1",True,"Applies deterministic label-safe contract_rerank_v1 to every fixed candidate set.",always_contract_rerank_v1),
 "clean_contract_router_v2":MethodSpec("clean_contract_router_v2",True,"Deterministic three-route SearchContractV1 router with label-safe contract reranking.",clean_contract_router_v2),
 "current_reranker":MethodSpec("current_reranker",False,"Legacy policy compiler directly consumes esci_label; faithful label-safe parity is impossible without changing ranking semantics.",None),
 "always_cortex":MethodSpec("always_cortex",True,"legacy_contaminated_for_paper_eval: historical Q-table used ESCI-label-derived reward.",always_cortex,True),
 "gated_cortex":MethodSpec("gated_cortex",True,"legacy_contaminated_for_paper_eval: gate calibration provenance is not proven train_fit-only.",gated_cortex,True),
 "semantic_baseline":MethodSpec("semantic_baseline",False,"Full provenance, model identity, corpus coverage, and reproducibility are not established.",None),
}

def enabled_methods(names:list[str], *, allow_unsafe_contaminated_state:bool=False):
    unknown=[name for name in names if name not in REGISTRY]
    if unknown: raise ValueError(f"Unknown methods: {unknown}")
    disabled={name:REGISTRY[name].reason for name in names if not REGISTRY[name].enabled or (REGISTRY[name].contaminated and not allow_unsafe_contaminated_state)}
    specs=[]
    for name in names:
        spec=REGISTRY[name]
        if not spec.enabled or (spec.contaminated and not allow_unsafe_contaminated_state): continue
        if spec.contaminated:
            original=spec.evaluator
            specs.append(MethodSpec(spec.name,True,spec.reason,lambda q,c,k,_fn=original:_fn(q,c,k,allow_unsafe_debug=True),True))
        else: specs.append(spec)
    return specs,disabled
