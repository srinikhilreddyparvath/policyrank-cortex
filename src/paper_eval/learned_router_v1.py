from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np

from src.full_esci_retrieval_engine import clean_text, tokenize
from src.paper_eval.adapter import FORBIDDEN_RANKER_COLUMNS, CandidateBoundaryError
from src.paper_eval.contracts_v1 import SearchContractV1

GENERIC_FEATURE_VERSION="generic_router_features_v1.0.0"
CONTRACT_FEATURE_VERSION="contract_router_features_v1.0.0"
LEARNED_ROUTER_VERSION="ridge_delta_router_v1.0.0"

GENERIC_FEATURES=(
    "query_token_count","query_character_length","candidate_count","baseline_top_score",
    "baseline_score_mean","baseline_score_std","baseline_top1_top2_margin","baseline_top1_top5_margin",
    "candidate_query_coverage_mean","candidate_query_coverage_std","candidate_query_coverage_max",
)
CONTRACT_ONLY_FEATURES=(
    "contract_status_resolved","contract_status_partial","product_type_resolved","positive_term_count",
    "negative_term_count","must_have_count","must_not_have_count","hard_exclusion_present",
    "brand_signal_present","price_signal_present","constraint_strength_hard","constraint_strength_soft",
    "ambiguity_low","ambiguity_medium","ambiguity_high","candidate_positive_coverage_mean",
    "candidate_positive_coverage_max","candidate_product_type_coverage_mean","candidate_product_type_coverage_max",
)
CONTRACT_FEATURES=GENERIC_FEATURES+CONTRACT_ONLY_FEATURES


def _coverage(terms:Iterable[str],tokens:set[str])->float:
    values=tuple(terms); return 0.0 if not values else sum(term in tokens for term in values)/len(values)


def extract_router_features(query:str, baseline_rows:list[dict], contract:SearchContractV1)->tuple[dict[str,float],dict[str,float]]:
    for row in baseline_rows:
        forbidden=FORBIDDEN_RANKER_COLUMNS.intersection(row)
        if forbidden: raise CandidateBoundaryError(f"Evaluator judgments reached learned-router feature extraction: {sorted(forbidden)}")
    query_tokens=tokenize(clean_text(query)); scores=np.asarray([float(row.get("score",0.0)) for row in baseline_rows],dtype=float)
    ordered=np.sort(scores)[::-1]; coverages=np.asarray([float(row.get("query_token_coverage",0.0)) for row in baseline_rows],dtype=float)
    generic={"query_token_count":float(len(query_tokens)),"query_character_length":float(len(str(query))),"candidate_count":float(len(baseline_rows)),
        "baseline_top_score":float(ordered[0]) if len(ordered) else 0.0,"baseline_score_mean":float(scores.mean()) if len(scores) else 0.0,
        "baseline_score_std":float(scores.std()) if len(scores) else 0.0,"baseline_top1_top2_margin":float(ordered[0]-ordered[1]) if len(ordered)>1 else 0.0,
        "baseline_top1_top5_margin":float(ordered[0]-ordered[min(4,len(ordered)-1)]) if len(ordered) else 0.0,
        "candidate_query_coverage_mean":float(coverages.mean()) if len(coverages) else 0.0,"candidate_query_coverage_std":float(coverages.std()) if len(coverages) else 0.0,
        "candidate_query_coverage_max":float(coverages.max()) if len(coverages) else 0.0}
    positive=[]; product=[]; product_terms=tuple(tokenize(contract.product_type or ""))
    for row in baseline_rows:
        tokens=set(tokenize(clean_text(" ".join(str(row.get(key) or "") for key in ("product_title","product_brand","product_description","product_bullet_point","product_color")))))
        title=set(tokenize(clean_text(row.get("product_title")))); positive.append(_coverage(contract.positive_terms,tokens)); product.append(_coverage(product_terms,title))
    contract_only={"contract_status_resolved":float(contract.contract_status=="resolved"),"contract_status_partial":float(contract.contract_status=="partial"),
        "product_type_resolved":float(bool(contract.product_type)),"positive_term_count":float(len(contract.positive_terms)),"negative_term_count":float(len(contract.negative_terms)),
        "must_have_count":float(len(contract.must_have)),"must_not_have_count":float(len(contract.must_not_have)),"hard_exclusion_present":float(bool(contract.must_not_have)),
        "brand_signal_present":float(bool(contract.brand_signal)),"price_signal_present":float(bool(contract.price_signal)),"constraint_strength_hard":float(contract.constraint_strength=="hard"),
        "constraint_strength_soft":float(contract.constraint_strength=="soft"),"ambiguity_low":float(contract.ambiguity=="low"),"ambiguity_medium":float(contract.ambiguity=="medium"),
        "ambiguity_high":float(contract.ambiguity=="high"),"candidate_positive_coverage_mean":float(np.mean(positive)) if positive else 0.0,
        "candidate_positive_coverage_max":float(np.max(positive)) if positive else 0.0,"candidate_product_type_coverage_mean":float(np.mean(product)) if product else 0.0,
        "candidate_product_type_coverage_max":float(np.max(product)) if product else 0.0}
    return generic,{**generic,**contract_only}


@dataclass
class RidgeModel:
    feature_names:list[str]; alpha:float; mean:list[float]; scale:list[float]; coefficients:list[float]; intercept:float
    def predict(self,frame)->np.ndarray:
        x=frame[self.feature_names].to_numpy(dtype=float); return ((x-np.asarray(self.mean))/np.asarray(self.scale))@np.asarray(self.coefficients)+self.intercept
    def to_dict(self): return asdict(self)


def fit_ridge(frame,features:tuple[str,...],target:str,alpha:float)->RidgeModel:
    x=frame[list(features)].to_numpy(dtype=float); y=frame[target].to_numpy(dtype=float); mean=x.mean(axis=0); scale=x.std(axis=0); scale[scale==0]=1.0; z=(x-mean)/scale; centered=y-y.mean()
    coefficients=np.linalg.solve(z.T@z+alpha*np.eye(z.shape[1]),z.T@centered)
    return RidgeModel(list(features),float(alpha),mean.tolist(),scale.tolist(),coefficients.tolist(),float(y.mean()))


def choose_routes(filter_delta:np.ndarray,rerank_delta:np.ndarray,threshold:float)->np.ndarray:
    routes=np.full(len(filter_delta),"PRESERVE",dtype=object); best=np.maximum(filter_delta,rerank_delta); active=best>threshold
    routes[active & (filter_delta>=rerank_delta)]="STRICT_FILTER"; routes[active & (rerank_delta>filter_delta)]="CONTRACT_RERANK"; return routes
