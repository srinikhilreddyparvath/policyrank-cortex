from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from src.full_esci_retrieval_engine import clean_text, tokenize
from src.paper_eval.adapter import FORBIDDEN_RANKER_COLUMNS, CandidateBoundaryError
from src.paper_eval.contracts_v1 import SearchContractV1

RERANKER_VERSION = "contract_rerank_v1.0.0"


@dataclass(frozen=True)
class ContractRerankConfig:
    baseline_weight: float = 0.65
    positive_coverage_weight: float = 0.20
    product_type_weight: float = 0.10
    brand_weight: float = 0.03
    price_weight: float = 0.02
    provenance: str = "a_priori_interpretable_v1; no outcome-based tuning"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


DEFAULT_CONFIG = ContractRerankConfig()


def _coverage(terms: tuple[str, ...], tokens: set[str]) -> float:
    return 0.0 if not terms else sum(term in tokens for term in terms) / len(terms)


def _normalized_baselines(rows: list[dict]) -> list[float]:
    scores=[float(row.get("score",0.0)) for row in rows]
    if not scores: return []
    high=max(scores)
    if high<=0: return [0.5]*len(scores)
    return [max(score,0.0)/high for score in scores]


def contract_rerank_v1(rows: list[dict], contract: SearchContractV1, config: ContractRerankConfig = DEFAULT_CONFIG) -> tuple[list[dict], dict]:
    """Rerank label-free candidates; hard exclusions are diagnostic only here."""
    for row in rows:
        forbidden=FORBIDDEN_RANKER_COLUMNS.intersection(row)
        if forbidden: raise CandidateBoundaryError(f"Evaluator judgments reached contract_rerank_v1: {sorted(forbidden)}")
    baselines=_normalized_baselines(rows); scored=[]
    totals={"soft_preference_matches":0,"must_have_matches":0,"must_not_have_violations":0,"brand_matches":0,"price_signal_matches":0,"product_type_matches":0}
    product_type_terms=tuple(tokenize(contract.product_type or ""))
    for position,(row,baseline_norm) in enumerate(zip(rows,baselines)):
        text=clean_text(" ".join(str(row.get(key) or "") for key in ("product_title","product_brand","product_description","product_bullet_point","product_color")))
        tokens=set(tokenize(text)); title_tokens=set(tokenize(clean_text(row.get("product_title")))); brand_tokens=set(tokenize(clean_text(row.get("product_brand"))))
        positive=_coverage(contract.positive_terms,tokens)
        product_type=_coverage(product_type_terms,title_tokens) if product_type_terms else 0.0
        brand=1.0 if contract.brand_signal and contract.brand_signal in brand_tokens else 0.0
        price=1.0 if contract.price_signal and contract.price_signal in tokens else 0.0
        must_have=sum(term in tokens for term in contract.must_have)
        violations=sum(term in tokens for term in contract.must_not_have)
        contract_score=(config.baseline_weight*baseline_norm + config.positive_coverage_weight*positive +
                        config.product_type_weight*product_type + config.brand_weight*brand + config.price_weight*price)
        enriched=dict(row,baseline_score=float(row.get("score",0.0)),contract_score=contract_score,score=contract_score,
                      positive_term_coverage=positive,product_type_compatibility=product_type,brand_compatibility=brand,
                      price_compatibility=price,must_not_have_violation_count=violations,_baseline_position=position)
        scored.append(enriched); totals["soft_preference_matches"]+=int(positive>0); totals["must_have_matches"]+=must_have
        totals["must_not_have_violations"]+=violations; totals["brand_matches"]+=int(brand); totals["price_signal_matches"]+=int(price); totals["product_type_matches"]+=int(product_type>0)
    output=sorted(scored,key=lambda row:(-float(row["contract_score"]),int(row["_baseline_position"]),str(row.get("product_id",""))))
    new_positions={str(row.get("product_id")):index for index,row in enumerate(output)}
    movements=[abs(index-new_positions[str(row.get("product_id"))]) for index,row in enumerate(rows)]
    scores=[float(row["contract_score"]) for row in output]
    for row in output: row.pop("_baseline_position",None)
    diagnostics={**totals,"hard_compatibility_changes":0,"contract_score_range":(max(scores)-min(scores)) if scores else 0.0,
                 "mean_candidate_movement":sum(movements)/len(movements) if movements else 0.0,
                 "top_1_changed":bool(rows and output and str(rows[0].get("product_id"))!=str(output[0].get("product_id"))),
                 "rerank_changed":[str(row.get("product_id")) for row in rows]!=[str(row.get("product_id")) for row in output],
                 "reranker_version":RERANKER_VERSION,"rerank_config":config.to_dict(),
                 "hard_constraint_policy":"diagnostic_only; STRICT_FILTER owns enforcement"}
    return output,diagnostics
