from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from src.full_esci_retrieval_engine import clean_text
from src.paper_eval.contract_esci import SELECTION_SEED, stable_order_key

CF_ESCI_VERSION="cf_esci_feasibility_v1.0.0"
GROUND_TRUTH_STATUS="NOT_HUMAN_GROUND_TRUTH"
COUNTERFACTUAL_STATUSES={"PROPOSED","REJECTED","NEEDS_HUMAN_REVIEW"}
OPERATIONS={"REPLACE","ADD","REMOVE","NEGATE"}
EVIDENCE_STATES={"SUPPORTS_OLD","SUPPORTS_NEW","CONTRADICTS_NEW","UNKNOWN","NOT_APPLICABLE"}
REJECTION_REASONS={"NO_ATOMIC_REQUIREMENT","NO_GROUNDED_REPLACEMENT","MULTIPLE_REQUIREMENTS_CHANGE","PRODUCT_TYPE_NOT_STABLE","INSUFFICIENT_CANDIDATE_EVIDENCE","AMBIGUOUS_SEMANTICS","UNNATURAL_COUNTERFACTUAL","OTHER_EXPLICIT_REASON"}
FAMILIES=("COLOR","BRAND","NUMERIC_SPECIFICATION","EXPLICIT_ATTRIBUTE","NEGATION_EXCLUSION")
ADDITIONAL_QUOTAS={family:50 for family in FAMILIES}

COLORS=("black","blue","brown","gray","green","orange","pink","purple","red","silver","white","yellow")
BRANDS=("nike","adidas","apple","samsung","sony","lego","disney","dell","hp","lenovo","canon","keurig","dewalt","milwaukee","stanley","kitchenaid","nintendo","xbox","playstation","pampers","gillette","colgate")
ATTRIBUTE_GROUPS={
    "material":("cotton","metal","wood","plastic","leather","silicone","stainless steel"),
    "connectivity":("wired","wireless","bluetooth"),
    "size":("small","medium","large","xl"),
    "finish":("matte","glossy","clear"),
    "scent":("scented","unscented"),
}
NUMERIC=re.compile(r"(?<![\w/])(\d+(?:\.\d+)?)\s*(inch|inches|in|mm|cm|oz|ounce|ounces|lb|lbs|pound|pounds|gb|tb|w|watt|watts|v|volt|volts|count|pack)(?!\w)",re.I)
NEGATED=re.compile(r"\b(without|no)\s+([a-z][a-z0-9-]{1,24})\b",re.I)


def _contains(text:str,value:str)->bool:
    return re.search(rf"(?<!\w){re.escape(value)}(?!\w)",text,re.I) is not None


def _replace_once(text:str,old:str,new:str)->str|None:
    pattern=re.compile(rf"(?<!\w){re.escape(old)}(?!\w)",re.I)
    if len(pattern.findall(text))!=1:return None
    return pattern.sub(new,text,count=1)


def family_hint(query:str)->str|None:
    text=clean_text(query)
    if any(_contains(text,value) for value in COLORS):return "COLOR"
    if any(_contains(text,value) for value in BRANDS):return "BRAND"
    if NUMERIC.search(text):return "NUMERIC_SPECIFICATION"
    if any(_contains(text,value) for values in ATTRIBUTE_GROUPS.values() for value in values):return "EXPLICIT_ATTRIBUTE"
    if NEGATED.search(text):return "NEGATION_EXCLUSION"
    return None


def select_additional_sources(queries:pd.DataFrame,existing_ids:set[int],*,seed:int=SELECTION_SEED)->pd.DataFrame:
    required={"query_id","query_text","source_partition"}; missing=required-set(queries)
    if missing:raise ValueError(f"Missing source columns: {sorted(missing)}")
    if not set(queries.source_partition).issubset({"policy_train","calibration"}):raise ValueError("Only policy_train/calibration allowed")
    work=queries[~queries.query_id.isin(existing_ids)].copy(); work["feasibility_family_hint"]=work.query_text.map(family_hint); work["_order"]=work.query_id.map(lambda value:stable_order_key(int(value),seed+1))
    chosen=[]
    for family,quota in ADDITIONAL_QUOTAS.items():
        candidates=work[work.feasibility_family_hint==family].sort_values("_order")
        if len(candidates)<quota:raise ValueError(f"Insufficient {family} sources: {len(candidates)} < {quota}")
        chosen.append(candidates.head(quota))
    result=pd.concat(chosen).sort_values("query_id").drop(columns="_order").reset_index(drop=True)
    if len(result)!=sum(ADDITIONAL_QUOTAS.values()) or result.query_id.duplicated().any():raise AssertionError("Additional selection failure")
    return result


def validate_counterfactual(proposal:dict[str,Any])->None:
    required={"counterfactual_id","source_query_id","source_query_text","counterfactual_query_text","invariant_product_type","changed_requirement_id","changed_requirement_type","operation","old_value","new_value","unit","requirement_strength","generation_provenance","evidence_sufficiency","counterfactual_status","ground_truth_status","changed_atom_count"}
    missing=required-set(proposal)
    if missing:raise ValueError(f"Missing counterfactual fields: {sorted(missing)}")
    if proposal["operation"] not in OPERATIONS:raise ValueError("Invalid operation")
    if proposal["counterfactual_status"] not in COUNTERFACTUAL_STATUSES:raise ValueError("Invalid counterfactual status")
    if proposal["ground_truth_status"]!=GROUND_TRUTH_STATUS:raise ValueError("Machine proposal cannot be human ground truth")
    if proposal["changed_atom_count"]!=1:raise ValueError("Exactly one requirement atom must change")
    if proposal["source_query_text"]==proposal["counterfactual_query_text"]:raise ValueError("Counterfactual query must differ")


def validate_evidence_state(state:str)->None:
    if state not in EVIDENCE_STATES:raise ValueError("Invalid directional evidence state")


def counterfactual_direction_accuracy(before:dict[str,int],after:dict[str,int],pairs:list[tuple[str,str]])->float|None:
    values=[]
    for new_id,old_id in pairs:
        if not all(pid in before and pid in after for pid in (new_id,old_id)):continue
        before_preference=before[old_id]-before[new_id]; after_preference=after[old_id]-after[new_id]
        values.append(float(after_preference>before_preference))
    return float(np.mean(values)) if values else None


def requirement_flip_consistency(before:dict[str,int],after:dict[str,int],new_ids:list[str],contradicting_ids:list[str])->float|None:
    return counterfactual_direction_accuracy(before,after,[(new_id,old_id) for new_id in new_ids for old_id in contradicting_ids])


def unaffected_candidate_stability(before:dict[str,int],after:dict[str,int],unaffected_ids:list[str])->float|None:
    ids=[pid for pid in unaffected_ids if pid in before and pid in after]; pairs=[(ids[i],ids[j]) for i in range(len(ids)) for j in range(i+1,len(ids))]
    if not pairs:return None
    concordant=sum(np.sign(before[a]-before[b])==np.sign(after[a]-after[b]) for a,b in pairs)
    return float(concordant/len(pairs))


def counterfactual_selectivity(before:dict[str,int],after:dict[str,int],affected_ids:list[str],unaffected_ids:list[str])->float|None:
    def movement(ids):return sum(abs(1/before[pid]-1/after[pid]) for pid in ids if pid in before and pid in after and before[pid]>0 and after[pid]>0)
    affected=movement(affected_ids); unaffected=movement(unaffected_ids); total=affected+unaffected
    return float(affected/total) if total else None


@dataclass(frozen=True)
class ProposalBundle:
    proposal:dict[str,Any]; evidence:list[dict[str,Any]]; pairs:list[dict[str,Any]]; unaffected:list[str]


def _candidate_text(row)->str:
    return clean_text(" ".join(str(row.get(key) or "") for key in ("product_title","product_brand","product_color","product_bullet_point","product_description")))


def _make_bundle(query_id:int,query:str,product_type:str|None,family:str,attribute:str,old:str,new:str,operation:str,rows:pd.DataFrame,unit:str|None=None,old_pattern:str|None=None,new_pattern:str|None=None)->ProposalBundle|None:
    counterfactual=_replace_once(query,old_pattern or old,new_pattern or new)
    if not counterfactual or clean_text(counterfactual)==clean_text(query):return None
    evidence=[]; new_ids=[]; old_ids=[]; unaffected=[]
    known_values=set(COLORS if family=="COLOR" else BRANDS if family=="BRAND" else next((values for name,values in ATTRIBUTE_GROUPS.items() if name==attribute),()))
    for row in rows.to_dict("records"):
        pid=str(row["product_id"]); text=_candidate_text(row); supports_new=_contains(text,new); supports_old=_contains(text,old)
        if supports_new and not supports_old:state="SUPPORTS_NEW"; new_ids.append(pid)
        elif supports_old and not supports_new:state="SUPPORTS_OLD"; old_ids.append(pid)
        elif any(_contains(text,value) for value in known_values-set((old,new))):state="CONTRADICTS_NEW"; unaffected.append(pid)
        else:state="UNKNOWN"
        evidence.append({"counterfactual_id":None,"source_query_id":query_id,"product_id":pid,"changed_requirement_family":family,"directional_state":state,"supports_old":supports_old,"supports_new":supports_new,"evidence_text":next((field for field in (str(row.get("product_color") or ""),str(row.get("product_brand") or ""),str(row.get("product_title") or ""),str(row.get("product_bullet_point") or ""),str(row.get("product_description") or "")) if _contains(clean_text(field),new) or _contains(clean_text(field),old)),None),"ground_truth_status":GROUND_TRUTH_STATUS})
    if not new_ids or not old_ids:return None
    cfid=hashlib.sha256(f"{query_id}|{family}|{clean_text(counterfactual)}".encode()).hexdigest()[:20]
    for record in evidence:record["counterfactual_id"]=cfid
    evidence_by_id={record["product_id"]:record["evidence_text"] for record in evidence}
    pairs=[{"counterfactual_id":cfid,"source_query_id":query_id,"p_new":new_id,"p_old":old_id,"p_new_evidence":evidence_by_id.get(new_id),"p_old_evidence":evidence_by_id.get(old_id),"expected_direction":"NEW_PRODUCT_SHOULD_GAIN","derivation":"catalog_evidence_only","ground_truth_status":GROUND_TRUTH_STATUS} for new_id in new_ids[:3] for old_id in old_ids[:3]]
    proposal={"counterfactual_id":cfid,"source_query_id":query_id,"source_query_text":query,"counterfactual_query_text":counterfactual,"invariant_product_type":product_type,"changed_requirement_id":"r_cf_1","changed_requirement_type":family,"operation":operation,"old_value":old,"new_value":new,"unit":unit,"requirement_strength":"hard","generation_provenance":{"generator_version":CF_ESCI_VERSION,"replacement_grounding":"old and new values explicitly evidenced by different candidates in the same fixed candidate pool","old_support_count":len(old_ids),"new_support_count":len(new_ids)},"evidence_sufficiency":"PAIR_GROUNDED","counterfactual_status":"NEEDS_HUMAN_REVIEW","ground_truth_status":GROUND_TRUTH_STATUS,"changed_atom_count":1}
    validate_counterfactual(proposal); return ProposalBundle(proposal,evidence,pairs,sorted(unaffected))


def propose_counterfactual(query_id:int,query:str,rows:pd.DataFrame,product_type:str|None)->ProposalBundle|None:
    text=clean_text(query)
    for old in COLORS:
        if _contains(text,old):
            available=[value for value in COLORS if value!=old and any(_contains(_candidate_text(row),value) for row in rows.to_dict("records"))]
            for new in sorted(available):
                bundle=_make_bundle(query_id,query,product_type,"COLOR","color",old,new,"REPLACE",rows)
                if bundle:return bundle
    for old in BRANDS:
        if _contains(text,old):
            available=[value for value in BRANDS if value!=old and any(_contains(_candidate_text(row),value) for row in rows.to_dict("records"))]
            for new in sorted(available):
                bundle=_make_bundle(query_id,query,product_type,"BRAND","brand",old,new,"REPLACE",rows)
                if bundle:return bundle
    match=NUMERIC.search(text)
    if match:
        old,unit=match.group(1),match.group(2).lower(); values=sorted({m.group(1) for row in rows.to_dict("records") for m in NUMERIC.finditer(_candidate_text(row)) if m.group(2).lower()==unit and m.group(1)!=old},key=lambda value:float(value))
        for new in values:
            bundle=_make_bundle(query_id,query,product_type,"NUMERIC_SPECIFICATION","numeric_specification",old,new,"REPLACE",rows,unit=unit)
            if bundle:return bundle
    for attribute,values in ATTRIBUTE_GROUPS.items():
        for old in values:
            if _contains(text,old):
                for new in sorted(value for value in values if value!=old and any(_contains(_candidate_text(row),value) for row in rows.to_dict("records"))):
                    bundle=_make_bundle(query_id,query,product_type,"EXPLICIT_ATTRIBUTE",attribute,old,new,"REPLACE",rows)
                    if bundle:return bundle
    negated=NEGATED.search(text)
    if negated:
        term=negated.group(2); bundle=_make_bundle(query_id,query,product_type,"NEGATION_EXCLUSION","excluded_attribute",term,term,"NEGATE",rows,old_pattern=negated.group(0),new_pattern=f"with {term}")
        if bundle:return bundle
    return None
