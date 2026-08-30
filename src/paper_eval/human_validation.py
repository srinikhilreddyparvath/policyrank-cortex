from __future__ import annotations

import hashlib
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.paper_eval.cf_esci import GROUND_TRUTH_STATUS

HUMAN_VALIDATION_VERSION="cf_esci_human_validation_v1.0.0"
SAMPLE_SEED=30
SAMPLE_SIZE=120
CALIBRATION_SIZE=20
FINAL_SIZE=100
MAX_POOL_SIZE=30
GENERATED_FAMILY_QUOTAS={"COLOR":15,"BRAND":15,"NUMERIC_SPECIFICATION":15,"EXPLICIT_ATTRIBUTE":15}
CALIBRATION_QUOTAS={
    "GENERATED":{"COLOR":3,"BRAND":2,"NUMERIC_SPECIFICATION":2,"EXPLICIT_ATTRIBUTE":3},
    "NATURAL":{"COLOR":1,"BRAND":1,"NUMERIC_SPECIFICATION":2,"EXPLICIT_ATTRIBUTE":2,"NEGATION_EXCLUSION":4},
}
QUERY_LABEL_FIELDS=("core_intent_preserved","single_requirement_change","changed_requirement_correct","requirement_family","operation","query_naturalness","semantic_plausibility","confidence","old_value_normalized","new_value_normalized")
KAPPA_FIELDS=("core_intent_preserved","single_requirement_change","requirement_family","operation","query_naturalness")
FORBIDDEN_ANNOTATOR_TOKENS=("esci_label","ndcg","rank","score","route","oracle","cortex","method","winner","expected_direction_machine","machine_confidence")
ADJUDICATION_REASONS=("GUIDELINE_CLARIFICATION","ATTRIBUTE_ROLE_AMBIGUITY","PRODUCT_TYPE_DISAGREEMENT","INSUFFICIENT_METADATA","DIRECTION_AMBIGUITY","QUERY_UNNATURAL","MULTIPLE_REQUIREMENTS_CHANGED","OTHER")


def stable_key(example_id:str,seed:int=SAMPLE_SEED)->str:return hashlib.sha256(f"{seed}:{example_id}".encode()).hexdigest()


def normalize_generated(records:list[dict])->pd.DataFrame:
    rows=[]
    for p in records:
        rows.append({"example_id":"G_"+p["counterfactual_id"],"source_type":"GENERATED","source_query_id_a":int(p["source_query_id"]),"source_query_id_b":None,"query_a":p["source_query_text"],"query_b":p["counterfactual_query_text"],"invariant_product_type":p["invariant_product_type"],"requirement_family":p["changed_requirement_type"],"operation":p["operation"],"old_value":p["old_value"],"new_value":p["new_value"],"source_record_id":p["counterfactual_id"]})
    return pd.DataFrame(rows)


def normalize_natural(records:list[dict])->pd.DataFrame:
    rows=[]
    for p in records:
        rows.append({"example_id":"N_"+p["canonical_pair_id"],"source_type":"NATURAL","source_query_id_a":int(p["query_id_a"]),"source_query_id_b":int(p["query_id_b"]),"query_a":p["query_text_a"],"query_b":p["query_text_b"],"invariant_product_type":p["product_type_a"],"requirement_family":p["requirement_family"],"operation":p["operation"],"old_value":p["old_value"],"new_value":p["new_value"],"source_record_id":p["canonical_pair_id"]})
    return pd.DataFrame(rows)


def select_human_validation_sample(generated:pd.DataFrame,natural:pd.DataFrame)->tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    chosen=[]
    for family,quota in GENERATED_FAMILY_QUOTAS.items():
        candidates=generated[generated.requirement_family==family].copy(); candidates["_key"]=candidates.example_id.map(stable_key)
        if len(candidates)<quota:raise ValueError(f"Insufficient generated {family}: {len(candidates)}")
        chosen.append(candidates.sort_values("_key").head(quota).drop(columns="_key"))
    generated_selected=pd.concat(chosen); natural_selected=natural.copy()
    if len(natural_selected)!=60:raise ValueError(f"Expected all 60 natural pairs, got {len(natural_selected)}")
    sample=pd.concat([generated_selected,natural_selected],ignore_index=True).sort_values("example_id").reset_index(drop=True)
    if len(sample)!=SAMPLE_SIZE or sample.example_id.duplicated().any():raise AssertionError("Human-validation sample failure")
    calibration=[]
    for source,quotas in CALIBRATION_QUOTAS.items():
        for family,quota in quotas.items():
            candidates=sample[(sample.source_type==source)&(sample.requirement_family==family)].copy(); candidates["_key"]=candidates.example_id.map(lambda value:stable_key(value,SAMPLE_SEED+1))
            if len(candidates)<quota:raise ValueError(f"Insufficient calibration {source}/{family}")
            calibration.append(candidates.sort_values("_key").head(quota).drop(columns="_key"))
    calibration_frame=pd.concat(calibration).sort_values("example_id").reset_index(drop=True); final=sample[~sample.example_id.isin(calibration_frame.example_id)].reset_index(drop=True)
    if len(calibration_frame)!=CALIBRATION_SIZE or len(final)!=FINAL_SIZE:raise AssertionError("Calibration/final split failure")
    return sample,calibration_frame,final


def blank_query_template(examples:pd.DataFrame,annotator_id:str)->pd.DataFrame:
    visible=examples[["example_id","query_a","query_b","invariant_product_type","requirement_family","operation","old_value","new_value"]].copy(); visible=visible.rename(columns={"invariant_product_type":"proposed_invariant_product_type","requirement_family":"proposed_requirement_family","operation":"proposed_operation","old_value":"proposed_old_value","new_value":"proposed_new_value"}); visible.insert(1,"annotator_id",annotator_id)
    for field in QUERY_LABEL_FIELDS:visible[field]=""
    return visible


def assert_annotation_blind(frame:pd.DataFrame)->None:
    lowered={str(column).lower() for column in frame.columns}
    violations=[token for token in FORBIDDEN_ANNOTATOR_TOKENS if token in lowered]
    if violations:raise ValueError(f"Annotator-facing columns violate blindness: {violations}")


def accepted_query_label(row:dict)->bool:
    return row.get("core_intent_preserved")=="YES" and row.get("single_requirement_change")=="YES" and row.get("changed_requirement_correct")=="YES" and row.get("query_naturalness")=="YES" and row.get("semantic_plausibility")!="NO"


def direction_eligible(value:str)->bool:return value not in {"INSUFFICIENT_EVIDENCE","NO_REQUIRED_DIRECTION","",None}


def raw_agreement(left:Iterable,right:Iterable)->float|None:
    pairs=[(str(a),str(b)) for a,b in zip(left,right) if pd.notna(a) and pd.notna(b) and str(a)!="" and str(b)!=""]
    return float(np.mean([a==b for a,b in pairs])) if pairs else None


def cohens_kappa(left:Iterable,right:Iterable)->float|None:
    pairs=[(str(a),str(b)) for a,b in zip(left,right) if pd.notna(a) and pd.notna(b) and str(a)!="" and str(b)!=""]
    if not pairs:return None
    labels=sorted({value for pair in pairs for value in pair}); observed=np.mean([a==b for a,b in pairs]); n=len(pairs); ca=Counter(a for a,_ in pairs); cb=Counter(b for _,b in pairs); expected=sum((ca[label]/n)*(cb[label]/n) for label in labels)
    return float((observed-expected)/(1-expected)) if expected<1 else (1.0 if observed==1 else None)


def prevalence(frame:pd.DataFrame,field:str,annotator_id:str)->list[dict]:
    values=frame[field].dropna().astype(str); values=values[values!=""]; counts=values.value_counts(); total=len(values)
    return [{"field":field,"annotator_id":annotator_id,"label":label,"count":int(count),"prevalence":float(count/total)} for label,count in counts.sort_index().items()] if total else []


def analyze_independent_annotations(a:pd.DataFrame,b:pd.DataFrame)->tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    keys=["example_id"]; merged=a.merge(b,on=keys,suffixes=("_A","_B"),validate="one_to_one"); summary=[]; prev=[]
    for field in QUERY_LABEL_FIELDS:
        left=merged[f"{field}_A"]; right=merged[f"{field}_B"]; summary.append({"field":field,"n":int(((left.astype(str)!="")&(right.astype(str)!="")).sum()),"raw_agreement":raw_agreement(left,right),"cohens_kappa":cohens_kappa(left,right) if field in KAPPA_FIELDS else None}); prev.extend(prevalence(a,field,"A")); prev.extend(prevalence(b,field,"B"))
    adjudication=[]
    for row in merged.to_dict("records"):
        for field in QUERY_LABEL_FIELDS:
            adjudication.append({"example_id":row["example_id"],"field":field,"annotator_A_raw":row.get(f"{field}_A"),"annotator_B_raw":row.get(f"{field}_B"),"adjudicated_label":"","adjudication_reason":"","adjudication_notes":""})
    direction=pd.DataFrame(columns=["field","n","raw_agreement","cohens_kappa"])
    return pd.DataFrame(summary),pd.DataFrame(prev),direction,pd.DataFrame(adjudication)


def analyze_direction_annotations(a:pd.DataFrame,b:pd.DataFrame)->tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    merged=a.merge(b,on=["direction_item_id"],suffixes=("_A","_B"),validate="one_to_one"); summary=[]; prev=[]
    for field in ("candidate_direction","direction_confidence"):
        left=merged[f"{field}_A"]; right=merged[f"{field}_B"]
        summary.append({"field":field,"n":int(((left.astype(str)!="")&(right.astype(str)!="")).sum()),"raw_agreement":raw_agreement(left,right),"cohens_kappa":cohens_kappa(left,right) if field=="candidate_direction" else None})
        prev.extend(prevalence(a,field,"A")); prev.extend(prevalence(b,field,"B"))
    adjudication=[{"direction_item_id":row["direction_item_id"],"field":field,"annotator_A_raw":row.get(f"{field}_A"),"annotator_B_raw":row.get(f"{field}_B"),"adjudicated_label":"","adjudication_reason":"","adjudication_notes":""} for row in merged.to_dict("records") for field in ("candidate_direction","direction_confidence")]
    return pd.DataFrame(summary),pd.DataFrame(prev),pd.DataFrame(adjudication)


def accepted_benchmark_pairs(examples:pd.DataFrame,adjudicated_query:pd.DataFrame,adjudicated_direction:pd.DataFrame)->pd.DataFrame:
    labels=adjudicated_query.pivot(index="example_id",columns="field",values="adjudicated_label").reset_index()
    accepted=labels[labels.apply(lambda row:accepted_query_label(row.to_dict()),axis=1)].copy()
    directions=adjudicated_direction[(adjudicated_direction.field=="candidate_direction")&adjudicated_direction.adjudicated_label.map(direction_eligible)][["direction_item_id","adjudicated_label"]].copy()
    directions["example_id"]=directions.direction_item_id.str.replace(r"_D\d+$","",regex=True)
    counts=directions.groupby("example_id").size().rename("eligible_direction_count")
    result=examples.merge(accepted[["example_id"]],on="example_id",how="inner").merge(counts,on="example_id",how="left"); result["eligible_direction_count"]=result.eligible_direction_count.fillna(0).astype(int); result["human_validation_status"]="ADJUDICATED_ACCEPTED"; result["annotator_count"]=2; result["adjudication_status"]="COMPLETE"
    return result
