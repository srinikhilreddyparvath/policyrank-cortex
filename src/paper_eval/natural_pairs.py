from __future__ import annotations

import hashlib
import itertools
import re
from dataclasses import dataclass
from fractions import Fraction
from typing import Any

import pandas as pd

from src.full_esci_retrieval_engine import clean_text
from src.paper_eval.cf_esci import ATTRIBUTE_GROUPS,BRANDS,COLORS,GROUND_TRUTH_STATUS

NATURAL_PAIR_VERSION="cf_esci_natural_pairs_v1.0.0"
VALIDATION_STATUS="NEEDS_HUMAN_REVIEW"
FAMILIES=("COLOR","BRAND","NUMERIC_SPECIFICATION","EXPLICIT_ATTRIBUTE","NEGATION_EXCLUSION","QUANTITY_PACK_SIZE")
OPERATIONS={"REPLACE","ADD","REMOVE","NEGATE"}
EVIDENCE_STATES={"SUPPORTS_A","SUPPORTS_B","CONTRADICTS_A","CONTRADICTS_B","UNKNOWN","NOT_APPLICABLE"}
UNIT_ROLES={"inch":"length","inches":"length","in":"length","mm":"length","cm":"length","oz":"capacity_or_weight","ounce":"capacity_or_weight","ounces":"capacity_or_weight","lb":"weight","lbs":"weight","pound":"weight","pounds":"weight","gb":"digital_capacity","tb":"digital_capacity","w":"power","watt":"power","watts":"power","v":"voltage","volt":"voltage","volts":"voltage"}
NATURAL_NUMERIC=re.compile(r"(?<!\w)(\d+/\d+|\d+(?:\.\d+)?)\s*(inch|inches|in|mm|cm|oz|ounce|ounces|lb|lbs|pound|pounds|gb|tb|w|watt|watts|v|volt|volts)(?!\w)",re.I)
PACK_PATTERNS=(re.compile(r"\bpack\s+of\s+(\d+)\b",re.I),re.compile(r"\b(\d+)\s*[- ]?(pack|count|ct)\b",re.I))
NEGATION=re.compile(r"\b(without|no|not|excluding|except)\s+([a-z][a-z0-9-]{1,30})\b",re.I)
WITH=re.compile(r"\bwith\s+([a-z][a-z0-9-]{1,30})\b",re.I)


@dataclass(frozen=True)
class Atom:
    family:str; attribute:str; value:str; normalized_value:str; start:int; end:int; unit:str|None=None; semantic_role:str|None=None; polarity:str="positive"


def normalize_number(value:str)->str:
    result=Fraction(value); return str(result.numerator) if result.denominator==1 else f"{float(result):.8g}"


def extract_atoms(query:str)->list[Atom]:
    text=clean_text(query); atoms=[]
    for value in COLORS:
        for match in re.finditer(rf"(?<!\w){re.escape(value)}(?!\w)",text):atoms.append(Atom("COLOR","color",value,value,match.start(),match.end()))
    for value in BRANDS:
        for match in re.finditer(rf"(?<!\w){re.escape(value)}(?!\w)",text):atoms.append(Atom("BRAND","brand",value,value,match.start(),match.end()))
    for match in NATURAL_NUMERIC.finditer(text):
        unit=match.group(2).lower(); atoms.append(Atom("NUMERIC_SPECIFICATION","numeric_specification",match.group(1),normalize_number(match.group(1)),match.start(1),match.end(1),unit,UNIT_ROLES[unit]))
    for pattern in PACK_PATTERNS:
        for match in pattern.finditer(text):
            value=match.group(1); atoms.append(Atom("QUANTITY_PACK_SIZE","quantity",value,normalize_number(value),match.start(1),match.end(1),"count","quantity"))
    for attribute,values in ATTRIBUTE_GROUPS.items():
        for value in values:
            for match in re.finditer(rf"(?<!\w){re.escape(value)}(?!\w)",text):atoms.append(Atom("EXPLICIT_ATTRIBUTE",attribute,value,value,match.start(),match.end()))
    for match in NEGATION.finditer(text):atoms.append(Atom("NEGATION_EXCLUSION","excluded_attribute",match.group(2),match.group(2).lower(),match.start(),match.end(),polarity="negative"))
    # Prefer the longest atom when controlled vocabularies overlap (e.g. steel / stainless steel).
    atoms=sorted(atoms,key=lambda atom:(atom.start,-(atom.end-atom.start),atom.family)); accepted=[]
    for atom in atoms:
        if not any(atom.start<other.end and other.start<atom.end for other in accepted):accepted.append(atom)
    return accepted


def query_skeleton(query:str,atom:Atom,placeholder:str|None=None,remove:bool=False)->str:
    text=clean_text(query); replacement="" if remove else placeholder or f"<{atom.family}>"; return " ".join((text[:atom.start]+replacement+text[atom.end:]).split())


def canonical_pair_id(query_id_a:int,query_id_b:int,family:str)->str:
    low,high=sorted((int(query_id_a),int(query_id_b))); return hashlib.sha256(f"{low}|{high}|{family}".encode()).hexdigest()[:20]


def _representatives(queries:pd.DataFrame)->pd.DataFrame:
    work=queries.copy(); work["normalized_query"]=work.query_text.map(clean_text); return work.sort_values("query_id").drop_duplicates("normalized_query",keep="first")


def mine_raw_pairs(queries:pd.DataFrame)->tuple[list[dict[str,Any]],list[dict[str,Any]]]:
    required={"query_id","query_text","source_partition","product_type"}; missing=required-set(queries)
    if missing:raise ValueError(f"Missing natural mining columns: {sorted(missing)}")
    if not set(queries.source_partition).issubset({"policy_train","calibration"}):raise ValueError("Natural mining accepts only policy_train/calibration")
    work=_representatives(queries); by_text={clean_text(row.query_text):row for row in work.itertuples(index=False)}; replacement={}; raw=[]; failures=[]
    for row in work.itertuples(index=False):
        atoms=extract_atoms(row.query_text)
        for atom in atoms:
            key=(atom.family,atom.attribute,atom.unit,atom.semantic_role,query_skeleton(row.query_text,atom))
            replacement.setdefault(key,[]).append((row,atom))
            base=query_skeleton(row.query_text,atom,remove=True)
            if base and base in by_text:
                other=by_text[base]
                a,b=sorted((row,other),key=lambda item:int(item.query_id)); operation="ADD" if int(a.query_id)==int(other.query_id) else "REMOVE"
                raw.append(_raw_record(a,b,atom.family,operation,None if operation=="ADD" else atom.normalized_value,atom.normalized_value if operation=="ADD" else None,atom,base))
        neg=NEGATION.search(clean_text(row.query_text))
        if neg:
            alternate=NEGATION.sub(f"with {neg.group(2)}",clean_text(row.query_text),count=1)
            if alternate in by_text:
                other=by_text[alternate]; a,b=sorted((row,other),key=lambda item:int(item.query_id)); raw.append(_raw_record(a,b,"NEGATION_EXCLUSION","NEGATE",neg.group(2),neg.group(2),Atom("NEGATION_EXCLUSION","excluded_attribute",neg.group(2),neg.group(2),neg.start(),neg.end(),polarity="negative"),f"<POLARITY> {neg.group(2)}"))
    for key,items in replacement.items():
        by_value={}
        for row,atom in items:by_value.setdefault(atom.normalized_value,(row,atom))
        for (_,left),(_,right) in itertools.combinations(sorted(by_value.items()),2):
            a,b=sorted((left[0],right[0]),key=lambda item:int(item.query_id)); atom_a=left[1] if int(a.query_id)==int(left[0].query_id) else right[1]; atom_b=right[1] if int(b.query_id)==int(right[0].query_id) else left[1]
            raw.append(_raw_record(a,b,key[0],"REPLACE",atom_a.normalized_value,atom_b.normalized_value,atom_a,key[-1],new_atom=atom_b))
    unique={}
    for record in raw:
        key=record["canonical_pair_id"]
        if key not in unique:unique[key]=record
        else:failures.append({"query_id_a":record["query_id_a"],"query_id_b":record["query_id_b"],"reason":"DUPLICATE_CANONICAL_PAIR","family":record["requirement_family"]})
    return sorted(unique.values(),key=lambda row:(row["query_id_a"],row["query_id_b"],row["requirement_family"])),failures


def _raw_record(a,b,family,operation,old_value,new_value,atom,skeleton,new_atom=None):
    return {"canonical_pair_id":canonical_pair_id(a.query_id,b.query_id,family),"query_id_a":int(a.query_id),"query_text_a":str(a.query_text),"query_id_b":int(b.query_id),"query_text_b":str(b.query_text),"product_type_a":a.product_type,"product_type_b":b.product_type,"requirement_family":family,"changed_attribute":atom.attribute,"operation":operation,"old_value":old_value,"new_value":new_value,"unit":atom.unit,"semantic_role":atom.semantic_role,"normalized_skeleton":skeleton,"changed_atom_count":1,"ground_truth_status":GROUND_TRUTH_STATUS,"validation_status":VALIDATION_STATUS}


def validate_natural_pair(record:dict[str,Any])->None:
    if record["query_id_a"]==record["query_id_b"] or clean_text(record["query_text_a"])==clean_text(record["query_text_b"]):raise ValueError("Natural queries must be distinct")
    if record["operation"] not in OPERATIONS:raise ValueError("Invalid natural operation")
    if record["changed_atom_count"]!=1:raise ValueError("Natural pair must change one atom")
    if not record["product_type_a"] or record["product_type_a"]!=record["product_type_b"]:raise ValueError("Product type invariant failed")
    if record["ground_truth_status"]!=GROUND_TRUTH_STATUS or record["validation_status"]!=VALIDATION_STATUS:raise ValueError("Natural pair cannot be machine ground truth")


def filter_valid_pairs(raw:list[dict[str,Any]])->tuple[list[dict[str,Any]],list[dict[str,Any]]]:
    valid=[]; failures=[]
    for record in raw:
        try:validate_natural_pair(record); valid.append(record)
        except ValueError as exc:failures.append({**record,"reason":str(exc)})
    return valid,failures


def candidate_text(row:dict[str,Any])->str:
    return clean_text(" ".join(str(row.get(key) or "") for key in ("product_title","product_brand","product_color","product_bullet_point","product_description")))


def _known_values(record:dict[str,Any])->tuple[str,...]:
    family=record["requirement_family"]
    if family=="COLOR":return COLORS
    if family=="BRAND":return BRANDS
    if family=="EXPLICIT_ATTRIBUTE":return ATTRIBUTE_GROUPS.get(record["changed_attribute"],())
    return ()


def build_candidate_evidence(record:dict[str,Any],candidate_rows:pd.DataFrame)->tuple[list[dict[str,Any]],list[dict[str,Any]],list[str]]:
    old=record.get("old_value"); new=record.get("new_value"); family=record["requirement_family"]; evidence=[]; supports_a=[]; supports_b=[]; locality=[]; known=_known_values(record)
    query_a_negative=bool(NEGATION.search(clean_text(record["query_text_a"]))); query_b_negative=bool(NEGATION.search(clean_text(record["query_text_b"])))
    for row in candidate_rows.drop_duplicates("product_id").to_dict("records"):
        pid=str(row["product_id"]); text=candidate_text(row); has_old=bool(old and re.search(rf"(?<!\w){re.escape(str(old))}(?!\w)",text)); has_new=bool(new and re.search(rf"(?<!\w){re.escape(str(new))}(?!\w)",text))
        if family in {"NUMERIC_SPECIFICATION","QUANTITY_PACK_SIZE"}:
            numeric_atoms=[atom for atom in extract_atoms(text) if atom.family==family and atom.unit==record.get("unit") and atom.semantic_role==record.get("semantic_role")]; numeric_values={atom.normalized_value for atom in numeric_atoms}; has_old=old in numeric_values if old is not None else False; has_new=new in numeric_values if new is not None else False; other=bool(numeric_values-{old,new})
        else:other=any(re.search(rf"(?<!\w){re.escape(value)}(?!\w)",text) for value in known if value not in {old,new})
        if family=="NEGATION_EXCLUSION" and old:
            negative_evidence=bool(re.search(rf"\b(without|no)\s+{re.escape(str(old))}\b",text)); positive_evidence=bool(re.search(rf"\bwith\s+{re.escape(str(old))}\b",text))
            a_support=negative_evidence if query_a_negative else positive_evidence; b_support=negative_evidence if query_b_negative else positive_evidence
        else:a_support=has_old if old is not None else False; b_support=has_new if new is not None else False
        if a_support and not b_support:state="SUPPORTS_A"; supports_a.append(pid)
        elif b_support and not a_support:state="SUPPORTS_B"; supports_b.append(pid)
        elif other:state="CONTRADICTS_A"; locality.append(pid)
        else:state="UNKNOWN"
        evidence.append({"canonical_pair_id":record["canonical_pair_id"],"product_id":pid,"directional_state":state,"supports_a":bool(a_support),"supports_b":bool(b_support),"contradicts_a":bool(other or (b_support and not a_support)),"contradicts_b":bool(other or (a_support and not b_support)),"evidence_text":next((str(row.get(key)) for key in ("product_color","product_brand","product_title","product_bullet_point","product_description") if row.get(key) and ((old and re.search(rf"(?<!\w){re.escape(str(old))}(?!\w)",clean_text(str(row.get(key))))) or (new and re.search(rf"(?<!\w){re.escape(str(new))}(?!\w)",clean_text(str(row.get(key))))))),None),"ground_truth_status":GROUND_TRUTH_STATUS})
    pairs=[{"canonical_pair_id":record["canonical_pair_id"],"p_b":pid_b,"p_a":pid_a,"expected_direction":"B_PRODUCT_SHOULD_GAIN","derivation":"natural_requirement_delta_plus_catalog_evidence","ground_truth_status":GROUND_TRUTH_STATUS} for pid_b in supports_b[:3] for pid_a in supports_a[:3]]
    if record["operation"]=="ADD" and supports_b:
        contradictors=[row["product_id"] for row in evidence if row["contradicts_b"] and not row["supports_b"]]
        pairs=[{"canonical_pair_id":record["canonical_pair_id"],"p_b":pid_b,"p_a":pid_a,"expected_direction":"B_PRODUCT_SHOULD_GAIN","derivation":"added_requirement_plus_catalog_evidence","ground_truth_status":GROUND_TRUTH_STATUS} for pid_b in supports_b[:3] for pid_a in contradictors[:3]]
    return evidence,pairs,sorted(set(locality))
