from __future__ import annotations
from dataclasses import dataclass
import json
import pandas as pd

FORBIDDEN_RANKER_COLUMNS=frozenset({"esci_label","relevance_label","label","grade"})

class CandidateBoundaryError(ValueError): pass

@dataclass(frozen=True)
class CandidateSetRequest:
    query:str; query_id:int; candidates:pd.DataFrame
    def ranker_frame(self)->pd.DataFrame:
        forbidden=FORBIDDEN_RANKER_COLUMNS.intersection(self.candidates.columns)
        if forbidden: raise CandidateBoundaryError(f"Evaluator-only judgment columns reached method adapter: {sorted(forbidden)}")
        return self.candidates.copy()
    @property
    def input_ids(self)->set[str]: return set(self.candidates["product_id"].astype(str))

def validate_output_subset(request:CandidateSetRequest, rows:list[dict])->None:
    output={str(row.get("product_id","")) for row in rows}
    external=sorted(output-request.input_ids)
    if external: raise CandidateBoundaryError(f"Method introduced external candidate IDs: {external[:10]}")
    if len(output)!=len(rows): raise CandidateBoundaryError("Method returned duplicate or empty product IDs")

def diagnostics_json(metadata:dict)->str:
    return json.dumps(metadata,sort_keys=True,default=str)
