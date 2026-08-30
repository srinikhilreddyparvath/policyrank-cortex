from __future__ import annotations
import hashlib,json

class ResumeFingerprintMismatch(ValueError): pass

def configuration_fingerprint(payload:dict)->str:
    canonical=json.dumps(payload,sort_keys=True,separators=(",",":"),default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()

def verify_resume_fingerprint(existing:dict,current:str)->None:
    observed=existing.get("configuration_fingerprint")
    if observed!=current: raise ResumeFingerprintMismatch(f"Resume fingerprint mismatch: existing={observed}, current={current}")
