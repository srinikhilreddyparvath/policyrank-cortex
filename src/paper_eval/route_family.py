ROUTE_FAMILY_VERSION="cortex_candidate_routes_v1"
ROUTES={
 "preserve":{"active":False,"implementation":"baseline_preservation_gate.preserve_baseline_slate"},
 "strict_filter":{"active":True,"implementation":"calibrated_route_execution_adapter.apply_strict_constraint_filter"},
 "strict_boost":{"active":True,"implementation":"calibrated_route_execution_adapter.apply_scale_aware_strict_boost"},
 "contract_rerank":{"active":True,"implementation":"contract filter + policy compiler + final enforcer + diversifier"},
 "fallback":{"active":False,"implementation":"adapter fallback flag"},
}
ORACLE_REQUIRED_ROUTES=tuple(ROUTES)
