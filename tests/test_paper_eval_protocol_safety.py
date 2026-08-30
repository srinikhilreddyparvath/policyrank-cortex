import unittest
from src.paper_eval.fingerprint import ResumeFingerprintMismatch, configuration_fingerprint, verify_resume_fingerprint
from src.paper_eval.route_family import ORACLE_REQUIRED_ROUTES, ROUTE_FAMILY_VERSION, ROUTES
from src.paper_eval.provenance import LEGACY_GATE_THRESHOLDS, LEGACY_Q_TABLE

class ProtocolSafetyTests(unittest.TestCase):
    def test_route_family_is_versioned_and_real(self):
        self.assertEqual(ROUTE_FAMILY_VERSION,"cortex_candidate_routes_v1"); self.assertEqual(set(ORACLE_REQUIRED_ROUTES),set(ROUTES)); self.assertIn("contract_rerank",ROUTES)
    def test_resume_fingerprint_mismatch_rejected(self):
        first=configuration_fingerprint({"seed":29,"methods":["a"]}); second=configuration_fingerprint({"seed":30,"methods":["a"]})
        verify_resume_fingerprint({"configuration_fingerprint":first},first)
        with self.assertRaises(ResumeFingerprintMismatch): verify_resume_fingerprint({"configuration_fingerprint":first},second)
    def test_legacy_threshold_and_policy_provenance_is_explicit(self):
        self.assertEqual(LEGACY_Q_TABLE.status,"legacy_contaminated_for_paper_eval")
        self.assertIn("ESCI-label-derived",LEGACY_Q_TABLE.reason)
        self.assertEqual(LEGACY_GATE_THRESHOLDS.status,"legacy_contaminated_for_paper_eval")
        self.assertIn("train_fit-only",LEGACY_GATE_THRESHOLDS.reason)

if __name__=="__main__": unittest.main()
