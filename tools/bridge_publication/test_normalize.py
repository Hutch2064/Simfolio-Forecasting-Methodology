"""Read-only saved-fixture/stdlib consumer checks; no scientific execution."""
import copy
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch
import normalize as N

ROOT = Path(__file__).resolve().parents[6]
HERE = ROOT/N.OUT
spec = importlib.util.spec_from_file_location('bridge_publication_strict_consumer', ROOT/N.U/'catalogue_support/preserve_study.py')
P = importlib.util.module_from_spec(spec)
spec.loader.exec_module(P)


class SavedProjection(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads((HERE/'actual-consumer-panel-fixture.json').read_text())

    def test_exact_deterministic_projection(self):
        for name, content in N.build(ROOT).items():
            self.assertEqual((HERE/name).read_bytes(), content)

    def test_unchanged_strict_consumer_accepts_only80_completed(self):
        for panel, row in self.fixture['panels'].items():
            refs = P.validate_panel(ROOT, panel, row,
                {'rough_only_covariance_on_sampled_conventional_SV_corrected_nodes'}, self.fixture['sources'])
            self.assertEqual(bool(refs), panel == '80')

    def test_false_fresh52_or_changed80_score_rejected(self):
        row = copy.deepcopy(self.fixture['panels']['52'])
        row['scores'] = {'invented': 1.}
        with self.assertRaises(ValueError):
            P.validate_panel(ROOT, '52', row, set(), self.fixture['sources'])
        row = copy.deepcopy(self.fixture['panels']['80'])
        row['scores']['rough_only_covariance_on_sampled_conventional_SV_corrected_nodes'] += .1
        with self.assertRaises(ValueError):
            P.validate_panel(ROOT, '80', row, set(), self.fixture['sources'])

    def test_wrong_prior_independent_proof_pin_rejected(self):
        bad = dict(N.PINS)
        bad[str(N.REVIEW)] = '0'*64
        with patch.object(N, 'PINS', bad):
            with self.assertRaises(AssertionError):
                N.build(ROOT)


if __name__ == '__main__':
    unittest.main()
