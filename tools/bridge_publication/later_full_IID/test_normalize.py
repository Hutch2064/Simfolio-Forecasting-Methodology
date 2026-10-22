"""Small stdlib/saved fixture checks, never rerun either scientific study."""
import copy
import importlib.util
import json
import unittest
from pathlib import Path
from unittest.mock import patch

import normalize as N

ROOT = Path(__file__).resolve().parents[6]
HERE = ROOT/N.OUT
s = importlib.util.spec_from_file_location('strict_IID_publication_consumer', ROOT/N.U/'catalogue_support/preserve_study.py')
P = importlib.util.module_from_spec(s)
s.loader.exec_module(P)


class SavedProjection(unittest.TestCase):
    def fixture(self, name):
        return json.loads((HERE/name/'actual-consumer-panel-fixture.json').read_text())

    def test_deterministic_source_projection(self):
        for name, content in N.build(ROOT).items():
            self.assertEqual((HERE/name).read_bytes(), content)

    def test_all_four_completed_panels_pass_unchanged_consumer(self):
        for name in N.STUDIES:
            f = self.fixture(name)
            for panel, row in f['panels'].items():
                self.assertEqual(row['status'], 'complete_scored_and_audited')
                self.assertTrue(P.validate_panel(ROOT, panel, row, set(), f['sources']))

    def test_changed_score_missing_control_and_partial_evidence_rejected(self):
        f = self.fixture('rough_IID')
        for problem in ('score', 'control', 'evidence'):
            row = copy.deepcopy(f['panels']['80'])
            if problem == 'score':
                row['scores']['rough_only_IID_on_sampled_conventional_SV_corrected_nodes'] += .1
            elif problem == 'control':
                row['controls'].pop('M261_saved_control')
            else:
                row['evidence'].pop('analysis')
            with self.assertRaises(ValueError):
                P.validate_panel(ROOT, '80', row, set(), f['sources'])

    def test_wrong_independent_audit_pin_rejected(self):
        bad = dict(N.STUDIES)
        record = list(bad['rough_IID'])
        record[2] = '0'*64
        bad['rough_IID'] = tuple(record)
        with patch.object(N, 'STUDIES', bad), self.assertRaises(AssertionError):
            N.build(ROOT)


if __name__ == '__main__':
    unittest.main()
