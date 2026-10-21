from pathlib import Path
import sys, unittest
import numpy as np
O = Path(__file__).resolve().parent; V = O/'asset_level_v6'; sys.path.insert(0, str(V))
import common
from candidates.asset_selector_design import candidate as c
from candidates.asset_selector_design.validation import RULES, schedule
from candidates.asset_selector_inner_precision_batched import candidate as reference


class DesignTests(unittest.TestCase):
    def fixture(self, rows=40):
        x = np.random.default_rng(638).normal(0, .01, (rows, 3))
        return x, np.array([.2, .3, .5]), np.datetime64('2000-01-01')+np.arange(rows)

    def test_every_validation_design_is_chronological_complete_and_uncapped(self):
        for rule in RULES:
            for rows in (3, 25, 40, 200, 1000):
                for requested in (1, 3, 2000):
                    eligible, folds = schedule(rows, requested, rule)
                    self.assertTrue(eligible); self.assertTrue(folds)
                    self.assertEqual(folds, sorted(set(folds)))
                    for start, h in folds:
                        self.assertGreaterEqual(start, 24 if rows > 24 else 2)
                        self.assertGreaterEqual(h, 1); self.assertLessEqual(h, requested); self.assertLessEqual(start+h, rows)
        self.assertGreater(len(schedule(1000, 3, 'established')[1]), 300)

    def test_overlap_adds_midpoints_without_removing_existing_origins(self):
        _, old = schedule(200, 80, 'established'); _, overlap = schedule(200, 80, 'overlap_midpoints')
        self.assertTrue(set(old).issubset(overlap)); self.assertGreater(len(overlap), len(old))

    def test_balanced_history_and_requested_horizon_designs_change_support(self):
        _, old = schedule(200, 100, 'established'); _, balanced = schedule(200, 100, 'support_balanced'); _, target = schedule(200, 100, 'target_first')
        self.assertTrue(all(start >= h for start, h in balanced))
        self.assertLess(balanced[0][1], old[0][1]); self.assertEqual(target[0][1], 100)
        self.assertEqual(len(target), 1); self.assertGreater(len(old), 1)

    def test_established_stack_is_exact_to_current_selector(self):
        x, w, dates = self.fixture(); old = reference.prepare(x, w, 4, dates, inner_paths=10)
        new = c.prepare(x, w, 4, dates, inner_paths=10, schedule_rule='established')
        for key in ('A', 'B'):
            np.testing.assert_array_equal(old['coefficients']['path'][key], new['coefficients'][key])
        np.testing.assert_array_equal(old['weights_by_mode']['path'], new['weights_by_mode']['stack'])
        a = reference.PathCriterionPool({'criterion': 'path', 'inner_paths': 10}, dates, old)
        b = c.DesignPool({'schedule': 'established', 'inner_paths': 10, 'pool': 'stack'}, dates, new)
        pa = np.concatenate([v for _, v in a.terminal_chunks(x, w, 4, 1000, 19)], axis=1)
        pb = np.concatenate([v for _, v in b.terminal_chunks(x, w, 4, 1000, 19)], axis=1)
        np.testing.assert_array_equal(pa, pb); self.assertEqual(a.model_id, b.model_id)

    def test_pick_one_is_the_lowest_scoring_vertex_and_gets_all_final_paths(self):
        x, w, dates = self.fixture(); b = c.prepare(x, w, 4, dates, inner_paths=2, schedule_rule='established')
        valid = b['trace'][-1]['eligible_this_fold']; valid = [i for i in valid if i not in b['invalid_internal_options']]
        expected = valid[int(np.argmin(b['coefficients']['A']-.5*np.diag(b['coefficients']['B'])))]
        self.assertEqual(np.flatnonzero(b['weights_by_mode']['select']).tolist(), [expected])
        model = c.DesignPool({'schedule': 'established', 'inner_paths': 2, 'pool': 'select'}, dates, b)
        paths = np.concatenate([v for _, v in model.terminal_chunks(x, w, 4, 1000, 51)], axis=1)
        self.assertEqual(paths.shape, (1000, 4)); self.assertEqual(model.last_selection['path_counts'][expected], 1000)

    def test_request_binding_and_seed_repetition(self):
        x, w, dates = self.fixture(); a = c.prepare(x, w, 4, dates, inner_paths=2, schedule_rule='target_first')
        b = c.prepare(x, w, 4, dates, inner_paths=2, schedule_rule='target_first', seed_replication=1)
        self.assertNotEqual([f['seed'] for f in a['trace']], [f['seed'] for f in b['trace']])
        self.assertEqual([f['training_rows'] for f in a['trace']], [f['training_rows'] for f in b['trace']])
        wrong = c.DesignPool({'schedule': 'established', 'inner_paths': 2, 'pool': 'stack'}, dates, a)
        with self.assertRaises(ValueError):
            next(wrong.terminal_chunks(x, w, 4, 1000, 51))


if __name__ == '__main__':
    unittest.main()
