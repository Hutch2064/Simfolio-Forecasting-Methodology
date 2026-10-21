from pathlib import Path
import sys, unittest
import numpy as np
O = Path(__file__).resolve().parent; V = O/'asset_level_v6'; sys.path.insert(0, str(V))
import common
from candidates.asset_selector_menu_components import candidate as c, bank
from candidates.asset_selector_inner_precision_batched import candidate as old
from candidates.asset_trained_pool.candidate import TrainedPool
from candidates.asset_fast_selector.candidate import BASE_CONFIG
from candidates.asset_selector_universe_certified import energy


class MenuTests(unittest.TestCase):
    def fixture(self):
        x = np.random.default_rng(278).normal(.0002, .012, (40, 3))
        return x, np.array([.2, .3, .5]), np.datetime64('2001-01-01')+np.arange(len(x))

    def test_shared_coefficients_equal_separate_menu_reductions(self):
        rng = np.random.default_rng(42); cube = rng.normal(size=(170, 4, 5)); cube[166] = cube[0]
        actual = rng.normal(size=5); masses = np.ones(5)/5
        A = np.zeros(170); B = np.zeros((170, 170)); c.accumulate(cube, actual, masses, A, B)
        for ids in bank.MENUS.values():
            a = np.zeros(len(ids)); b = np.zeros((len(ids), len(ids))); c.accumulate(cube[ids], actual, masses, a, b)
            np.testing.assert_array_equal(a, A[ids]); np.testing.assert_array_equal(b, B[np.ix_(ids, ids)])

    def test_original_menu_is_exact_and_new_menus_exclude_parent(self):
        x, w, dates = self.fixture(); a = old.prepare(x, w, 4, dates, inner_paths=4); b = c.prepare(x, w, 4, dates, inner_paths=4)
        for key in ('A', 'B'):
            np.testing.assert_array_equal(a['coefficients']['path'][key], b['coefficients']['original'][key])
        np.testing.assert_array_equal(a['weights_by_mode']['path'], b['weights_by_mode']['original'][:166])
        first = old.PathCriterionPool({'criterion': 'path', 'inner_paths': 4}, dates, a)
        second = c.MenuPool({'menu': 'original', 'inner_paths': 4}, dates, b)
        pa = np.concatenate([v for _, v in first.terminal_chunks(x, w, 4, 1000, 71)], axis=1)
        pb = np.concatenate([v for _, v in second.terminal_chunks(x, w, 4, 1000, 71)], axis=1)
        np.testing.assert_array_equal(pa, pb); self.assertEqual(first.model_id, second.model_id)
        for name in ('no_parent', 'flat', 'flat_PPW'):
            self.assertEqual(b['weights_by_mode'][name][0], 0)
            self.assertEqual(energy.even_allocation(b['weights_by_mode'][name], 1000)[0], 0)
        self.assertEqual({k: len(v) for k, v in bank.MENUS.items()}, {'original': 166, 'no_parent': 165, 'flat': 167, 'original_PPW': 168, 'flat_PPW': 169})

    def test_exact_legs_reproduce_parent_with_matching_component_counts(self):
        x, w, _ = self.fixture(); prepared, failures = bank.prefix_fits(x, [0, 166, 167], w); self.assertFalse(failures)
        parent = TrainedPool(BASE_CONFIG, prepared['parent']); mixed = np.concatenate([v for _, v in parent.terminal_chunks(x, w, 11, 1000, 97)], axis=1)
        raw, filtered = parent.last_component_counts
        for index, start, count in ((166, 0, raw), (167, raw, filtered)):
            if count:
                leg = bank.model(index, prepared); pieces = np.concatenate([v for _, v in leg.terminal_chunks(x, w, 11, count, 97)], axis=1)
                np.testing.assert_array_equal(pieces, mixed[start:start+count])

    def test_PPWs_preserve_joint_assets_and_training_only_fits(self):
        x, w, dates = self.fixture(); b = c.prepare(x, w, 4, dates, inner_paths=2)
        for trace in b['trace']:
            self.assertLessEqual(trace['validation_stop'], len(x))
            self.assertEqual(trace['PPW_dependence_fit']['history_sha256'], c.signature(x[:trace['training_rows']]))
        for index in (168, 169):
            generator = bank.model(index, b); values = np.concatenate([v for _, v in generator.terminal_chunks(x, w, 7, 10, 81)], axis=1)
            self.assertEqual(values.shape, (10, 7)); self.assertTrue(np.isfinite(values).all())
            self.assertEqual(generator.last_asset_log_paths.shape, (10, 3))

    def test_menus_use_no_outside_components_and_bind_request(self):
        x, w, dates = self.fixture(); b = c.prepare(x, w, 4, dates, inner_paths=2)
        for name, ids in bank.MENUS.items():
            weights = b['weights_by_mode'][name]
            self.assertTrue(np.all(weights[[i for i in range(170) if i not in ids]] == 0))
            m = c.MenuPool({'menu': name, 'inner_paths': 2}, dates, b)
            output = np.concatenate([v for _, v in m.terminal_chunks(x, w, 4, 1000, 91)], axis=1)
            self.assertEqual(output.shape, (1000, 4)); self.assertEqual(sum(m.last_selection['path_counts']), 1000)
        with self.assertRaises(ValueError):
            next(m.terminal_chunks(x, w, 5, 1000, 91))


if __name__ == '__main__':
    unittest.main()
