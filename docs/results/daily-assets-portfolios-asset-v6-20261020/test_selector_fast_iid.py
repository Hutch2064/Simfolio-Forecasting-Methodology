from pathlib import Path
import sys, unittest
import numpy as np
O = Path(__file__).resolve().parent; V = O/'asset_level_v6'; sys.path.insert(0, str(V))
import common
from blocks import BlockLimits, JointBlockStream
from asset_paths import JointAssetBootstrap
from candidates.asset_selector_fast_iid.paths import FastBootstrap, all_chunks
from candidates.asset_selector_batched_paths.paths import all_chunks as old_chunks
from candidates.asset_selector_universe.bank import prefix_fits
from candidates.asset_selector_fast_iid import candidate as fast
from candidates.asset_selector_inner_precision_batched import candidate as old


class ExactIIDTests(unittest.TestCase):
    def test_vector_rows_match_original_scalar_seed_stream(self):
        for n in (2, 7, 252, 1001):
            for horizon in (1, 2, 65, 129):
                original = JointBlockStream(n, horizon, 6, 918, replacement=True, limits=BlockLimits(maximum=1))
                expected = original.take(horizon)
                actual = np.stack([np.random.default_rng(s).integers(0, n, size=horizon) for s in np.random.SeedSequence(918).spawn(6)])
                np.testing.assert_array_equal(actual, expected)

    def test_full_asset_and_portfolio_paths_match_all_rebalance_policies_and_chunks(self):
        x = np.random.default_rng(12).normal(0, .02, (43, 3)); w = np.array([.2, .0, .8])
        for policy in (0, 1, 5):
            for width in (1, 7, 64):
                a = JointAssetBootstrap(replacement=True, limits=BlockLimits(maximum=1)); b = FastBootstrap(replacement=True, limits=BlockLimits(maximum=1))
                pa = np.concatenate([v for _, v in a.terminal_chunks(x, w, 75, 12, 889, width=width, rebalance_every=policy)], axis=1)
                pb = np.concatenate([v for _, v in b.terminal_chunks(x, w, 75, 12, 889, width=width, rebalance_every=policy)], axis=1)
                np.testing.assert_array_equal(pa, pb); np.testing.assert_array_equal(a.last_asset_log_paths, b.last_asset_log_paths); np.testing.assert_array_equal(a.last_asset_weights, b.last_asset_weights)

    def test_other_block_streams_delegate_without_changing_outputs(self):
        x = np.random.default_rng(12).normal(0, .02, (43, 2)); w = np.array([.3, .7])
        for replacement in (False, True):
            a = JointAssetBootstrap(replacement=replacement); b = FastBootstrap(replacement=replacement)
            pa = np.concatenate([v for _, v in a.terminal_chunks(x, w, 120, 8, 18, width=7)], axis=1)
            pb = np.concatenate([v for _, v in b.terminal_chunks(x, w, 120, 8, 18, width=7)], axis=1)
            np.testing.assert_array_equal(pa, pb)

    def test_all166_inner_option_paths_are_exact(self):
        x = np.random.default_rng(12).normal(0, .01, (40, 3)); w = np.array([.2, .3, .5]); ids = list(range(166)); prepared, errors = prefix_fits(x, ids)
        self.assertFalse(errors)
        for (a, av), (b, bv) in zip(old_chunks(x, w, 70, 10, 88, ids, prepared, width=7), all_chunks(x, w, 70, 10, 88, ids, prepared, width=7)):
            self.assertEqual(a, b); np.testing.assert_array_equal(av, bv)

    def test_full_selection_coefficients_weights_and1000_final_paths_are_exact(self):
        x = np.random.default_rng(638).normal(0, .01, (40, 3)); w = np.array([.2, .3, .5]); d = np.datetime64('2000-01-01') + np.arange(40)
        a = old.prepare(x, w, 4, d, inner_paths=10); b = fast.prepare(x, w, 4, d, inner_paths=10)
        for mode in ('terminal', 'path'):
            for term in ('A', 'B'):
                np.testing.assert_array_equal(a['coefficients'][mode][term], b['coefficients'][mode][term])
            np.testing.assert_array_equal(a['weights_by_mode'][mode], b['weights_by_mode'][mode])
            cfg = {'criterion': mode, 'inner_paths': 10}; ma = old.PathCriterionPool(cfg, d, a); mb = fast.PathCriterionPool(cfg, d, b)
            np.testing.assert_array_equal(np.concatenate([v for _, v in ma.terminal_chunks(x, w, 4, 1000, 19)], axis=1), np.concatenate([v for _, v in mb.terminal_chunks(x, w, 4, 1000, 19)], axis=1))
            self.assertEqual(ma.model_id, mb.model_id)


if __name__ == '__main__':
    unittest.main()
