"""Training-law and joint-path checks, without new held-out forecast queues."""
from pathlib import Path
import os
for k in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMBA_NUM_THREADS']:
    os.environ[k] = '1'
import sys
import unittest
import ast
import json
import numpy as np
O = Path(__file__).resolve().parent
sys.path.insert(0, str(O/'asset_level_v6'))
from candidates.asset_prior_best import candidate, fit, mean, kernels


class AssetPriorTests(unittest.TestCase):
    def test_garch_fitting_functions_copied_exactly(self):
        receipt = json.loads((candidate.OWN/'fit_source_receipt.json').read_text())
        old = ast.parse(Path(receipt['original_source']).read_text())
        new = ast.parse((candidate.OWN/'fit.py').read_text())
        for name in receipt['functions_copied_verbatim']:
            a = next(x for x in old.body if isinstance(x, ast.FunctionDef) and x.name == name)
            b = next(x for x in new.body if isinstance(x, ast.FunctionDef) and x.name == name)
            self.assertEqual(ast.dump(a, include_attributes=False), ast.dump(b, include_attributes=False))

    def test_univariate_batch_variance_matches_prior_formula(self):
        x = np.random.default_rng(55).normal(.001, .01, 503)
        q = int(np.sqrt(len(x))); mu = x.mean()
        rss = sum(len(g)*(float(g.mean())-mu)**2 for g in np.array_split(x, q))
        expected = rss/(q-1)/len(x)*((q-1)/(q-3))
        actual = mean.estimate(x[:, None])
        self.assertAlmostEqual(actual['covariance'][0, 0], expected, places=20)

    def test_joint_uncertainty_retains_linear_asset_dependence(self):
        x = np.random.default_rng(51).normal(0, .01, 503)
        fitted = mean.estimate(np.column_stack((x, 2*x)))
        delta = mean.draws(fitted, 1000, 337)
        np.testing.assert_allclose(delta[:, 1], 2*delta[:, 0], rtol=0, atol=1e-12)
        np.testing.assert_allclose(delta.mean(axis=0), fitted['center']-fitted['sample_mean'], atol=1e-18)

    def test_chunk_invariance_for_joint_asset_pool(self):
        x = np.random.default_rng(17).normal(.0002, .01, (160, 2))
        for replacement in [False, True]:
            a = candidate.AssetPool(replacement)
            b = candidate.AssetPool(replacement)
            p = np.concatenate([v for _, v in a.terminal_chunks(x, [.7, .3], 83, 20, 123, width=1)], axis=1)
            q = np.concatenate([v for _, v in b.terminal_chunks(x, [.7, .3], 83, 20, 123, width=64)], axis=1)
            np.testing.assert_array_equal(p, q)
            self.assertEqual(a.last_asset_log_paths.shape, (20, 2))
            np.testing.assert_allclose(a.last_asset_weights.sum(axis=1), 1., atol=1e-14)

    def test_zero_shift_raw_kernel_matches_direct_joint_accounting(self):
        x = np.random.default_rng(77).normal(0, .02, (70, 3))
        idx = np.array([[3, 20, 50, 17], [0, 2, 4, 7]])
        w = np.array([.5, .3, .2]); target = np.log(w)
        arrays = [np.zeros(3), np.zeros(3), np.zeros(3), np.zeros(3)]
        got = kernels.advance(x, x, idx, np.zeros((2, 3)), *arrays,
            np.ones((2, 3)), False, target, np.tile(target, (2, 1)), np.zeros((2, 3)), np.zeros(2))
        expected = np.cumsum(np.log1p(np.sum(w*np.expm1(x[idx]), axis=2)), axis=1)
        np.testing.assert_allclose(got, expected, atol=3e-16)

    def test_fhs_uses_same_residual_row_for_all_assets(self):
        x = np.zeros((5, 2)); z = np.linspace(-1., 1., 5)
        pool = np.column_stack((z, -z)); indices = np.array([[0, 3, 1, 4]])
        assets = np.zeros((1, 2))
        kernels.advance(x, pool, indices, np.zeros((1, 2)), np.zeros(2),
            np.ones(2), np.zeros(2), np.zeros(2), np.ones((1, 2)), True,
            np.log([.5, .5]), np.log([[.5, .5]]), assets, np.zeros(1))
        self.assertAlmostEqual(assets[0, 0], -assets[0, 1], places=15)

    def test_factory_refuses_unselected_or_missing_parent(self):
        self.assertFalse((candidate.OWN/'parent_selection.json').exists())
        with self.assertRaises(FileNotFoundError):
            candidate.build()


if __name__ == '__main__':
    unittest.main()
