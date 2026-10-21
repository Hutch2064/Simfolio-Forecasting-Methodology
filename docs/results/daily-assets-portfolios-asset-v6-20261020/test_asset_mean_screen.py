from pathlib import Path
import os
for k in ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'NUMBA_NUM_THREADS'):
    os.environ[k] = '1'
import sys
import unittest
import numpy as np
O = Path(__file__).resolve().parent
sys.path.insert(0, str(O/'asset_level_v6'))
from asset_paths import JointAssetBootstrap
from candidates.asset_mean_uncertainty.model import JointMeanBootstrap


class MeanScreenTests(unittest.TestCase):
    def test_unshrunk_single_asset_adds_only_fixed_path_mean_delta(self):
        x = np.random.default_rng(61).normal(.001, .01, (520, 1))
        for replacement in (True, False):
            raw = JointAssetBootstrap(replacement=replacement)
            changed = JointMeanBootstrap(replacement, False)
            a = np.concatenate([p for _, p in raw.terminal_chunks(x, [1.], 91, 100, 55)], axis=1)
            b = np.concatenate([p for _, p in changed.terminal_chunks(x, [1.], 91, 100, 55)], axis=1)
            np.testing.assert_allclose(b-a, changed.last_mean_delta*np.arange(1, 92), atol=5e-16)
            np.testing.assert_allclose(changed.last_mean_delta.mean(axis=0), 0., atol=1e-18)

    def test_joint_mean_law_is_chunk_invariant(self):
        x = np.random.default_rng(91).normal(0, .01, (510, 2))
        for replacement in (True, False):
            for shrink in (True, False):
                a, b = JointMeanBootstrap(replacement, shrink), JointMeanBootstrap(replacement, shrink)
                p = np.concatenate([p for _, p in a.terminal_chunks(x, [.7, .3], 95, 100, 137, width=1)], axis=1)
                q = np.concatenate([p for _, p in b.terminal_chunks(x, [.7, .3], 95, 100, 137, width=64)], axis=1)
                np.testing.assert_array_equal(p, q)
                self.assertEqual(a.last_asset_log_paths.shape, (100, 2))
                np.testing.assert_allclose(a.last_asset_weights.sum(axis=1), 1., atol=1e-14)


if __name__ == '__main__':
    unittest.main()
