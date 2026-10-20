import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from blocks import BlockLimits, JointBlockStream, partition
from asset_paths import JointAssetBootstrap, advance
from inputs import AssetInputs


def materialize(model, x, weights, h, paths=8, seed=991, **kwargs):
    return np.concatenate([v for _, v in model.terminal_chunks(x, weights, h, paths, seed, **kwargs)], axis=1)


class JointSamplingTests(unittest.TestCase):
    def test_without_replacement_complete_cycles_and_refill(self):
        stream = JointBlockStream(23, 23*4+7, 11, 71, replacement=False,
                                  limits=BlockLimits(3, 8))
        indices = np.concatenate([stream.take(17) for _ in range(5)]+[stream.take(14)], axis=1)
        for path in indices:
            for cycle in range(4):
                np.testing.assert_array_equal(np.sort(path[cycle*23:(cycle+1)*23]), np.arange(23))
            self.assertEqual(len(np.unique(path[92:])), 7)

    def test_partition_covers_every_row_once(self):
        for n in (1, 2, 13, 101):
            starts, lengths = partition(n, 2, 7, np.random.default_rng(4))
            np.testing.assert_array_equal(np.concatenate([np.arange(s, s+l) for s, l in zip(starts, lengths)]), np.arange(n))
            self.assertTrue(np.all(lengths[:-1] >= 2))
            self.assertTrue(np.all(lengths <= 7))

    def test_replacement_can_repeat_and_blocks_are_contiguous(self):
        stream = JointBlockStream(17, 250, 4, 78, replacement=True,
                                  limits=BlockLimits(4, 4))
        a = stream.take(250)
        self.assertLess(len(np.unique(a[0])), 250)
        for path in a:
            for start in range(0, 248, 4):
                np.testing.assert_array_equal(np.diff(path[start:start+4]), np.ones(3))
        self.assertGreaterEqual(a.min(), 0); self.assertLess(a.max(), 17)

    def test_chunk_width_does_not_change_sampling_or_wealth(self):
        x = np.random.default_rng(55).normal(0, .01, (53, 3))
        for replacement in (False, True):
            a = materialize(JointAssetBootstrap(replacement=replacement), x, [.5, .3, .2], 137, width=1)
            b = materialize(JointAssetBootstrap(replacement=replacement), x, [.5, .3, .2], 137, width=64)
            np.testing.assert_array_equal(a, b)

    def test_asset_paths_preserve_joint_row_relation(self):
        a = np.linspace(-.3, .3, 40)
        model = JointAssetBootstrap(replacement=True)
        materialize(model, np.column_stack((a, -2*a)), [.5, .5], 200)
        np.testing.assert_allclose(model.last_asset_log_paths[:, 1], -2*model.last_asset_log_paths[:, 0], atol=1e-14)
        np.testing.assert_allclose(model.last_asset_weights.sum(axis=1), 1., atol=1e-14)

    def test_within_path_rebalancing_matches_direct_asset_accounting(self):
        x = np.log1p(np.array([[.2, -.1], [-.05, .3], [.1, -.2]]))
        indices = np.array([[0, 1, 2, 0, 2]])
        target = np.log([.7, .3])
        for frequency in (0, 1, 2):
            holding = np.array([.7, .3]); expected = []
            for step, index in enumerate(indices[0]):
                if frequency and step % frequency == 0:
                    holding = holding.sum()*np.array([.7, .3])
                holding *= np.exp(x[index]); expected.append(np.log(holding.sum()))
            out = advance(x, indices, target, target.reshape(1, 2).copy(),
                          np.zeros((1, 2)), np.zeros(1), 0, frequency)
            np.testing.assert_allclose(out[0], expected, atol=3e-16)
        daily = materialize(JointAssetBootstrap(replacement=True), x, [.7, .3], 50, rebalance_every=1)
        hold = materialize(JointAssetBootstrap(replacement=True), x, [.7, .3], 50, rebalance_every=0)
        self.assertFalse(np.allclose(daily, hold))

    def test_single_asset_and_extreme_wealth_remain_log_stable(self):
        x = np.array([[1000., -1000.], [-1000., 1000.], [0., 0.]])
        out = materialize(JointAssetBootstrap(replacement=True), x, [.5, .5], 100)
        self.assertTrue(np.isfinite(out).all())
        one = np.array([[.01], [-.02], [.03]])
        stream = JointBlockStream(3, 20, 8, 991, replacement=True)
        expected = np.cumsum(one[stream.take(20), 0], axis=1)
        actual = materialize(JointAssetBootstrap(replacement=True), one, [1.], 20)
        np.testing.assert_array_equal(actual, expected)

    def test_training_matrix_uses_all_intervening_observations_and_no_future(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            da = np.array(['2001-01-01', '2001-01-02', '2001-01-03', '2001-01-04', '2001-01-05'], dtype='datetime64[D]')
            db = da[[0, 2, 4]]
            xa, xb = np.array([0., .01, .02, .03, .04]), np.array([0., -.03, .07])
            rows = []
            for sid, dates, values in [('A', da, xa), ('B', db, xb)]:
                np.save(root/(sid+'d.npy'), dates); np.save(root/(sid+'x.npy'), values)
                rows.append({'series_id': sid, 'dates_file': sid+'d.npy', 'returns_file': sid+'x.npy', 'panel_branch': 'assets'})
            matrix = np.array([[.03, -.03], [.07, .07]])
            target = np.log1p(np.expm1(matrix).mean(axis=1))
            np.save(root/'Pd.npy', db[1:]); np.save(root/'Px.npy', target)
            row = {'series_id': 'P', 'dates_file': 'Pd.npy', 'returns_file': 'Px.npy',
                   'panel_branch': 'portfolios', 'construction': 'constant_weights_common_observed_endpoint',
                   'rational_weights': [['A', 1, 2], ['B', 1, 2]], 'provenance': {'valuation_anchor': str(da[0])}}
            rows.append(row); (root/'panel.json').write_text(json.dumps({'series': rows}))
            inputs = AssetInputs(root)
            got, w, ids = inputs.training(row, 2)
            np.testing.assert_allclose(got, matrix, atol=1e-16)
            self.assertEqual(ids, ('A', 'B')); self.assertFalse(got.flags.writeable)
            # Extend the target to permit a training-only earlier origin test.
            np.save(root/'Pd.npy', np.append(db[1:], np.datetime64('2001-01-07')))
            np.save(root/'Px.npy', np.append(target, 0.))
            before = inputs.training(row, 2)[0].copy()
            np.save(root/'Ax.npy', np.append(xa, 9999.))
            np.save(root/'Ad.npy', np.append(da, np.datetime64('2001-01-06')))
            after = inputs.training(row, 2)[0]
            np.testing.assert_array_equal(before, after)


if __name__ == '__main__':
    unittest.main()
