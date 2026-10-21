"""Analytic, temporal, invariance and real dated joint-path contracts."""
import json
from pathlib import Path
import tempfile
import unittest
import numpy as np
from candidates.asset_selector_sequential_learning.learning import learn, pack_B, unpack_B
from candidates.asset_selector_sequential_learning import candidate as new
from candidates.asset_selector_endpoint_flat import candidate as parent


class LearningTests(unittest.TestCase):
    def coefficients(self):
        rng = np.random.default_rng(821)
        terms = [parent.energy.coefficients(rng.normal(size=(5, 12)), y) for y in [.2, -.3, .7, 0., 1.]]
        return np.asarray([a for a, b in terms]), np.asarray([b for a, b in terms]), np.array([1, 2, 1, 3, 2])

    def test_gradient_is_empirical_cdf_mixture_directional_derivative(self):
        a, b = parent.energy.coefficients(np.array([[-1., 0., 1.], [1., 2., 3.]]), .4)
        w = np.array([.3, .7]); direction = np.array([1., -1.]); step = 1e-5
        derivative = (parent.energy.objective(w+step*direction, a, b)-parent.energy.objective(w-step*direction, a, b))/(2*step)
        self.assertAlmostEqual(derivative, direction @ (a-b@w), places=10)
        # Direct weighted empirical CRPS for the same mixture.
        x = np.array([-1., 0., 1., 1., 2., 3.]); p = np.repeat(w/3, 3)
        expected = p @ abs(x-.4) - .5 * p @ abs(x[:, None]-x[None, :]) @ p
        self.assertAlmostEqual(expected, parent.energy.objective(w, a, b), places=14)

    def test_known_good_expert_moves_in_correct_direction(self):
        # First-step Fig.2: excess=(-.5,+.5), E=1, eta=1,
        # L=(-.25,+.25), so p_good=logistic(.5).
        result = learn(np.array([[0., 1.]]), np.zeros((1, 2, 2)), [1])
        np.testing.assert_allclose(result['weights'], [1/(1+np.exp(-.5)), 1/(1+np.exp(.5))], rtol=1e-14)

    def test_prefix_permutation_scale_and_progressive(self):
        a, b, m = self.coefficients(); result = learn(a, b, m)
        np.testing.assert_array_equal(result['trajectory'][:4], learn(a[:3], b[:3], m[:3])['trajectory'])
        permutation = np.array([3, 1, 4, 0, 2])
        permuted = learn(a[:, permutation], b[:, permutation][:, :, permutation], m)
        np.testing.assert_allclose(permuted['trajectory'], result['trajectory'][:, permutation], atol=2e-15)
        for scale in [1e-8, 1e8]:
            np.testing.assert_allclose(learn(scale*a, scale*b, m)['trajectory'], result['trajectory'], atol=3e-15)
        np.testing.assert_array_equal(result['progressive_weights'], result['trajectory'].mean(axis=0))
        np.testing.assert_allclose(result['trajectory'].sum(axis=1), 1., atol=2e-15)
        np.testing.assert_array_equal(unpack_B(pack_B(b), a.shape[1]), b)

    def test_degenerate_singleton_zero_prefix_and_bad_inputs(self):
        self.assertEqual(learn([[1.], [2.]], [[[0.]], [[0.]]], [1, 5])['weights'][0], 1.)
        r = learn(np.ones((3, 4)), np.ones((3, 4, 4)), [1, 1, 3])
        np.testing.assert_array_equal(r['trajectory'], np.full((4, 4), .25))
        a, b, m = self.coefficients()
        with_zero = learn(np.concatenate([np.ones((1, 5)), a]), np.concatenate([np.ones((1, 5, 5)), b]), [1, *m])
        np.testing.assert_array_equal(with_zero['weights'], learn(a, b, m)['weights'])
        with self.assertRaises(ValueError): learn([[np.nan, 0.]], np.zeros((1, 2, 2)), [1])
        with self.assertRaises(ValueError): learn([[1., 0.]], np.zeros((1, 2, 2)), [1.5])
        # Explicit numerical failure rather than silently losing squared gradients.
        for scale in [1e-300, 1e300]:
            with self.assertRaises(ArithmeticError): learn([[0., scale]], np.zeros((1, 2, 2)), [1])


class DatedJointForecastTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(9211)
        cls.x = rng.normal(.0002, .01, size=(32, 2)); cls.w = np.array([.35, .65])
        cls.dates = np.arange(np.datetime64('2020-01-01'), np.datetime64('2020-02-02'))
        cls.kw = {'inner_paths': 4, 'schedule_rule': 'half_doubling', 'width': 2}
        cls.control = parent.prepare(cls.x, cls.w, 3, cls.dates, **cls.kw)
        cls.bundle = new.prepare(cls.x, cls.w, 3, cls.dates, **cls.kw)

    def test_exact_parent_control_coefficients_weights_trace_and_paths(self):
        c, b = self.control, self.bundle
        self.assertEqual(c['trace'], b['trace'])
        self.assertEqual(c['invalid_internal_options'], b['invalid_internal_options'])
        for key in ['A', 'B']:
            np.testing.assert_array_equal(c['coefficients']['flat'][key], b['coefficients']['batch'][key])
        np.testing.assert_array_equal(c['weights_by_mode']['flat'], b['weights_by_mode']['batch'])
        # Identical allocation, component order, seeds and request joint accounting.
        for width in [1, 2, 3]:
            p = parent.MenuPool({'menu':'flat','inner_paths':4,'schedule':'half_doubling'}, self.dates, c)
            q = new.MenuPool({'menu':'batch','inner_paths':4,'schedule':'half_doubling'}, self.dates, b)
            x = list(p.terminal_chunks(self.x, self.w, 3, 20, 928, width=width))
            y = list(q.terminal_chunks(self.x, self.w, 3, 20, 928, width=width))
            for (i, px), (j, qx) in zip(x, y):
                self.assertEqual(i, j); np.testing.assert_array_equal(px, qx)
            self.assertEqual(p.last_selection['path_counts'], q.last_selection['path_counts'])

    def test_all_modes_dated_request_rejection_and_compact_replay(self):
        b = self.bundle
        self.assertTrue(all(t['validation_stop'] <= len(self.x) for t in b['trace']))
        terms = b['fold_coefficients']; k = len(terms['valid'])
        replay = learn(terms['A'], unpack_B(terms['B_upper'], k), terms['represented'])
        np.testing.assert_array_equal(replay['weights'], b['sequential']['weights'])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'folds.npz'; new.save_fold_coefficients(b, path)
            with np.load(path, allow_pickle=False) as stored:
                np.testing.assert_array_equal(stored['A'], terms['A'])
                self.assertEqual(json.loads(str(stored['metadata']))['trace'], b['trace'])
            cached, metadata = new.load_fold_coefficients(path, self.x, self.w, 3, self.dates,
                inner_paths=4, schedule_rule='half_doubling')
            np.testing.assert_array_equal(cached['B_upper'], terms['B_upper'])
            with self.assertRaises(ValueError):
                new.load_fold_coefficients(path, self.x, self.w[::-1], 3, self.dates,
                    inner_paths=4, schedule_rule='half_doubling')
        for mode in new.MODES:
            q = new.MenuPool({'menu':mode,'inner_paths':4,'schedule':'half_doubling'}, self.dates, b)
            chunks = list(q.terminal_chunks(self.x, self.w, 3, 20, 928, width=2))
            self.assertEqual(np.concatenate([a for _, a in chunks], axis=1).shape, (20, 3))
            counts = np.asarray(q.last_selection['path_counts'])
            self.assertEqual(counts.sum(), 20); self.assertTrue(np.all(counts % 2 == 0))
            with self.assertRaises(ValueError): list(q.terminal_chunks(self.x+.01, self.w, 3, 20, 928))
            shifted = new.MenuPool(q.config, self.dates+np.timedelta64(1, 'D'), b)
            with self.assertRaises(ValueError): list(shifted.terminal_chunks(self.x, self.w, 3, 20, 928))


if __name__ == '__main__':
    unittest.main(verbosity=2)
