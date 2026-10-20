"""Numeric-law and exact-score checks before any new benchmark admission."""
import unittest
import numpy as np
import streaming as s
import bounded_public
import bounded_public_without_replacement as cycle
from simfolio_forecasting_methodology.runner import empirical_crps_by_horizon


class StreamingTests(unittest.TestCase):
    def test_original_block_draws_exact(self):
        for n, horizon in [(1, 17), (7, 4), (13, 58), (101, 303)]:
            for cyc, reference in [(False, bounded_public.bounded_block_indices), (True, cycle.bounded_cycle_indices)]:
                stream = s.BlockStream(n, horizon, 24, 373, cycles=cyc)
                chunks = [stream.take(min(7, horizon-i)) for i in range(0, horizon, 7)]
                actual = np.concatenate(chunks, axis=1)
                expected = reference(n, horizon, 24, np.random.default_rng(373))
                np.testing.assert_array_equal(actual, expected)
                if cyc:
                    for row in actual:
                        for i in range(0, horizon, n):
                            self.assertEqual(len(set(row[i:i+n])), len(row[i:i+n]))

    def test_chunk_invariant_indices(self):
        history = np.random.default_rng(52).normal(.001, .02, 600)
        for cls, args in [(s.AdaptiveStream, (history, 357, 240, 1)),
                          (s.FhsStream, (600, 240, 1))]:
            a, b = cls(*args), cls(*args)
            full = a.take(357)
            split = np.concatenate([b.take(min(19, 357-i)) for i in range(0, 357, 19)], axis=1)
            np.testing.assert_array_equal(full, split)

    def test_native_accounting_exact_same_indices(self):
        history = np.array([-.93, -.1, .001, .07, .85])
        source = np.clip(np.expm1(history), -.999999, np.finfo(float).max/10)
        indices = np.random.default_rng(3).integers(0, len(history), (24, 171))
        delta = np.linspace(-.03, .03, 24)
        ref, _, _ = s.incumbent.retained.parent.incumbent.prior.shifted_native_paths(history, source, indices, delta)
        states = (np.ones(24), np.zeros(24), np.zeros(24))
        actual = []
        for i in range(0, 171, 17):
            part, _, _ = s.native_block(history, source, indices[:, i:i+17], delta, *states)
            actual.append(part)
        np.testing.assert_array_equal(np.concatenate(actual, axis=1), np.cumsum(ref, axis=1))

    def test_FHS_accounting_exact_same_indices(self):
        history = np.random.default_rng(28).normal(.0004, .014, 800)
        fit = s.incumbent.retained.FHS.base.fit_variance_target(history)
        indices = np.random.default_rng(31).integers(0, len(history), (24, 173))
        delta = np.linspace(-.0001, .0001, 24)
        ref, _, _ = s.incumbent.retained.FHS.parameter_paths(fit, indices, delta)
        states = (np.ones(24), np.zeros(24), np.zeros(24))
        variance = np.full(24, fit['next_variance_normalized'])
        actual = []
        for i in range(0, 173, 17):
            part, _, _ = s.fhs_block(fit['pool'], indices[:, i:i+17], fit['mean']+delta,
                fit['target_variance'], fit['alpha'], fit['beta'], variance, *states)
            actual.append(part)
        np.testing.assert_array_equal(np.concatenate(actual, axis=1), np.cumsum(ref, axis=1))

    def test_four_models_terminal_and_CRPS_chunk_invariance(self):
        history = np.random.default_rng(32).normal(.0002, .012, 600)
        target = np.random.default_rng(55).normal(0, .1, 137)
        for mid in [s.ADAPTIVE, s.REPLACEMENT, s.CYCLES, s.BEST]:
            full = np.concatenate([x for _, x in s.terminal_chunks(mid, history, 137, 240, 95, width=137)], axis=1)
            chunks = list(s.terminal_chunks(mid, history, 137, 240, 95, width=11))
            np.testing.assert_array_equal(np.concatenate([x for _, x in chunks], axis=1), full)
            scores = np.concatenate([empirical_crps_by_horizon(x, target[i:i+x.shape[1]]) for i, x in chunks])
            np.testing.assert_array_equal(scores, empirical_crps_by_horizon(full, target))
            independent = np.mean(np.abs(full-target), axis=0)-.5*np.mean(np.abs(full[:, None, :]-full[None, :, :]), axis=(0, 1))
            np.testing.assert_allclose(scores, independent, rtol=2e-13, atol=1e-13)

    def test_real_1000_members_all_four_models(self):
        history = np.random.default_rng(52).normal(.0005, .01, 600)
        for mid in [s.ADAPTIVE, s.REPLACEMENT, s.CYCLES, s.BEST]:
            for _, terminal in s.terminal_chunks(mid, history, 9, 1000, 58, width=4):
                self.assertEqual(terminal.shape[0], 1000)
                self.assertTrue(np.isfinite(terminal).all())
                self.assertGreater(len(np.unique(terminal[:, -1])), 500)


if __name__ == '__main__':
    unittest.main()
