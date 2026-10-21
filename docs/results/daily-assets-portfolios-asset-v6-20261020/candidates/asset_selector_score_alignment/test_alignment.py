"""Bounded analytic and real complete-bank contract verification, no panel."""
import inspect
import itertools
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from candidates.asset_selector_score_alignment import candidate as new
from candidates.asset_selector_endpoint_flat import candidate as parent


class AnalyticTests(unittest.TestCase):
    def test_independent_36_axis_enumeration(self):
        expected = set(itertools.product(['endpoint', 'uniform_days', 'log_calendar'],
            ['raw', 'train_SD', 'train_SD_sqrt_lead', 'sqrt_lead'], ['batch', 'boa', 'progressive']))
        configs = new.configurations(8)
        actual = [(c['criterion'], c['scale'], c['learner']) for c in configs]
        self.assertEqual(len(configs), 36); self.assertEqual(len(set(actual)), 36)
        self.assertEqual(set(actual), expected)
        self.assertEqual(len(set(new.identity(c) for c in configs)), 36)
        self.assertTrue(all(c['inner_paths'] == 8 and c['schedule'] == 'half_doubling' for c in configs))
        self.assertNotIn('heldout', inspect.signature(new.prepare).parameters)
        self.assertEqual(set(inspect.signature(new.replay).parameters), {'cache'})

    def test_exact_analytic_criterion_masses_and_scales(self):
        dates = np.array(['2020-01-03', '2020-01-06', '2020-01-07'], dtype='datetime64[D]')
        expected = np.diff([0., np.log(2.), np.log(5.), np.log(6.)])/np.log(6.)
        np.testing.assert_allclose(new.lead_masses('log_calendar', '2020-01-02', dates), expected, atol=1e-15)
        np.testing.assert_array_equal(new.lead_masses('uniform_days', '2020-01-02', dates), [1., 1., 1.])
        np.testing.assert_array_equal(new.lead_masses('endpoint', '2020-01-02', dates), [0., 0., 1.])
        self.assertEqual(new.denominator('raw', .25, 4), 1.)
        self.assertEqual(new.denominator('train_SD', .25, 4), .25)
        self.assertEqual(new.denominator('train_SD_sqrt_lead', .25, 4), .5)
        self.assertEqual(new.denominator('sqrt_lead', .25, 4), 2.)
        for bad in [0., np.nan, np.inf]:
            with self.assertRaises(ArithmeticError): new.denominator('train_SD', bad, 4)
        with self.assertRaises(ValueError): new.lead_masses('log_calendar', '2020-01-04', dates)


class DatedFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng = np.random.default_rng(9211)
        cls.x = rng.normal(.0002, .01, size=(32, 2)); cls.w = np.array([.35, .65])
        # Weekend gaps exercise actual historical calendar weighting.
        cls.dates = np.busday_offset(np.datetime64('2020-01-01'), np.arange(32))
        cls.kw = dict(inner_paths=2, schedule_rule='half_doubling', width=2)
        cls.control = parent.prepare(cls.x, cls.w, 3, cls.dates, **cls.kw)
        with patch.object(parent, 'all_chunks', wraps=parent.all_chunks) as generated:
            cls.bundle = new.prepare(cls.x, cls.w, 3, cls.dates, **cls.kw)
            cls.fold_cube_calls = generated.call_count

    def test_exact_frozen_control_and_one_cube_stream_per_fold(self):
        c, b = self.control, self.bundle
        self.assertEqual(self.fold_cube_calls, len(b['trace']))
        for old, current in zip(c['trace'], b['trace']):
            for k, value in old.items(): self.assertEqual(value, current[k])
        for axis in ['A', 'B']:
            np.testing.assert_array_equal(c['coefficients']['flat'][axis], b['coefficients'][('endpoint','sqrt_lead')][axis])
        np.testing.assert_array_equal(c['weights_by_mode']['flat'], b['weights_by_mode']['endpoint:sqrt_lead:batch'])
        for width in [1, 2, 3]:
            p = parent.MenuPool({'menu':'flat','inner_paths':2,'schedule':'half_doubling'}, self.dates, c)
            q = new.ScorePool({'criterion':'endpoint','scale':'sqrt_lead','learner':'batch','inner_paths':2,'schedule':'half_doubling'}, self.dates, b)
            control = list(p.terminal_chunks(self.x, self.w, 3, 20, 928, width=width))
            current = list(q.terminal_chunks(self.x, self.w, 3, 20, 928, width=width))
            for (i, x), (j, y) in zip(control, current):
                self.assertEqual(i,j); np.testing.assert_array_equal(x, y)
            self.assertEqual(p.last_selection['path_counts'], q.last_selection['path_counts'])

    def test_36_finite_forecasts_and_request_binding(self):
        for config in new.configurations(2):
            q = new.ScorePool(config, self.dates, self.bundle)
            chunks = list(q.terminal_chunks(self.x, self.w, 3, 20, 928, width=2))
            values = np.concatenate([v for _, v in chunks], axis=1)
            self.assertEqual(values.shape, (20,3)); self.assertTrue(np.isfinite(values).all())
            counts = np.asarray(q.last_selection['path_counts'])
            self.assertEqual(counts.sum(), 20); self.assertTrue(np.all(counts % 2 == 0))
            with self.assertRaises(ValueError): list(q.terminal_chunks(self.x+.001, self.w, 3, 20, 928))
        q = new.ScorePool(new.configurations(2)[0], self.dates+np.timedelta64(1,'D'), self.bundle)
        with self.assertRaises(ValueError): list(q.terminal_chunks(self.x, self.w, 3, 20, 928))

    def test_compact_replay_prefix_sd_and_temporal_consistency(self):
        b = self.bundle; cache = b['fold_coefficients']
        for trace, sd in zip(b['trace'], cache['training_SD']):
            self.assertEqual(sd, np.std(parent.portfolio_returns(self.x[:trace['training_rows']], self.w), ddof=1))
            self.assertLessEqual(trace['validation_stop'], len(self.x))
        learned, fits, coefficients, online = new.replay(cache)
        for mode in learned: np.testing.assert_array_equal(learned[mode], b['weights_by_mode'][mode])
        # An untouched prefix of fold summaries has the same pre/post-update BOA trajectory.
        short = {key: (value[:, :1].copy() if key in ('A','B_upper') else value[:1].copy()
                 if key != 'valid' else value.copy()) for key, value in cache.items()}
        prefix_online = new.replay(short)[3]
        for axis in new.AXES:
            np.testing.assert_array_equal(prefix_online[axis]['trajectory'], online[axis]['trajectory'][:2])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'request.npz'; new.save_cache(b,path)
            loaded, metadata = new.load_cache(path,self.x,self.w,3,self.dates,inner_paths=2)
            for name in cache: np.testing.assert_array_equal(cache[name],loaded[name])
            with self.assertRaises(ValueError): new.load_cache(path,self.x,self.w,3,self.dates,inner_paths=8)

    def test_later_validation_outcome_cannot_change_earlier_coefficient_streams(self):
        changed = self.x.copy(); changed[-1] += .003
        later = new.prepare(changed,self.w,3,self.dates,**self.kw)
        cache = self.bundle['fold_coefficients']; updated = later['fold_coefficients']
        for name in ['A','B_upper']:
            np.testing.assert_array_equal(cache[name][:,:-1],updated[name][:,:-1])
        np.testing.assert_array_equal(cache['training_SD'],updated['training_SD'])
        self.assertFalse(np.array_equal(cache['A'][:,-1],updated['A'][:,-1]))

    def test_daily_sum_and_log_normalization_independent_reconstruction(self):
        # Rebuild one fold's raw per-lead laws independently of the accumulator.
        b = self.bundle; t = b['trace'][0]; stop = t['training_rows']; h = t['validation_horizon']
        valid = t['eligible_this_fold']; past = self.x[:stop]
        prepared, failures = parent.prefix_fits(past,valid,self.w); self.assertFalse(failures)
        cube = np.concatenate([values for _,values in parent.all_chunks(past,self.w,h,2,t['seed'],valid,prepared,width=1)], axis=2)
        returns = parent.portfolio_returns(self.x[stop:stop+h],self.w)
        observations = np.cumsum(returns); observations[-1] = returns.sum()
        terms = [parent.energy.coefficients(cube[:,:,i], observations[i]) for i in range(h)]
        raw = np.asarray([a for a,b in terms]); sd = np.std(parent.portfolio_returns(past,self.w),ddof=1)
        for axis, (criterion,scale) in enumerate(new.AXES):
            mass = new.lead_masses(criterion,self.dates[stop-1],self.dates[stop:stop+h])
            denom = np.array([new.denominator(scale,sd,i+1) for i in range(h)])
            expected = np.zeros(len(valid))
            for a, divisor, m in zip(raw,denom,mass): expected += m*(a/divisor)
            np.testing.assert_array_equal(expected,b['fold_coefficients']['A'][axis,0])


if __name__ == '__main__': unittest.main(verbosity=2)
