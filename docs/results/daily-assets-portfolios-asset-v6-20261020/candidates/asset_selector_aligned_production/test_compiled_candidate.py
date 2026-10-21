"""Exact all-work reference parity; no recipe, fit, lead or path approximation."""
import unittest
from unittest.mock import patch
import numpy as np
from candidates.asset_selector_score_alignment import candidate as reference
from candidates.asset_selector_aligned_production import compiled_candidate as candidate
from simfolio_forecasting_methodology.evaluation import empirical_crps_by_horizon


class AlignedTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng=np.random.default_rng(9211);cls.x=rng.normal(.0002,.01,size=(32,2));cls.w=np.array([.35,.65])
        cls.dates=np.busday_offset(np.datetime64('2020-01-01'),np.arange(32))
        cls.r=reference.prepare(cls.x,cls.w,3,cls.dates,inner_paths=2,width=2)
        with patch.object(reference.control.energy,'fit',wraps=reference.control.energy.fit)as fit, \
             patch.object(reference.control,'prefix_fits',wraps=reference.control.prefix_fits)as prefix, \
             patch.object(candidate,'coefficients',wraps=candidate.coefficients)as coefficients:
            cls.c=candidate.prepare(cls.x,cls.w,3,cls.dates,width=2)
            cls.fit_calls=fit.call_count;cls.prefix_calls=prefix.call_args_list;cls.coefficient_calls=coefficients.call_count
    def test_exact_coefficients_weights_trace_certificate_and_all_work(self):
        self.assertEqual(self.fit_calls,1)
        self.assertEqual(self.coefficient_calls,sum(t['validation_horizon']for t in self.r['trace']))
        self.assertEqual(len(self.prefix_calls),len(self.r['trace'])+1)
        self.assertEqual(self.c['trace'],self.r['trace'])
        for a in ('A','B'):
            np.testing.assert_array_equal(self.c['coefficients'][candidate.AXIS][a],self.r['coefficients'][candidate.AXIS][a])
        np.testing.assert_array_equal(self.c['weights_by_mode'][candidate.MODE],self.r['weights_by_mode'][candidate.MODE])
        for k,v in self.r['fit'][candidate.AXIS].items():
            if isinstance(v,np.ndarray):np.testing.assert_array_equal(v,self.c['fit'][candidate.AXIS][k])
            else:self.assertEqual(v,self.c['fit'][candidate.AXIS][k])
        self.assertEqual(list(self.prefix_calls[-1].args[1]),self.r['coefficients'][candidate.AXIS]['valid'])
    def test_full1000_paths_and_every_lead_loss_multiple_seeds_and_widths(self):
        actual=np.array([.01,-.004,.008])
        for seed in [928,1928]:
            for width in [1,2,64]:
                r=reference.ScorePool(candidate.CONFIG,self.dates,self.r)
                c=candidate.ScorePool(candidate.CONFIG,self.dates,self.c)
                left=list(r.terminal_chunks(self.x,self.w,3,1000,seed,width=width))
                right=list(c.terminal_chunks(self.x,self.w,3,1000,seed,width=width))
                for (start,a),(position,b)in zip(left,right):
                    self.assertEqual(start,position);np.testing.assert_array_equal(a,b)
                    np.testing.assert_array_equal(empirical_crps_by_horizon(a,actual[start:start+a.shape[1]]),empirical_crps_by_horizon(b,actual[start:start+b.shape[1]]))
                self.assertEqual(r.last_selection['path_counts'],c.last_selection['path_counts'])
    def test_dated_binding_and_invalid_inputs(self):
        model=candidate.ScorePool(candidate.CONFIG,self.dates,self.c)
        for x,w,h in [(self.x+.001,self.w,3),(self.x,self.w[::-1],3),(self.x,self.w,2)]:
            with self.assertRaises(ValueError):list(model.terminal_chunks(x,w,h,1000,928))
        with self.assertRaises(ValueError):candidate.ScorePool({**candidate.CONFIG,'inner_paths':8},self.dates)
        with self.assertRaises(ArithmeticError):candidate.prepare(np.zeros_like(self.x),self.w,3,self.dates)


class PairLoopTests(unittest.TestCase):
    def test_exact_canonical_pairs_for_all_members_not_P2_specific(self):
        from candidates.asset_selector_aligned_production.coefficients import coefficients,pair_matrix
        rng=np.random.default_rng(771)
        for recipes,members in [(1,2),(2,4),(9,8),(167,2),(13,32)]:
            x=rng.normal(size=(recipes,members))
            for values in [x,np.zeros_like(x),np.round(x),x*1e-8,x*1e8]:
                expected=reference.control.energy.coefficients(values,.3)
                actual=coefficients(values,.3)
                for a,b in zip(expected,actual):np.testing.assert_array_equal(a,b)
        self.assertFalse(pair_matrix.targetoptions['fastmath'])

if __name__=='__main__':unittest.main(verbosity=2)
