"""Synthetic uncached wrapper qualification; no real panel request."""
import unittest
from unittest.mock import patch
import numpy as np
from candidates.asset_additional_learners import builder,shrinkage,gate
from candidates.asset_additional_learners.candidate import LearnerPool,identity
from . import candidate

class AdapterTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x=np.random.default_rng(514).normal(.0002,.01,(32,2));cls.w=np.array([.35,.65]);cls.h=3
        cls.dates=np.busday_offset(np.datetime64('2020-01-01'),np.arange(32))
        prefix=builder.control.prefix_fits
        with patch.object(builder,'current_geometry_record',side_effect=AssertionError('No probe allowed')),patch.object(gate,'fit',side_effect=AssertionError('No gate allowed')),patch.object(builder.control,'prefix_fits',wraps=prefix) as spy:
            cls.bundle=candidate.prepare(cls.x,cls.w,cls.h,cls.dates,width=2)
            cls.prefix_lengths=[len(call.args[0]) for call in spy.call_args_list]

    def test_uncached_builder_once_current_refit_without_probe_or_gates(self):
        self.assertEqual(self.prefix_lengths.count(len(self.x)),1)
        self.assertEqual(len(self.prefix_lengths),len(self.bundle['fold_records'])+1)
        self.assertTrue(all(r['validation_stop']<=len(self.x) for r in self.bundle['fold_records']))
        expected=shrinkage.fit(self.bundle['fold_records']);actual=self.bundle['uniform_shrinkage_fit']
        np.testing.assert_array_equal(actual['weights'],expected['weights']);self.assertEqual(actual['rho'],expected['rho']);self.assertEqual(actual['episodes'],expected['episodes'])
        self.assertEqual(self.bundle['inner_paths'],2);self.assertEqual(self.bundle['schedule_rule'],'half_doubling')

    def test_existing_uniform_adapter_bitexact_100_and_1000_joint_paths(self):
        for paths in (100,1000):
            original=LearnerPool('uniform_shrinkage',self.dates,self.bundle,self.bundle['uniform_shrinkage_fit'])
            actual=candidate.ShrinkagePool(candidate.CONFIG,self.dates,self.bundle)
            with patch.object(builder.control,'prefix_fits',side_effect=AssertionError('Final sampler must reuse the one current-prefix fit')):
                left=list(original.terminal_chunks(self.x,self.w,self.h,paths,519,width=2));right=list(actual.terminal_chunks(self.x,self.w,self.h,paths,519,width=2))
            self.assertEqual(len(left),len(right))
            for (start,a),(other,b) in zip(left,right):self.assertEqual(start,other);np.testing.assert_array_equal(a,b)
            for key,value in original.last_selection.items():
                if isinstance(value,np.ndarray):np.testing.assert_array_equal(actual.last_selection[key],value)
                else:self.assertEqual(actual.last_selection[key],value)
            self.assertTrue(actual.last_selection['one_simplex_for_whole_joint_trajectory']);self.assertFalse(actual.last_selection['automatic_fallback'])

    def test_identity_and_frozen_settings(self):
        self.assertEqual(candidate.identity(candidate.CONFIG),identity('uniform_shrinkage'))
        with self.assertRaises(ValueError):candidate.identity({**candidate.CONFIG,'inner_paths':4})
        for kwargs in ({'inner_paths':4},{'schedule_rule':'all'}):
            with patch.object(builder,'prepare',side_effect=AssertionError('Invalid settings must fail before fitting')):
                with self.assertRaises(ValueError):candidate.prepare(self.x,self.w,self.h,self.dates,**kwargs)

if __name__=='__main__':unittest.main(verbosity=2)
