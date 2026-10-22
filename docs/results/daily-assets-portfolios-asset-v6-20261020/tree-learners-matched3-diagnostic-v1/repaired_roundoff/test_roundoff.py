import unittest
import numpy as np
from .models import canonical_alphas
from .objective import NumericalFailure

class RoundoffTests(unittest.TestCase):
    def test_tiny_negative_is_exact_positive_zero_only(self):
        raw=np.array([0.,-2.0543252740130515e-33,.1,1.]);clean,receipt=canonical_alphas(raw,np.array([.5,1.]),np.array([.3,.7]))
        self.assertEqual(clean[1].tobytes(),np.float64(0.).tobytes());np.testing.assert_array_equal(clean[[0,2,3]],raw[[0,2,3]])
        self.assertEqual(receipt['canonicalized_negative_count'],1);self.assertGreater(receipt['roundoff_bound'],abs(raw[1]));np.testing.assert_array_equal(raw,[0.,-2.0543252740130515e-33,.1,1.])

    def test_material_negative_and_nonfinite_fail_closed(self):
        for raw in ([-1e-3,0.],[np.nan],[np.inf]):
            with self.assertRaises(NumericalFailure):canonical_alphas(raw,[.5,1.],[.3,.7])
        with self.assertRaises(NumericalFailure):canonical_alphas([-1e-33],[0.,0.],[.3,.7])

    def test_all_nonnegative_bits_and_path_order_unchanged(self):
        raw=np.array([-0.,0.,np.nextafter(0.,1.),.1,1.,np.finfo(float).max]);clean,receipt=canonical_alphas(raw,[np.inf],[np.nan])
        self.assertEqual(clean.tobytes(),raw.tobytes());self.assertEqual(receipt['canonicalized_negative_count'],0);self.assertIsNone(receipt['roundoff_bound'])
        self.assertEqual(sorted(set([0.,*map(float,clean),float('inf')])),sorted(set([0.,*map(float,raw),float('inf')])))

if __name__=='__main__':unittest.main(verbosity=2)
