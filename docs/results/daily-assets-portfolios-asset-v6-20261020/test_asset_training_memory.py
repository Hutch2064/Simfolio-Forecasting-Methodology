import unittest
import numpy as np
from diagnose_asset_training_memory import stats

class TrainingMemoryTests(unittest.TestCase):
    def test_known_autoregression_is_recovered_from_training(self):
        rng=np.random.default_rng(117);x=np.zeros(12000);eps=rng.normal(0,1,len(x))
        for i in range(1,len(x)):x[i]=.6*x[i-1]+eps[i]
        fit=stats(x);self.assertLess(abs(fit['raw_training_AR1']-.6),.025)
        self.assertGreater(fit['Bartlett_HAC_standard_error'],0.)
        self.assertEqual(fit['training_rows'],len(x))

    def test_scale_and_location_invariance(self):
        x=np.random.default_rng(919).normal(.0002,.01,800)
        a,b=stats(x),stats(100*x+3)
        for key in ('raw_training_AR1','Bartlett_HAC_standard_error','absolute_phi_over_se'):
            self.assertAlmostEqual(a[key],b[key],places=12)

    def test_hac_matches_positive_semidefinite_bartlett_matrix(self):
        x=np.random.default_rng(99).normal(size=60);fit=stats(x);y=x-x.mean();lag=y[:-1]
        score=lag*(y[1:]-fit['raw_training_AR1']*lag);score-=score.mean()
        bandwidth=fit['training_bandwidth'];distance=np.abs(np.arange(len(score))[:,None]-np.arange(len(score))[None,:])
        kernel=np.maximum(0.,1-distance/(bandwidth+1));self.assertGreater(np.linalg.eigvalsh(kernel).min(),-1e-14)
        expected=np.sqrt(score@kernel@score)/(lag@lag)
        self.assertAlmostEqual(fit['Bartlett_HAC_standard_error'],expected,places=15)
if __name__=='__main__':unittest.main()
