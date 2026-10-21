from pathlib import Path
import os,sys,unittest
for k in('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[k]='1'
import numpy as np
O=Path(__file__).resolve().parent;V=O/'asset_level_v6' if(O/'asset_level_v6').exists() else O;sys.path.insert(0,str(V))
from candidates.asset_selector_path_objective import coefficients as c
from candidates.asset_selector_stacking import energy
from candidates.asset_selector_universe_certified.energy import fit

class PathCoefficientTests(unittest.TestCase):
    def setUp(self):
        self.x=np.random.default_rng(911).normal(size=(4,37,9));self.y=np.linspace(-.4,.3,9);self.days=np.array([1,2,5,6,7,8,9,12,13]);self.mass=c.lead_masses(self.days)
    def test_exact_finite_ensemble_proper_score_identity(self):
        A=np.zeros(4);B=np.zeros((4,4));c.accumulate(self.x,self.y,self.mass,A,B)
        expected_A=np.zeros(4);expected_B=np.zeros((4,4))
        for t in range(9):
            a,b=energy.coefficients(self.x[:,:,t],self.y[t]);expected_A+=self.mass[t]*a;expected_B+=self.mass[t]*b
        np.testing.assert_allclose(A,expected_A,rtol=3e-15,atol=3e-15);np.testing.assert_allclose(B,expected_B,rtol=3e-15,atol=3e-15)
        w=np.array([.1,.2,.3,.4]);direct=0.
        for t in range(9):
            flat=self.x[:,:,t].ravel();p=np.repeat(w/37,37)
            direct+=self.mass[t]*(p@np.abs(flat-self.y[t])-.5*np.sum(p[:,None]*p[None,:]*np.abs(flat[:,None]-flat[None,:])))
        self.assertAlmostEqual(energy.objective(w,A,B),direct,places=14)
    def test_chronological_chunk_accumulation_is_byte_identical(self):
        A=np.zeros(4);B=np.zeros((4,4));c.accumulate(self.x,self.y,self.mass,A,B)
        for width in(1,2,4):
            a=np.zeros(4);b=np.zeros((4,4))
            for start in range(0,9,width):c.accumulate(self.x[:,:,start:start+width],self.y[start:start+width],self.mass[start:start+width],a,b)
            self.assertEqual(A.tobytes(),a.tobytes());self.assertEqual(B.tobytes(),b.tobytes())
    def test_calendar_mass_and_convex_certificate(self):
        self.assertAlmostEqual(np.sum(self.mass*np.sqrt(np.arange(1,10))),1.,places=15)
        A=np.zeros(4);B=np.zeros((4,4));c.accumulate(self.x,self.y,self.mass,A,B);f=fit(A,B)
        self.assertLessEqual(f['objective'],min(A-.5*np.diag(B))+1e-12)
        for bad in([0,1],[1,1],[2,1],[]):
            with self.assertRaises(ValueError):c.lead_masses(bad)

if __name__=='__main__':unittest.main()
