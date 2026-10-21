"""All-pair byte parity, ties, continuation and convex fit equivalence."""
from pathlib import Path
import sys,unittest
import numpy as np
O=Path(__file__).resolve().parent;sys.path.insert(0,str(O/'asset_level_v6'))
import common
from candidates.asset_selector_path_objective.coefficients import accumulate as original
from candidates.asset_selector_coefficient_sweep.coefficients import accumulate as sweep
from candidates.asset_selector_universe_certified.energy import fit

class CoefficientSweepTests(unittest.TestCase):
 def compare(self,x,y,m):
  k=x.shape[0];A=np.zeros(k);B=np.zeros((k,k));a=A.copy();b=B.copy();original(x,y,m,A,B);sweep(x,y,m,a,b)
  np.testing.assert_array_equal(a,A);np.testing.assert_array_equal(b,B);return A,B
 def test_random_ensembles_and_every_pair(self):
  rng=np.random.default_rng(2461)
  for k,p,t in((1,2,3),(4,17,9),(17,100,4),(166,1000,2)):
   x=rng.normal(size=(k,p,t));self.compare(x,rng.normal(size=t),rng.uniform(.01,1,t))
 def test_ties_point_masses_and_scale_extremes(self):
  rng=np.random.default_rng(995)
  for scale in(1e-8,1.,1e8):
   x=rng.integers(-3,4,size=(12,100,7)).astype(float)*scale;x[0]=0.;x[1]=x[2];self.compare(x,np.zeros(7),np.linspace(.01,.9,7))
 def test_chunking_and_preexisting_accumulators(self):
  rng=np.random.default_rng(619);x=rng.normal(size=(9,100,23));y=rng.normal(size=23);m=rng.uniform(.01,1,23);A=rng.normal(size=9);B=rng.normal(size=(9,9));a=A.copy();b=B.copy();original(x,y,m,A,B)
  for start in range(0,23,5):sweep(x[:,:,start:start+5],y[start:start+5],m[start:start+5],a,b)
  np.testing.assert_array_equal(a,A);np.testing.assert_array_equal(b,B)
 def test_certified_fit_identical(self):
  rng=np.random.default_rng(996);x=rng.normal(size=(18,100,11));y=rng.normal(size=11);m=np.linspace(.01,.2,11);A,B=self.compare(x,y,m);a=np.zeros(18);b=np.zeros((18,18));sweep(x,y,m,a,b)
  first=fit(A,B);second=fit(a,b);np.testing.assert_array_equal(first['weights'],second['weights']);self.assertEqual(first['objective'],second['objective']);self.assertEqual(first['KKT_gap'],second['KKT_gap'])
if __name__=='__main__':unittest.main()
