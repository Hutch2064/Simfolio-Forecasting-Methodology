"""Ensure the parent contrast changes raw blocks only, preserving filtered paths."""
from pathlib import Path
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[k]='1'
import sys,unittest
import numpy as np
O=Path(__file__).resolve().parent
sys.path.insert(0,str(O/'asset_level_v6'))
from candidates.asset_prior_best.candidate import AssetPool

class ParentContrastTests(unittest.TestCase):
    def test_only_raw_component_changes(self):
        x=np.random.default_rng(281).normal(.0003,.012,(157,3))
        a,b=AssetPool(False),AssetPool(True)
        p=np.concatenate([p for _,p in a.terminal_chunks(x,[.6,.3,.1],211,100,781)],axis=1)
        q=np.concatenate([p for _,p in b.terminal_chunks(x,[.6,.3,.1],211,100,781)],axis=1)
        np.testing.assert_array_equal(p[50:],q[50:])
        self.assertFalse(np.array_equal(p[:50],q[:50]))
        np.testing.assert_array_equal(a.last_batch['covariance'],b.last_batch['covariance'])

if __name__=='__main__':unittest.main()
