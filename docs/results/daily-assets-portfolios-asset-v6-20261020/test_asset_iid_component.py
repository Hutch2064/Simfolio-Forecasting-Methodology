from pathlib import Path
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[k]='1'
import sys,unittest
import numpy as np
O=Path(__file__).resolve().parent;sys.path.insert(0,str(O/'asset_level_v6'))
from candidates.asset_iid_component.iid import JointIIDStream
from candidates.asset_iid_component.candidate import IIDPool
from candidates.asset_prior_best.candidate import AssetPool

class IIDComponentTests(unittest.TestCase):
    def test_joint_rows_are_bounded_and_chunk_invariant(self):
        a,b=JointIIDStream(21,107,100,925),JointIIDStream(21,107,100,925)
        x=a.take(107);y=np.concatenate([b.take(1),b.take(64),b.take(42)],axis=1)
        np.testing.assert_array_equal(x,y)
        self.assertTrue(np.all((x>=0)&(x<21)))
        # A shared index selects the whole row, retaining exact constituent relationships.
        history=np.stack([np.arange(21),2*np.arange(21)],axis=1)
        sampled=history[x]
        np.testing.assert_array_equal(sampled[:,:,1],2*sampled[:,:,0])

    def test_filtered_component_is_unchanged(self):
        x=np.random.default_rng(319).normal(.0001,.01,(183,2))
        a,b=IIDPool(True),AssetPool(False)
        p=np.concatenate([p for _,p in a.terminal_chunks(x,[.8,.2],117,100,231)],axis=1)
        q=np.concatenate([p for _,p in b.terminal_chunks(x,[.8,.2],117,100,231)],axis=1)
        np.testing.assert_array_equal(p[50:],q[50:])

if __name__=='__main__':unittest.main()
