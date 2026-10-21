from pathlib import Path
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[k]='1'
import sys,unittest
import numpy as np
O=Path(__file__).resolve().parent;sys.path.insert(0,str(O/'asset_level_v6'))
from candidates.asset_adaptive_blocks.stationary import StationaryStream,block_length
from candidates.asset_adaptive_blocks.candidate import AdaptivePool
from candidates.asset_circular_blocks.candidate import CircularPool

class AdaptiveBlockTests(unittest.TestCase):
    def test_index_stream_is_chunk_invariant_and_preserves_rows(self):
        a,b=StationaryStream(17,93,100,191,7.),StationaryStream(17,93,100,191,7.)
        whole=a.take(93);parts=np.concatenate([b.take(1),b.take(64),b.take(28)],axis=1)
        np.testing.assert_array_equal(whole,parts)
        self.assertTrue(np.all((whole>=0)&(whole<17)))
        history=np.stack([np.arange(17),3*np.arange(17)],axis=1)
        rows=history[whole];np.testing.assert_array_equal(rows[:,:,1],3*rows[:,:,0])

    def test_no_restart_limit_continues_and_wraps(self):
        result=StationaryStream(13,41,20,122,1e20).take(41)
        np.testing.assert_array_equal(result,(result[:,0,None]+np.arange(41))%13)

    def test_selector_is_scale_column_and_location_invariant(self):
        rng=np.random.default_rng(588);x=rng.normal(size=(1200,2))
        for t in range(1,len(x)):x[t]=.8*x[t-1]+x[t]
        a=block_length(x)
        self.assertAlmostEqual(a,block_length(x[:,::-1]*[.3,5.]+[2.,-1.]),places=8)
        iid=block_length(rng.normal(size=(1200,2)))
        self.assertGreater(a,iid)

    def test_filtered_component_and_horizon_prefix_are_unchanged(self):
        x=np.random.default_rng(737).normal(.0003,.011,(500,2))
        a,b=AdaptivePool(),CircularPool(True)
        p=np.concatenate([p for _,p in a.terminal_chunks(x,[.7,.3],93,100,551)],axis=1)
        q=np.concatenate([p for _,p in b.terminal_chunks(x,[.7,.3],93,100,551)],axis=1)
        np.testing.assert_array_equal(p[50:],q[50:])
        first=np.concatenate([p for _,p in AdaptivePool().terminal_chunks(x,[.7,.3],13,100,551)],axis=1)
        np.testing.assert_array_equal(first,p[:,:13])

if __name__=='__main__':unittest.main()
