from pathlib import Path
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[k]='1'
import sys,unittest
import numpy as np
O=Path(__file__).resolve().parent;sys.path.insert(0,str(O/'asset_level_v6'))
from blocks import BlockLimits
from candidates.asset_circular_blocks.circular import CircularBlockStream
from candidates.asset_circular_blocks.candidate import CircularPool
from candidates.asset_prior_best.candidate import AssetPool

class CircularTests(unittest.TestCase):
    def test_wrap_covers_complete_history_and_streaming_is_invariant(self):
        a=CircularBlockStream(13,39,20,331,limits=BlockLimits(13,13))
        whole=a.take(39)
        self.assertTrue(np.all((whole>=0)&(whole<13)))
        for path in whole:
            for first in (0,13,26):np.testing.assert_array_equal(np.sort(path[first:first+13]),np.arange(13))
        b=CircularBlockStream(13,39,20,331,limits=BlockLimits(13,13))
        np.testing.assert_array_equal(whole,np.concatenate([b.take(7),b.take(11),b.take(21)],axis=1))

    def test_filtered_component_is_unchanged(self):
        x=np.random.default_rng(981).normal(.0002,.013,(171,2))
        a,b=CircularPool(True),AssetPool(False)
        p=np.concatenate([p for _,p in a.terminal_chunks(x,[.7,.3],93,100,441)],axis=1)
        q=np.concatenate([p for _,p in b.terminal_chunks(x,[.7,.3],93,100,441)],axis=1)
        np.testing.assert_array_equal(p[50:],q[50:])
        self.assertFalse(np.array_equal(p[:50],q[:50]))

if __name__=='__main__':unittest.main()
