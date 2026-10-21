from pathlib import Path
import os,sys,unittest
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[key]='1'
from unittest.mock import patch
import numpy as np
O=Path(__file__).resolve().parent;sys.path.insert(0,str(O/'asset_level_v6'))
from candidates.asset_equilibrium_blocks.renewal import EquilibriumBlockStream,initial_residual
from candidates.asset_equilibrium_blocks.candidate import EquilibriumPool
from candidates.asset_fading_vol.candidate import FadingVolPool
from candidates.asset_circular_blocks.circular import CircularBlockStream

class EquilibriumBlockTests(unittest.TestCase):
    def test_exact_integer_cutpoint_distribution(self):
        for m in (1,2,7,31):
            counts=np.zeros(m)
            class Draws:
                def __init__(self,a,b):self.values=iter((a,b))
                def integers(self,*args):return next(self.values)
            for first in range(m+1):
                for second in range(m):counts[initial_residual(Draws(first,second),m)-1]+=1
            np.testing.assert_array_equal(counts,2*np.arange(m,0,-1))
            self.assertEqual(int(counts.sum()),m*(m+1))

    def test_joint_row_and_residual_distribution_is_invariant(self):
        from fractions import Fraction
        for n,m in ((1,1),(3,2),(5,7)):
            pi=[[Fraction(2*(m-r),n*m*(m+1)) for r in range(m)] for i in range(n)]
            value=pi
            for _ in range(12):
                update=[[Fraction(0) for r in range(m)] for i in range(n)]
                for i in range(n):
                    for r in range(1,m+1):
                        if r>1:update[(i+1)%n][r-2]+=value[i][r-1]
                        else:
                            for j in range(n):
                                for length in range(m):update[j][length]+=value[i][0]/(n*m)
                self.assertEqual(update,pi);value=update
            if m>1:self.assertNotEqual(pi,[[Fraction(1,n*m) for r in range(m)] for i in range(n)])

    def test_bounds_historical_wrap_and_chunk_invariance(self):
        a,b=EquilibriumBlockStream(17,83,100,298),EquilibriumBlockStream(17,83,100,298)
        x=a.take(83);y=np.concatenate([b.take(1) for _ in range(83)],axis=1)
        np.testing.assert_array_equal(x,y);self.assertTrue(np.all((0<=x)&(x<17)))
        for starts,lengths in a.descriptors:
            self.assertEqual(lengths.sum(),83);self.assertTrue(np.all((lengths>=1)&(lengths<=17)))
        np.testing.assert_array_equal(EquilibriumBlockStream(1,12,4,97).take(12),np.zeros((4,12),dtype=int))

    def test_same_financial_law_and_unchanged_filtered_half(self):
        x=np.random.default_rng(61).normal(.0002,.01,(260,2));w=[.6,.4]
        expected=np.concatenate([p for _,p in FadingVolPool(False).terminal_chunks(x,w,91,100,314)],axis=1)
        with patch('candidates.asset_equilibrium_blocks.candidate.EquilibriumBlockStream',CircularBlockStream):
            same=np.concatenate([p for _,p in EquilibriumPool().terminal_chunks(x,w,91,100,314)],axis=1)
        np.testing.assert_array_equal(same,expected)
        changed=np.concatenate([p for _,p in EquilibriumPool().terminal_chunks(x,w,91,100,314)],axis=1)
        np.testing.assert_array_equal(changed[50:],expected[50:]);self.assertFalse(np.array_equal(changed[:50],expected[:50]))
if __name__=='__main__':unittest.main()
