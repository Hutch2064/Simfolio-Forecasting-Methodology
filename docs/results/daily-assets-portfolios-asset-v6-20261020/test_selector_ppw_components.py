from pathlib import Path
import os,sys,unittest
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[key]='1'
import numpy as np
from arch.bootstrap import optimal_block_length
O=Path(__file__).resolve().parent;V=O/'asset_level_v6'if(O/'asset_level_v6').exists()else O;sys.path.insert(0,str(V))
from candidates.asset_selector_ppw.blocks import fit_lengths,CircularStream,StationaryStream,PPWJointBootstrap

class CorrectedPPWComponents(unittest.TestCase):
    def setUp(self):
        self.x=np.random.default_rng(2091).normal(.0003,.01,(400,2))
        for i in range(1,len(self.x)):self.x[i]+=.6*self.x[i-1]
        self.w=np.array([.65,.35])
    def test_corrected_library_estimates_and_zero_weight_exclusion(self):
        fitted=fit_lengths(self.x,self.w);x=self.x-self.x.mean(axis=0);expected=optimal_block_length(np.column_stack((x,x*x)))
        for mode in ('stationary','circular'):self.assertEqual(fitted['lengths'][mode],max(1.,min(float(len(x)),float(expected[mode].max()))))
        extra=np.column_stack((self.x,np.linspace(-1.,1.,len(self.x))));other=fit_lengths(extra,np.r_[self.w,0.])
        self.assertEqual(fitted['lengths'],other['lengths']);self.assertFalse(fitted['estimator']['forecast_optimality_claim'])
    def test_circular_boundaries_wrap_and_chunk_invariance(self):
        whole=CircularStream(5,19,20,2092,3).take(19);stream=CircularStream(5,19,20,2092,3)
        chunks=np.concatenate([stream.take(n)for n in (1,4,2,12)],axis=1);np.testing.assert_array_equal(whole,chunks)
        self.assertTrue(np.all((whole>=0)&(whole<5)))
        for t in range(1,19):
            if t%3:np.testing.assert_array_equal(whole[:,t],(whole[:,t-1]+1)%5)
        self.assertTrue(np.any((whole[:,:-1]==4)&(whole[:,1:]==0)))
    def test_stationary_chunk_invariance(self):
        whole=StationaryStream(11,50,30,2093,4.3).take(50);stream=StationaryStream(11,50,30,2093,4.3)
        np.testing.assert_array_equal(whole,np.concatenate([stream.take(n)for n in (3,1,29,17)],axis=1))
    def test_joint_indices_and_holdings_accounting(self):
        fitted=fit_lengths(self.x,self.w);H=17;P=8;seed=2094
        for mode in ('stationary','circular'):
            length=fitted['lengths'][mode];stream=StationaryStream(len(self.x),H,P,seed,length)if mode=='stationary'else CircularStream(len(self.x),H,P,seed,max(1,min(len(self.x),int(np.floor(length+.5)))))
            indices=stream.take(H);sampled=self.x[indices]
            for interval in (0,1):
                model=PPWJointBootstrap(mode,fitted);actual=np.concatenate([v for _,v in model.terminal_chunks(self.x,self.w,H,P,seed,width=3,rebalance_every=interval)],axis=1)
                expected=np.log(np.sum(np.exp(np.cumsum(sampled,axis=1))*self.w,axis=2))if interval==0 else np.cumsum(np.log(np.sum(np.exp(sampled)*self.w,axis=2)),axis=1)
                np.testing.assert_allclose(actual,expected,rtol=0,atol=2e-14);np.testing.assert_allclose(model.last_asset_log_paths,sampled.sum(axis=1),rtol=0,atol=2e-15)
                second=PPWJointBootstrap(mode,fitted);other=np.concatenate([v for _,v in second.terminal_chunks(self.x,self.w,H,P,seed,width=1,rebalance_every=interval)],axis=1)
                self.assertEqual(actual.tobytes(),other.tobytes())
    def test_short_identification_and_prepared_request_binding(self):
        with self.assertRaises(ValueError):fit_lengths(self.x[:3],self.w)
        constant=np.tile([.01,-.02],(3,1));f=fit_lengths(constant,self.w);self.assertEqual(f['lengths'],{'stationary':1.,'circular':1.})
        for mode in ('stationary','circular'):
            paths=np.concatenate([v for _,v in PPWJointBootstrap(mode,f).terminal_chunks(constant,self.w,20,7,2095,width=4)],axis=1)
            expected=np.arange(1,21)*np.log(np.sum(self.w*np.exp(constant[0])))
            np.testing.assert_allclose(paths,np.tile(expected,(7,1)),rtol=0,atol=2e-14)
        f=fit_lengths(self.x,self.w);changed=self.x.copy();changed[-1,0]+=.01
        with self.assertRaises(ValueError):list(PPWJointBootstrap('stationary',f).terminal_chunks(changed,self.w,2,4,2096))
        with self.assertRaises(ValueError):PPWJointBootstrap('unsupported')

if __name__=='__main__':unittest.main()
