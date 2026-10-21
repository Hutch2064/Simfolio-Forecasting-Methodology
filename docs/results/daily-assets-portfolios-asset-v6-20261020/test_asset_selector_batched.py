from pathlib import Path
import hashlib,os,sys,unittest
for k in('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[k]='1'
import numpy as np
O=Path(__file__).resolve().parent;V=O/'asset_level_v6'if(O/'asset_level_v6').exists()else O;sys.path.insert(0,str(V))
from candidates.asset_selector_universe import bank
from candidates.asset_selector_batched.terminal import group_terminals
from candidates.asset_selector_batched import candidate as batch
from candidates.asset_selector_universe_certified import candidate as reference

class SharedInnerSimulationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x=np.random.default_rng(191).normal(.0002,.01,size=(96,3));cls.w=np.array([.6,.3,.1])
        cls.prepared,invalid=bank.prefix_fits(cls.x,list(range(166)));assert not invalid
    def assert_bytes_equal(self,a,b):
        np.testing.assert_array_equal(a,b);self.assertEqual(hashlib.sha256(a.tobytes()).hexdigest(),hashlib.sha256(b.tobytes()).hexdigest())
    def check_group(self,family,law,*,paths=100,horizon=21):
        ids=[i for i,s in enumerate(bank.BANK)if s['kind']=='filtered'and s['family']==family and s['innovations']==law]
        bundle=self.prepared['families'][family];result=group_terminals(self.x,self.w,horizon,paths,991,ids,bundle)
        for c,index in enumerate(ids):
            model=bank.model(index,self.prepared)
            reference=np.concatenate([v for _,v in model.terminal_chunks(self.x,self.w,horizon,paths,991)],axis=1)
            self.assert_bytes_equal(result['terminals'][c],reference[:,-1])
            self.assert_bytes_equal(result['last_asset_log_paths'][c],model.last_asset_log_paths)
            self.assert_bytes_equal(result['last_asset_weights'][c],model.last_asset_weights)
        return ids,result
    def test_all162_options_and_joint_asset_states_exact(self):
        count=0
        for family in bank.fits.FAMILIES:
            for law in('Gaussian','empirical','Student'):ids,_=self.check_group(family,law);count+=len(ids)
        self.assertEqual(count,162)
    def test_chunk_invariance_for_every_family(self):
        for family in bank.fits.FAMILIES:
            ids,a=self.check_group(family,'Student',horizon=71)
            b=group_terminals(self.x,self.w,71,100,991,ids,self.prepared['families'][family],width=1)
            for key in a:self.assert_bytes_equal(a[key],b[key])
    def test1000_member_precision_and_degenerate_weights(self):
        old_w=self.w.copy();self.w=np.array([1.,0.,0.])
        try:
            for family in('constant','AR1_GARCH','SV_path'):self.check_group(family,'Gaussian',paths=1000,horizon=65)
        finally:self.w=old_w

    def test_causal_complete_request_ensemble_hash_weights_and_outputs(self):
        x=self.x[:32];a=reference.prepare(x,self.w,20);b=batch.prepare(x,self.w,20)
        self.assertEqual(a['trace'],b['trace'])
        for key in('A','B'):self.assert_bytes_equal(a['coefficients'][key],b['coefficients'][key])
        for key in a['weights_by_mode']:self.assert_bytes_equal(a['weights_by_mode'][key],b['weights_by_mode'][key])
        for config in reference.CONFIGS:
            old=np.concatenate([v for _,v in reference.HorizonPool(config,a).terminal_chunks(x,self.w,20,1000,191)],axis=1)
            new=np.concatenate([v for _,v in batch.HorizonPool(config,b).terminal_chunks(x,self.w,20,1000,191)],axis=1)
            self.assert_bytes_equal(old,new)

if __name__=='__main__':unittest.main()
