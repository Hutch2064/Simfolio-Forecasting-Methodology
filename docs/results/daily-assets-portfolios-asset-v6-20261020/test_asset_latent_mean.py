from pathlib import Path
import os,sys,unittest
for name in('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[name]='1'
import numpy as np
O=Path(__file__).resolve().parent;V=O/'asset_level_v6'if(O/'asset_level_v6').exists()else O;sys.path.insert(0,str(V))
from candidates.asset_latent_mean.candidate import LatentMean,CONFIGS,BASE_CONFIG,identity,prepare
from candidates.asset_trained_pool.candidate import TrainedPool


class LatentPathTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng=np.random.default_rng(981);n=310;state=np.zeros(2);cls.x=np.empty((n,2))
        for t in range(n):state=.92*state+.001*rng.normal(size=2);cls.x[t]=state+.009*rng.normal(size=2)
        cls.x[:,1]+=.4*cls.x[:,0];cls.bundle=prepare(cls.x)

    def test_zero_curve_replays_exact_working_parent_at_both_precisions(self):
        zero={**self.bundle,'drift':{name:[{**value,'final_mean':0.}for value in rows]for name,rows in self.bundle['drift'].items()}}
        for config in CONFIGS:
            for paths in(100,1000):
                a=np.concatenate([p for _,p in LatentMean(config,zero).terminal_chunks(self.x,[.6,.4],79,paths,982)],1)
                b=np.concatenate([p for _,p in TrainedPool(BASE_CONFIG,self.bundle['parent']).terminal_chunks(self.x,[.6,.4],79,paths,982)],1)
                self.assertEqual(a.tobytes(),b.tobytes())

    def test_all_curves_preserve_joint_paths_and_chunk_invariance(self):
        self.assertEqual(len(CONFIGS),4);self.assertEqual(len({identity(c)for c in CONFIGS}),4)
        for config in CONFIGS:
            model=LatentMean(config,self.bundle);a=np.concatenate([p for _,p in model.terminal_chunks(self.x,[.6,.4],81,100,983)],1)
            b=np.concatenate([p for _,p in LatentMean(config,self.bundle).terminal_chunks(self.x,[.6,.4],81,100,983,width=1)],1)
            self.assertEqual(a.shape,(100,81));self.assertEqual(a.tobytes(),b.tobytes());self.assertTrue(np.isfinite(a).all());np.testing.assert_allclose(model.last_asset_weights.sum(1),np.ones(100),atol=5e-15,rtol=0)

    def test_single_asset_curve_is_exact_expected_cumulative_asset_shift(self):
        from candidates.asset_latent_mean import fit
        x=self.x[:,:1];bundle=prepare(x);config={'noise':'parent_variance','selection':'likelihood'}
        a=np.concatenate([p for _,p in LatentMean(config,bundle).terminal_chunks(x,[1.],63,100,984)],1)
        b=np.concatenate([p for _,p in TrainedPool(BASE_CONFIG,bundle['parent']).terminal_chunks(x,[1.],63,100,984)],1)
        curve=fit.expected_curve(bundle['drift']['parent_variance'][0],0,63)*np.sqrt(bundle['parent']['raw']['target_variance'][0]);expected=np.cumsum(curve)
        np.testing.assert_allclose(a-b,np.tile(expected,(100,1)),atol=5e-15,rtol=0)

    def test_fit_prefix_identity_and_parent_weight_are_preserved(self):
        before=self.bundle['parent']['pool_fit']['raw_fraction'];changed=self.x.copy();changed[-1,0]+=.001
        for config in CONFIGS:
            with self.assertRaises(AssertionError):list(LatentMean(config,self.bundle).terminal_chunks(changed,[.6,.4],2,100,985))
        self.assertEqual(self.bundle['parent']['pool_fit']['raw_fraction'],before)

if __name__=='__main__':unittest.main()
