from pathlib import Path
import os,sys,unittest
for name in('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[name]='1'
import numpy as np
O=Path(__file__).resolve().parent;V=O/'asset_level_v6'if(O/'asset_level_v6').exists()else O;sys.path.insert(0,str(V))
from candidates.asset_random_measure import measure


class MeasureTests(unittest.TestCase):
    def test_geometric_conditional_means_and_stationary_duality_are_exact(self):
        rng=np.random.default_rng(961);x=rng.normal(size=(17,2));q=rng.dirichlet(np.ones(17));length=3.7;r=1-1/length;p=1/length
        b=measure.conditional_means(x,length);expected=np.stack([sum(p*r**k*x[(i+k)%len(x)]for k in range(len(x)))/(1-r**len(x))for i in range(len(x))])
        np.testing.assert_allclose(b,expected,rtol=3e-15,atol=5e-16)
        shift=np.roll(np.eye(len(x)),1,axis=1);transition=r*shift+p*np.tile(q,(len(x),1));system=transition.T-np.eye(len(x));system[-1]=1.;target=np.zeros(len(x));target[-1]=1.;stationary=np.linalg.solve(system,target)
        np.testing.assert_allclose(q@b,stationary@x,rtol=2e-14,atol=5e-16)

    def test_random_stationary_mean_covariance_matches_dirichlet_formula(self):
        rng=np.random.default_rng(962);x=rng.normal(size=(13,2));length=2.8;concentration=9.;b=measure.conditional_means(x,length)
        stream=measure.RandomRows(len(x),100000,963,length,concentration);q=np.diff(np.column_stack((np.zeros(len(stream.cdf)),stream.cdf)),axis=1);means=q@b;centered=b-b.mean(0);expected=centered.T@centered/len(b)/(concentration+1)
        np.testing.assert_allclose(np.cov(means.T,bias=True),expected,rtol=.03,atol=.0001)

    def test_stationary_initial_row_distribution_matches_markov_solution(self):
        rng=np.random.default_rng(964);q=rng.dirichlet(np.ones(11));length=3.2;r=1-1/length
        transition=r*np.roll(np.eye(len(q)),1,axis=1)+(1-r)*np.tile(q,(len(q),1));system=transition.T-np.eye(len(q));system[-1]=1.;rhs=np.zeros(len(q));rhs[-1]=1.;expected=np.linalg.solve(system,rhs)
        sampled=measure.initial_rows(np.tile(q,(180000,1)),length,rng.random(180000));actual=np.bincount(sampled,minlength=len(q))/len(sampled)
        np.testing.assert_allclose(actual,expected,atol=.0017,rtol=.01)

    def test_stream_chunk_identity_and_zero_concentration_boundary(self):
        a=measure.RandomRows(31,100,965,4.2,0.);q=np.diff(np.column_stack((np.zeros(100),a.cdf)),axis=1);self.assertTrue(np.all((q==0)|(q==1)))
        for concentration in(0.,31.,6.2):
            first=measure.RandomRows(31,100,966,4.2,concentration).take(83);stream=measure.RandomRows(31,100,966,4.2,concentration);pieces=np.concatenate([stream.take(1)for _ in range(83)],1)
            np.testing.assert_array_equal(first,pieces);self.assertTrue(((first>=0)&(first<31)).all())

    def test_all_six_joint_laws_have_exact_path_counts_and_no_extra_mean_draw(self):
        from candidates.asset_random_measure.candidate import RandomMeasure,CONFIGS,identity,prepare
        rng=np.random.default_rng(967);x=rng.normal(size=(180,2))*.013;x[:,1]+=.4*x[:,0];bundle=prepare(x)
        self.assertEqual(len(CONFIGS),6);self.assertEqual(len({identity(c)for c in CONFIGS}),6);self.assertGreaterEqual(bundle['matched_concentration'],0)
        for config in CONFIGS:
            for paths in(100,1000):
                model=RandomMeasure(config,bundle);a=np.concatenate([p for _,p in model.terminal_chunks(x,[.6,.4],71,paths,968)],1);b=np.concatenate([p for _,p in RandomMeasure(config,bundle).terminal_chunks(x,[.6,.4],71,paths,968,width=1)],1)
                self.assertEqual(a.shape,(paths,71));np.testing.assert_array_equal(a,b);self.assertTrue(np.isfinite(a).all());np.testing.assert_allclose(model.last_asset_weights.sum(1),np.ones(paths),atol=5e-15,rtol=0)

if __name__=='__main__':unittest.main()
