from pathlib import Path
import os,sys,unittest
for name in('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[name]='1'
import numpy as np
from scipy.integrate import quad
from scipy.special import ndtri
from scipy.stats import norm
O=Path(__file__).resolve().parent;V=O/'asset_level_v6'if(O/'asset_level_v6').exists()else O;sys.path.insert(0,str(V))
from candidates.asset_SAS_skew import skew
from candidates.asset_SAS_skew.candidate import SkewPool,CONFIGS,BASE_CONFIG,prepare
from candidates.asset_trained_pool.candidate import TrainedPool,prepare as parent_prepare


class DistributionTests(unittest.TestCase):
    def test_transform_and_inverse_against_hyperbolic_definition(self):
        z=np.linspace(-20.,20.,401);m,v=skew.constants()
        for t in(-.99,-.6,.6,.99):
            epsilon=np.arctanh(t);raw=np.sinh(np.arcsinh(z)+epsilon)
            expected=(raw-np.sinh(epsilon)*m)/(np.cosh(epsilon)*np.sqrt(1+t*t*v))
            actual=skew.transform(z,t);np.testing.assert_allclose(actual,expected,rtol=0,atol=3e-13)
            np.testing.assert_allclose(skew.inverse(actual,t),z,rtol=0,atol=3e-11)

    def test_density_normalization_and_independent_Jacobian(self):
        m,v=skew.constants()
        for t in(-.8,.8):
            integral,error=quad(lambda x:np.exp(-skew.negative_log_density(x,t)),-np.inf,np.inf,epsabs=1e-10,epsrel=1e-10)
            self.assertAlmostEqual(integral,1.,places=9)
            x=np.linspace(-2.,2.,21);u=x*np.sqrt(1+t*t*v)+t*m
            z=np.sinh(np.arcsinh(u/np.sqrt(1-t*t))-np.arctanh(t))
            dz=np.sqrt(1+t*t*v)*np.cosh(np.arcsinh(u/np.sqrt(1-t*t))-np.arctanh(t))/np.sqrt(u*u+1-t*t)
            np.testing.assert_allclose(skew.negative_log_density(x,t),-norm.logpdf(z)-np.log(dz),rtol=0,atol=2e-12)

    def test_mean_variance_and_monotone_Gaussian_copula(self):
        for t in(-.8,.3,.8):
            first=quad(lambda z:float(skew.transform(z,t))*norm.pdf(z),-np.inf,np.inf,epsabs=1e-11)[0]
            second=quad(lambda z:float(skew.transform(z,t))**2*norm.pdf(z),-np.inf,np.inf,epsabs=1e-11)[0]
            self.assertLess(abs(first),2e-12);self.assertAlmostEqual(second,1.,places=10)
        z=np.random.default_rng(7601).normal(size=(4000,2))@np.array([[1.,0.],[.4,np.sqrt(.84)]]).T
        transformed=skew.transform(z,np.array([-.8,.6]))
        np.testing.assert_array_equal(np.argsort(z,axis=0),np.argsort(transformed,axis=0))

    def test_training_likelihood_recovery_and_BIC_Gaussian_endpoint(self):
        n=4000;z=ndtri((np.arange(n)+.5)/n);fit=skew.fit(skew.transform(z,.55))
        self.assertLess(abs(fit['likelihood_t']-.55),.015);self.assertNotEqual(fit['BIC_t'],0.)
        self.assertLess(fit['fitted_mean_NLL'],fit['normal_mean_NLL'])
        self.assertEqual(skew.fit(z)['BIC_t'],0.)

    def test_exact_Gaussian_array_identity_and_invalid_support(self):
        z=np.random.default_rng(7602).normal(size=(4,7,2));self.assertIs(skew.transform(z,np.zeros(2)),z)
        np.testing.assert_array_equal(skew.inverse(z,0.),z)
        for t in(-1.,1.,np.inf):
            with self.assertRaises(ValueError):skew.transform(z,t)


class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng=np.random.default_rng(7603);z=rng.normal(size=(301,2))@np.array([[1.,0.],[.4,.9]]).T;cls.x=skew.transform(z,np.array([-.5,.4]))*.01+.0002;cls.bundle=prepare(cls.x)

    def test_exact_parent_and_Gaussian_limit_at100and1000(self):
        zero=dict(self.bundle);zero['skew']={k:np.zeros(2)for k in('likelihood','BIC')}
        for n in(100,1000):
            parent=np.concatenate([v for _,v in TrainedPool(BASE_CONFIG,self.bundle).terminal_chunks(self.x,[.6,.4],71,n,7604)],axis=1)
            for config,bundle in[({'skew':'parent'},self.bundle),*[(c,zero)for c in CONFIGS]]:
                actual=np.concatenate([v for _,v in SkewPool(config,bundle).terminal_chunks(self.x,[.6,.4],71,n,7604)],axis=1);np.testing.assert_array_equal(actual,parent)

    def test_all_parent_fits_and_training_blend_unchanged(self):
        parent=parent_prepare(self.x,BASE_CONFIG['family'],BASE_CONFIG['raw_fade'])
        for key in('raw','mean','filter','pool_fit'):np.testing.assert_equal(parent[key],self.bundle[key])
        for r in self.bundle['skew_fit']:
            self.assertEqual(r['training_rows'],len(self.x));self.assertLess(abs(r['likelihood_t']),1.)

    def test_joint_path_counts_accounting_chunk_and_prefix_guard(self):
        for config in CONFIGS:
            model=SkewPool(config,self.bundle);a=np.concatenate([v for _,v in model.terminal_chunks(self.x,[.6,.4],73,100,7605)],axis=1)
            b=np.concatenate([v for _,v in SkewPool(config,self.bundle).terminal_chunks(self.x,[.6,.4],73,100,7605,width=1)],axis=1)
            self.assertEqual(a.shape,(100,73));self.assertTrue(np.isfinite(a).all());np.testing.assert_array_equal(a,b);np.testing.assert_allclose(model.last_asset_weights.sum(1),1.,atol=8e-15)
        changed=self.x.copy();changed[-1,0]+=.001
        with self.assertRaises(AssertionError):list(model.terminal_chunks(changed,[.6,.4],2,100,7605))

if __name__=='__main__':unittest.main()
