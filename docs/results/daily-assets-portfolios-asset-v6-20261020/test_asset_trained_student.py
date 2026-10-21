from pathlib import Path
import os,sys,unittest
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[name]='1'
import numpy as np
from scipy.integrate import quad
from scipy.stats import t
O=Path(__file__).resolve().parent;V=O/'asset_level_v6'if(O/'asset_level_v6').exists()else O;sys.path.insert(0,str(V))
from candidates.asset_trained_student import weight

def integral_crps(y,nu,scale):
    s=scale*np.sqrt((nu-2)/nu)
    return quad(lambda x:t.cdf(x/s,nu)**2,-np.inf,y,epsabs=2e-10)[0]+quad(lambda x:t.sf(x/s,nu)**2,y,np.inf,epsabs=2e-10)[0]

class FormulaTests(unittest.TestCase):
    def test_Student_CRPS_matches_independent_CDF_integral(self):
        for nu,y,scale in ((2.2,-.7,.8),(3.,.2,1.3),(6.,2.3,.5),(35.,-.1,2.)):
            actual=float(weight.absolute(y,scale,nu)-scale*weight.self_half(nu));expected=integral_crps(y,nu,scale)
            self.assertAlmostEqual(actual,expected,places=9)

    def test_derived_chord_interpolation_bound(self):
        rng=np.random.default_rng(7201);pool=np.r_[rng.normal(size=29),-4.,8.];scales=np.geomspace(.11,13.,57)
        for nu in (2.2,5.,40.):
            actual,bound,count=weight.cross_interpolated(pool,scales,nu);expected=np.array([weight.absolute(pool,s,nu).mean()for s in scales]);self.assertGreater(count,1)
            self.assertGreaterEqual(float(np.min(actual-expected)),-2e-13);self.assertLessEqual(float(np.max(actual-expected)),bound+2e-13)

    def test_leave_one_out_mixture_coefficients_match_integrated_mixture_CDF(self):
        pool=np.array([-.9,0.,.3,.3,1.1,-.2,.7]);n=len(pool);observed=pool+.12;raw=np.linspace(.8,1.2,n);filtered=np.linspace(.6,1.3,n);nu=5.
        c,l,q,bound,_=weight.coefficients(pool,observed,raw,filtered,nu,relative_tolerance=1e-7)
        for fraction in (.0,.37,1.):
            expected=0.
            for i in range(n):
                empirical=np.delete(pool,i)*raw[i];s=filtered[i]*np.sqrt((nu-2)/nu);y=observed[i]
                def cdf(x):return fraction*np.mean(empirical<=x)+(1-fraction)*t.cdf(x/s,nu)
                edges=np.r_[-np.inf,np.unique(np.r_[empirical,y]),np.inf]
                for left,right in zip(edges[:-1],edges[1:]):
                    expected+=quad(lambda x:(cdf(x)-float(x>=y))**2,left,right,epsabs=2e-10)[0]
            self.assertLessEqual(abs(c+l*fraction+q*fraction*fraction-expected),bound*n+2e-9)

class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from candidates.asset_feature_batch import fits
        from candidates.asset_trained_student.candidate import prepare
        rng=np.random.default_rng(7202);cls.x=rng.standard_t(4,size=(230,2))*.009;cls.x[:,1]+=.4*cls.x[:,0];cls.base=fits.base_fit(cls.x);cls.bundle=prepare(cls.x,cls.base)

    def test_Gaussian_endpoint_exact_parent_at100_and1000_paths(self):
        from candidates.asset_trained_student.candidate import TrainedStudentPool,BASE_CONFIG
        from candidates.asset_trained_pool.candidate import TrainedPool
        from candidates.asset_trained_pool import weight as original
        b={**self.bundle,'filter':{**self.bundle['filter'],'inverse_df':0.}}
        b['pool_fit']=weight.estimate(self.x,b,self.base,'EWMA');expected_fit=original.estimate(self.x,b,self.base,'EWMA');np.testing.assert_equal(b['pool_fit']['quadratic_coefficients'],expected_fit['quadratic_coefficients'])
        self.assertEqual(b['pool_fit']['raw_fraction'],expected_fit['raw_fraction'])
        for n in (100,1000):
            a=np.concatenate([x for _,x in TrainedStudentPool({'innovations':'Student'},b).terminal_chunks(self.x,[.6,.4],73,n,7203)],axis=1)
            c=np.concatenate([x for _,x in TrainedPool(BASE_CONFIG,b).terminal_chunks(self.x,[.6,.4],73,n,7203)],axis=1);np.testing.assert_array_equal(a,c)

    def test_joint_Student_law_counts_chunking_and_prefix_identity(self):
        from candidates.asset_trained_student.candidate import TrainedStudentPool
        self.assertGreater(self.bundle['filter']['inverse_df'],0.)
        b={**self.bundle,'pool_fit':{**self.bundle['pool_fit'],'raw_fraction':.5}}
        for n in (100,1000):
            m=TrainedStudentPool({'innovations':'Student'},b);a=np.concatenate([x for _,x in m.terminal_chunks(self.x,[.6,.4],75,n,7204)],axis=1)
            c=np.concatenate([x for _,x in TrainedStudentPool({'innovations':'Student'},b).terminal_chunks(self.x,[.6,.4],75,n,7204,width=1)],axis=1)
            self.assertEqual(a.shape,(n,75));np.testing.assert_array_equal(a,c);self.assertTrue(np.isfinite(a).all());np.testing.assert_allclose(m.last_asset_weights.sum(1),1.,rtol=0,atol=6e-15)
            changed=self.x.copy();changed[-1,0]+=.01
            with self.assertRaises(AssertionError):list(TrainedStudentPool({'innovations':'Student'},b).terminal_chunks(changed,[.6,.4],2,n,7204))

    def test_actual_fitted_weight_has_no_arbitrary_fraction(self):
        f=self.bundle['pool_fit'];c,l,q=f['quadratic_coefficients'];self.assertGreaterEqual(f['raw_fraction'],0);self.assertLessEqual(f['raw_fraction'],1)
        objective=lambda w:c+l*w+q*w*w
        self.assertLessEqual(objective(f['raw_fraction']),min(objective(0.),objective(1.))+1e-10);self.assertEqual(f['Student_df'],1/self.bundle['filter']['inverse_df'])

if __name__=='__main__':unittest.main()
