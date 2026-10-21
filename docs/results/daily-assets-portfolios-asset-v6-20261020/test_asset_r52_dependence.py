from pathlib import Path
import os,sys,unittest,ast,json,hashlib
for name in('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','NUMBA_NUM_THREADS'):os.environ[name]='1'
import numpy as np
from scipy.special import stdtr,ndtr,ndtri
from scipy.stats import t,norm
from scipy.integrate import quad
O=Path(__file__).resolve().parent;V=O/'asset_level_v6'if(O/'asset_level_v6').exists()else O;sys.path.insert(0,str(V))
from candidates.asset_r52_dependence import student_state,likelihood_kernel,dependence,path_native
from candidates.asset_r52_dependence.candidate import R52DependencePool,BASE_CONFIG,prepare
from candidates.asset_trained_pool.candidate import TrainedPool
C=V/'candidates/asset_r52_dependence'

class InfrastructureTests(unittest.TestCase):
    def test_copied_source_and_numeric_function_AST(self):
        d=json.loads((C/'source_derivation.json').read_text())
        self.assertGreater(d['sealed_source_files_verified'],0)
        for r in d['copied_modules']:
            p=C/r['target']
            if r.get('adaptation')=='byte-identical copied source':self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(),r['original_sha256'])
            fns=r.get('original_function_ast_sha256')
            if isinstance(fns,dict):
                actual={}
                for node in ast.parse(p.read_text()).body:
                    if not isinstance(node,ast.FunctionDef):continue
                    for sub in ast.walk(node):
                        if isinstance(sub,ast.ImportFrom)and sub.module=='lapack_native':sub.level=0
                        if 'type_params' in sub._fields and not sub.type_params:
                            sub._fields=tuple(field for field in sub._fields if field!='type_params')
                    actual[node.name]=hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()
                self.assertEqual(actual,fns)

    def test_likelihood_gradient_and_reference_recursion(self):
        rng=np.random.default_rng(7401);u=rng.uniform(.02,.98,size=(43,3));s=np.array([[1.,.3,.1],[.3,1.,.2],[.1,.2,1.]])
        a=.06;b=.85;nu=7.;z=student_state.standardized_scores(u,nu);constant=student_state.copula_constant(nu,3)
        actual=likelihood_kernel.evaluate(z,s,a,b,s,nu,constant);reference=student_state.likelihood(z,s,a,b,s,nu,constant)
        for x,y in zip(actual,reference):np.testing.assert_allclose(x,y,rtol=0,atol=2e-10)
        for j in range(2):
            theta=np.array([a,b]);theta[j]+=1e-6;plus=likelihood_kernel.evaluate(z,s,*theta,s,nu,constant,False)[0];theta[j]-=2e-6;minus=likelihood_kernel.evaluate(z,s,*theta,s,nu,constant,False)[0]
            self.assertAlmostEqual(actual[1][j],(plus-minus)/2e-6,places=5)

    def test_large_matrix_native_likelihood_agrees_with_reference(self):
        rng=np.random.default_rng(7402);p=16;s=np.eye(p)*.8+np.ones((p,p))*.2;z=rng.standard_t(8,size=(17,p))*np.sqrt(6/8);constant=student_state.copula_constant(8.,p)
        a=likelihood_kernel.evaluate(z,s,.04,.9,s,8.,constant);b=student_state.likelihood(z,s,.04,.9,s,8.,constant)
        for x,y in zip(a,b):np.testing.assert_allclose(x,y,rtol=0,atol=2e-9)

    def test_native_path_CDF_mapping_matches_independent_recursion_and_chunks(self):
        rng=np.random.default_rng(7403);normal=rng.normal(size=(11,7,2));s=np.array([[1.,.3],[.3,1.]]);q0=np.array([[1.2,.45],[.45,.8]]);fit={'nu':6.,'target':s,'a':.08,'b':.83};q=np.tile(q0,(7,1,1));expected=q.copy();scales=np.random.default_rng(7404).gamma(3.,2.,size=(11,7,1));noise=normal*np.sqrt(4/scales);probs=np.empty_like(normal)
        for k in range(11):
            for j in range(7):
                v=np.linalg.cholesky(expected[j])@noise[k,j];z=v/np.sqrt(np.diag(expected[j]));probs[k,j]=np.clip(stdtr(6.,z*np.sqrt(6/4)),1e-8,1-1e-8);expected[j]=.09*s+.08*np.outer(v,v)+.83*expected[j]
        actual=dependence.gaussianize(normal.copy(),q,fit,np.random.default_rng(7404));np.testing.assert_allclose(ndtr(actual),probs,rtol=0,atol=1.1e-12);np.testing.assert_allclose(q,expected,rtol=0,atol=2e-13)
        split=np.tile(q0,(7,1,1));radial=np.random.default_rng(7404);b=np.concatenate([dependence.gaussianize(normal[:3].copy(),split,fit,radial),dependence.gaussianize(normal[3:].copy(),split,fit,radial)],axis=0)
        np.testing.assert_array_equal(actual,b);np.testing.assert_array_equal(q,split)

    def test_Gaussian_margins_with_Student_joint_tail_dependence(self):
        n=50000;nu=4.;rng=np.random.default_rng(7405);fit={'nu':nu,'target':np.eye(2),'a':0.,'b':0.};q=np.tile(np.eye(2),(n,1,1));z=dependence.gaussianize(rng.normal(size=(1,n,2)),q,fit,np.random.default_rng(7406))[0]
        bound=norm.ppf(.95);marginal=np.mean(z>bound,axis=0);self.assertTrue(np.all(np.abs(marginal-.05)<.005))
        cut=t.ppf(.95,nu);expected=quad(lambda x:t.pdf(x,nu)*t.sf(cut/np.sqrt((nu+x*x)/(nu+1)),nu+1),cut,np.inf,epsabs=1e-11)[0];observed=np.mean(np.all(z>bound,axis=1));self.assertLess(abs(observed-expected),5*np.sqrt(expected*(1-expected)/n));self.assertGreater(expected,.05**2)

class ModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rng=np.random.default_rng(7407);radial=np.sqrt(4/rng.chisquare(6,size=(230,1)));cls.x=(rng.normal(size=(230,2))@np.array([[1.,0.],[.4,.9]]).T)*radial*.01;cls.bundle=prepare(cls.x)

    def test_exact_parent_and_scalar_controls_at100_and1000(self):
        scalar=self.x[:,:1];scalar_bundle=prepare(scalar)
        for x,w,bundle,cfg in((self.x,[.6,.4],self.bundle,{'copula':'parent'}),(scalar,[1.],scalar_bundle,{'copula':'R52_Student_cDCC'})):
            for n in(100,1000):
                a=np.concatenate([p for _,p in R52DependencePool(cfg,bundle).terminal_chunks(x,w,71,n,7408)],axis=1);b=np.concatenate([p for _,p in TrainedPool(BASE_CONFIG,bundle).terminal_chunks(x,w,71,n,7408)],axis=1);np.testing.assert_array_equal(a,b)

    def test_joint_counts_accounting_chunk_invariance_and_prefix_guard(self):
        cfg={'copula':'R52_Student_cDCC'};m=R52DependencePool(cfg,self.bundle);a=np.concatenate([p for _,p in m.terminal_chunks(self.x,[.6,.4],73,100,7409)],axis=1);b=np.concatenate([p for _,p in R52DependencePool(cfg,self.bundle).terminal_chunks(self.x,[.6,.4],73,100,7409,width=1)],axis=1)
        self.assertEqual(a.shape,(100,73));np.testing.assert_array_equal(a,b);self.assertTrue(np.isfinite(a).all());np.testing.assert_allclose(m.last_asset_weights.sum(1),1.,rtol=0,atol=6e-15)
        changed=self.x.copy();changed[-1,0]+=.001
        with self.assertRaises(AssertionError):list(R52DependencePool(cfg,self.bundle).terminal_chunks(changed,[.6,.4],2,100,7409))

    def test_current_marginal_mean_and_weight_fits_are_unchanged(self):
        from candidates.asset_trained_pool.candidate import prepare as parent_prepare
        p=parent_prepare(self.x,BASE_CONFIG['family'],BASE_CONFIG['raw_fade']);np.testing.assert_equal(p['pool_fit'],self.bundle['pool_fit']);np.testing.assert_equal(p['mean'],self.bundle['mean']);np.testing.assert_equal(p['filter'],self.bundle['filter']);dep=self.bundle['dependence'];self.assertGreaterEqual(dep['a'],0);self.assertGreaterEqual(dep['b'],0);self.assertLess(dep['a']+dep['b'],1);self.assertTrue(np.isinf(dep['nu'])or dep['nu']>2)

if __name__=='__main__':unittest.main()
