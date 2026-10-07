"""Independent Student copula, corrected moment and forecast-law references."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
from scipy.special import ndtri, stdtr
from scipy.stats import kstest, multivariate_t, t

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('student_cdcc_test_models',ROOT/'tools/student_corrected_dcc_innovation_copula/models.py')
models=importlib.util.module_from_spec(spec);sys.modules[spec.name]=models;spec.loader.exec_module(models)


def dense_reference(z,s,a,b,nu):
    q=s.copy();loss=0.
    for row in z:
        sd=np.sqrt(np.diag(q));r=q/sd[:,None]/sd[None,:]
        unscaled=row*np.sqrt(nu/(nu-2))
        loss-=multivariate_t.logpdf(unscaled,shape=r,df=nu)-t.logpdf(unscaled,df=nu).sum()
        w=sd*row;q=(1-a-b)*s+a*np.outer(w,w)+b*q
    return loss,q


def test_student_likelihood_and_recursive_a_b_gradient_match_dense_references():
    s=np.array([[1.,.4,-.1],[.4,1.,.2],[-.1,.2,1.]])
    u=np.random.default_rng(541).uniform(.01,.99,(43,3))
    for nu in (2.3,8.,31.):
        z=models.state.standardized_scores(u,nu);constant=models.state.copula_constant(nu,3)
        for theta in (np.array([.06,.91]),np.array([.2,0.])):
            value,gradient,q=models.state.likelihood(z,s,*theta,s,nu,constant)
            fast,_,fast_q=models.state.likelihood(z,s,*theta,s,nu,constant,False)
            assert fast==value
            np.testing.assert_array_equal(q,fast_q)
            ref,rq=dense_reference(z,s,*theta,nu)
            np.testing.assert_allclose(value,ref,atol=1e-11,rtol=1e-11)
            np.testing.assert_allclose(q,rq,atol=1e-13,rtol=1e-13)
            fd=[]
            for j in range(2):
                delta=np.zeros(2);delta[j]=1e-6
                fd.append((models.state.likelihood(z,s,*(theta+delta),s,nu,constant)[0]-models.state.likelihood(z,s,*(theta-delta),s,nu,constant)[0])/2e-6)
            np.testing.assert_allclose(gradient,fd,rtol=3e-6,atol=2e-7)


def test_standardized_student_margins_and_corrected_conditional_moment():
    s=np.array([[1.,.4],[.4,1.]]);q=np.array([[2.,.7],[.7,.5]])
    nu=8.;n=180000;rng=np.random.default_rng(601);states=np.repeat(q[None],n,axis=0)
    scores=rng.normal(size=(1,n,2))*np.sqrt((nu-2)/rng.chisquare(nu,size=(1,n,1)))
    models.native.load_paths().scores(scores,s,states,.1,.8)
    u=stdtr(nu,scores[0]*np.sqrt(nu/(nu-2)))
    for margin in u.T:assert kstest(margin,'uniform').statistic<.006
    np.testing.assert_allclose(np.cov(scores[0].T),q/np.sqrt(np.diag(q))[:,None]/np.sqrt(np.diag(q))[None,:],atol=.013)
    np.testing.assert_allclose(states.mean(0),.1*s+.9*q,atol=.006)


def test_native_student_paths_match_dense_recursion_and_exact_blocking():
    s=np.array([[1.,.3],[.3,1.]]);q=2*s;nu=7.;rng=np.random.default_rng(102)
    raw=rng.normal(size=(1030,4,2))*np.sqrt((nu-2)/rng.chisquare(nu,size=(1030,4,1)))
    expected=raw.copy();rq=np.repeat(q[None],4,axis=0)
    for i in range(len(raw)):
        for k in range(4):
            w=np.linalg.cholesky(rq[k])@raw[i,k]
            expected[i,k]=w/np.sqrt(np.diag(rq[k]));rq[k]=.05*s+.03*np.outer(w,w)+.92*rq[k]
    actual=raw.copy();states=np.repeat(q[None],4,axis=0)
    models.native.load_paths().scores(actual,s,states,.03,.92)
    np.testing.assert_allclose(actual,expected,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(states,rq,rtol=1e-12,atol=1e-12)
    blocked=raw.copy();bstates=np.repeat(q[None],4,axis=0)
    models.native.load_paths().scores(blocked[:1024],s,bstates,.03,.92)
    models.native.load_paths().scores(blocked[1024:],s,bstates,.03,.92)
    np.testing.assert_array_equal(blocked,actual);np.testing.assert_array_equal(bstates,states)


def test_uniform_stream_prefix_and_gaussian_endpoint(monkeypatch):
    s=np.array([[1.,.3],[.3,1.]]);q=2*s
    monkeypatch.setattr(models,'configuration',lambda *args:(s,q,.04,.9,8.,None,None))
    models.dcc_uniforms.cache_clear();args=((100,2),b'',4,1030,85)
    full=models.dcc_uniforms(*args)
    np.testing.assert_array_equal(full[:,:100],models.dcc_uniforms(args[0],b'',4,100,85))
    expected=np.full((4,12,2),.7)
    monkeypatch.setattr(models,'configuration',lambda *args:(s,q,.04,.9,np.inf,None,None))
    monkeypatch.setattr(models.parent,'dcc_uniforms',lambda *args:expected)
    models.dcc_uniforms.cache_clear()
    np.testing.assert_array_equal(models.dcc_uniforms(args[0],b'',4,12,85),expected)
    models.dcc_uniforms.cache_clear()


def test_joint_fit_recovers_simulated_student_dynamics_and_beats_gaussian_endpoint():
    s=np.array([[1.,.4],[.4,1.]]);nu=7.;a,b=.07,.85;n=7000;rng=np.random.default_rng(835)
    z=rng.normal(size=(n,1,2))*np.sqrt((nu-2)/rng.chisquare(nu,size=(n,1,1)))
    states=s[None].copy();models.native.load_paths().scores(z,s,states,a,b)
    u=stdtr(nu,z[:,0]*np.sqrt(nu/(nu-2)))
    theta,q,fitted,diagnostic=models.state.fit(u,s)
    assert abs(theta[0]-a)<.035 and abs(theta[1]-b)<.09
    assert 4.<fitted<12. and theta.sum()<1.
    assert np.linalg.eigvalsh(q).min()>0.
    assert diagnostic['training_nll']<diagnostic['gaussian_training_nll']


def test_large_nu_copula_limit_matches_gaussian():
    s=np.array([[1.,.3],[.3,1.]]);u=np.random.default_rng(263).uniform(.01,.99,(39,2));nu=1e7
    z=models.state.standardized_scores(u,nu)
    student=models.state.likelihood(z,s,.03,.9,s,nu,models.state.copula_constant(nu,2))[0]
    gaussian=models.state.gaussian.likelihood(ndtri(u),s,.03,.9,s)[0]
    np.testing.assert_allclose(student,gaussian,atol=2e-6,rtol=2e-6)


def test_allocation_free_likelihood_matches_original_for_full_histories():
    from likelihood_kernel import evaluate, gaussian_likelihood
    rng=np.random.default_rng(256)
    for p in (2,6,12):
        root=rng.normal(size=(p,p));s=root@root.T+p*np.eye(p)
        s=s/np.sqrt(np.diag(s))[:,None]/np.sqrt(np.diag(s))[None,:]
        u=rng.uniform(.0001,.9999,(700,p))
        for a,b in ((0.,0.),(.07,.9),(.2,0.)):
            z=ndtri(u)
            expected=models.state.gaussian.likelihood(z,s,a,b,s)
            actual=gaussian_likelihood(z,s,a,b,s)
            for measured,reference in zip(actual,expected):
                np.testing.assert_allclose(measured,reference,rtol=2e-11,atol=2e-10)
            for nu in (2.1,8.,1e7):
                z=models.state.standardized_scores(u,nu);c=models.state.copula_constant(nu,p)
                expected=models.state.likelihood(z,s,a,b,s,nu,c)
                actual=evaluate(z,s,a,b,s,nu,c)
                for measured,reference in zip(actual,expected):
                    np.testing.assert_allclose(measured,reference,rtol=2e-10,atol=2e-9)
                value,_,q=evaluate(z,s,a,b,s,nu,c,False)
                np.testing.assert_allclose(value,actual[0],rtol=2e-11,atol=2e-10)
                np.testing.assert_array_equal(q,actual[2])


def test_student_cdf_interpolation_has_uniform_probability_error_bound():
    from student_cdf import ERROR, grid, interpolate, resolution
    rng=np.random.default_rng(257)
    for nu in (2.01,2.1,3.,8.,31.,100.,1000.):
        limit,n,_=resolution(nu)
        # Include every interval midpoint, all grid knots and rare tail draws.
        angles=np.linspace(-limit,limit,2*n+1)
        x=np.r_[np.sqrt(nu)*np.tan(angles),rng.standard_t(nu,size=10000),-1e10,1e10]
        expected=np.clip(stdtr(nu,x),1e-8,1-1e-8)
        actual=x.copy();interpolate(actual,nu,*grid(nu))
        np.testing.assert_allclose(actual,expected,rtol=0,atol=ERROR+2e-14)


def test_joint_tail_derivative_matches_whole_likelihood_finite_difference():
    from likelihood_kernel import evaluate
    u=np.random.default_rng(259).uniform(.0001,.9999,(700,6))
    s=.8*np.eye(6)+.2*np.ones((6,6));step=np.finfo(float).eps**(1/3)
    for nu in (2.1,8.,31.,1000.):
        eta=np.log(nu-2);h=step*max(1,abs(eta));plus=2+np.exp(eta+h);minus=2+np.exp(eta-h)
        z=models.state.standardized_scores(u,nu);zp=models.state.standardized_scores(u,plus);zm=models.state.standardized_scores(u,minus)
        cp=models.state.copula_constant(plus,6);cm=models.state.copula_constant(minus,6)
        for a,b in ((.07,.9),(.2,0.)):
            value,gradient,q=evaluate(z,s,a,b,s,nu,models.state.copula_constant(nu,6),True,(zp-zm)/(2*h),(cp-cm)/(2*h))
            fd=(evaluate(zp,s,a,b,s,plus,cp,False)[0]-evaluate(zm,s,a,b,s,minus,cm,False)[0])/(2*h)
            np.testing.assert_allclose(gradient[2],fd,rtol=2e-5,atol=2e-6)
            reference=evaluate(z,s,a,b,s,nu,models.state.copula_constant(nu,6))
            assert value==reference[0]
            np.testing.assert_array_equal(q,reference[2])
            np.testing.assert_array_equal(gradient[:2],reference[1])
