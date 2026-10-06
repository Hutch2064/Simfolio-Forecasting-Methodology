"""Joint-density normalization, independent derivatives and sparse curvature."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.optimize._numdiff import approx_derivative
from scipy.stats import norm, t

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools/student_return_laplace_rough'))
sys.path.insert(0,str(ROOT/'tools/joint_leverage_rough'))
import leverage_laplace as model


@pytest.mark.parametrize('u',[0.,.05,.2])
def test_rank_and_exact_state_derivatives(u):
    h=np.array([-.8,.3,1.4]);returns=np.array([-.5,.2,1.3]);actual=model.scores(h,returns,u)
    x=returns*np.exp(-h/2)
    expected=x if u==0 else norm.ppf(t.cdf(x/math.sqrt(1-2*u),1/u))
    np.testing.assert_allclose(actual[0],expected,rtol=2e-12,atol=2e-12)
    for index in range(3):
        for order in range(3):
            def value(q,index=index,order=order):
                changed=h.copy();changed[index]=q[0];return np.array([model.scores(changed,returns,u)[order][index]])
            derivative=approx_derivative(value,np.array([h[index]]))[0]
            assert actual[order+1][index]==pytest.approx(derivative,rel=2e-7,abs=2e-8)


def test_joint_transition_is_normalized_with_student_marginal():
    u=.15;rho=-.6;eta=.3;h=.2;level=.1;phi=.9;r=.7
    z=model.scores(np.array([h]),np.array([r]),u)[0][0]
    center=level+phi*(h-level)+eta*rho*z
    sd=eta*math.sqrt(1-rho*rho)
    value=quad(lambda v:norm.pdf(v,loc=center,scale=sd),-np.inf,np.inf)[0]
    assert value==pytest.approx(1.,abs=1e-10)
    density=t.pdf(r*np.exp(-h/2)/math.sqrt(1-2*u),1/u)*np.exp(-h/2)/math.sqrt(1-2*u)
    # Independent conditional density integrates to the original return marginal.
    assert value*density==pytest.approx(density,rel=1e-12)


@pytest.mark.parametrize('rho',[0.,-.35,.25])
def test_state_gradient_hessian_match_independent_joint_density(rho):
    returns=np.array([-.7,.1,1.,-.3,.5]);theta=(.1,.8,.4,.12,rho);h=np.array([.1,.2,.3,.2,.1]);level,phi,eta,u,_=theta
    def independent(values):
        scaled=returns*np.exp(-values/2)/math.sqrt(1-2*u)
        obs=-t.logpdf(scaled,1/u).sum()+len(values)*.5*math.log(1-2*u)+values.sum()/2
        initial=-norm.logpdf(values[0],loc=level,scale=eta/math.sqrt(1-phi*phi))
        z=norm.ppf(t.cdf(scaled[:-1],1/u));pred=level+phi*(values[:-1]-level)+eta*rho*z
        transition=-norm.logpdf(values[1:],loc=pred,scale=eta*math.sqrt(1-rho*rho)).sum()
        return obs+initial+transition
    energy,g,diagonal,off,_=model.quantities(h,returns,*theta)
    numerical=approx_derivative(lambda x:np.array([independent(x)]),h).reshape(-1)
    np.testing.assert_allclose(g,numerical,rtol=2e-7,atol=2e-7)
    dense=approx_derivative(lambda x:approx_derivative(lambda y:np.array([independent(y)]),x).reshape(-1),h)
    calculated=np.diag(diagonal)+np.diag(off,1)+np.diag(off,-1)
    np.testing.assert_allclose(calculated,dense,rtol=2e-5,atol=2e-5)
    assert np.isfinite(energy)


@pytest.mark.parametrize('rho',[0.,-.3,.3])
def test_laplace_gradient_matches_independent_finite_difference(rho):
    returns=np.random.default_rng(81).normal(size=31);theta=np.array([.1,.8,math.log(.4),.12,rho])
    def likelihood(v):return model.likelihood_gradient(returns,v[0],v[1],math.exp(v[2]),v[3],v[4])[0]
    result=model.likelihood_gradient(returns,theta[0],theta[1],math.exp(theta[2]),theta[3],theta[4]);assert result[4]
    numerical=approx_derivative(lambda x:np.array([likelihood(x)]),theta).reshape(-1)
    np.testing.assert_allclose(result[1],numerical,rtol=3e-5,atol=5e-5)


def test_zero_leverage_matches_original_student_laplace_likelihood():
    returns=np.random.default_rng(37).normal(size=31);theta=(.1,.8,.4,.12)
    a=model.likelihood_gradient(returns,*theta,0.);b=model.original.likelihood_gradient(returns*returns,*theta)
    assert a[0]==pytest.approx(b[0],abs=2e-12)
    np.testing.assert_allclose(a[1][:4],b[1],rtol=2e-9,atol=2e-9)
    np.testing.assert_allclose(a[2],b[2],rtol=2e-12,atol=2e-12)


def test_zero_leverage_native_paths_and_random_streams_match_m221():
    import importlib.util

    from leverage_native import load_paths
    spec=importlib.util.spec_from_file_location('leverage_native_reference',ROOT/'tools/pathwise_multiscale_rough/pathwise_native.py')
    reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)
    native,dot=load_paths();old,_=reference.load_paths();horizon=126
    args=(np.array([.9]),np.array([.3]),np.array([[.1]]),np.array([.2]),np.zeros(horizon),np.ones(horizon))
    mean=np.full(horizon,.0002);base=np.linspace(-.01,.02,horizon);ms=(np.array([.8,.9]),np.array([.4,.6]),np.array([.1,.2]),np.array([0.,.1]),np.array([.1,.2]),-.5,np.zeros(horizon),np.ones(horizon))
    r1=np.random.default_rng(52);s1=np.random.default_rng(91);r2=np.random.default_rng(52);s2=np.random.default_rng(91);expected=np.empty(horizon);actual=np.empty(horizon)
    old.map_path(*args,r1.bit_generator.capsule,dot,mean,base,expected,*ms,s1.bit_generator.capsule,True)
    native.map_path(*args,r2.bit_generator.capsule,dot,mean,base,actual,*ms,s2.bit_generator.capsule,True,0.,np.zeros(horizon),-1.3)
    assert actual.tobytes()==expected.tobytes()
    assert r1.bit_generator.state==r2.bit_generator.state
    assert s1.bit_generator.state==s2.bit_generator.state


def test_native_leverage_matches_independent_lagged_gaussian_recursion():
    from leverage_native import load_paths
    native,dot=load_paths();horizon=31;rho=-.4
    phi=np.array([.7]);weights=np.array([.3]);root=np.array([[.1]]);initial=np.array([.2])
    means=np.zeros(horizon);variances=np.ones(horizon);mean=np.full(horizon,.0002);base=np.linspace(-.02,.03,horizon)
    mp=np.array([.6,.9]);ml=np.array([.4,.6]);mc=np.array([.1,.2]);mi=np.array([0.,.1]);state0=np.array([.1,.2]);ell=-.5
    mm=np.zeros(horizon);mv=np.ones(horizon);scores=np.linspace(-2,2,horizon)
    r=np.random.default_rng(71);s=np.random.default_rng(21);actual=np.empty(horizon)
    native.map_path(phi,weights,root,initial,means,variances,r.bit_generator.capsule,dot,mean,base,actual,mp,ml,mc,mi,state0,ell,mm,mv,s.bit_generator.capsule,True,rho,scores,-1.3)
    rr=np.random.default_rng(71);ss=np.random.default_rng(21);state=initial.copy();ms=state0.copy();expected=[]
    aggregate=np.array([ml@mc,ml@mi]);a=aggregate/np.linalg.norm(aggregate)
    for day in range(horizon):
        z=ss.normal(size=2)
        z=z+a*(rho*(scores[day-1] if day else -1.3)+(math.sqrt(1-rho*rho)-1)*(a@z))
        ms=mp*ms+mc*z[0]+mi*z[1]
        scale=np.exp(.5*(weights@state)-.25)*np.exp(.5*(ell+ml@ms)-.25)
        expected.append(np.clip(mean[day]+(base[day]-mean[day])*scale,-1,1))
        state=phi*state+np.diag(root)*rr.normal(size=1)
    np.testing.assert_allclose(actual,expected,rtol=2e-13,atol=2e-13)
    assert r.bit_generator.state==rr.bit_generator.state
    assert s.bit_generator.state==ss.bit_generator.state
    # Conditional covariance plus the Gaussian return-rank contribution equals I.
    transform=np.eye(2)+(math.sqrt(1-rho*rho)-1)*np.outer(a,a)
    np.testing.assert_allclose(transform@transform.T+rho*rho*np.outer(a,a),np.eye(2),atol=1e-14)
    assert a@(rho*a)==pytest.approx(rho)


def test_gaussian_boundary_gradient_and_dense_laplace_value():
    returns=np.random.default_rng(317).normal(size=17);level=.1;phi=.8;eta=.4;rho=-.3
    result=model.likelihood_gradient(returns,level,phi,eta,0.,rho);assert result[4]
    step=1e-5
    values=[model.likelihood_gradient(returns,level,phi,eta,u,rho)[0] for u in [0.,step,2*step]]
    derivative=(-3*values[0]+4*values[1]-values[2])/(2*step)
    assert result[1][3]==pytest.approx(derivative,rel=3e-5,abs=3e-5)
    h=result[2]
    energy,_,diagonal,off,_=model.quantities(h,returns,level,phi,eta,0.,rho)
    precision=np.diag(diagonal)+np.diag(off,1)+np.diag(off,-1)
    determinant=len(h)*math.log(eta*eta)-math.log1p(-phi*phi)+(len(h)-1)*math.log1p(-rho*rho)
    dense=len(h)*(-.5*math.log(2*math.pi))-.5*determinant-energy-.5*np.linalg.slogdet(precision)[1]
    assert result[0]==pytest.approx(dense,abs=2e-12)


@pytest.mark.parametrize('rho',[0.,-.6,.4])
def test_last_observation_conditional_multiscale_moments(rho):
    phi=np.array([.6,.9]);loading=np.array([.4,.6])
    common=np.array([.1,.2]);independent=np.array([0.,.1]);horizon=17;score=-1.3
    state=np.array([.1,.2]);cov=np.zeros((2,2));means=[];variances=[]
    covariance=np.outer(common,common)+np.outer(independent,independent)
    for _ in range(horizon):
        state=phi*state;cov=phi[:,None]*cov*phi[None,:]+covariance
        means.append(float(loading@state));variances.append(float(loading@cov@loading))
    actual=model.conditioned_multiscale_moments(phi,loading,common,independent,np.array(means),np.array(variances),rho,score)
    aggregate=np.array([loading@common,loading@independent]);a=aggregate/np.linalg.norm(aggregate)
    matrix=np.column_stack((common,independent));first_mean=rho*score*a
    first_cov=np.eye(2)-rho*rho*np.outer(a,a)
    state=np.array([.1,.2]);cov=np.zeros((2,2));expected_means=[];expected_variances=[]
    for day in range(horizon):
        state=phi*state+(matrix@first_mean if day==0 else 0.)
        cov=phi[:,None]*cov*phi[None,:]+matrix@(first_cov if day==0 else np.eye(2))@matrix.T
        expected_means.append(float(loading@state));expected_variances.append(float(loading@cov@loading))
    np.testing.assert_allclose(actual[0],expected_means,atol=2e-15)
    np.testing.assert_allclose(actual[1],expected_variances,atol=2e-15)
    if rho==0.:
        assert actual[0].tobytes()==np.array(means).tobytes()
        assert actual[1].tobytes()==np.array(variances).tobytes()


def test_underflowed_student_tail_matches_high_precision_incomplete_beta():
    u=1e-4;returns=np.array([-50.,50.]);h=np.zeros(2)
    # Independent 80-decimal incomplete-beta reference:
    # log(I_{nu/(nu+x^2)}(nu/2,1/2)/2), x=50/sqrt(1-2u), nu=1/u.
    logtail=-1120.6376867881663378028747981459467559069703333073460933878973563723244531741189
    from scipy.special import ndtri_exp
    expected=-ndtri_exp(float(logtail))
    actual=model.scores(h,returns,u)
    np.testing.assert_allclose(actual[0],[-expected,expected],rtol=2e-13,atol=2e-13)
    assert all(np.isfinite(x).all() for x in actual)
