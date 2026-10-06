"""Independent Yule-Walker/dense Gaussian references for adaptive AR dynamics."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import toeplitz
from scipy.stats import multivariate_normal

p=Path(__file__).resolve().parents[1]/'tools/adaptive_ar_order_rough/state.py'
spec=importlib.util.spec_from_file_location('adaptive_ar_order_test',p)
state=importlib.util.module_from_spec(spec);spec.loader.exec_module(state)


def covariance(phi,n):
    p=len(phi);matrix=np.eye(p+1);rhs=np.zeros(p+1);rhs[0]=1.
    for j in range(p+1):
        for k in range(1,p+1):matrix[j,abs(j-k)]-=phi[k-1]
    gamma=list(np.linalg.solve(matrix,rhs))
    for j in range(p+1,n):gamma.append(sum(phi[k-1]*gamma[j-k] for k in range(1,p+1)))
    return toeplitz(gamma[:n])


@pytest.mark.parametrize('partials',[[.9],[.8,-.7],[.5,.2,-.3],[.95,-.8,.4,.2]])
def test_profile_matches_independent_dense_stationary_gaussian(partials):
    phi=state.coefficients(partials)
    assert max(abs(np.linalg.eigvals(state.transition([],phi))))<1.
    h=np.random.default_rng(931).normal(size=39)+3.
    nll,level,variance=state.Profile(h,len(phi)).evaluate(phi)
    unit=covariance(phi,len(h));inv=np.linalg.inv(unit);one=np.ones(len(h))
    expected_level=(one@inv@h)/(one@inv@one)
    expected_variance=((h-expected_level)@inv@(h-expected_level))/len(h)
    assert np.isclose(level,expected_level,atol=1e-10)
    assert np.isclose(variance,expected_variance,rtol=1e-9)
    expected=-multivariate_normal.logpdf(h,mean=np.full(len(h),level),cov=variance*unit)
    assert np.isclose(nll,expected,rtol=1e-9,atol=1e-8)


def test_unrestricted_ar2_selection_and_scalar_control():
    rng=np.random.default_rng(500);h=np.zeros(1500)
    for t in range(2,len(h)):h[t]=1.5*h[t-1]-.7*h[t-2]+.2*rng.normal()
    fit=state.fit(h)
    assert fit['ar_order']>=2
    assert fit['coefficients'][1]<0.
    assert fit['tested_orders'][-1]['bic']>=fit['selected_bic']
    assert state.fit(h,force_scalar=True)==state.scalar.fit(h)


def test_negative_ar1_supported_and_constant_boundary():
    rng=np.random.default_rng(132);h=np.zeros(800)
    for t in range(1,len(h)):h[t]=-.8*h[t-1]+rng.normal()
    fit=state.fit(h);assert fit['coefficients'][0]<0.
    fit=state.fit(np.full(30,3.));assert fit['innovation_sd']==0.


@pytest.fixture(scope='module')
def model():
    import sys
    path=Path(__file__).resolve().parents[1]/'tools/adaptive_ar_order_rough/models.py'
    spec=importlib.util.spec_from_file_location('adaptive_ar_forecast_test',path)
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    return m


def test_forecast_covariance_matches_independent_innovation_map(model):
    phi=state.coefficients([.8,-.6,.2]);f=state.transition([],phi)
    initial=np.array([.4,-.1,.2]);sd=.15;horizon=35
    mean,variance=model.state.moments(f,initial,sd*np.r_[1.,0.,0.],horizon)
    impulses=[];current=initial.copy()
    for t in range(horizon):
        current=f@current
        impulses=[f@v for v in impulses]+[np.array([sd,0.,0.])]
        assert mean[t]==pytest.approx(current[0],abs=1e-12)
        assert variance[t]==pytest.approx(sum(v[0]**2 for v in impulses),abs=1e-12)


def test_native_ar_shift_and_rng_match_scalar_reference(model):
    import math
    native,dot=model.load_paths();c=state.coefficients([.8,-.6,.2]);memory=np.array([.1,-.2,.3]);days=31;sd=.2;level=-1.
    f=state.transition([],c);mm,mv=model.state.moments(f,memory,sd*np.r_[1.,0.,0.],days);mm+=level
    mean=np.full(days,.0003);base=np.linspace(-.01,.01,days);expected=np.empty(days)
    rng=np.random.default_rng(73);current=memory.copy()
    for t in range(days):
        value=0.
        for j in range(3):value+=c[j]*current[j]
        value+=sd*rng.standard_normal()
        current[1:]=current[:-1];current[0]=value
        multiplier=math.exp(.5*(level+value-mm[t])-.25*mv[t])
        expected[t]=np.clip(mean[t]+(base[t]-mean[t])*multiplier,-1.,1.)
    r_rng=np.random.default_rng(72);m_rng=np.random.default_rng(73);output=np.empty((days,2))[:,0]
    native.map_path(np.array([.9]),np.zeros(1),np.zeros((1,1)),np.zeros(1),np.zeros(days),np.zeros(days),r_rng.bit_generator.capsule,dot,mean,base,output,np.zeros(2),c,sd,memory,level,mm,mv,m_rng.bit_generator.capsule,True)
    assert output.tobytes()==expected.tobytes()
    assert rng.bit_generator.state==m_rng.bit_generator.state
