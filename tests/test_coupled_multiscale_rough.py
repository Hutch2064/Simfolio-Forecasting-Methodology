"""Independent likelihood, coupled state covariance, stability and RNG controls."""
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope='module')
def model():
    path=Path(__file__).resolve().parents[1]/'tools/coupled_multiscale_rough/models.py'
    spec=importlib.util.spec_from_file_location('test_coupled_multiscale',path)
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    return m


def test_historical_innovations_match_causal_scalar_recursion(model):
    h=np.random.default_rng(68).normal(size=200);phi=np.array([.8,.98]);c=np.array([.6,.2,.1])
    q=np.zeros(2);errors=[]
    for t,value in enumerate(h):
        if t:errors.append(value-c[0]*h[t-1]-c[1]*q[0]-c[2]*q[1])
        q=phi*q+(1-phi)*value
    actual,last=model.state.innovations(h,phi,c)
    np.testing.assert_allclose(actual,errors,rtol=1e-12,atol=1e-12)
    np.testing.assert_allclose(last,q,rtol=1e-12,atol=1e-12)


@pytest.mark.parametrize('k',[0,1,3])
def test_positive_contractive_coefficients_give_stationary_transition(model,k):
    phi=np.linspace(.7,.999,k);c=np.full(k+1,.98/(k+1));f=model.state.transition(phi,c)
    assert (f>=0).all();assert np.max(np.abs(np.linalg.eigvals(f)))<1
    previous=np.arange(k+1,dtype=float)*.2
    next_h=c@previous
    expected=np.r_[next_h,phi*previous[1:]+(1-phi)*next_h]
    np.testing.assert_allclose(f@previous,expected,rtol=1e-12,atol=1e-12)


def test_conditional_moments_match_stationary_covariance_identity(model):
    from scipy.linalg import solve_discrete_lyapunov
    phi=np.array([.8,.99]);c=np.array([.6,.2,.1]);f=model.state.transition(phi,c)
    initial=np.array([.3,-.2,.1]);root=.15*np.r_[1.,1.-phi];p=solve_discrete_lyapunov(f,np.outer(root,root))
    mean,variance=model.state.moments(f,initial,root,30)
    for t in range(1,31):
        power=np.linalg.matrix_power(f,t)
        assert mean[t-1]==pytest.approx((power@initial)[0],abs=1e-12)
        assert variance[t-1]==pytest.approx((p-power@p@power.T)[0,0],abs=1e-12)


def test_native_memory_recursion_and_rng_match_scalar_reference(model):
    native,dot=model.load_paths();phi=np.array([.8,.98]);c=np.array([.6,.2,.1]);memory=np.array([.1,-.2,.3]);days=31;sd=.2;level=-1.
    f=model.state.transition(phi,c);mm,mv=model.state.moments(f,memory,sd*np.r_[1.,1-phi],days);mm+=level
    mean=np.full(days,.0003);base=np.linspace(-.01,.01,days);expected=np.empty(days)
    rng=np.random.default_rng(73);current=memory.copy()
    for t in range(days):
        value=0.
        for j in range(3):value+=c[j]*current[j]
        value+=sd*rng.standard_normal()
        for j in range(2):current[j+1]=phi[j]*current[j+1]+(1-phi[j])*value
        current[0]=value
        multiplier=math.exp(.5*(level+value-mm[t])-.25*mv[t])
        expected[t]=np.clip(mean[t]+(base[t]-mean[t])*multiplier,-1.,1.)
    r_rng=np.random.default_rng(72);m_rng=np.random.default_rng(73);output=np.empty((days,2))[:,0]
    native.map_path(np.array([.9]),np.zeros(1),np.zeros((1,1)),np.zeros(1),np.zeros(days),np.zeros(days),r_rng.bit_generator.capsule,dot,mean,base,output,phi,c,sd,memory,level,mm,mv,m_rng.bit_generator.capsule,True)
    assert output.tobytes()==expected.tobytes()
    assert rng.bit_generator.state==m_rng.bit_generator.state


def test_disabled_memory_matches_original_rough_native_bytes_and_rng(model):
    native,dot=model.load_paths();original,old_dot=model.pathwise.load_paths()
    phi=np.array([.8,.99]);w=np.array([.4,.8]);root=np.diag([.2,.1]);initial=np.array([.1,-.2]);days=30
    mean=np.full(days,.0003);base=np.linspace(-.01,.01,days);rm=np.zeros(days);rv=np.full(days,.2)
    r1=np.random.default_rng(75);r2=np.random.default_rng(75);m1=np.random.default_rng(76);m2=np.random.default_rng(76)
    old=np.empty(days);new=np.empty(days)
    original.map_path(phi,w,root,initial,rm,rv,r1.bit_generator.capsule,old_dot,mean,base,old,np.array([.8]),np.ones(1),np.ones(1),np.zeros(1),np.zeros(1),0.,rm,rv,m1.bit_generator.capsule,False)
    native.map_path(phi,w,root,initial,rm,rv,r2.bit_generator.capsule,dot,mean,base,new,np.empty(0),np.array([.8]),.1,np.zeros(1),0.,rm,rv,m2.bit_generator.capsule,False)
    assert old.tobytes()==new.tobytes()
    assert r1.bit_generator.state==r2.bit_generator.state
    assert m1.bit_generator.state==m2.bit_generator.state


def test_adaptive_memory_count_uses_conditional_bic(model):
    rng=np.random.default_rng(77);h=np.empty(400);h[0]=0.
    for t in range(1,len(h)):h[t]=.9*h[t-1]+rng.normal(scale=.2)
    record=model.state.fit(h);k=record['component_count']
    assert record['success'] and sum(record['coefficients'])<1
    assert record['bic_parameter_count']==2*k+3
    assert record['selected_bic']==min(row['bic'] for row in record['tested_orders'])


def test_constant_proxy_is_exact_deterministic_boundary(model):
    record=model.state.fit(np.full(200,-1.))
    assert record['component_count']==0 and record['innovation_sd']==0
    assert record['level']==-1. and record['initial']==[0.]
