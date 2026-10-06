"""Independent Gaussian-state moments and original rough-kernel parity."""
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'tools/pathwise_multiscale_rough'))
from pathwise_native import load_paths

spec=importlib.util.spec_from_file_location('pathwise_reference_native',ROOT/'tools/streamed_rough_paths/stream_native.py')
reference=importlib.util.module_from_spec(spec);spec.loader.exec_module(reference)


@pytest.mark.parametrize('n,horizon',[(1,1),(3,31),(14,126),(33,252)])
def test_disabled_multiscale_matches_original_rough_paths_and_rng_bytes(n,horizon):
    native,dot=load_paths();old,_=reference.load_paths();rng=np.random.default_rng(915)
    phi=rng.uniform(.05,.99,n);weights=rng.normal(size=n);root=np.diag(rng.uniform(0,.2,n));initial=rng.normal(size=n)
    means=rng.normal(size=horizon);variances=rng.uniform(0,1,horizon);mean=np.full(horizon,.0003);base=rng.normal(.0003,.02,horizon)
    expected=np.empty(horizon);actual=np.empty(horizon)
    r1=np.random.default_rng(41);r2=np.random.default_rng(41);ms=np.random.default_rng(531)
    args=(phi,weights,root,initial,means,variances)
    old.map_path(*args,r1.bit_generator.capsule,dot,mean,base,expected)
    native.map_path(*args,r2.bit_generator.capsule,dot,mean,base,actual,
                    np.array([.9]),np.ones(1),np.array([.1]),np.zeros(1),np.zeros(1),0.,
                    np.zeros(horizon),np.ones(horizon),ms.bit_generator.capsule,False)
    assert expected.tobytes()==actual.tobytes()
    assert json.dumps(r1.bit_generator.state,sort_keys=True)==json.dumps(r2.bit_generator.state,sort_keys=True)


def test_native_state_recursion_matches_independent_numpy_reference():
    native,dot=load_paths();horizon=126
    phi=np.array([.7,.9]);weights=np.array([.4,.6]);root=np.diag([.1,.2]);initial=np.array([.2,-.1])
    means=np.zeros(horizon);variances=np.full(horizon,.1);mean=np.full(horizon,.0003);base=np.linspace(-.03,.04,horizon)
    mp=np.array([.5,.8,.95]);ml=np.array([.4,.6,1.]);mc=np.array([.1,.2,.05]);mi=np.array([0.,0.,.1]);state0=np.array([.1,.2,-.1]);ell=-.4
    ms_mean=np.empty(horizon);ms_variance=np.empty(horizon);m=state0.copy();cov=np.zeros((3,3));q=np.outer(mc,mc)+np.outer(mi,mi)
    for t in range(horizon):
        m*=mp;cov*=mp[:,None]*mp[None,:];cov+=q
        ms_mean[t]=ell+ml@m;ms_variance[t]=ml@cov@ml
    args=(mp,ml,mc,mi,state0,ell,ms_mean,ms_variance)
    r=np.random.default_rng(33);s=np.random.default_rng(44);actual=np.empty(horizon)
    native.map_path(phi,weights,root,initial,means,variances,r.bit_generator.capsule,dot,mean,base,actual,*args,s.bit_generator.capsule,True)
    rr=np.random.default_rng(33);ss=np.random.default_rng(44);state=initial.copy();ms_state=state0.copy();expected=[]
    for t in range(horizon):
        z=ss.normal(size=2);ms_state=mp*ms_state+mc*z[0]+mi*z[1]
        scale=np.exp(.5*(weights@state-means[t])-.25*variances[t])
        scale*=np.exp(.5*(ell+ml@ms_state-ms_mean[t])-.25*ms_variance[t])
        expected.append(np.clip(mean[t]+(base[t]-mean[t])*scale,-1.,1.))
        state=phi*state+np.diag(root)*rr.normal(size=len(phi))
    np.testing.assert_allclose(actual,expected,rtol=2e-14,atol=2e-14)
    assert json.dumps(r.bit_generator.state,sort_keys=True)==json.dumps(rr.bit_generator.state,sort_keys=True)
    assert json.dumps(s.bit_generator.state,sort_keys=True)==json.dumps(ss.bit_generator.state,sort_keys=True)


def test_existing_multiscale_moments_equal_independent_matrix_recursion():
    spec=importlib.util.spec_from_file_location('pathwise_multiscale_test',ROOT/'tools/pathwise_multiscale_rough/models.py')
    model=importlib.util.module_from_spec(spec);sys.modules[spec.name]=model;spec.loader.exec_module(model)
    model.initialize(None,1,1)
    x=np.random.default_rng(492).normal(.0003,.01,507);data=x.tobytes()
    phi,loading,common,independent,initial,ell,mean,variance=model.multiscale_configuration(data,126)
    assert len(phi)==5 # Existing four timescales plus their fitted residual.
    covariance=np.zeros((len(phi),len(phi)));state=initial.copy();q=np.outer(common,common)+np.outer(independent,independent)
    calculated=[]
    for _ in range(126):
        state*=phi;covariance*=phi[:,None]*phi[None,:];covariance+=q
        calculated.append((ell+loading@state,loading@covariance@loading))
    np.testing.assert_allclose(np.array(calculated).T,[mean,variance],rtol=2e-12,atol=2e-12)
    # For Gaussian h, E[exp(h-Eh-var(h)/2)] = 1 exactly: the second
    # moment of the volatility multiplier remains one at every horizon.
    for index in [0,20,125]:
        v=variance[index]
        integral=quad(lambda z,v=v:math.exp(math.sqrt(v)*z-.5*v-.5*z*z)/math.sqrt(2*math.pi),-np.inf,np.inf)[0]
        assert integral==pytest.approx(1.,abs=1e-10)
