"""Decay-only construction, independent predictive objective, and fixed mean."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.special import logit


@pytest.fixture(scope='module')
def model():
    p=Path(__file__).resolve().parents[1]/'tools/learned_multiscale_rates_rough/models.py'
    spec=importlib.util.spec_from_file_location('test_learned_rates',p)
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    return m


def test_existing_rates_preserve_complete_construction(model):
    h=np.random.default_rng(42).normal(-1., .3, 600)
    original=model.bd._bdes_multiscale_components(h,4)
    new=model.components(h,original['phis'],original['half_lives'])
    for k,v in original.items():
        if isinstance(v,np.ndarray):assert v.tobytes()==new[k].tobytes(),k
        else:assert v==new[k],k


@pytest.mark.parametrize('phis',[[.9],[.7,.97],[.2,.7,.95,.999]])
def test_conditional_objective_matches_independent_recursion(model,phis):
    h=np.random.default_rng(5).normal(size=120);h-=h.mean();phis=np.array(phis)
    q=np.empty((len(h),len(phis)))
    for j,phi in enumerate(phis):
        level=float(np.median(h))
        for i,x in enumerate(h):level=(1-phi)*x+phi*level;q[i,j]=level
        q[:,j]-=q[:,j].mean()
    b=np.linalg.solve(q.T@q+.05*max(h.var(),1e-8)*np.eye(len(phis)),q.T@h)
    signal=np.abs(b)*q.std(axis=0);positive=signal[signal>0]
    if len(positive):b*=signal/(signal+np.median(positive)+1e-8)
    residual=h-q@b;residual-=residual.mean()
    rho=np.clip(residual[:-1]@residual[1:]/(residual[:-1]@residual[:-1]),-.999999,.999999)
    error=h[1:]-(q[:-1]*phis)@b-rho*residual[:-1]
    expected=.5*len(error)*(np.log(2*np.pi*np.mean(error**2))+1)
    assert model.rate_fit.objective(logit(phis),h)==pytest.approx(expected,rel=1e-12)


def test_fit_improves_training_target_and_retains_stationarity(model):
    rng=np.random.default_rng(98);h=np.empty(800);h[0]=0
    for i in range(1,len(h)):h[i]=.85*h[i-1]+rng.normal(scale=.2)
    phis,receipt=model.rate_fit.fit(h,np.array([.7,.9,.97,.995]))
    assert np.all((phis>0)&(phis<1));assert receipt['negative_loglikelihood']<=receipt['initial_negative_loglikelihood']+1e-7


def test_century_return_mean_unchanged(model):
    from simfolio_forecasting_methodology.models.asset_level.sv_moment_functions import (
        moment_return_curves,
    )
    d=np.random.default_rng(10).normal(.0005,.01,600).tobytes()
    expected,_=moment_return_curves(model.student.original_fit(d),25200)
    mean,paths=model.asset_paths(d,np.full((2,25200),.5))
    assert mean.tobytes()==expected.tobytes();assert np.isfinite(paths).all()


def test_adaptive_order_uses_penalized_conditional_likelihood(model):
    rng=np.random.default_rng(70);h=np.empty(600);h[0]=0.
    for i in range(1,len(h)):h[i]=.85*h[i-1]+rng.normal(scale=.15)
    rates,receipt=model.rate_fit.fit_adaptive(h,np.array([.9,.95,.98,.99]))
    assert receipt['component_count']==len(rates)
    assert receipt['bic_parameter_count']==2*len(rates)+3
    assert receipt['selected_bic']==pytest.approx(2*receipt['negative_loglikelihood']+(2*len(rates)+3)*np.log(len(h)-1))
    tested=receipt['tested_orders']
    assert tested[0]['count']==1
    assert receipt['selected_bic']==min(x['bic'] for x in tested)
    assert len(tested)==len(rates)+1


def test_single_component_matches_adaptive_first_fit(model):
    rng=np.random.default_rng(70);h=np.empty(600);h[0]=0.
    for i in range(1,len(h)):h[i]=.85*h[i-1]+rng.normal(scale=.15)
    adaptive,receipt=model.rate_fit.fit_adaptive(h,np.array([.9,.95,.98,.99]))
    single,control=model.rate_fit.fit_one(h,np.array([.9,.95,.98,.99]))
    assert len(adaptive)==len(single)==1
    assert adaptive.tobytes()==single.tobytes()
    assert receipt['negative_loglikelihood']==control['negative_loglikelihood']
    assert control['count_selection']=='fixed_one_learned_decay_control'


def test_unshrunk_objective_and_construction_match_loading_rule(model):
    from scipy.special import logit
    h=np.random.default_rng(88).normal(-1.,.3,500);phi=np.array([.9])
    shrunk=model.components(h,phi);plain=model.components(h,phi,shrink_loadings=False)
    assert np.allclose(shrunk['b']*2,plain['b'],rtol=1e-6)
    centered=h-h.mean();q=model.rate_fit.ewma(centered,1-phi[0]);q-=q.mean()
    residual=centered-q*plain['b'][0];residual-=residual.mean()
    rho=np.clip(residual[:-1]@residual[1:]/(residual[:-1]@residual[:-1]),-.999999,.999999)
    error=centered[1:]-q[:-1]*phi[0]*plain['b'][0]-rho*residual[:-1]
    expected=.5*len(error)*(np.log(2*np.pi*np.mean(error**2))+1)
    assert model.rate_fit.objective(logit(phi),centered,False)==pytest.approx(expected,rel=1e-12)
