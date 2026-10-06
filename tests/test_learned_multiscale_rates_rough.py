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


def test_predictive_loading_objective_matches_constructed_states(model):
    h=np.random.default_rng(101).normal(-1.,.3,600);phi=np.array([.85,.99]);scale=.7
    fitted=model.components(h,phi,loading_scale=scale)
    plain=model.components(h,phi,shrink_loadings=False)
    assert np.array_equal(fitted['b'],plain['b']*scale)
    centered=h-h.mean();z=np.r_[logit(phi),logit(scale)]
    error=model.rate_fit.residual_innovations(centered,phi,False,scale)
    expected=.5*len(error)*(np.log(2*np.pi*np.mean(error**2))+1)
    assert model.rate_fit.objective_loading(z,centered)==pytest.approx(expected,rel=1e-12)

def test_predictive_loading_selects_by_penalized_target(model):
    h=np.random.default_rng(123).normal(size=500)
    rates,record=model.rate_fit.fit_adaptive_loading(h,np.array([.9]))
    assert 0<record['loading_scale']<1
    assert record['negative_loglikelihood']<=record['initial_negative_loglikelihood']+1e-7
    assert record['bic_parameter_count']==2*len(rates)+4
    assert record['selected_bic']==min(x['bic'] for x in record['tested_orders'])


def test_full_hurst_domain_retains_interior_prior_and_mean():
    path=Path(__file__).resolve().parents[1]/'tools/learned_loading_full_hurst_rough/models.py'
    spec=importlib.util.spec_from_file_location('test_learned_loading_full_hurst',path)
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    theta=np.array([.1,np.log(1/63),np.log(.7)])
    assert m.log_prior(theta)==0.
    for h in (.001,.499):
        theta[0]=h;assert m.log_prior(theta)==0.
        spectrum=m.rough.backend.unit_expected_periodogram(h,theta[1],100)
        assert np.isfinite(spectrum).all() and (spectrum>0).all()
    for h in (0.,.5):
        theta[0]=h;assert m.log_prior(theta)==-np.inf
    assert m.map_impl.physical(np.array([.2,0.,0.]))[0]==.1


def test_zero_ridge_svd_likelihood_matches_independent_projection():
    folder=Path(__file__).resolve().parents[1]/'tools/learned_loading_zero_ridge_rough'
    spec=importlib.util.spec_from_file_location('test_zero_ridge_model',folder/'models.py')
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    h=np.random.default_rng(39).normal(size=300);h-=h.mean();phi=np.array([.8,.98]);scale=.3
    q=np.column_stack([m.rate_fit.ewma(h,1-p) for p in phi]);q-=q.mean(axis=0)
    b=np.linalg.lstsq(q,h,rcond=np.finfo(float).eps*max(q.shape))[0]*scale
    residual=h-q@b;residual-=residual.mean()
    rho=np.clip(residual[:-1]@residual[1:]/(residual[:-1]@residual[:-1]),-.999999,.999999)
    error=h[1:]-(q[:-1]*phi)@b-rho*residual[:-1]
    expected=.5*len(error)*(np.log(2*np.pi*np.mean(error**2))+1)
    assert m.rate_fit.objective_loading(np.r_[logit(phi),logit(scale)],h)==pytest.approx(expected,rel=1e-12)
    assert np.allclose(m.construction.components(h,phi,loading_scale=scale)['b'],b,rtol=1e-12)


def test_single_predictive_loading_control_is_adaptive_first_fit():
    path=Path(__file__).resolve().parents[1]/'tools/single_predictive_loading_full_hurst_rough/models.py'
    spec=importlib.util.spec_from_file_location('test_single_predictive_loading_model',path)
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    h=np.random.default_rng(32).normal(size=500);centered=h-h.mean()
    rho=float(centered[:-1]@centered[1:]/(centered[:-1]@centered[:-1]))
    first=np.array([np.clip(rho,np.finfo(float).eps,1.-np.finfo(float).eps)])
    expected,receipt=m.learned.rate_fit.fit_loading(h,first)
    actual,control=m.fit_rates(h,np.array([.9,.99]))
    assert expected.tobytes()==actual.tobytes()
    assert receipt['loading_scale']==control['loading_scale']
    assert control['component_count']==1


def test_unclipped_rough_observations_match_untransformed_log_square():
    p=Path(__file__).resolve().parents[1]/'tools/learned_loading_unclipped_rough/models.py'
    spec=importlib.util.spec_from_file_location('test_unclipped_learned_loading',p)
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    x=np.random.default_rng(813).normal(.0003,.012,600)
    x[0]=.35;x[1]=-.3
    data=x.tobytes();m.initialize(None,1,1)
    y,eps=m.observed(data)
    original=m.rough.parent.predecessor_fit(data)
    bias,noise=m.rough.parent.parent.log_square_moments(original['student_return_laplace_fit']['inverse_df'])
    gaussian_bias,_=m.rough.parent.parent.log_square_moments(0.)
    squared=((x-x.mean())*100)**2
    floor=max(np.quantile(squared[squared>0],.001)*.1,1e-10)
    raw=np.log(np.maximum(squared,floor))-m.shell.bd.SV_LOG_CHI_SQUARE_MEAN+gaussian_bias-bias
    expected=raw-m.causal_predictor(raw,*original['posterior_center'][:3],noise)
    assert np.array_equal(y,expected)
    assert np.array_equal(eps,(x-x.mean())*100)
    assert raw.max()>np.quantile(raw,.995)
