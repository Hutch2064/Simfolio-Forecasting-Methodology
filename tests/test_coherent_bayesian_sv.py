"""Independent likelihood/augmentation, joint posterior, and execution contracts."""
import math
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import gammaln
from scipy.stats import norm, t

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/rough_bayesian'))
import coherent
import mixture
from mixture_kernels import exact_return_loglik


def test_ar1_covariance_and_transformed_prior():
    for phi in [-.8,0.,.98]:
        theta = np.array([np.arctanh(phi),math.log(.7)])
        transition,weights,q = coherent.configuration(theta,'ar1',500)
        assert transition[0] == pytest.approx(phi)
        assert weights.tolist() == [1.]
        assert q[0,0]/(1-transition[0]**2) == pytest.approx(.49)
        expected = 19*math.log1p(phi)+.5*math.log1p(-phi)+math.log1p(-phi*phi)
        assert coherent.log_prior(theta,'ar1') == pytest.approx(expected)


def test_student_raw_likelihood_and_unit_variance_scale():
    squared = np.array([0.,.3,4.,80.])
    h = np.array([.1,-.2,1.,3.])
    for nu in [2.1,5.,30.]:
        scale = np.exp(h/2)*math.sqrt((nu-2)/nu)
        expected = np.sum(t.logpdf(np.sqrt(squared)/scale,nu)-np.log(scale))
        assert coherent.student_squared_loglik(squared,h,nu) == pytest.approx(expected,abs=2e-13)
        assert np.allclose(t.var(nu)*scale*scale,np.exp(h))


def test_gamma_augmentation_integrates_to_student_likelihood():
    value,h,nu = 2.4,.3,5.2
    shape,rate = nu/2,nu/2
    def augmented(tau):
        log_gamma = shape*math.log(rate)-gammaln(shape)+(shape-1)*math.log(tau)-rate*tau
        return math.exp(log_gamma+norm.logpdf(value,scale=math.exp(h/2)*math.sqrt((nu-2)/(nu*tau))))
    integrated = quad(augmented,0,np.inf,epsabs=1e-12)[0]
    actual = coherent.student_squared_loglik(np.array([value*value]),np.array([h]),nu)
    assert math.log(integrated) == pytest.approx(actual,abs=1e-10)
    rates = coherent.scale_rates(np.array([0.,value*value]),np.array([h,h]),nu)
    assert rates[1] == pytest.approx(2/(nu+value*value*math.exp(-h)*nu/(nu-2)))


@pytest.mark.parametrize('kind,student',[('ar1',False),('ar1',True),('rough',True)])
def test_chain_resume_preserves_complete_trace_history_reservoir_and_rng(kind,student):
    data = (np.random.default_rng(29).normal(size=25)*.01).tobytes()
    whole,a = coherent.parameter_chain(data,63,kind,student,42,64,80,24)
    first,state = coherent.parameter_chain(data,63,kind,student,42,64,40,24)
    second,b = coherent.parameter_chain(data,63,kind,student,0,0,40,24,state)
    np.testing.assert_array_equal(whole,np.concatenate([first,second]))
    for name in ('theta','root','h'):
        np.testing.assert_array_equal(a[name],b[name])
    for name in ('parameters','histories'):
        np.testing.assert_array_equal(a['reservoir'][name],b['reservoir'][name])
    assert a['rng_state'] == b['rng_state']
    assert a['reservoir_rng_state'] == b['reservoir_rng_state']


def test_generalized_gaussian_rough_chain_preserves_existing_sampler_exactly():
    data = (np.random.default_rng(39).normal(size=25)*.01).tobytes()
    original,a = mixture.parameter_chain(data,63,True,52,64,80,24)
    generalized,b = coherent.parameter_chain(data,63,'rough',False,52,64,80,24)
    np.testing.assert_array_equal(original,generalized)
    for name in ('theta','root','h'):
        np.testing.assert_array_equal(a[name],b[name])
    for name in ('parameters','histories'):
        np.testing.assert_array_equal(a['reservoir'][name],b['reservoir'][name])
    assert a['rng_state'] == b['rng_state']


@pytest.fixture
def coherent_shell(monkeypatch,tmp_path):
    import importlib.util
    directory = Path(__file__).resolve().parents[1]/'tools/rough_bayesian'
    spec = importlib.util.spec_from_file_location('models',directory/'models.py')
    shell = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules,'models',shell)
    spec.loader.exec_module(shell)
    monkeypatch.setattr(shell.controls,'CACHE_ROOT',tmp_path/'cache')
    coherent.fit.cache_clear()
    return shell


@pytest.mark.parametrize('kind,student',[('ar1',False),('ar1',True),('rough',True)])
def test_asset_adapter_preserves_full_cached_paths_and_never_fits_production_sv(coherent_shell,monkeypatch,kind,student):
    import pandas as pd
    from simfolio_forecasting_methodology.runner import TrainingData,PortfolioPolicy,ForecastContext
    from simfolio_forecasting_methodology.evaluation import empirical_crps_by_horizon
    def forbidden(*args):
        raise AssertionError('standalone SV called conventional production anchor')
    monkeypatch.setattr(coherent_shell.controls,'asset_fit',forbidden)
    assets = np.random.default_rng(85).normal(size=(80,6))*.01
    dates = pd.bdate_range('1980-01-02',periods=86)
    weights = (1/6,)*6
    training = TrainingData(assets@np.array(weights),assets,
        PortfolioPolicy(('SPYSIM','GLDSIM','IWMSIM','XLPSIM','IEISIM','REITSIM'),weights,'monthly'),dates[:80].values)
    context = ForecastContext('coherent_test','test','origin',6,24,38,dates[80:].values,str(dates[79].date()))
    candidate = coherent.CoherentCandidate('coherent_test',kind,student,32,32,32)
    fresh = candidate.simulate_daily_log_returns(training,context)
    coherent.fit.cache_clear()
    cached = candidate.simulate_daily_log_returns(training,context)
    np.testing.assert_array_equal(fresh,cached)
    assert fresh.shape == (24,6) and np.isfinite(fresh).all()
    realized = np.zeros(6)
    np.testing.assert_array_equal(empirical_crps_by_horizon(np.cumsum(fresh,axis=1),realized),
                                  empirical_crps_by_horizon(np.cumsum(cached,axis=1),realized))


def test_student_collapsed_nu_gamma_and_corrected_latent_block_match_joint_quadrature(monkeypatch):
    # Fix AR1 hyperparameters and level prior range, then integrate level out
    # independently to obtain the two-dimensional exact h/nu posterior.
    from scipy.special import ndtr
    data = np.array([.02,-.01]).tobytes()
    squared,y,active,center,lower,upper = mixture.observed(data,True)
    # A one-observation independent target, retaining exactly zero observations
    # in a separate likelihood test above; center/level support are fixed here.
    squared,y,active = squared[:1],y[:1],active[:1]
    monkeypatch.setattr(mixture,'observed',lambda *_:(squared,y,active,center,lower,upper))
    monkeypatch.setattr(coherent,'log_prior',lambda point,kind: 0. if np.array_equal(point,np.array([math.atanh(math.exp(-1/63)),math.log(.7)])) else -np.inf)
    eta,prior_sd = .7,4.
    total_sd = math.sqrt(eta*eta+prior_sd*prior_sd)
    def log_h_prior(h):
        conditional_sd = 1/math.sqrt(1/eta**2+1/prior_sd**2)
        conditional_mean = (h/eta**2+center/prior_sd**2)*conditional_sd**2
        mass = ndtr((upper-conditional_mean)/conditional_sd)-ndtr((lower-conditional_mean)/conditional_sd)
        return norm.logpdf(h,center,total_sd)+math.log(max(mass,1e-300))
    def density(h,z):
        return math.exp(log_h_prior(h)+coherent.tail_prior(z)+coherent.student_squared_loglik(squared,np.array([h]),2+math.exp(z)))
    def integrate(feature):
        return quad(lambda z: quad(lambda h: feature(h,z)*density(h,z),center-18,center+18,epsabs=2e-7)[0],-8,6,epsabs=2e-7)[0]
    mass = integrate(lambda h,z:1.)
    expected_h,expected_z = integrate(lambda h,z:h)/mass,integrate(lambda h,z:z)/mass
    draws,_ = coherent.parameter_chain(data,63,'ar1',True,271,1024,24000,24)
    assert draws[:,3].mean() == pytest.approx(expected_z,abs=.10)
    # h is recorded after log likelihood in the diagnostics, not theta level.
    assert draws[:,5].mean() == pytest.approx(expected_h,abs=.10)
