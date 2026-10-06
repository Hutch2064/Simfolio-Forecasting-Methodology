"""Independent stationary covariance and unchanged mean checks."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.linalg import solve_discrete_lyapunov
from simfolio_forecasting_methodology.models.asset_level.sv_moment_functions import moment_return_curves


@pytest.fixture(scope='module')
def model():
    p=Path(__file__).resolve().parents[1]/'tools/variance_targeted_rough/models.py'
    spec=importlib.util.spec_from_file_location('test_variance_targeted_model',p)
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    m.initialize();return m


def fit():
    return {'bdes_multiscale_vol':dict(phis=np.array([.7,.96]),b=np.array([.4,.8]),q_var=np.array([.02,.003]),
        residual_phi=-.1,residual_innovation_sd=.1,residual_common_loading=-.2,
        ell=1.,q_last=np.array([.2,-.1]),residual_last=.15),
        'base_fit':dict(sigma=.02,dlm_state_noise_var=0.,dlm_state_transition_phi=.8,
            dlm_long_run_anchor_mean=.0004,dlm_state_posterior_deviation_mean=0.,dlm_state_posterior_deviation_var=0.),
        'mean_meta':dict(hac_long_run_variance=.0004),'n_obs':500,
        'innovation_pool':np.linspace(-3,3,101)}


def test_stationary_variance_matches_lyapunov_and_long_horizon(model):
    f=fit();q=f['bdes_multiscale_vol'];sd=np.r_[np.sqrt(q['q_var']),q['residual_innovation_sd']]
    common=np.r_[sd[:-1],sd[-1]*q['residual_common_loading']]
    noise=np.outer(common,common);noise[-1,-1]=sd[-1]**2
    p=solve_discrete_lyapunov(np.diag(np.r_[q['phis'],q['residual_phi']]),noise)
    loading=np.r_[q['b'],1.];v=float(loading@p@loading)
    expected=np.exp(q['ell']+.5*v)/10000*(1+(.0004/.02)**2)-(.0004*np.exp(.5*q['ell']+.125*v)/100/.02)**2
    assert model.stationary_return_variance(f)==pytest.approx(expected,rel=2e-15)
    _,sigma=moment_return_curves(f,25200)
    assert sigma[-1]**2==pytest.approx(expected,rel=2e-15)


def test_100_year_mean_curve_is_byte_exact_and_variance_target_is_training_only(model,monkeypatch):
    f=fit();x=np.linspace(-.02,.02,500);data=x.tobytes()
    monkeypatch.setattr(model.streamed,'predecessor_fit',lambda d:f)
    model.variance_scale.cache_clear()
    mean,sigma=moment_return_curves(f,25200)
    targeted_mean,paths=model.predecessor_asset_paths(data,np.full((8,25200),.5))
    assert mean.tobytes()==targeted_mean.tobytes()
    target=np.var(x,ddof=1)
    assert (sigma[-1]*model.variance_scale(data))**2==pytest.approx(target,rel=3e-15)
    assert model.streamed.rough_fit is model.parent.rough_fit
    assert model.streamed.prepared is model.streamed.base.prepared
    assert np.isfinite(paths).all()
