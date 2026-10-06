"""Causal residual-volatility adapter correctness."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope="module",params=["volatility_frontier_research","conditional_residual_rough"])
def model(request):
    path=Path(__file__).resolve().parents[1]/"tools"/request.param/"models.py"
    spec=importlib.util.spec_from_file_location("residual_frontier_test_"+request.param,path)
    module=importlib.util.module_from_spec(spec);sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module

def test_offset_matches_conventional_one_step_filter(model):
    y=np.random.default_rng(3).normal(size=100)
    level,phi,eta=.2,.9,.3
    _,filtered,_=model.shell.bd._sv_kalman_filter(y,level,phi,eta,return_path=True)
    expected=np.r_[level,level+phi*(filtered[:-1]-level)]
    np.testing.assert_allclose(model.causal_log_variance_predictor(y,level,phi,eta),expected,atol=1e-15,rtol=0)

def test_current_and_future_observations_cannot_change_offset(model):
    y=np.random.default_rng(4).normal(size=100)
    changed=y.copy();changed[40:]+=100
    a=model.causal_log_variance_predictor(y,.2,.9,.3)
    b=model.causal_log_variance_predictor(changed,.2,.9,.3)
    np.testing.assert_array_equal(a[:41],b[:41])
    assert a[41]!=b[41]

def test_mean_innovations_and_multiscale_backbone_unchanged(model):
    from simfolio_forecasting_methodology.models.asset_level.filtered_innovation_moment_sv import (
        sorted_moment_marginals,
    )
    from simfolio_forecasting_methodology.models.numerical.dynamic_gaussian import (
        map_uniforms_to_marginal_paths,
    )
    x=np.random.default_rng(5).normal(.0001,.012,300)
    u=np.random.default_rng(6).random((24,17))
    fit=model.predecessor_fit(x.tobytes())
    expected=map_uniforms_to_marginal_paths(sorted_moment_marginals(fit,24,17)[:,:,None],u[:,:,None])[:,:,0]
    _,actual=model.predecessor_asset_paths(x.tobytes(),u)
    np.testing.assert_array_equal(actual,expected)
    y,_=model.shell.bd._sv_observed_log_variance(x,float(x.mean()))
    level,phi,eta,_,_=fit["posterior_center"]
    residual,_=model.observed(x.tobytes())
    np.testing.assert_array_equal(residual,y-model.causal_log_variance_predictor(y,level,phi,eta))


def test_stationary_normalization_retains_conditional_signal():
    from scipy.special import roots_hermitenorm
    nodes,mass=roots_hermitenorm(40);mass/=mass.sum()
    stationary_variance=.8
    for conditional_mean,conditional_variance in [(.3,.2),(-.3,.4),(0.,.8)]:
        state=conditional_mean+np.sqrt(conditional_variance)*nodes
        multiplier=np.exp(.5*state-.25*stationary_variance)
        expected=np.exp(conditional_mean+.5*(conditional_variance-stationary_variance))
        np.testing.assert_allclose(mass@multiplier**2,expected,rtol=1e-13)
