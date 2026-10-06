"""Independent generative and stream contracts for proxy leverage."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from simfolio_forecasting_methodology.models.numerical import dynamic_gaussian as dg

path=Path(__file__).resolve().parents[1]/'tools/proxy_leverage_rough/leverage_fit.py'
spec=importlib.util.spec_from_file_location('proxy_leverage_fit_test',path)
fit=importlib.util.module_from_spec(spec);sys.modules[spec.name]=fit;spec.loader.exec_module(fit)


def test_captured_scores_preserve_gaussian_uniform_stream_exactly():
    assets=np.random.default_rng(77).normal(size=(300,6))
    u,e,model=fit.capture_uniforms(assets,8,20,81,dg)
    reference=dg.simulate_future_gaussian_uniforms(model,8,20,np.random.default_rng(81))
    assert np.array_equal(u,reference)
    rng=np.random.default_rng(81)
    mean,cov=dg.kalman_terminal_posterior(model)
    rng.multivariate_normal(mean,cov,size=8,check_valid='raise')
    draws=rng.normal(size=(20,8*(6+model['factor_count'])))
    expected=draws[:,8*model['factor_count']:].reshape(20,8,6).transpose(1,0,2)
    assert np.array_equal(e,expected)


@pytest.mark.parametrize('rho',[-.8,0.,.6])
def test_lagged_leverage_preserves_gaussian_marginals_and_current_independence(rho):
    rng=np.random.default_rng(914);n=120000
    previous=rng.normal(size=n);current=rng.normal(size=n);independent=rng.normal(size=n)
    shock=rho*previous+np.sqrt(1.-rho*rho)*independent
    assert abs(shock.mean())<.012 and abs(shock.var()-1)<.018
    assert abs(np.corrcoef(previous,shock)[0,1]-rho)<.012
    assert abs(np.corrcoef(current,shock)[0,1])<.012
    # E[M^2]=1 for a Gaussian log-vol shock; current return shock is independent.
    multiplier=np.exp(.5*.4*shock-.25*.4**2)
    assert abs(np.mean(multiplier**2)-1)<.012
    assert abs(np.mean(current*multiplier))<.012


def test_proxy_estimator_recovers_lag_direction_and_constant_boundary():
    rng=np.random.default_rng(197);x=rng.normal(size=(30000,2))
    noise=rng.normal(size=(29999,2));vol=.4*x[:-1]+np.sqrt(1-.4**2)*noise
    result=fit.fit_correlations(x,vol,None)
    np.testing.assert_allclose(result,.4,atol=.018)
    assert np.array_equal(fit.fit_correlations(np.ones((50,2)),np.ones((49,2)),None),np.zeros(2))


def test_native_precomputed_innovations_match_scalar_forecast():
    import math
    import sys
    spec=importlib.util.spec_from_file_location('proxy_leverage_native_test',Path(__file__).parents[1]/'tools/proxy_leverage_rough/leverage_native.py')
    native_loader=importlib.util.module_from_spec(spec);sys.modules[spec.name]=native_loader;spec.loader.exec_module(native_loader)
    native,dot=native_loader.load_paths();days=31;rho=.9;sd=.2;initial=np.array([.3]);mu=-1.
    mean=np.full(days,.0003);base=np.linspace(-.01,.01,days)
    mm=mu+rho**np.arange(1,days+1)*initial[0];mv=sd**2*(1-rho**(2*np.arange(1,days+1)))/(1-rho*rho)
    shocks=np.random.default_rng(73).standard_normal(days);current=initial[0];expected=np.empty(days)
    for t in range(days):
        current=rho*current+sd*shocks[t]
        expected[t]=np.clip(mean[t]+(base[t]-mean[t])*math.exp(.5*(mu+current-mm[t])-.25*mv[t]),-1.,1.)
    rough_rng=np.random.default_rng(72);memory_rng=np.random.default_rng(73);state=memory_rng.bit_generator.state;output=np.empty(days)
    native.map_path(np.array([.9]),np.zeros(1),np.zeros((1,1)),np.zeros(1),np.zeros(days),np.zeros(days),rough_rng.bit_generator.capsule,dot,mean,base,output,np.empty(0),np.array([rho]),sd,initial,mu,mm,mv,memory_rng.bit_generator.capsule,True,shocks)
    assert output.tobytes()==expected.tobytes()
    assert memory_rng.bit_generator.state==state
