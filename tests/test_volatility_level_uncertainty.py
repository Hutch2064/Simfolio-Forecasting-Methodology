"""Independent dense stationary Gaussian posterior and predictive references."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import toeplitz

root=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('volatility_level_test',root/'tools/volatility_level_uncertainty/level.py')
level=importlib.util.module_from_spec(spec);spec.loader.exec_module(level)
spec=importlib.util.spec_from_file_location('volatility_scalar_test',root/'tools/stationary_ar1_rough/state.py')
scalar=importlib.util.module_from_spec(spec);spec.loader.exec_module(scalar)


@pytest.mark.parametrize('rho',[0.,.3,.9,.999])
def test_level_and_future_moments_match_dense_gaussian_conditioning(rho):
    h=np.random.default_rng(732).normal(size=31)+2.;n=len(h);future=15
    _,mu,variance=scalar.profile(h,rho)
    covariance=variance/(1-rho*rho)*toeplitz(rho**np.arange(n+future))
    history=covariance[:n,:n];inverse=np.linalg.inv(history);one=np.ones(n)
    posterior_variance=1/(one@inverse@one)
    posterior_mean=posterior_variance*(one@inverse@h)
    fit={'coefficients':[rho],'innovation_sd':np.sqrt(variance)}
    assert level.posterior_variance(fit,n)==pytest.approx(posterior_variance,rel=1e-10)
    assert mu==pytest.approx(posterior_mean,abs=1e-9)
    cross=covariance[n:,:n];mapping=cross@inverse
    gain=np.ones(future)-mapping@one
    expected_mean=mapping@h+gain*posterior_mean
    expected_covariance=covariance[n:,n:]-mapping@cross.T+np.outer(gain,gain)*posterior_variance
    powers=rho**np.arange(1,future+1)
    actual_mean=mu+powers*(h[-1]-mu)
    process_variance=variance*(1-powers*powers)/(1-rho*rho)
    actual_variance=level.predictive_variance(process_variance,fit,n)
    np.testing.assert_allclose(actual_mean,expected_mean,rtol=1e-9,atol=1e-9)
    np.testing.assert_allclose(actual_variance,np.diag(expected_covariance),rtol=1e-9,atol=1e-9)


def test_constant_proxy_boundary_has_no_added_uncertainty():
    fit={'coefficients':[0.],'innovation_sd':0.}
    assert level.posterior_variance(fit,31)==0.
    assert np.array_equal(level.predictive_variance(np.zeros(15),fit,31),np.zeros(15))


def test_native_level_response_and_future_rng_match_scalar_reference():
    import math
    import sys
    path=root/'tools/volatility_level_uncertainty/models.py'
    spec=importlib.util.spec_from_file_location('level_uncertainty_native_test',path)
    model=importlib.util.module_from_spec(spec);sys.modules[spec.name]=model;spec.loader.exec_module(model)
    native,dot=model.load_paths();days=31;rho=.9;sd=.2;initial=np.array([.3]);mu=-1.;delta=.25
    gain=level.response(rho,days);mean=np.full(days,.0003);base=np.linspace(-.01,.01,days)
    mm=mu+rho**np.arange(1,days+1)*initial[0]
    mv=sd**2*(1-rho**(2*np.arange(1,days+1)))/(1-rho*rho)+.1*gain*gain
    rng=np.random.default_rng(73);current=initial[0];expected=np.empty(days)
    for t in range(days):
        current=rho*current+sd*rng.standard_normal()
        conventional=mu+current-mm[t];conventional+=delta*gain[t]
        expected[t]=np.clip(mean[t]+(base[t]-mean[t])*math.exp(.5*conventional-.25*mv[t]),-1.,1.)
    rough_rng=np.random.default_rng(72);memory_rng=np.random.default_rng(73);output=np.empty(days)
    native.map_path(np.array([.9]),np.zeros(1),np.zeros((1,1)),np.zeros(1),np.zeros(days),np.zeros(days),rough_rng.bit_generator.capsule,dot,mean,base,output,np.empty(0),np.array([rho]),sd,initial,mu,mm,mv,memory_rng.bit_generator.capsule,True,gain,delta)
    assert output.tobytes()==expected.tobytes()
    assert memory_rng.bit_generator.state==rng.bit_generator.state
