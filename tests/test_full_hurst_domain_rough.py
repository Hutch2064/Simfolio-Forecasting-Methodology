"""Theoretical-domain support and independent covariance/lift checks."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import beta


@pytest.fixture(scope='module')
def model():
    p=Path(__file__).resolve().parents[1]/'tools/full_hurst_domain_rough/models.py'
    spec=importlib.util.spec_from_file_location('test_full_hurst_model',p)
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    return m


@pytest.mark.parametrize('h',[1e-6,.001,.01,.03,.1,.49,.499999])
def test_full_support_and_exact_bessel_covariance(model,h):
    k=1/63;n=32
    assert np.isfinite(model.log_prior([h,np.log(k),np.log(.7)]))
    lags=np.arange(n);a,b=.5-h,2*h
    covariance=np.array([1. if t==0 else quad(lambda z, t=t:np.exp(-k*(1+z)/(1-z)*t) if z<1 else 0.,
        0.,1.,weight='alg',wvar=(a-1,b-1),epsabs=1e-12,epsrel=1e-12)[0]/beta(a,b) for t in lags])
    expected=2*np.fft.rfft(covariance*(1-lags/n)).real-1
    np.testing.assert_allclose(model.unit_expected_periodogram(h,np.log(k),n),expected[1:],rtol=2e-8,atol=2e-10)


@pytest.mark.parametrize('h',[1e-6,.001,.01,.499999])
def test_expanded_domain_dynamic_lift_certifies_all_requested_daily_lags(model,h):
    from dynamic import selected_kernel
    k=1/2520;lag=2520
    phi,mass,receipt=selected_kernel(h,k,lag)
    exact=model.overlay.exact_covariance(h,k,np.arange(lag+1))
    approximate=phi[None,:]**np.arange(lag+1)[:,None]@mass
    assert receipt['autocorrelation_error_upper_bound']<=.001
    assert np.max(np.abs(exact-approximate))<=.001
    assert mass.min()>=0 and mass.sum()==pytest.approx(1.)


def test_mathematical_endpoints_excluded_and_other_prior_unchanged(model):
    t=np.array([.1,np.log(1/63),np.log(.7)])
    assert model.log_prior(t)==model.overlay.log_prior(t)
    for h in [-.01,0.,.5,.6]:
        t[0]=h;assert model.log_prior(t)==-np.inf
    assert model.streamed.predecessor_fit is model.streamed.base.predecessor_fit
    assert model.streamed.predecessor_asset_paths is model.streamed.base.predecessor_asset_paths
