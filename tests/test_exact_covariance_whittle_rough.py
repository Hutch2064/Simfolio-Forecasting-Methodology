"""Independent references for analytic covariance spectral estimation."""
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.linalg import toeplitz
from scipy.special import beta


@pytest.fixture(scope='module')
def model():
    path=Path(__file__).resolve().parents[1]/'tools/exact_covariance_whittle_rough/models.py'
    spec=importlib.util.spec_from_file_location('test_exact_covariance_model',path)
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module;spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('h,k',[(.03,1/2520),(.1,1/63),(.49,.5)])
def test_covariance_matches_independent_beta_integral(model,h,k):
    lags=np.array([0,1,7,63,252,2520])
    a,b=.5-h,2*h
    expected=[1. if t==0 else quad(lambda z, t=t:np.exp(-k*(1+z)/(1-z)*t) if z<1 else 0.,
        0.,1.,weight='alg',wvar=(a-1,b-1),epsabs=1e-12,epsrel=1e-12)[0]/beta(a,b) for t in lags]
    np.testing.assert_allclose(model.overlay.exact_covariance(h,k,lags),expected,rtol=2e-10,atol=2e-12)


@pytest.mark.parametrize('n',[9,10,63,64])
def test_spectral_mean_matches_dense_covariance_quadratic_form(model,n):
    h,k=.14,1/43
    c=model.overlay.exact_covariance(h,k,np.arange(n))
    covariance=toeplitz(c)
    fourier=np.exp(-2j*np.pi*np.arange(1,n//2+1)[:,None]*np.arange(n)[None,:]/n)
    expected=np.real(np.einsum('ij,jk,ik->i',fourier,covariance,fourier.conj()))/n
    np.testing.assert_allclose(model.unit_expected_periodogram(h,np.log(k),n),expected,rtol=3e-13,atol=3e-13)


def test_likelihood_does_not_resolve_factors_and_retains_map_prior(model,monkeypatch):
    def forbidden(*args):raise AssertionError('factor resolution inside fitting')
    monkeypatch.setattr(model.overlay,'configuration',forbidden)
    theta=np.array([.1,np.log(1/63),np.log(.7)])
    for n in [31,32]:
        spectrum=np.arange(1,n//2+1)/n
        expected=np.exp(2*theta[2])*model.unit_expected_periodogram(theta[0],theta[1],n)+.4
        terms=np.log(expected)+spectrum/expected
        if n%2==0:terms[-1]*=.5
        reference=model.overlay.log_prior(theta)-terms.sum()
        assert model.whittle_target(theta,spectrum,n,.4)==reference
    assert model.streamed.prepared is model.streamed.base.prepared
    assert model.streamed.predecessor_fit is model.streamed.base.predecessor_fit
