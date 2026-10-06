"""Dense Gaussian and difference-operator references for the spectral fit."""
import importlib.util
from pathlib import Path
import sys
import numpy as np
import pytest
from scipy.linalg import toeplitz


@pytest.fixture(scope='module')
def model():
    p=Path(__file__).resolve().parents[1]/'tools/full_hurst_differenced_rough/models.py'
    spec=importlib.util.spec_from_file_location('test_full_hurst_differenced_model',p)
    m=importlib.util.module_from_spec(spec);sys.modules[spec.name]=m;spec.loader.exec_module(m)
    return m


@pytest.mark.parametrize('n',[9,10,63,64])
def test_colored_noise_and_rough_spectrum_against_dense_difference_covariance(model,n):
    h,k=.014,1/43;noise=.4
    c=model.overlay.exact_covariance(h,k,np.arange(n+1))
    difference=np.diff(np.eye(n+1),axis=0)
    covariance=difference@(toeplitz(c)+noise*np.eye(n+1))@difference.T
    fourier=np.exp(-2j*np.pi*np.arange(n//2+1)[:,None]*np.arange(n)[None,:]/n)
    expected=np.real(np.einsum('ij,jk,ik->i',fourier,covariance,fourier.conj()))/n
    actual=model.unit_expected_periodogram(h,np.log(k),n)+noise*model.noise_expected_periodogram(n)
    np.testing.assert_allclose(actual,expected,rtol=5e-10,atol=2e-11)


def test_real_fourier_weights_and_mean_model_unchanged(model):
    theta=np.array([.1,np.log(1/63),np.log(.7)])
    for n in [31,32]:
        p=np.arange(n//2+1)/n
        expected=np.exp(2*theta[2])*model.unit_expected_periodogram(theta[0],theta[1],n)+.4*model.noise_expected_periodogram(n)
        terms=np.log(expected)+p/expected;terms[0]*=.5
        if n%2==0:terms[-1]*=.5
        assert model.whittle_target(theta,p,n,.4)==model.log_prior(theta)-terms.sum()
    assert model.streamed.prepared is model.streamed.base.prepared
    assert model.streamed.predecessor_asset_paths is model.streamed.base.predecessor_asset_paths


def test_full_hurst_support(model):
    import math
    for h in [.001,.025,.1,.499]:
        theta=np.array([h,math.log(1/63),math.log(.7)])
        assert np.isfinite(model.log_prior(theta))
        if .03<h<.49:assert model.log_prior(theta)==model.streamed.overlay.log_prior(theta)
    for h in [0.,-.001,.5]:
        assert model.log_prior(np.array([h,math.log(1/63),math.log(.7)]))==-np.inf
