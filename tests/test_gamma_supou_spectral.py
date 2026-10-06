import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.linalg import toeplitz


@pytest.fixture(scope='module')
def model():
    path = Path(__file__).resolve().parents[1]/'tools/gamma_supou_spectral/models.py'
    spec = importlib.util.spec_from_file_location('gamma_spectral_test_model', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('n,shape,kappa', [(9,.1,.001), (10,.7,.02), (127,3.,.3), (128,10.,.0004)])
def test_exact_gamma_covariance_spectrum_matches_dense_reference(model,n,shape,kappa):
    covariance = toeplitz((1+kappa*np.arange(n))**(-shape))
    fourier = np.exp(-2j*np.pi*np.arange(n//2+1)[:, None]*np.arange(n)[None, :]/n)
    expected = np.real(np.einsum('ij,jk,ik->i',fourier,covariance,fourier.conj()))/n
    np.testing.assert_allclose(model.expected_unit_periodogram(np.log(shape),np.log(kappa),n),expected,atol=2e-12,rtol=2e-12)


def test_parameter_likelihood_does_not_resolve_factor_count(model,monkeypatch):
    monkeypatch.setattr(model.overlay,'configuration',lambda *a: pytest.fail('lift evaluated during spectral parameter target'))
    theta = np.array([0.,np.log(1/63),np.log(.7)])
    y = np.random.default_rng(165).normal(size=301)
    periodogram = np.abs(np.fft.rfft(y-y.mean())[1:])**2/len(y)
    assert np.isfinite(model.whittle_target(theta,periodogram,len(y),4.))
