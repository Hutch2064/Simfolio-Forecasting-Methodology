"""Independent covariance and likelihood references for spectral inference."""
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.linalg import toeplitz


@pytest.fixture(scope='module')
def model():
    path = Path(__file__).resolve().parents[1]/'tools/debiased_whittle_rough/models.py'
    spec = importlib.util.spec_from_file_location('whittle_test_model', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('n', [9, 10, 127, 128])
def test_expected_periodogram_matches_dense_toeplitz_quadratic_form(model, n):
    phi = np.array([0., .2, .93, .99999])
    mass = np.array([.2, .1, .3, .4])
    covariance = toeplitz((phi[None, :]**np.arange(n)[:, None])@mass)
    fourier = np.exp(-2j*np.pi*np.arange(1, n//2+1)[:, None]*np.arange(n)[None, :]/n)
    expected = np.real(np.einsum('ij,jk,ik->i', fourier, covariance, fourier.conj()))/n
    np.testing.assert_allclose(model.expected_periodogram(phi, mass, n), expected, atol=2e-11, rtol=2e-11)


@pytest.mark.parametrize('n', [31, 32])
def test_white_noise_likelihood_restores_gaussian_half_and_nuisance_mean(model, monkeypatch, n):
    y = np.random.default_rng(134).normal(size=n)
    y -= y.mean()
    spectrum = np.abs(np.fft.rfft(y)[1:])**2/n
    monkeypatch.setattr(model.base.overlay, 'log_prior', lambda theta: 0.)
    monkeypatch.setattr(model.base.overlay, 'configuration',
        lambda theta, lag: (np.array([0.]), np.ones(1), np.array([[float(theta[0])]])))
    for latent_variance in [.1, 2., 10.]:
        total = latent_variance+.3
        expected = -.5*((n-1)*np.log(total) + np.sum(y*y)/total)
        assert model.whittle_target([latent_variance], spectrum, n, n, .3) == pytest.approx(expected, abs=2e-13)


def test_nonzero_periodogram_is_unchanged_by_sample_mean(model):
    y = np.random.default_rng(18).normal(size=501)
    np.testing.assert_allclose(np.fft.rfft(y)[1:], np.fft.rfft(y-y.mean())[1:], atol=2e-14, rtol=2e-14)
