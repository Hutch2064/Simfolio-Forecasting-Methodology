import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import toeplitz


@pytest.fixture(scope='module')
def model():
    path = Path(__file__).resolve().parents[1]/'tools/differenced_whittle_rough/models.py'
    spec = importlib.util.spec_from_file_location('differenced_whittle_test_model', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('n', [9, 10, 127, 128])
def test_differenced_expectation_matches_dense_covariance_including_zero(model, n):
    phi = np.array([0., .2, .93, .99999])
    mass = np.array([.2, .1, .3, .4])
    original = toeplitz((phi[None, :]**np.arange(n+1)[:, None])@mass)
    differences = np.diff(np.eye(n+1), axis=0)
    covariance = differences@original@differences.T
    fourier = np.exp(-2j*np.pi*np.arange(n//2+1)[:, None]*np.arange(n)[None, :]/n)
    expected = np.real(np.einsum('ij,jk,ik->i', fourier, covariance, fourier.conj()))/n
    np.testing.assert_allclose(model.differenced_expected_periodogram(phi, mass, n), expected, atol=3e-11, rtol=3e-11)


def test_differenced_measurement_noise_is_ma1_not_independent(model):
    n = 301
    variance = 4.2
    actual = model.differenced_expected_periodogram(np.zeros(1), np.array([variance]), n)
    frequencies = 2*np.pi*np.arange(n//2+1)/n
    np.testing.assert_allclose(actual, 2*variance*(1-(1-1/n)*np.cos(frequencies)), atol=3e-15, rtol=1e-14)
    assert actual[0] == pytest.approx(2*variance/n)


def test_differenced_data_eliminate_arbitrary_observation_level():
    y = np.random.default_rng(154).normal(size=501)
    np.testing.assert_allclose(np.diff(y+100), np.diff(y), atol=2e-14, rtol=2e-14)
