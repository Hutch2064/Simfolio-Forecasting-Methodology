import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.linalg import toeplitz


@pytest.fixture(scope='module')
def model():
    path = Path(__file__).resolve().parents[1]/'tools/whittle_gls_rough/models.py'
    spec = importlib.util.spec_from_file_location('gls_test_model', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    module.initialize()
    return module


@pytest.mark.parametrize('n,R', [(31,.7), (32,4.934802200544679), (181,9.)])
def test_terminal_matches_dense_gls_point_conditioning(model, n, R):
    phi = np.array([0.,.3,.95])
    mass = np.array([.1,.2,.4])
    w = np.ones(3)
    q = np.diag(mass*(1-phi*phi))
    y = np.random.default_rng(186).normal(size=n)+2.
    covariance = toeplitz((phi[None,:]**np.arange(n)[:,None])@mass)+R*np.eye(n)
    inverse_one = np.linalg.solve(covariance, np.ones(n))
    level = inverse_one@y/inverse_one.sum()
    cross = mass[:,None]*phi[:,None]**np.arange(n-1,-1,-1)[None,:]
    expected_state = cross@np.linalg.solve(covariance, y-level)
    expected_p = np.diag(mass)-cross@np.linalg.solve(covariance, cross.T)
    actual = model.conditional_terminal(y, phi, w, q, np.sqrt(4.934802200544679/R), R)
    assert actual[0] == pytest.approx(level, abs=2e-14)
    np.testing.assert_allclose(actual[1], expected_state, atol=2e-14, rtol=2e-14)
    np.testing.assert_allclose(actual[2], expected_p, atol=2e-14, rtol=2e-14)


def test_covariance_inference_and_return_backbone_are_inherited(model):
    assert model.parent.base.base.rough_fit is model.parent.rough_fit
    assert model.parent.base.base.prepared is model.prepared
    assert model.Candidate is model.parent.Candidate
    assert model.parent.base.noise.predecessor_fit is model.parent.base.base.predecessor_fit


def test_terminal_is_invariant_to_constant_observation_shift(model):
    y = np.random.default_rng(189).normal(size=91)
    args = (np.array([.8]), np.ones(1), np.array([[.1]]), 1., 4.934802200544679)
    left = model.conditional_terminal(y, *args)
    right = model.conditional_terminal(y+3, *args)
    assert right[0]-left[0] == pytest.approx(3., abs=2e-14)
    np.testing.assert_allclose(left[1], right[1], atol=2e-14, rtol=0)
    np.testing.assert_array_equal(left[2], right[2])
