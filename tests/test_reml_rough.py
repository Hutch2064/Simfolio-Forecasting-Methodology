import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import cholesky, toeplitz


@pytest.fixture(scope='module')
def model():
    path = Path(__file__).resolve().parents[1]/'tools/reml_rough/models.py'
    spec = importlib.util.spec_from_file_location('reml_test_model',path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('n,R', [(31,.7),(32,4.934802200544679),(181,9.)])
def test_restricted_filter_matches_dense_gls_likelihood_and_integrated_state(model,n,R):
    phi = np.array([0.,.3,.95])
    mass = np.array([.1,.2,.4])
    w = np.ones(3)
    q = np.diag(mass*(1-phi*phi))
    y = np.random.default_rng(186).normal(size=n)+2.
    covariance = toeplitz((phi[None,:]**np.arange(n)[:,None])@mass)+R*np.eye(n)
    np.linalg.solve(covariance,y)
    inverse_one = np.linalg.solve(covariance,np.ones(n))
    precision = inverse_one.sum()
    level = inverse_one@y/precision
    residual = y-level
    restricted = -.5*(n*np.log(2*np.pi)+2*np.log(np.diag(cholesky(covariance))).sum()
                      +residual@np.linalg.solve(covariance,residual)+np.log(precision))
    cross = mass[:,None]*phi[:,None]**np.arange(n-1,-1,-1)[None,:]
    expected_state = cross@np.linalg.solve(covariance,residual)
    response = cross@inverse_one
    expected_p = np.diag(mass)-cross@np.linalg.solve(covariance,cross.T)+np.outer(response,response)/precision
    actual = model.restricted_filter(y,phi,w,q,R)
    assert actual[0] == pytest.approx(restricted,abs=2e-12)
    assert actual[1] == pytest.approx(level,abs=2e-14)
    assert actual[2] == pytest.approx(1/precision,abs=2e-14)
    np.testing.assert_allclose(actual[3],expected_state,atol=2e-14,rtol=2e-14)
    np.testing.assert_allclose(actual[4],expected_p,atol=2e-14,rtol=2e-14)


def test_restricted_likelihood_is_invariant_to_observation_level_shift(model):
    y = np.random.default_rng(187).normal(size=100)
    args = (np.array([.8]),np.ones(1),np.array([[.1]]),4.)
    left = model.restricted_filter(y,*args)
    right = model.restricted_filter(y+3,*args)
    assert left[0] == pytest.approx(right[0],abs=2e-13)
    assert right[1]-left[1] == pytest.approx(3.,abs=2e-14)
    np.testing.assert_allclose(left[3],right[3],atol=2e-14,rtol=0)
