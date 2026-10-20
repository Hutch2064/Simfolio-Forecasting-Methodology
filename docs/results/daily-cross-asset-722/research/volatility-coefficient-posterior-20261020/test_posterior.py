import numpy as np
import pytest
from numpy.polynomial.legendre import leggauss
from scipy.special import logsumexp
from scipy.optimize._numdiff import approx_derivative
import posterior


def test_target_jacobian_and_mode_gradient():
    y = np.random.default_rng(18).normal(size=90)
    points = np.array([[-.4, 1.4], [-2., -1.], [1., -3.]])
    for point, value in zip(points, posterior.log_targets(points, y)):
        expected, gradient = posterior.mode_objective(point, y)
        assert value == pytest.approx(-expected, abs=2e-12)
        numeric = approx_derivative(lambda x: posterior.mode_objective(x, y)[0], point, method='3-point').ravel()
        np.testing.assert_allclose(gradient, numeric, atol=1e-7, rtol=1e-6)


def test_flat_simplex_prior_recovers_analytic_moments():
    covariance = np.array([[2., 1.], [1., 2.]])
    result = posterior.importance_nodes(np.empty(0), np.zeros(2), covariance, 31, 15)
    np.testing.assert_allclose(result['mean'], np.full(2, posterior.SCALE/3), atol=.0015, rtol=0)
    np.testing.assert_allclose(np.diag(result['covariance']), np.full(2, posterior.SCALE**2/18), atol=.0008, rtol=0)
    assert result['covariance'][0, 1] == pytest.approx(-posterior.SCALE**2/36, abs=.0008)
    assert result['ESS'] > 5000


def test_posterior_matches_independent_simplex_quadrature():
    y = np.random.default_rng(52).normal(size=70)
    y -= y.mean()
    y /= np.sqrt(np.mean(y*y))
    mode, covariance = posterior.fit_proposal(y)
    nodes = posterior.importance_nodes(y, mode, covariance, 51, 16)
    x, w = leggauss(70)
    x, w = (x+1)/2, w/2
    uv = np.array([(a, b) for a in x for b in x])
    ab = posterior.SCALE*np.c_[uv[:, 0], (1.-uv[:, 0])*uv[:, 1]]
    z0 = posterior.SCALE-ab.sum(axis=1)
    coordinates = np.log(ab/z0[:, None])
    logjac = np.log(ab[:, 0])+np.log(ab[:, 1])+np.log(z0)-np.log(posterior.SCALE)
    loglikelihood = posterior.log_targets(coordinates, y)-logjac
    quadrature = np.array([wa*wb for wa in w for wb in w])*(1.-uv[:, 0])*posterior.SCALE**2
    lw = loglikelihood+np.log(quadrature)
    exact = np.exp(lw-logsumexp(lw))@ab
    np.testing.assert_allclose(nodes['mean'], exact, atol=.0015, rtol=0)
    assert nodes['ESS'] > 1000
