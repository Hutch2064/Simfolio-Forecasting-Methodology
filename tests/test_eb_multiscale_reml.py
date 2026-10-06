"""Independent covariance reference and differentiation of the REML objective."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative

spec = importlib.util.spec_from_file_location('eb_reml_test', Path(__file__).parents[1] / 'tools/eb_multiscale_rough/reml.py')
reml = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reml)


@pytest.mark.parametrize('phi', [-.8, 0., .9, .999])
@pytest.mark.parametrize('penalty', [.01, 1., 100.])
def test_dense_covariance_reference(phi, penalty):
    rng = np.random.default_rng(52)
    n = 31
    z = rng.normal(size=(n, 4))
    y = rng.normal(size=n) + 2
    covariance = phi ** np.abs(np.arange(n)[:, None] - np.arange(n)[None, :]) / (1 - phi ** 2) + z @ z.T / penalty
    inverse = np.linalg.inv(covariance)
    ones = np.ones(n)
    precision = ones @ inverse @ ones
    level = ones @ inverse @ y / precision
    rss = (y - level) @ inverse @ (y - level)
    reference = .5 * ((n - 1) * np.log(rss / (n - 1)) + np.linalg.slogdet(covariance)[1] + np.log(precision)) / (n - 1)
    point = np.array([np.arctanh(phi), np.log(penalty)])
    stats = reml.statistics(y, z)
    value, gradient, estimate = reml.evaluate(point, stats)
    assert value == pytest.approx(reference, abs=2e-9)
    assert estimate[0] == pytest.approx(level, abs=2e-9)
    numerical = approx_derivative(lambda p: reml.evaluate(p, stats)[0], point).ravel()
    np.testing.assert_allclose(gradient, numerical, rtol=2e-5, atol=2e-8)


def test_null_and_optimized_fit():
    rng = np.random.default_rng(15)
    z = rng.normal(size=(400, 4))
    errors = rng.normal(scale=.2, size=400)
    for t in range(1, len(errors)):
        errors[t] += .6 * errors[t - 1]
    y = 2 + z @ np.array([.5, -.2, 0., .1]) + errors
    estimate, diagnostics = reml.fit(y, z, .5, .1)
    level, beta, phi, variance = estimate
    assert level == pytest.approx(2, abs=.04)
    np.testing.assert_allclose(beta, [.5, -.2, 0., .1], atol=.04)
    assert phi == pytest.approx(.6, abs=.12)
    assert variance == pytest.approx(.04, abs=.01)
    assert not diagnostics['null_loading_variance']
    assert diagnostics['gradient_inf_norm'] < 1e-6


@pytest.mark.parametrize('phi', [-.7, .8, .9999])
def test_null_gradient_and_stationary_whitening(phi):
    rng = np.random.default_rng(87)
    y = rng.normal(size=100)
    z = rng.normal(size=(100, 4))
    stats = reml.statistics(y, z)
    design = np.column_stack((np.ones(len(y)), z, y))
    white = np.vstack((np.sqrt(1 - phi ** 2) * design[0], design[1:] - phi * design[:-1]))
    gd, gc, gp, first, _ = stats
    delta = 1 - phi
    np.testing.assert_allclose(gd + delta * gc + delta ** 2 * gp + delta * (2 - delta) * first, white.T @ white, atol=1e-12)
    point = np.array([np.arctanh(phi)])
    _, gradient, _ = reml.evaluate(point, stats, null=True)
    numerical = approx_derivative(lambda p: reml.evaluate(p, stats, null=True)[0], point).ravel()
    np.testing.assert_allclose(gradient, numerical, rtol=2e-5, atol=2e-8)
