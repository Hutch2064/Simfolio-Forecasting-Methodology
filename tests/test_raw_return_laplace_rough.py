"""Dense independent references for sparse raw-return Laplace integration."""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import toeplitz
from scipy.optimize import minimize
from scipy.optimize._numdiff import approx_derivative

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/raw_return_laplace_rough'))
import laplace_sv


@pytest.mark.parametrize('n,phi', [(1, .8), (7, .05), (80, .99)])
def test_tridiagonal_solve_and_selected_inverse_match_dense(n, phi):
    diagonal, off = laplace_sv.precision(n, phi, .3)
    diagonal += np.random.default_rng(n).uniform(.1, 3., n)
    matrix = np.diag(diagonal) + np.diag(np.full(n - 1, off), 1) + np.diag(np.full(n - 1, off), -1)
    pivots = laplace_sv.factor(diagonal, off)
    rhs = np.random.default_rng(41).normal(size=n)
    np.testing.assert_allclose(laplace_sv.solve(pivots, off, rhs), np.linalg.solve(matrix, rhs), rtol=1e-12, atol=1e-12)
    variance, adjacent = laplace_sv.inverse_diagonal(pivots, off)
    inverse = np.linalg.inv(matrix)
    np.testing.assert_allclose(variance, np.diag(inverse), rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(adjacent, np.diag(inverse, 1), rtol=1e-12, atol=1e-12)
    assert np.log(pivots).sum() == pytest.approx(np.linalg.slogdet(matrix)[1], abs=1e-12)


@pytest.mark.parametrize('n', [1, 7])
def test_zero_return_density_has_exact_gaussian_integral(n):
    level, phi, eta = .2, .8, .3
    covariance = toeplitz(eta * eta / (1 - phi * phi) * phi ** np.arange(n))
    expected = -.5 * n * np.log(2 * np.pi) - .5 * n * level + .125 * covariance.sum()
    actual = laplace_sv.likelihood_gradient(np.zeros(n), level, phi, eta)
    assert actual[4]
    assert actual[0] == pytest.approx(expected, abs=1e-12)
    np.testing.assert_allclose(actual[2], level - .5 * covariance @ np.ones(n), rtol=1e-12, atol=1e-12)


@pytest.mark.parametrize('n,phi', [(1, .5), (9, .8), (101, .98)])
def test_latent_mode_log_density_and_hyperparameter_gradient_match_dense(n, phi):
    squared = np.random.default_rng(87).normal(size=n) ** 2
    level, eta = .1, .3
    covariance = toeplitz(eta * eta / (1 - phi * phi) * phi ** np.arange(n))
    precision = np.linalg.inv(covariance)
    def objective(h):
        return .5 * ((h - level) @ precision @ (h - level) + (h + squared * np.exp(-h)).sum())
    def gradient(h):
        return precision @ (h - level) + .5 - .5 * squared * np.exp(-h)
    reference = minimize(objective, np.full(n, level), jac=gradient, method='BFGS', options={'gtol': 1e-8})
    h = reference.x
    hessian = precision + np.diag(.5 * squared * np.exp(-h))
    expected = (-.5 * n * np.log(2 * np.pi) - .5 * np.linalg.slogdet(covariance)[1]
                - objective(h) - .5 * np.linalg.slogdet(hessian)[1])
    result = laplace_sv.likelihood_gradient(squared, level, phi, eta)
    assert result[4]
    assert result[0] == pytest.approx(expected, abs=2e-7)
    np.testing.assert_allclose(result[2], h, rtol=1e-5, atol=1e-5)
    def target(parameters):
        return laplace_sv.likelihood_gradient(squared, parameters[0], parameters[1], np.exp(parameters[2]))[0]
    numerical = approx_derivative(target, [level, phi, np.log(eta)], method='3-point').ravel()
    np.testing.assert_allclose(result[1], numerical, rtol=2e-5, atol=2e-5)


def test_century_mean_curve_remains_the_original_predecessor():
    import importlib.util

    path = Path(__file__).resolve().parents[1] / 'tools/raw_return_laplace_rough/models.py'
    spec = importlib.util.spec_from_file_location('raw_laplace_mean_guard', path)
    model = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = model
    spec.loader.exec_module(model)
    model.initialize()
    x = np.random.default_rng(492).normal(.0003, .01, 507)
    data = x.tobytes()
    from simfolio_forecasting_methodology.models.asset_level.sv_moment_functions import (
        moment_return_curves,
    )

    expected, _ = moment_return_curves(model.original_fit(data), 25200)
    uniforms = np.full((8, 25200), .5)
    actual, paths = model.asset_paths(data, uniforms)
    assert expected.tobytes() == actual.tobytes()
    assert np.isfinite(paths).all()
    assert model.refit(data)['raw_return_laplace_fit']['success']
