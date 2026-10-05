"""Scientific target, numerical approximation, and compiled recurrence checks."""
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.signal import lfilter
from scipy.stats import norm, t

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools/rough_bayesian'))
from kernels import coefficients, fou_cells, fractional_cells, grid, quadrature
from inference import chain, gaussian_path, gaussian_gradient, heston_path, heston_gradient, log_likelihood, student_likelihood, terminal_pgas


@pytest.mark.parametrize('tolerance', [.01, .001])
def test_fractional_grid_and_signed_fou_error(tolerance):
    edges, evidence = grid(tolerance)
    assert evidence['fractional_cell_max_relative_error'] < tolerance
    for h in [.031, .1, .3, .489]:
        rates, weights = quadrature(h, edges)
        lags = np.arange(25201)
        approx = np.exp(-lags[:, None] * rates) @ weights
        exact = fractional_cells(h, lags)
        assert np.max(np.abs(approx / exact - 1)) < tolerance
        for kappa in [1 / 2500, 1 / 63, .49]:
            phi, innovation, signed, root, _ = coefficients(h, kappa, tolerance)
            impulse = (phi[None, :] ** lags[:, None]) @ (signed * innovation)
            target = fou_cells(h, kappa, len(lags))
            # Coefficients use stationary SD as the eta parameter. Compare
            # normalized kernels and preserve the signed long-lag response.
            scale = impulse[0] / target[0]
            error = np.linalg.norm(impulse - scale * target) / np.linalg.norm(scale * target)
            assert error < tolerance
            assert impulse[-1] < 0
            covariance = np.outer(innovation, innovation) / (1 - np.outer(phi, phi))
            assert np.max(np.abs(root @ root.T - covariance)) < 1e-5


def test_gaussian_recurrence_against_independent_linear_filters():
    phi, innovation, weights, root, _ = coefficients(.11, 1 / 63, .01)
    rng = np.random.default_rng(38)
    initial = root @ rng.normal(size=phi.size)
    noise = rng.normal(size=502)
    actual, terminal = gaussian_path(phi, innovation, weights, initial, noise, -.3, .6)
    expected = np.zeros(noise.size)
    end = np.empty(phi.size)
    for j in range(phi.size):
        states, last = lfilter([innovation[j]], [1, -phi[j]], noise, zi=[phi[j] * initial[j]])
        before = np.r_[initial[j], states[:-1]]
        expected += weights[j] * before
        end[j] = states[-1]
    np.testing.assert_allclose(actual, -.3 + .6 * expected, rtol=0, atol=2e-12)
    np.testing.assert_allclose(terminal, end, rtol=0, atol=2e-12)


def test_return_likelihoods_match_predictive_laws():
    rng = np.random.default_rng(7)
    y, h, driving = rng.normal(size=(3, 64))
    rho = -.6
    expected = norm.logpdf(y / np.exp(h / 2), rho * driving, math.sqrt(1 - rho ** 2)) - h / 2
    # Gaussian function omits the common -n/2 log(2pi) term.
    assert log_likelihood(y, h, driving, rho) == pytest.approx(expected.sum() + 32 * math.log(2 * math.pi))
    nu = 6.2
    sd = np.exp(h / 2) * math.sqrt((nu - 2) / nu)
    assert student_likelihood(y, h, nu) == pytest.approx(np.sum(t.logpdf(y / sd, nu) - np.log(sd)))


def test_heston_implicit_equation_and_truncation():
    phi = np.array([.8, .3])
    weights = np.array([.4, .6])
    step = np.array([.2, .7])
    initial = np.array([-.1, .2])
    theta, kappa, eta, white = .7, .3, .4, -.8
    h, terminal = heston_path(phi, weights, step, initial, np.array([white]), math.log(theta), kappa, eta)
    current = theta + weights @ initial
    next_v = theta + weights @ terminal
    expected = phi * initial + step * (eta * math.sqrt(current) * white - kappa * (next_v - theta))
    np.testing.assert_allclose(terminal, expected, atol=1e-15)
    assert h[0] == pytest.approx(math.log(current))
    h, _ = heston_path(phi, weights, step, np.array([-100., -100.]), np.zeros(20), math.log(theta), kappa, eta)
    assert np.isfinite(h).all()


def test_heston_adjoint_matches_finite_differences():
    phi, weights, step = np.array([.8, .3]), np.array([1., 1.]), np.array([.2, .7])
    y = np.random.default_rng(182).normal(size=30)
    noise = np.random.default_rng(284).normal(size=30) * .1
    theta, kappa, eta, rho = .7, .3, .15, -.4
    target, gradient, _, _ = heston_gradient(y, phi, weights, step, noise, math.log(theta), kappa, eta, rho)
    for j in range(noise.size):
        plus, minus = noise.copy(), noise.copy()
        plus[j] += 1e-5
        minus[j] -= 1e-5
        upper = heston_gradient(y, phi, weights, step, plus, math.log(theta), kappa, eta, rho)[0]
        lower = heston_gradient(y, phi, weights, step, minus, math.log(theta), kappa, eta, rho)[0]
        assert gradient[j] == pytest.approx((upper - lower) / 2e-5, abs=2e-8)


@pytest.mark.parametrize('rho,nu', [(-.4, 0.), (0., 7.5)])
def test_fractional_gaussian_adjoint_matches_finite_differences(rho, nu):
    phi, innovation, weights, root, _ = coefficients(.11, 1 / 63, .01)
    y = np.random.default_rng(821).normal(size=32)
    white = np.random.default_rng(291).normal(size=phi.size + y.size) * .2
    target, gradient = gaussian_gradient(y, phi, innovation, weights, root, white, -.2, .6, rho, nu)
    for j in range(white.size):
        plus, minus = white.copy(), white.copy()
        plus[j] += 1e-5
        minus[j] -= 1e-5
        upper = gaussian_gradient(y, phi, innovation, weights, root, plus, -.2, .6, rho, nu)[0]
        lower = gaussian_gradient(y, phi, innovation, weights, root, minus, -.2, .6, rho, nu)[0]
        assert gradient[j] == pytest.approx((upper - lower) / 2e-5, abs=2e-8)


def test_pgas_preserves_nonmarkovian_innovation_posterior():
    # The second return informs W0 through its volatility, while W1 remains
    # independent N(0,1). Quadrature gives the exact conditional target.
    y = np.array([.2, 2.])
    def density(z):
        h = .3 * z
        return math.exp(-.5 * z * z - .5 * (h + 4 * math.exp(-h)))
    mass = quad(density, -10, 10)[0]
    expected = quad(lambda z: z * density(z), -10, 10)[0] / mass
    rng = np.random.default_rng(183)
    reference = np.zeros(2)
    draws = []
    for iteration in range(14000):
        reference = terminal_pgas(y, np.array([.8]), np.array([.6]), np.array([1.]),
            np.array([0.]), reference, 0., .5, 8, rng.normal(size=(2, 8)), rng.random(size=(3, 8)))
        if iteration >= 2000:
            draws.append(reference.copy())
    draws = np.asarray(draws)
    assert draws[:, 0].mean() == pytest.approx(expected, abs=.04)
    assert abs(draws[:, 1].mean()) < .04
    assert draws[:, 1].var() == pytest.approx(1., abs=.06)


def test_sampling_resume_keeps_exact_rng_and_frozen_adaptation():
    y = np.random.default_rng(38).normal(size=64)
    whole = chain(y, 92, .01, 'fou', burn=64, kept=40)
    first = chain(y, 92, .01, 'fou', burn=64, kept=20)
    second = chain(y, 0, .01, 'fou', burn=0, kept=20, resume=first['state'])
    for name in ('parameters', 'terminal', 'diagnostic_trace'):
        np.testing.assert_array_equal(whole[name], np.concatenate([first[name], second[name]]))


@pytest.fixture
def overlay_shell(monkeypatch):
    import importlib.util
    directory = Path(__file__).resolve().parents[1] / 'tools/rough_bayesian'
    spec = importlib.util.spec_from_file_location('models', directory / 'models.py')
    shell = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, 'models', shell)
    spec.loader.exec_module(shell)
    return shell


def test_tempered_overlay_covariance_and_original_eight_factor_target(overlay_shell):
    import overlay
    models = overlay_shell
    bins, evidence = overlay.accuracy_grid()
    assert evidence['maximum_absolute_autocorrelation_error'] < .001
    lags = np.arange(25201)
    for h, k in [(.031, 1 / 2500), (.1, 1 / 63), (.3, .49), (.489, .004)]:
        phi, covariance = overlay.lifted_covariance(h, k, bins)
        assert covariance.sum() == pytest.approx(1, abs=2e-15)
        assert np.linalg.eigvalsh(covariance).min() > -1e-12
        actual = phi[None, :] ** lags[:, None] @ covariance.sum(axis=1)
        assert np.max(np.abs(actual - overlay.exact_covariance(h, k, lags))) < .001
        point = np.array([h, math.log(k), math.log(.7)])
        for actual, expected in zip(overlay.configuration(point, False), models.controls.rough_parameters(point)):
            np.testing.assert_array_equal(actual, expected)


def test_overlay_multiplier_and_resume_against_independent_reference(overlay_shell):
    import overlay
    models = overlay_shell
    def reference_multiplier(phi, weights, root, state, variance, normals):
        state, variance, mean = state.copy(), variance.copy(), normals[0].copy()
        out = []
        for noise in normals[1:]:
            out.append(np.exp(.5 * (weights @ state - weights @ mean) - .25 * (weights @ variance @ weights)))
            state = phi * state + root @ noise
            mean = phi * mean
            variance = variance * np.outer(phi, phi) + root @ root.T
        return np.array(out)
    phi, weights, covariance = models.controls.rough_parameters(np.array([.1, math.log(1 / 63), math.log(.7)]))
    values, vectors = np.linalg.eigh(covariance)
    root = vectors * np.sqrt(np.maximum(values, 0))
    posterior = covariance / (1 - np.outer(phi, phi))
    rng = np.random.default_rng(98)
    normals = rng.normal(size=(33, len(phi)))
    initial = rng.normal(size=len(phi))
    np.testing.assert_allclose(overlay.multiplier_path(phi, weights, root, initial, posterior, normals),
                               reference_multiplier(phi, weights, root, initial, posterior, normals), atol=2e-14, rtol=0)
    y = rng.normal(size=64)
    whole, _ = overlay.parameter_chain(y, 0., False, 42, 32, 128)
    first, state = overlay.parameter_chain(y, 0., False, 42, 32, 64)
    last, _ = overlay.parameter_chain(y, 0., False, 0, 0, 64, state)
    np.testing.assert_array_equal(whole, np.concatenate([first, last]))


def test_collapsed_overlay_filter_against_original_reference(overlay_shell):
    import overlay
    rng = np.random.default_rng(839)
    y = rng.normal(size=507)
    for adaptive in (False, True):
        phi, weights, covariance = overlay.configuration(np.array([.1, math.log(1 / 63), math.log(.7)]), adaptive)
        fast = overlay.filter_rough(y, phi, weights, covariance, .1)
        reference = overlay_shell.controls.filter_rough(y, phi, weights, covariance, .1)
        for actual, expected in zip(fast, reference):
            np.testing.assert_allclose(actual, expected, atol=2e-11, rtol=0)
