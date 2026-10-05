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


@pytest.mark.parametrize('hurst,kappa', [(.031, 1/2500), (.1, 1/63), (.3, .49), (.489, .004)])
def test_dynamic_spectral_measure_matches_independent_beta_integral(hurst, kappa):
    from scipy.special import beta
    from overlay import exact_covariance
    a, b = .5 - hurst, 2 * hurst
    for lag in (1, 12, 126, 2520):
        def integrand(z):
            return math.exp(-kappa * (1 + z) / (1 - z) * lag) if z < 1 else 0.
        expected = quad(integrand, 0, 1, weight='alg', wvar=(a-1, b-1),
                        epsabs=1e-11, epsrel=1e-11)[0] / beta(a, b)
        assert exact_covariance(hurst, kappa, np.array([lag]))[0] == pytest.approx(expected, abs=2e-10)


@pytest.mark.parametrize('hurst,kappa', [(.031, 1/2500), (.1, 1/63), (.3, .49), (.489, .004)])
def test_dynamic_resolution_covers_every_requested_daily_lag(hurst, kappa):
    from dynamic import selected_kernel
    from overlay import exact_covariance
    lags = np.arange(25201)
    phi, mass, evidence = selected_kernel(hurst, kappa, int(lags[-1]))
    actual = phi[None, :] ** lags[:, None] @ mass
    assert (mass >= 0).all() and (phi >= 0).all() and (phi < 1).all()
    assert mass.sum() == pytest.approx(1., abs=2e-15)
    error = float(np.max(np.abs(actual - exact_covariance(hurst, kappa, lags))))
    assert error <= evidence['autocorrelation_error_upper_bound'] + 2e-14
    assert evidence['autocorrelation_error_upper_bound'] <= .001
    assert evidence['factors'] == len(phi)


def test_dynamic_filter_matches_dense_gaussian_likelihood(overlay_shell):
    from scipy.linalg import toeplitz, cho_factor, cho_solve
    from dynamic import configuration
    import overlay
    point = np.array([.11, math.log(1/63), math.log(.7)])
    phi, weights, q = configuration(point, 63)
    mass = np.diag(q) / (1 - phi * phi)
    covariance = toeplitz(phi[None, :] ** np.arange(64)[:, None] @ mass)
    covariance += np.eye(64) * math.pi**2/2
    y = np.random.default_rng(947).normal(size=64)
    factor = cho_factor(covariance, lower=True)
    expected = -.5 * (64*math.log(2*math.pi) + 2*np.log(np.diag(factor[0])).sum()
                       + y @ cho_solve(factor, y))
    assert overlay.filter_rough(y, phi, weights, q, 0.)[0] == pytest.approx(expected, abs=2e-11)


def test_dynamic_sampling_resume_preserves_trace_and_rng(overlay_shell):
    import overlay
    y = np.random.default_rng(38).normal(size=64)
    whole, whole_state = overlay.parameter_chain(y, 0., 'dynamic:126', 42, 32, 64)
    first, state = overlay.parameter_chain(y, 0., 'dynamic:126', 42, 32, 32)
    last, final_state = overlay.parameter_chain(y, 0., 'dynamic:126', 0, 0, 32, state)
    np.testing.assert_array_equal(whole, np.concatenate([first, last]))
    assert whole_state['rng_state'] == final_state['rng_state']


def test_reused_jacobi_geometry_preserves_complete_factor_arrays():
    from dynamic import JacobiGeometry, quadrature as spectral_quadrature
    for hurst in [.0301, .1, .3, .4899]:
        geometry = JacobiGeometry(hurst)
        for order in range(1, 34):
            reference = spectral_quadrature(hurst, 1/63, order, .001)
            actual = spectral_quadrature(hurst, 1/63, order, .001, geometry)
            assert all(a.tobytes() == b.tobytes() for a, b in zip(actual, reference))


@pytest.mark.parametrize('root_kind', ['permuted_diagonal', 'zero_row', 'dense'])
def test_independent_multiplier_preserves_complete_path(root_kind):
    import overlay
    rng = np.random.default_rng(948)
    root = np.diag([.1, .4, .2, .3])[:, [2, 0, 3, 1]]
    if root_kind == 'zero_row':
        root[2] = 0.
    elif root_kind == 'dense':
        root = rng.normal(size=(4, 4))
    args = (np.array([.1, .6, .8, .99]), np.ones(4), root,
            rng.normal(size=4), rng.normal(size=(25201, 4)),
            rng.normal(size=25200), rng.uniform(size=25200))
    expected = overlay.multiplier_prepared(*args)
    actual = overlay.multiplier_independent_prepared(*args)
    assert actual.tobytes() == expected.tobytes()


def test_asset_fit_process_lanes_preserve_trace_and_prediction_draws(overlay_shell):
    import overlay
    overlay_shell.initialize(None, 1, 1)
    rng = np.random.default_rng(195)
    data = tuple(rng.normal(0, .01, size=128).tobytes() for _ in range(2))
    args = (data, 'dynamic:191', 24, 32, 64, 64)
    serial = overlay.fitted_assets(*args, 1, None)
    parallel = overlay.fitted_assets(*args, 2, None)
    for expected, actual in zip(serial, parallel):
        assert actual['trace'].tobytes() == expected['trace'].tobytes()
        assert actual['parameters'].tobytes() == expected['parameters'].tobytes()
        assert actual['kernel'] == expected['kernel']


@pytest.mark.parametrize('adaptive', [False, True])
def test_prepared_overlay_paths_are_byte_exact(overlay_shell, adaptive):
    import overlay
    rng = np.random.default_rng(817)
    phi, weights, covariance = overlay.configuration(
        np.array([.11, math.log(1 / 63), math.log(.7)]), adaptive)
    values, vectors = np.linalg.eigh(covariance)
    root = vectors * np.sqrt(np.maximum(values, 0))
    posterior = covariance / (1 - phi[:, None] * phi[None, :])
    initial = rng.normal(size=phi.size)
    for horizon in (1, 126, 8766):
        normals = rng.normal(size=(horizon + 1, phi.size))
        means, variances = overlay.path_normalizers(
            phi, weights, root, posterior, normals[0], horizon)
        original = overlay.multiplier_path(phi, weights, root, initial, posterior, normals)
        prepared = overlay.multiplier_prepared(phi, weights, root, initial, normals, means, variances)
        assert prepared.tobytes() == original.tobytes()


@pytest.mark.parametrize('adaptive', [False, True])
def test_native_overlay_filter_is_byte_exact(overlay_shell, adaptive):
    import overlay
    rng = np.random.default_rng(721)
    y = rng.normal(size=11657)
    for h, k in ((.031, 1 / 2520), (.1, 1 / 63), (.48, .3)):
        configuration = overlay.configuration(np.array([h, math.log(k), math.log(.7)]), adaptive)
        original = overlay.filter_rough_numba(y, *configuration, .1)
        native = overlay.filter_rough(y, *configuration, .1)
        for expected, actual in zip(original, native):
            assert np.asarray(expected).tobytes() == np.asarray(actual).tobytes()


@pytest.mark.parametrize('adaptive', [False, True])
def test_native_terminal_filter_preserves_reference_bits(overlay_shell, adaptive):
    import overlay
    rng = np.random.default_rng(1261)
    for length in (1, 507, 11657):
        y = rng.normal(size=length)
        for h, k in ((.031, 1 / 2520), (.1, 1 / 63), (.48, .3)):
            configuration = overlay.configuration(np.array([h, math.log(k), math.log(.7)]), adaptive)
            original = overlay_shell.controls.filter_rough(y, *configuration, .1)[1:3]
            optimized = overlay.terminal_filter(y, *configuration, .1)
            for expected, actual in zip(original, optimized):
                assert expected.tobytes() == actual.tobytes()


def test_repeated_posterior_draws_reuse_preparation_without_removing_paths(overlay_shell, monkeypatch):
    import overlay
    monkeypatch.setattr(overlay_shell.controls, 'cache', lambda namespace, identity, build: build())
    original = overlay.terminal_filter
    calls = []
    def counted(*args):
        calls.append(1)
        return original(*args)
    monkeypatch.setattr(overlay, 'terminal_filter', counted)
    parameters = np.array([[.1, math.log(1 / 63), math.log(.7)],
                           [.12, math.log(1 / 70), math.log(.8)]])
    parameters = parameters[[0, 1, 0, 1, 0]]
    arguments = (np.random.default_rng(157).normal(size=64).tobytes(), False,
                 parameters.tobytes(), .1, 'duplicate-preparation-test')
    overlay.predictive_states.cache_clear()
    overlay.predictive_normalizers.cache_clear()
    prepared = overlay.predictive_states(*arguments)
    normalizers = overlay.predictive_normalizers(*arguments, 126)
    assert len(calls) == 2
    assert len(prepared) == len(normalizers) == len(parameters)
    assert prepared[0] is prepared[2] is prepared[4]
    assert normalizers[0] is normalizers[2] is normalizers[4]


def test_standalone_joint_level_target_against_dense_gaussian(overlay_shell):
    from scipy.linalg import toeplitz, cho_factor, cho_solve
    import standalone, overlay
    y = np.random.default_rng(846).normal(size=64)
    point = np.array([.11, math.log(1/63), math.log(.7), .3])
    phi, weights, q = overlay.configuration(point[:3], 'dynamic:126')
    stationary = np.diag(q) / (1 - phi * phi)
    covariance = toeplitz(phi[None, :] ** np.arange(len(y))[:, None] @ stationary)
    covariance += np.eye(len(y)) * math.pi**2/2
    residual = y - point[3]
    factor = cho_factor(covariance, lower=True)
    expected = (-.5 * (len(y)*math.log(2*math.pi) + 2*np.log(np.diag(factor[0])).sum()
                + residual @ cho_solve(factor, residual))
                + overlay.log_prior(point[:3]) - .5*((point[3]-y.mean())/4)**2)
    assert standalone.log_target(y, point, 126) == pytest.approx(expected, abs=2e-11)
    point[3] = y.max() + 10
    assert standalone.log_target(y, point, 126) == -np.inf


def test_standalone_joint_chain_resume_preserves_trace_and_rng(overlay_shell):
    import standalone
    y = np.random.default_rng(936).normal(size=64)
    whole, whole_state = standalone.parameter_chain(y, 126, 42, 64, 128)
    first, state = standalone.parameter_chain(y, 126, 42, 64, 64)
    last, final_state = standalone.parameter_chain(y, 126, 0, 0, 64, state)
    assert whole.tobytes() == np.concatenate([first, last]).tobytes()
    assert whole_state['rng_state'] == final_state['rng_state']
    assert np.ptp(whole[:, 3]) > 0


def test_standalone_volatility_has_its_own_scale_and_exact_daily_transitions():
    import standalone
    phi, weights, sd = np.array([.9, .3]), np.ones(2), np.array([.1, .4])
    initial = np.array([.7, -.3])
    normals = np.random.default_rng(138).normal(size=(126, 2))
    expected, state = [], initial.copy()
    for z in normals:
        state = phi*state + sd*z
        expected.append(math.exp(.5*(.3 + weights @ state))/100)
    actual = standalone.volatility_path(phi, weights, sd, initial, normals, .3)
    np.testing.assert_allclose(actual, expected, rtol=1e-14, atol=0)
    twice = standalone.volatility_path(phi, weights, sd, initial, normals, .3+2*math.log(2))
    np.testing.assert_allclose(twice, 2*actual, rtol=1e-14, atol=0)


def test_standalone_conditional_innovation_mapping_preserves_all_quantiles(overlay_shell):
    import standalone
    uniforms = np.random.default_rng(438).uniform(size=(240, 63))
    nodes = np.linspace(-3, 4, 240)
    expected = uniforms.copy()
    overlay_shell.controls.interpolate_nodes(expected, nodes)
    actual = np.vstack([standalone.map_shocks(row, nodes) for row in uniforms])
    assert expected.tobytes() == actual.tobytes()
    assert standalone.map_shocks(np.array([0., 1.]), nodes).tolist() == [-3., 4.]
