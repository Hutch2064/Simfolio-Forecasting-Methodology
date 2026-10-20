from types import SimpleNamespace
import numpy as np
import pytest
from scipy.integrate import quad
import candidate


def moments(z):
    a = z*z/2
    integral = lambda k: quad(lambda u: u**k*np.exp(a*(u*u-1)), 0, 1, epsabs=1e-13, epsrel=1e-12)[0]
    den = integral(0)
    return integral(2)/den, integral(4)/den


def test_cdf_and_inverse_against_independent_quadrature():
    p = np.array([0, 1e-6, .01, .1, .3, .5, .9, .99, 1-1e-6, 1])
    for z in (0., .01, 1., 3., 8., 30.):
        u = candidate.local_scale_quantile(p, z)
        assert np.all(np.diff(u) >= 0)
        assert np.max(np.abs(candidate.local_scale_cdf(u, z)-p)) < 2e-12
        a = z*z/2
        den = quad(lambda t: np.exp(a*(t*t-1)), 0, 1, epsabs=1e-13)[0]
        independent = [quad(lambda t: np.exp(a*(t*t-1)), 0, x, epsabs=1e-13)[0]/den for x in u]
        np.testing.assert_allclose(independent, p, atol=2e-12, rtol=0)


def test_weak_signal_retains_uncertainty_and_strong_signal_escapes_shrinkage():
    first, second = moments(0.)
    assert first == pytest.approx(1/3)
    assert second == pytest.approx(1/5)
    assert moments(10.)[0] > .97
    for z in (.01, 1., 3., 8.):
        expected_weight = quad(lambda p: float(candidate.local_scale_quantile(p, z))**2,
            0, 1, epsabs=1e-9, epsrel=1e-9, limit=300)[0]
        assert expected_weight == pytest.approx(moments(z)[0], abs=1e-8)


def test_pairs_share_scale_and_use_the_unchanged_gaussian_permutation():
    seed, n = 121, 120
    weights = candidate.paired_local_weights(.02, .01, n, seed)
    rng = np.random.default_rng(seed)
    rng.random(n//2)
    original = weights[np.argsort(rng.permutation(n))]
    np.testing.assert_array_equal(original[:n//2], original[n//2:])


def test_sign_scale_and_zero_SE_boundaries():
    a = candidate.paired_local_weights(.02, .01, 120, 121)
    np.testing.assert_array_equal(a, candidate.paired_local_weights(-.02, .01, 120, 121))
    np.testing.assert_array_equal(a, candidate.paired_local_weights(2., 1., 120, 121))
    np.testing.assert_array_equal(candidate.mean_offsets(.02, 20., 0., 120, 121), np.zeros(120))
    for mu, se, paths in [(1., -1., 120), (np.nan, 1., 120), (1., 1., 119)]:
        with pytest.raises(ValueError):
            candidate.paired_local_weights(mu, se, paths, 121)


def test_conditional_mixture_sampling_matches_integrated_moments():
    df, scale, se = 20., .01, candidate.retained.gaussian_sd(20., .01)
    for z in (0., 1., 3.):
        mu = z*se
        draws = np.concatenate([candidate.mean_offsets(mu, df, scale, 120, seed)+mu for seed in range(150)])
        ew, ew2 = moments(z)
        assert draws.mean() == pytest.approx(mu*ew, abs=se*.002)
        assert draws.var() == pytest.approx(se*se*ew+mu*mu*(ew2-ew*ew), rel=.035)


def test_current_mean_law_boundary_replays_complete_current_winner(monkeypatch):
    import sweep as api
    row = api.load_panel(candidate.ROOT/'inputs')['series'][0]
    dates, values, origins = api.load_series(candidate.ROOT/'inputs', row)
    def unchanged(mu, df, scale, paths, seed):
        se = candidate.retained.gaussian_sd(df, scale)
        return candidate.retained.parameter_draws(df, scale, paths, seed)+(candidate.parent.regularized_mean(mu, se)-mu)
    monkeypatch.setattr(candidate, 'mean_offsets', unchanged)
    expected, _ = api.original._forecast(candidate.build_smooth_mean_pool(), row['series_id'],
        values, dates, SimpleNamespace(**origins[0]), 1391)
    actual, _ = api.original._forecast(candidate.build_candidate(), row['series_id'],
        values, dates, SimpleNamespace(**origins[0]), 1391)
    np.testing.assert_array_equal(actual, expected)
