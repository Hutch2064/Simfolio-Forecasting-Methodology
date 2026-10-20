import numpy as np
from types import SimpleNamespace
import candidate


def test_coupling_iid_boundary_exact_original_draw_matrix():
    pool = np.arange(40, dtype=float)
    expected = np.random.default_rng(123).integers(0, len(pool), size=(12, 150))
    actual, block = candidate.residual_indices(pool, 12, 150, np.random.default_rng(123), 5, block=1)
    assert block == 1
    np.testing.assert_array_equal(actual, expected)


def test_restart_and_circular_continuation_are_exact():
    iid = np.array([[4, 1, 3, 0, 2], [1, 4, 0, 1, 3]], dtype=np.int64)
    flags = np.array([[False, False, True, False], [True, False, True, True]])
    expected = np.array([[4, 0, 1, 0, 1], [1, 4, 0, 1, 3]])
    np.testing.assert_array_equal(candidate.coupled_indices(iid, flags, 5), expected)


def test_stationary_uniform_marginals_and_expected_continuation_frequency():
    n, paths, horizon, block = 11, 4000, 40, 5
    indices, _ = candidate.residual_indices(np.arange(n, dtype=float), paths, horizon,
        np.random.default_rng(909), 18, block=block)
    counts = np.bincount(indices.ravel(), minlength=n)/indices.size
    assert np.max(np.abs(counts-1/n)) < .004
    continuation = np.mean(indices[:, 1:] == (indices[:, :-1]+1) % n)
    assert abs(continuation-((1-1/block)+(1/block)/n)) < .005


def test_gaussian_fit_and_mean_draws_are_shared_original_callables():
    model = candidate.build_candidate()
    assert type(model.components[0]) is candidate.retained.GaussianComponent
    assert model.simulate_daily_log_returns.__func__ is candidate.retained.GaussianPool.simulate_daily_log_returns


def test_iid_boundary_replays_full_native_winner_losses(monkeypatch):
    import sweep as api
    row = api.load_panel(candidate.ROOT/'inputs')['series'][0]
    dates, values, origins = api.load_series(candidate.ROOT/'inputs', row)
    original = candidate.residual_indices
    def iid(pool, paths, horizon, rng, seed, block=None):
        return original(pool, paths, horizon, rng, seed, block=1)
    monkeypatch.setattr(candidate, 'residual_indices', iid)
    expected, _ = api.original._forecast(candidate.build_gaussian_equal_pool(), row['series_id'],
        values, dates, SimpleNamespace(**origins[0]), 591)
    actual, _ = api.original._forecast(candidate.build_candidate(), row['series_id'],
        values, dates, SimpleNamespace(**origins[0]), 591)
    np.testing.assert_array_equal(expected, actual)
