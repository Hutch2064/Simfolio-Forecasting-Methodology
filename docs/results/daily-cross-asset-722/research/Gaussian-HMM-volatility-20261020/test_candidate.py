import itertools
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.optimize import approx_fprime, minimize
from scipy.stats import norm
import hmm
import candidate


def test_filter_smoother_and_counts_against_exhaustive_state_paths():
    y = np.array([.3, -2., .7, .1, 1.5])
    v = np.array([.4, 2.3])
    p = np.array([[.9, .1], [.2, .8]])
    total = 0.
    gamma = np.zeros((len(y), 2))
    counts = np.zeros((2, 2))
    for states in itertools.product(range(2), repeat=len(y)):
        weight = .5*np.prod(norm.pdf(y, scale=np.sqrt(v[list(states)])))
        for t in range(1, len(y)):
            weight *= p[states[t-1], states[t]]
        total += weight
        for t, state in enumerate(states):
            gamma[t, state] += weight
            if t:
                counts[states[t-1], state] += weight
    ll, actual_g, actual_c, filtered = hmm.smooth(y, v, p)
    assert ll == pytest.approx(np.log(total), abs=1e-13)
    np.testing.assert_allclose(actual_g, gamma/total, atol=1e-14)
    np.testing.assert_allclose(actual_c, counts/total, atol=1e-14)
    np.testing.assert_allclose(filtered, actual_g[-1], atol=1e-14)
    swapped = hmm.smooth(y, v[::-1].copy(), p[::-1, ::-1].copy())
    assert swapped[0] == pytest.approx(ll, abs=1e-13)
    np.testing.assert_allclose(swapped[1][:, ::-1], actual_g, atol=1e-14)


def test_MAP_M_step_monotonicity_and_analytic_gradient():
    y = np.random.default_rng(223).normal(size=110)
    v = np.array([.2, 2.])
    p = np.array([[.8, .2], [.1, .9]])
    for _ in range(25):
        ll, gamma, counts, _ = hmm.smooth(y, v, p)
        before = ll+hmm.prior(v, p)
        v, p = hmm.mstep(y, gamma, counts)
        after = hmm.smooth(y, v, p)[0]+hmm.prior(v, p)
        assert after >= before-1e-11
    theta = np.r_[np.log(v), np.log(p[:, 1]/p[:, 0])]
    _, gradient = hmm.objective_gradient(theta, y)
    numerical = approx_fprime(theta, lambda t: hmm.objective_gradient(t, y)[0], 1e-6)
    np.testing.assert_allclose(gradient, numerical, atol=2e-7, rtol=1e-3)


def test_fit_equivariance_support_and_independent_posterior_objective():
    rng = np.random.default_rng(38)
    x = np.r_[rng.normal(0, .005, 300), rng.normal(0, .03, 300)]
    fit = hmm.fit(x)
    moved = hmm.fit(-2*x+.04)
    np.testing.assert_allclose(moved['variance'], 4*fit['variance'], rtol=1e-5)
    np.testing.assert_allclose(moved['transition'], fit['transition'], atol=1e-6)
    np.testing.assert_allclose(moved['filtered'], fit['filtered'], atol=1e-6)
    assert np.all(fit['variance'] > 0) and np.all(fit['transition'] > 0)
    assert fit['variance'][0] <= fit['variance'][1]
    p = fit['transition']
    theta = np.r_[np.log(fit['variance']/fit['target_variance']), np.log(p[:, 1]/p[:, 0])]
    y = (x-x.mean())/np.std(x)
    independent = minimize(lambda z: hmm.objective_gradient(z, y)[0], theta,
                           method='Powell', options={'xtol': 1e-9, 'ftol': 1e-12})
    assert hmm.objective_gradient(theta, y)[0] <= independent.fun+1e-9


def test_constant_history_and_explicit_failed_fit_rejection(monkeypatch):
    fit = hmm.fit(np.zeros(20))
    assert fit['constant_history'] and np.all(fit['variance'] == 0)
    original = hmm.minimize
    def failed(*args, **kwargs):
        r = original(*args, **kwargs)
        r.success = False
        return r
    monkeypatch.setattr(hmm, 'minimize', failed)
    with pytest.raises(RuntimeError, match='No converged'):
        hmm.fit(np.random.default_rng(2).normal(size=70))


def golden_native(v, p, filtered, mean, delta, u, z):
    out = np.empty_like(z)
    clips = floors = 0
    for i in range(len(z)):
        state = int(u[i, 0] >= filtered[0])
        wealth = 1.
        previous = 0.
        for t in range(z.shape[1]):
            state = int(u[i, t+1] >= p[state, 0])
            raw = mean+delta[i]+np.sqrt(v[state])*z[i, t]
            if raw > np.log1p(np.finfo(float).max/10):
                simple = np.finfo(float).max/10
                clips += 1
            else:
                simple = np.expm1(raw)
                if simple < -.999999:
                    simple = -.999999
                    clips += 1
            wealth *= 1+simple
            floors += int(wealth < 1e-300)
            current = np.log(max(wealth, 1e-300))
            out[i, t] = current-previous
            previous = current
    return out, clips, floors


def test_native_accounting_and_equal_state_Normal_boundary():
    rng = np.random.default_rng(4)
    u = rng.random((3, 61))
    z = rng.normal(size=(3, 60))
    p = np.array([[.9, .1], [.2, .8]])
    filtered = np.array([.3, .7])
    v = np.array([.0001, .0001])
    delta = np.array([-.0003, 0, .0003])
    actual = hmm.native_paths(v, p, filtered, .0002, delta, u, z)
    golden = golden_native(v, p, filtered, .0002, delta, u, z)
    np.testing.assert_allclose(actual[0], golden[0], atol=1e-15)
    np.testing.assert_allclose(actual[0], .0002+delta[:, None]+.01*z, atol=1e-15)
    assert actual[1:] == golden[1:] == (0, 0)
    z[:] = -100
    extreme = hmm.native_paths(np.ones(2), p, filtered, 0., delta*0, u, z)
    golden = golden_native(np.ones(2), p, filtered, 0., delta*0, u, z)
    np.testing.assert_allclose(extreme[0], golden[0], atol=1e-13)
    assert extreme[1:] == golden[1:]
    assert extreme[1] == 180 and extreme[2] > 0


def test_native_retained_WR_and_mean_component_unchanged(monkeypatch):
    import sweep as api
    row = api.load_panel(candidate.ROOT/'inputs')['series'][0]
    dates, values, origins = api.load_series(candidate.ROOT/'inputs', row)
    # An identity component substitution tests the entire inherited pool wrapper,
    # seeds and accounting, without falsely claiming financial HMM/FHS parity.
    monkeypatch.setattr(candidate, 'RegimeComponent', lambda: candidate.parent.ShrunkComponent('FHS'))
    expected, _ = api.original._forecast(candidate.build_smooth_mean_pool(), row['series_id'],
        values, dates, SimpleNamespace(**origins[0]), 193)
    actual, _ = api.original._forecast(candidate.build_candidate(), row['series_id'],
        values, dates, SimpleNamespace(**origins[0]), 193)
    np.testing.assert_array_equal(actual, expected)
