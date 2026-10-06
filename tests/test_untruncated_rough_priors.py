import importlib.util
import math
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location('unbounded_rough_test', Path(__file__).parents[1] / 'tools/untruncated_rough_priors/unbounded_map.py')
optimizer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(optimizer)


def test_prior_coordinates_have_no_kappa_or_amplitude_box():
    mapped = optimizer.physical([.4, 4., 3.])
    assert mapped[1] > math.log(.5)
    assert mapped[2] > math.log(3.)
    np.testing.assert_allclose(mapped, [.202, math.log(1 / 63) + 8, math.log(.7) + 4.5])


def test_map_can_select_parameters_outside_old_support():
    target = np.array([.2, math.log(2.), math.log(5.)])
    fitted = optimizer.fit_map(lambda theta: -float(np.sum((theta - target) ** 2)))
    np.testing.assert_allclose(fitted['map'], target, atol=2e-6)
    assert fitted['optimizer']['success']


spec = importlib.util.spec_from_file_location('untruncated_rough_candidate_test', Path(__file__).parents[1] / 'tools/untruncated_rough_priors/models.py')
model = importlib.util.module_from_spec(spec)
import sys

sys.modules[spec.name] = model
spec.loader.exec_module(model)


def test_original_interior_target_is_byte_identical():
    for h in (.02, .1, .3, .48):
        theta = np.array([h, math.log(1 / 63), math.log(.7)])
        n = 101
        periodogram = np.abs(np.fft.rfft(np.random.default_rng(78).normal(size=n))) ** 2 / n
        a = model.whittle_target(theta, periodogram, n, 5.5)
        b = model.backend.whittle_target(theta, periodogram, n, 5.5)
        assert np.float64(a).tobytes() == np.float64(b).tobytes()


def test_direct_covariance_spectrum_beyond_old_support():
    for n in (17, 32):
        for kappa, scale in ((.02, .7), (2., 5.)):
            theta = np.array([.2, math.log(kappa), math.log(scale)])
            t = kappa * np.arange(1, n + 1)
            covariance = np.r_[1., 2 ** .8 / math.gamma(.2) * t ** .2 * model.backend.kv(.2, t)]
            g = np.r_[2 * (covariance[0] - covariance[1]),
                      2 * covariance[1:n] - covariance[:n - 1] - covariance[2:n + 1]]
            omega = 2 * np.pi * np.arange(n // 2 + 1) / n
            expected = np.array([g[0] + 2 * sum((1 - j / n) * g[j] * math.cos(w * j) for j in range(1, n)) for w in omega])
            np.testing.assert_allclose(model.backend.unit_expected_periodogram(.2, math.log(kappa), n), expected, atol=3e-15)
            assert np.isfinite(model.whittle_target(theta, np.ones(len(omega)), n, 5.4))
            if kappa > .5:
                assert not np.isfinite(model.backend.log_prior(theta))


def test_hurst_support_is_retained():
    for h in (0., .009, .49, .5):
        assert model.log_prior(np.array([h, 0., 0.])) == -np.inf
