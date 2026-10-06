import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[1] / 'tools/multiscale_offset_rough'))
from ms_causal import predict


@pytest.mark.parametrize('states', [1, 3, 5])
def test_predictions_match_dense_gaussian_conditioning(states):
    rng = np.random.default_rng(49)
    phi = np.linspace(.7, .99, states)
    root = rng.normal(size=(states, 2)) * .03
    covariance = root @ root.T
    stationary = covariance / (1 - phi[:, None] * phi[None, :])
    loading = rng.normal(size=states)
    y = rng.normal(size=15)
    level, noise = .4, 5.3
    observation_covariance = np.empty((len(y), len(y)))
    for i in range(len(y)):
        for j in range(len(y)):
            observation_covariance[i, j] = loading @ ((phi ** abs(i - j))[:, None] * stationary) @ loading + (noise if i == j else 0.)
    expected = np.full(len(y), level)
    for t in range(1, len(y)):
        expected[t] += observation_covariance[t, :t] @ np.linalg.solve(observation_covariance[:t, :t], y[:t] - level)
    np.testing.assert_allclose(predict(y, level, phi, loading, covariance, noise), expected, atol=3e-13)


def test_future_observations_cannot_change_past_predictions():
    y = np.arange(20, dtype=float)
    params = (0., np.array([.9]), np.ones(1), np.array([[.1]]), 5.)
    original = predict(y, *params)
    y[10:] += 100
    np.testing.assert_array_equal(predict(y, *params)[:11], original[:11])
