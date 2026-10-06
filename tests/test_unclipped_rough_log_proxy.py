import importlib.util
import sys
from pathlib import Path

import numpy as np

spec = importlib.util.spec_from_file_location('unclipped_proxy_test', Path(__file__).parents[1] / 'tools/unclipped_log_proxy_rough/models.py')
model = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = model
spec.loader.exec_module(model)


def test_only_tail_clipping_removed_and_dates_retained(monkeypatch):
    x = np.random.default_rng(14).normal(scale=.01, size=200)
    x[-1] = 2.
    eps = (x - x.mean()) * 100
    squared = eps * eps
    floor = max(np.quantile(squared[squared > 0], .001) * .1, 1e-10)
    raw = np.log(np.maximum(squared, floor)) - model.parent.shell.bd.SV_LOG_CHI_SQUARE_MEAN
    u = .2
    bias, noise = model.parent.parent.parent.log_square_moments(u)
    gaussian_bias, _ = model.parent.parent.parent.log_square_moments(0.)
    expected = raw + gaussian_bias - bias
    monkeypatch.setattr(model.parent.parent, 'predecessor_fit', lambda _: {'student_return_laplace_fit': {'inverse_df': u}, 'posterior_center': (0., .8, .2)})
    seen = {}

    def predictor(y, level, phi, eta, observation_variance):
        seen.update(noise=observation_variance, parameters=(level, phi, eta))
        return np.zeros_like(y)

    monkeypatch.setattr(model, 'causal_predictor', predictor)
    observed, innovations = model.observed(x.tobytes())
    np.testing.assert_array_equal(observed, expected)
    np.testing.assert_array_equal(innovations, eps)
    clipped, _ = model.parent.shell.bd._sv_observed_log_variance(x, x.mean())
    assert len(observed) == len(clipped) == len(x)
    assert raw[-1] > clipped[-1]
    assert seen['noise'] == noise
    assert seen['parameters'] == (0., .8, .2)
    model.observed.cache_clear()
