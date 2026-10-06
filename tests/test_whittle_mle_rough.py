import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture(scope='module')
def model():
    path = Path(__file__).resolve().parents[1]/'tools/whittle_mle_rough/models.py'
    spec = importlib.util.spec_from_file_location('mle_test_model', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_penalties_removed_but_support_retained(model):
    for theta in [np.array([.1,-4.,-.2]), np.array([.4,-2.,.2])]:
        assert model.overlay.log_prior(theta) == 0.
        assert model.original_prior(theta) < 0.
    assert model.overlay.log_prior(np.array([.8,-4.,-.2])) == -np.inf


def test_target_is_exactly_unpenalized_whittle_objective(model):
    theta = np.array([.1,-4.,-.2])
    n = 32
    noise = 4.
    periodogram = np.arange(1.,17.)/8
    phi,w,q = model.overlay.configuration(theta, 'dynamic:100')
    mass = np.diag(q)*w*w/(1-phi*phi)
    expected = model.parent.expected_periodogram(phi, mass, n)+noise
    terms = np.log(expected)+periodogram/expected
    terms[-1] *= .5
    assert model.parent.whittle_target(theta, periodogram, n, 100, noise) == pytest.approx(-terms.sum(), abs=1e-12)


def test_original_overlay_and_prediction_backbone_remain_unchanged(model):
    import overlay
    theta = np.array([.1,-4.,-.2])
    assert overlay.log_prior(theta) == model.original_prior(theta)
    assert overlay.log_prior(theta) != model.overlay.log_prior(theta)
    assert model.parent.base.base.prepared is model.parent.base.prepared
    assert model.Candidate is model.parent.Candidate
