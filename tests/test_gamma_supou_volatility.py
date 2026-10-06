import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.special import gamma


@pytest.fixture(scope='module')
def model():
    path = Path(__file__).resolve().parents[1]/'tools/gamma_supou_volatility/models.py'
    spec = importlib.util.spec_from_file_location('gamma_supou_test_model', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('shape,kappa', [(.1, 1/2520), (.7, 1/63), (3., .3)])
def test_lift_certificate_covers_every_requested_daily_lag(model, shape, kappa):
    from gamma_kernel import selected_kernel
    lag = 12000
    phi, mass, certificate = selected_kernel(np.log(shape), kappa, lag)
    actual = (phi[None, :]**np.arange(1, lag+1)[:, None])@mass
    exact = np.exp(-shape*np.log1p(kappa*np.arange(1, lag+1)))
    assert np.max(np.abs(actual-exact)) <= certificate['autocorrelation_error_upper_bound']
    assert certificate['autocorrelation_error_upper_bound'] <= .001
    assert np.all(mass >= 0)
    assert mass.sum() == pytest.approx(1., abs=3e-15)
    assert np.all((phi >= 0) & (phi < 1))


def test_gamma_rate_covariance_matches_independent_integral():
    shape, kappa, lag = .7, .02, 18
    actual = quad(lambda u: np.exp(-u*(1+kappa*lag))*u**(shape-1)/gamma(shape),
                  0, np.inf, epsabs=1e-10)[0]
    assert actual == pytest.approx((1+kappa*lag)**(-shape), abs=1e-9)


def test_new_covariance_does_not_mutate_rough_model_and_preserves_variance(model):
    import overlay
    assert overlay.configuration is not model.overlay.configuration
    theta = np.array([np.log(.7), np.log(1/63), np.log(.8)])
    phi, w, q = model.overlay.configuration(theta, 'dynamic:1000')
    assert w@(q/(1-phi[:, None]*phi[None, :]))@w == pytest.approx(.8**2, abs=2e-15)
    original = np.array([.1, np.log(1/63), np.log(.7)])
    assert np.isfinite(overlay.log_prior(original))
