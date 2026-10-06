import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.optimize._numdiff import approx_derivative

spec = importlib.util.spec_from_file_location('eb_loading_test', Path(__file__).parents[1] / 'tools/eb_loading_shrink_rough/reml.py')
kernel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(kernel)


@pytest.mark.parametrize('phi', [0., .8, .9999])
@pytest.mark.parametrize('penalty', [.01, 10.])
def test_independent_gaussian_covariance(phi, penalty):
    rng = np.random.default_rng(25)
    y = rng.normal(size=30)
    z = rng.normal(size=(30, 4))
    covariance = phi ** np.abs(np.arange(30)[:, None] - np.arange(30)[None, :]) / (1 - phi ** 2) + z @ z.T / penalty
    # The AR determinant is constant in this conditional penalty fit.
    value = .5 * (30 * np.log(y @ np.linalg.solve(covariance, y) / 30)
                  + np.linalg.slogdet(covariance)[1] + np.log1p(-phi ** 2)) / 30
    stats = kernel.statistics(y, z, phi)
    actual, gradient, _ = kernel.evaluate(np.log(penalty), stats)
    assert actual == pytest.approx(value, abs=1e-9)
    numerical = approx_derivative(lambda p: kernel.evaluate(float(p[0]), stats)[0], [np.log(penalty)]).item()
    assert gradient == pytest.approx(numerical, abs=1e-8)
