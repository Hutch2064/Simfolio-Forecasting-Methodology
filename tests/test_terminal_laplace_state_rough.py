"""Independent Gaussian linear-system covariance and disabled-state RNG checks."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest

p=Path(__file__).resolve().parents[1]/'tools/terminal_laplace_state_rough/terminal.py'
spec=importlib.util.spec_from_file_location('terminal_state_test_helpers',p)
helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)


@pytest.mark.parametrize('rho',[0.,.8,.999])
def test_future_variance_matches_full_innovation_linear_map(rho):
    n=30;sd=.2;initial_var=.3;basis=np.zeros((n,n+1))
    for t in range(n):
        basis[t,0]=rho**(t+1)*np.sqrt(initial_var)
        for j in range(t+1):basis[t,j+1]=rho**(t-j)*sd
    q=np.array([sd*sd*sum(rho**(2*j) for j in range(t+1)) for t in range(n)])
    actual=helper.variance_curve(q,rho,initial_var)
    np.testing.assert_allclose(actual,np.diag(basis@basis.T),rtol=1e-12,atol=1e-12)


def test_zero_uncertainty_preserves_bytes_and_draws_no_random_numbers():
    rng=np.random.default_rng(17);expected=np.random.default_rng(17)
    initial=np.array([.2]);variance=np.array([.1,.2])
    assert helper.initial_state(initial,0.,rng) is initial
    assert helper.variance_curve(variance,.9,0.) is variance
    assert rng.bit_generator.state==expected.bit_generator.state


def test_initial_state_uses_independent_rng_and_matches_direct_gaussian_transform():
    rng=np.random.default_rng(18);expected=np.random.default_rng(18)
    initial=np.array([.2]);variance=.3
    actual=helper.initial_state(initial,variance,rng)
    np.testing.assert_array_equal(actual,initial+np.sqrt(variance)*expected.standard_normal(size=1))
    assert rng.bit_generator.state==expected.bit_generator.state
