"""Independent quadrature verifies exact model-implied log-square moments."""
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import t

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'tools/student_implied_noise_rough'))
import student_log_moments as model


@pytest.mark.parametrize('nu',[2.1,3.,5.,20.,100.])
def test_student_log_square_moments_match_numerical_return_density(nu):
    scale=np.sqrt((nu-2)/nu)
    def integral(order):
        def density(z):
            value=np.log(z*z)
            return 2*value**order*t.pdf(z/scale,nu)/scale
        return quad(density,0,1,epsabs=2e-8)[0]+quad(density,1,np.inf,epsabs=2e-8)[0]
    mean=integral(1);variance=integral(2)-mean*mean
    actual=model.log_square_moments(1/nu)
    np.testing.assert_allclose(actual,[mean,variance],rtol=2e-7,atol=2e-7)


def test_gaussian_endpoint_log_chi_square_moments():
    mean,variance=model.log_square_moments(0.)
    assert mean==pytest.approx(-1.2703628454614782,abs=1e-14)
    assert variance==pytest.approx(np.pi*np.pi/2,abs=1e-14)
