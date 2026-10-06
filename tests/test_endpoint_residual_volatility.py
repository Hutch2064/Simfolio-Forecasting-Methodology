"""Dense finite-sample spectral and analytic-gradient endpoint references."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
from scipy.linalg import toeplitz
from scipy.optimize._numdiff import approx_derivative

p=Path(__file__).resolve().parents[1]/'tools/endpoint_residual_volatility/kernel.py'
spec=importlib.util.spec_from_file_location('endpoint_kernel_test',p)
kernel=importlib.util.module_from_spec(spec);spec.loader.exec_module(kernel)


@pytest.mark.parametrize('n',[17,18])
def test_ou_expected_periodogram_matches_dense_difference_covariance(n):
    covariance=np.exp(-.07*np.arange(n+1));matrix=toeplitz(covariance)
    difference=np.eye(n,n+1,k=1)-np.eye(n,n+1);observed=difference@matrix@difference.T
    expected=[]
    for j in range(n//2+1):
        fourier=np.exp(-2j*np.pi*j*np.arange(n)/n)
        expected.append(float((fourier.conj()@observed@fourier).real/n))
    actual,derivative=kernel.expected(np.log(.07),n)
    np.testing.assert_allclose(actual,expected,rtol=1e-10,atol=1e-12)
    numerical=approx_derivative(lambda x:kernel.expected(x[0],n)[0],[np.log(.07)])[:,0]
    np.testing.assert_allclose(derivative,numerical,rtol=1e-7,atol=1e-9)


@pytest.mark.parametrize('kind',['white','ou'])
@pytest.mark.parametrize('n',[17,18])
def test_map_gradient_matches_finite_difference(kind,n):
    rng=np.random.default_rng(923);periodogram=abs(np.fft.rfft(rng.normal(size=n)))**2/n
    u=np.array([.2]) if kind=='white' else np.array([.3,-.2])
    actual=kernel.objective_n(u,periodogram,n,2.,kind)[1]
    reference=approx_derivative(lambda z:np.array([kernel.objective_n(z,periodogram,n,2.,kind)[0]]),u).ravel()
    np.testing.assert_allclose(actual,reference,rtol=1e-6,atol=1e-7)


def test_scalar_ou_stationary_covariance_and_white_boundary():
    for marker,decay in [(0.,0.),(.5,.07)]:
        theta=np.array([marker,np.log(.07),np.log(.8)])
        phi,w,q,variance=kernel.configuration(theta)
        assert phi[0]==pytest.approx(np.exp(-decay) if marker else 0.)
        assert variance==pytest.approx(.8**2)
        assert q[0,0]/(1-phi[0]**2)==pytest.approx(variance)
        assert w.tolist()==[1.]
    phi,w,q,v=kernel.configuration(np.array([-1.,0.,0.]))
    assert v==0. and q[0,0]==0.
