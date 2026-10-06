"""Independent references for the extra observation-variance component."""
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.special import gamma, kv

ROOT=Path(__file__).resolve().parents[1]
spec=importlib.util.spec_from_file_location('student_joint_rough_noise_test',ROOT/'tools/student_pathwise_joint_rough_noise/models.py')
model=importlib.util.module_from_spec(spec);sys.modules[spec.name]=model;spec.loader.exec_module(model)


@pytest.mark.parametrize('n',[5,6,17])
def test_spectral_target_matches_dense_differenced_covariance(n):
    h=.17;k=.09;amplitude=.6;noise=3.7;reference=5.2
    lag=np.abs(np.arange(n+1)[:,None]-np.arange(n+1))
    c=np.ones(lag.shape);positive=lag>0;x=k*lag[positive]
    c[positive]=2**(1-h)/gamma(h)*x**h*kv(h,x)
    c=amplitude**2*c+noise*np.eye(n+1)
    d=np.diff(np.eye(n+1),axis=0);cov=d@c@d.T
    fourier=np.exp(-2j*np.pi*np.arange(n//2+1)[:,None]*np.arange(n)/n)
    expected=np.real(np.einsum('fi,ij,fj->f',fourier,cov,fourier.conj()))/n
    observed=np.abs(np.fft.rfft(np.random.default_rng(123).normal(size=n)))**2/n
    theta=np.array([h,math.log(k),math.log(amplitude),math.log1p(noise/reference)])
    terms=np.log(expected)+observed/expected;terms[0]*=.5
    if n%2==0:terms[-1]*=.5
    assert model.whittle_target(theta,observed,n,reference)==pytest.approx(model.backend.log_prior(theta[:3])-terms.sum(),abs=2e-12)


def test_fixed_noise_control_matches_m221_target():
    theta=np.array([.1,math.log(1/63),math.log(.7)])
    observed=np.arange(51,dtype=float)+.3
    a=model.whittle_target(np.r_[theta,math.log(2)],observed,101,5.5)
    b=model.backend.whittle_target(theta,observed,101,5.5)
    assert a==pytest.approx(b,abs=2e-12)


def test_data_derived_noise_bound_has_positive_likelihood_derivative():
    n=101;observed=np.abs(np.fft.rfft(np.random.default_rng(1).normal(size=n)))**2/n
    signal=.7**2*model.backend.unit_expected_periodogram(.1,math.log(1/63),n)
    noise=model.backend.noise_expected_periodogram(n);upper=float(np.max(observed/noise))
    expected=signal+1.001*upper*noise
    assert np.all((expected-observed)*noise/expected**2>0)


@pytest.mark.parametrize('noise',[.01,3.,8.])
def test_scaled_native_terminal_filter_matches_dense_gaussian_conditioning(noise):
    n=9;phi=np.array([.5,.9]);w=np.array([.6,.8]);q=np.diag([.3,.1]);p=q/(1-phi[:,None]*phi[None,:])
    y=np.random.default_rng(54).normal(size=n);level=float(y.mean())
    covariance=np.empty((n,n));cross=np.empty((2,n))
    for i in range(n):
        cross[:,i]=(phi**(n-1-i))*(p@w)
        for j in range(n):covariance[i,j]=w@((phi**abs(i-j))[:,None]*p)@w+noise*(i==j)
    mean=cross@np.linalg.solve(covariance,y-level)
    variance=p-cross@np.linalg.solve(covariance,cross.T)
    scale=math.sqrt(4.934802200544679/noise)
    got_mean,got_variance=model.overlay.terminal_filter(y*scale,phi,w,q*scale**2,level*scale)
    np.testing.assert_allclose(got_mean/scale,mean,atol=2e-12)
    np.testing.assert_allclose(got_variance/scale**2,variance,atol=2e-12)
