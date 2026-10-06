"""Finite-sample differenced spectra for a diagonal Gaussian state transition."""
import numpy as np


def stationary(phi,covariance):
    return covariance/(1-phi[:,None]*phi[None,:])


def expected_periodogram(phi,loading,covariance,n):
    p=stationary(phi,covariance)
    coefficients=loading*(p@loading)
    lags=np.arange(n+1)
    c=coefficients@(phi[:,None]**lags)
    differences=np.empty(n)
    differences[0]=2*(c[0]-c[1])
    differences[1:]=2*c[1:n]-c[:n-1]-c[2:n+1]
    return 2*np.fft.rfft(differences*(1-np.arange(n)/n)).real-differences[0]
