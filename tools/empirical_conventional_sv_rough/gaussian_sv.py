"""One-state Gaussian SV quasi-likelihood with explicit observation noise."""
import math

import numpy as np
from numba import njit
from scipy.optimize import minimize
from scipy.special import expit, logit


@njit(cache=True,nogil=True)
def filter_states(y,level,phi,eta,noise):
    n=len(y);mean=level;variance=max(eta*eta/max(1-phi*phi,1e-4),1e-6)
    means=np.empty(n);variances=np.empty(n);pred_m=np.empty(n);pred_v=np.empty(n);likelihood=0.
    for i in range(n):
        pred_m[i]=mean;pred_v[i]=variance
        innovation=y[i]-mean;v=variance+noise
        likelihood-=.5*(math.log(2*math.pi*v)+innovation*innovation/v)
        gain=variance/v;mean+=gain*innovation;variance=max((1-gain)*variance,1e-8)
        means[i]=mean;variances[i]=variance
        mean=level+phi*(mean-level);variance=phi*phi*variance+eta*eta
    return likelihood,means,variances,pred_m,pred_v

@njit(cache=True,nogil=True)
def smooth_states(y,level,phi,eta,noise):
    likelihood,mean,var,pm,pv=filter_states(y,level,phi,eta,noise)
    for i in range(len(y)-2,-1,-1):
        gain=var[i]*phi/pv[i+1]
        mean[i]+=gain*(mean[i+1]-pm[i+1]);var[i]+=gain*gain*(var[i+1]-pv[i+1])
    return likelihood,mean,var


def fit(y,noise,initial):
    center=float(y.mean());lo,hi=np.quantile(y,[.01,.99]);bounds=[(lo-4,hi+4),(-7.,7.),(math.log(.02),math.log(2.5))]
    def target(theta):
        level,p,e=theta;phi=expit(p);eta=math.exp(e)
        if phi>=.999:return 1e100
        ll=filter_states(y,level,phi,eta,noise)[0]
        prior=-.5*((level-center)/4)**2-.5*((phi-.94)/.20)**2-.5*((e-math.log(.35))/1.)**2+math.log(phi*(1-phi))+e
        return -float(ll+prior)
    start=[initial[0],logit(initial[1]),math.log(initial[2])]
    result=minimize(target,start,method='L-BFGS-B',bounds=bounds,options={'ftol':1e-10,'gtol':1e-5,'maxiter':200,'maxls':100})
    if not result.success:result=minimize(target,result.x,method='Powell',bounds=bounds,options={'ftol':1e-10,'xtol':1e-6,'maxiter':200})
    if not result.success or not np.isfinite(result.fun):raise ArithmeticError('empirical-noise conventional SV MAP failed')
    theta=(float(result.x[0]),float(expit(result.x[1])),float(math.exp(result.x[2])))
    return theta,{'success':bool(result.success),'evaluations':int(result.nfev),'objective':float(result.fun),'noise_variance':noise}
