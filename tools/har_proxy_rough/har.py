"""Stationary log-HAR on a filtered daily volatility proxy, not intraday RV.

Corsi's daily/weekly/monthly restrictions give an AR(22). Nonnegative aggregate
coefficients with sum below one guarantee stationarity. Conditional Gaussian
MLE estimates coefficients and innovation variance; no ridge or shrink constant.
"""
import numpy as np
from scipy.optimize import minimize
from scipy.signal import lfilter


def fit(history):
    h=np.asarray(history,float)
    windows=(1,5,22)
    t=np.arange(22,len(h))
    prefix=np.r_[0.,np.cumsum(h)]
    x=np.column_stack([(prefix[t]-prefix[t-w])/w for w in windows]);y=h[t]
    xc=x-x.mean(axis=0);yc=y-y.mean()
    gram=xc.T@xc/len(y);cross=xc.T@yc/len(y)
    start=np.maximum(np.linalg.lstsq(xc,yc,rcond=None)[0],0.)
    if start.sum()>=1:start/=(start.sum()+1)
    upper=1-np.sqrt(np.finfo(float).eps)
    result=minimize(lambda b:float(b@gram@b-2*cross@b),start,jac=lambda b:2*(gram@b-cross),method='SLSQP',bounds=[(0.,upper)]*3,constraints=[{'type':'ineq','fun':lambda b:upper-b.sum(),'jac':lambda b:-np.ones(3)}],options={'ftol':1e-12,'maxiter':200})
    if not result.success:raise ArithmeticError('stationary log-HAR Gaussian MLE failed')
    beta=result.x;intercept=float(y.mean()-x.mean(axis=0)@beta)
    residual=y-intercept-x@beta
    variance=float(np.mean(residual*residual))
    ar=np.zeros(22)
    for w,b in zip(windows,beta):ar[:w]+=b/w
    return {'beta':beta,'ar':ar,'intercept':intercept,'innovation_variance':variance,'last':h[-22:][::-1].copy(),'optimizer_success':bool(result.success),'estimator':'stationary_nonnegative_conditional_Gaussian_MLE','input':'original_conventional_MAP_RTS_log_volatility_proxy','horizons':windows}


def moments(fit,horizon):
    ar=fit['ar'];den=np.r_[1.,-ar];initial=np.asarray(fit['last'],float)
    # Exact conditional AR forecast with deterministic observed initial history.
    zi=np.array([np.dot(ar[k:],initial[:len(ar)-k]) for k in range(len(ar))])
    mean=lfilter([1.],den,np.full(horizon,fit['intercept']),zi=zi)[0]
    impulse=np.zeros(horizon);impulse[0]=1.
    psi=lfilter([1.],den,impulse)
    variance=fit['innovation_variance']*np.cumsum(psi*psi)
    if not np.isfinite(mean).all() or not np.isfinite(variance).all():raise ArithmeticError('nonfinite log-HAR forecast moments')
    return mean,variance
