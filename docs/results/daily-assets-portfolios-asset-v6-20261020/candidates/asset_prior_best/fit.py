"""Exact prior best GARCH fitting functions, now applied per constituent."""
from time import perf_counter
import numpy as np
from numba import njit
from scipy.optimize import minimize
STATIONARITY_MARGIN=1e-8

def parameters(theta):
    # Stable softmax coordinates; the margin is a numerical stationarity guard.
    z=np.array([0.,theta[0],theta[1]],dtype=float);z=np.exp(z-z.max());z/=z.sum()
    a,b=(1.-STATIONARITY_MARGIN)*z[1:]
    return float(a),float(b),z

@njit(cache=True,fastmath=False)
def filter_loss(y,a,b):
    n=len(y);v=np.empty(n);v[0]=1.;da=db=0.;ga=gb=0.;loss=0.
    for t in range(n):
        if t:
            old=v[t-1];v[t]=1.-a-b+a*y[t-1]**2+b*old
            da=-1.+y[t-1]**2+b*da
            db=-1.+old+b*db
        value=v[t]
        if value<=0. or not np.isfinite(value):
            raise ValueError('invalid conditional training variance')
        loss+=np.log(value)+y[t]**2/value
        slope=1./value-y[t]**2/value**2
        ga+=slope*da;gb+=slope*db
    next_v=1.-a-b+a*y[-1]**2+b*v[-1]
    return .5*loss/n,np.array([.5*ga/n,.5*gb/n]),v,next_v

def objective(theta,y):
    a,b,z=parameters(theta);loss,gradient,_,_=filter_loss(y,a,b)
    scale=1.-STATIONARITY_MARGIN
    jac=np.array([[scale*z[1]*(1.-z[1]),-scale*z[1]*z[2]],
                  [-scale*z[1]*z[2],scale*z[2]*(1.-z[2])]])
    return loss,jac.T@gradient

def fit_variance_target(history):
    started=perf_counter();mu=float(np.mean(history));centered=history-mu
    target=float(np.mean(centered*centered))
    if not np.isfinite(target) or target<=0:raise ValueError('positive finite training variance required')
    y=centered/np.sqrt(target);attempts=[];eligible=[]
    for a,b in ((.05,.90),(.10,.50)):
        theta=np.log(np.array([a,b])/(1.-a-b))
        fit=minimize(objective,theta,args=(y,),jac=True,method='L-BFGS-B',
                     options={'ftol':1e-12,'gtol':1e-7,'maxiter':500})
        value,gradient=objective(fit.x,y)
        valid=bool(fit.success and np.isfinite(value) and np.isfinite(gradient).all()
                   and np.max(np.abs(gradient))<=1e-4)
        attempts.append({'success':bool(fit.success),'eligible':valid,'objective':float(value),
                         'gradient_max':float(np.max(np.abs(gradient))),'iterations':int(fit.nit)})
        if valid:eligible.append((float(value),fit.x))
    if not eligible:raise ArithmeticError('both fixed GARCH training fits failed convergence')
    value,theta=min(eligible,key=lambda x:x[0]);a,b,_=parameters(theta)
    null=.5*float(np.mean(y*y))
    # The constant-variance boundary is an explicit member of this fitted law.
    if value>=null:a=b=0.;value=null
    value,_,variances,next_v=filter_loss(y,a,b)
    pool=y/np.sqrt(variances);pool=pool-pool.mean();pool=pool/np.sqrt(np.mean(pool*pool))
    return {'mean':mu,'target_variance':target,'alpha':a,'beta':b,'omega':1.-a-b,
            'next_variance_normalized':float(next_v),'pool':pool,'objective':float(value),
            'constant_variance_objective':null,'fixed_start_results':attempts,
            'fit_seconds':perf_counter()-started,'training_n':len(history)}
