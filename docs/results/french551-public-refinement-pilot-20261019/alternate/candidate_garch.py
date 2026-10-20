"""One-parameter addition to EWMA: data-fitted variance mean reversion."""
import math
import numpy as np
from numba import njit
from scipy.signal import lfilter
from scipy.optimize import minimize
from simfolio_forecasting_methodology.french_panel.control_adapters import fit_ewma_qmle


def variances(residuals, initial, decay, persistence):
    beta=persistence*decay
    alpha=persistence*(1-decay)
    omega=(1-persistence)*initial
    updated,_=lfilter([1.0],[1.0,-beta],alpha*residuals*residuals+omega,
                     zi=[beta*initial])
    return np.concatenate(([initial],updated[:-1])),float(updated[-1])


def objective(params,residuals,initial):
    v,_=variances(residuals,initial,*params)
    v=np.maximum(v,np.finfo(np.float64).tiny)
    return float(np.mean(np.log(v)+residuals*residuals/v))


def fit_garch(history):
    ewma=fit_ewma_qmle(history)
    residuals=history-ewma.mean
    if ewma.decay is None:
        raise ValueError('pilot requires nonconstant series')
    options=[]
    for initial in [(ewma.decay,.99),(.5,.5)]:
        opt=minimize(objective,initial,args=(residuals,ewma.initial_variance),
                     method='L-BFGS-B',bounds=[(0.,1.),(0.,1.)],
                     options={'ftol':1e-12,'gtol':1e-7,'maxiter':500})
        if not opt.success or not np.isfinite(opt.fun):
            raise ValueError(f'variance-targeted GARCH QMLE failed: {opt.message}')
        options.append((float(opt.fun),float(opt.x[0]),float(opt.x[1])))
    for decay,persistence in [(ewma.decay,1.),(0.,0.)]:
        options.append((objective((decay,persistence),residuals,ewma.initial_variance),
                        decay,persistence))
    value,decay,persistence=min(options)
    v,forecast=variances(residuals,ewma.initial_variance,decay,persistence)
    z=residuals/np.sqrt(v);z-=z.mean();z/=np.sqrt(np.mean(z*z))
    return z,{'mean':ewma.mean,'decay':decay,'persistence':persistence,
        'alpha':persistence*(1-decay),'beta':persistence*decay,
        'omega':(1-persistence)*ewma.initial_variance,
        'target_variance':ewma.initial_variance,'forecast_variance':forecast,
        'objective':value,'nested_ewma_objective':ewma.objective/len(history),
        'residual_mean':float(z.mean()),'residual_second_moment':float(np.mean(z*z)),
        'fit_status':'complete_training_only'}


@njit(cache=True,fastmath=False)
def paths(z,indices,mean,alpha,beta,omega,variance):
    pcount,horizon=indices.shape;out=np.empty((pcount,horizon))
    for p in range(pcount):
        v,wealth,previous=variance,1.,0.
        for h in range(horizon):
            residual=math.sqrt(v)*z[indices[p,h]]
            r=mean+residual
            simple=min(max(math.expm1(r),-.999999),np.finfo(np.float64).max/10)
            wealth*=1.+simple
            current=math.log(max(wealth,1.e-300))
            out[p,h]=current-previous;previous=current
            v=omega+alpha*residual*residual+beta*v
    return out


def simulate(history,indices):
    z,fit=fit_garch(history)
    return paths(z,indices,fit['mean'],fit['alpha'],fit['beta'],fit['omega'],
                 fit['forecast_variance']),fit
