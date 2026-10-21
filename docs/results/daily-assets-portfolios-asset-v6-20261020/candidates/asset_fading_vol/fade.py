"""A positive variance-ratio overlay that returns to historical variance."""
import numpy as np
from numba import njit
from scipy.optimize import minimize_scalar

@njit(cache=True,fastmath=False)
def ewma_loss(y,decay):
    variance=1.;total=0.
    for t in range(len(y)):
        if variance<=0 or not np.isfinite(variance):return np.inf,variance
        total+=np.log(variance)+y[t]*y[t]/variance
        variance=decay*variance+(1.-decay)*y[t]*y[t]
    return .5*total/len(y),variance

def fit_ewma(history):
    x=np.asarray(history,dtype=float);mu=x.mean();target=np.mean((x-mu)**2)
    if target<=0 or not np.isfinite(target):raise ValueError('Positive finite historical variance required')
    y=(x-mu)/np.sqrt(target)
    result=minimize_scalar(lambda d:ewma_loss(y,d)[0],method='bounded',bounds=(1e-8,1.-1e-8),options={'xatol':1e-10,'maxiter':200})
    value,initial=ewma_loss(y,result.x)
    if not result.success or not np.isfinite(value) or initial<=0 or not np.isfinite(initial):
        raise ArithmeticError('Training EWMA fit did not converge to a valid state')
    null=.5*float(np.mean(y*y))
    if value>=null:return {'decay':1.,'initial_variance_ratio':1.,'null_selected':True,'objective':null}
    return {'decay':float(result.x),'initial_variance_ratio':float(initial),'null_selected':False,'objective':float(value)}

def scale_table(initial,decay,start,count):
    weights=np.asarray(decay)[:,None]**np.arange(start,start+count)[None,:]
    # Stable convex combination: at the first lead this equals initial exactly;
    # later it approaches one, including for an initial ratio near zero.
    variance=(1.-weights)+weights*np.asarray(initial)[:,None]
    return np.sqrt(variance)
