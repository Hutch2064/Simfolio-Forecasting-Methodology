"""Mean/variance-standardized sinh-arcsinh skew law with delta fixed at one."""
from functools import lru_cache
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize_scalar


@lru_cache(maxsize=1)
def constants():
    value,error=quad(lambda z:np.hypot(1.,z)*np.exp(-.5*z*z)/np.sqrt(2*np.pi),-np.inf,np.inf,epsabs=1e-13,epsrel=1e-13)
    if error>1e-11:raise ArithmeticError('Skew moment integral unresolved')
    return value,2.-value*value


def transform(z,t):
    z=np.asarray(z,dtype=float);t=np.asarray(t,dtype=float)
    if np.any(np.abs(t)>=1)or not np.isfinite(t).all():raise ValueError('Finite open skew parameter required')
    if not np.any(t):return z
    m,v=constants()
    return (z+t*(np.hypot(1.,z)-m))/np.sqrt(1.+v*t*t)


def inverse(y,t):
    if not np.isfinite(t)or abs(t)>=1:raise ValueError('Finite open skew parameter required')
    if t==0:return np.asarray(y,dtype=float).copy()
    m,v=constants();u=np.asarray(y,dtype=float)*np.sqrt(1.+v*t*t)+t*m
    a=1.-t*t;r=np.sqrt(u*u+a)
    # Rationalize only the cancelling branch of the quadratic root.
    z=np.asarray((u-t*r)/a)
    active=u*t>0
    np.divide(u*u-t*t,u+t*r,out=z,where=active)
    return z


def negative_log_density(y,t):
    z=inverse(y,t);_,v=constants();jac=1.+t*z/np.hypot(1.,z)
    with np.errstate(divide='ignore',over='ignore',invalid='ignore'):
        return .5*(np.log(2*np.pi)+z*z)-.5*np.log1p(v*t*t)+np.log(jac)


def fit(pool):
    y=np.asarray(pool,dtype=float)
    if y.ndim!=1 or len(y)<24 or not np.isfinite(y).all():raise ValueError('Finite training innovation vector required')
    normal=float(np.mean(negative_log_density(y,0.)))
    bound=np.nextafter(1.,0.)
    def objective(t):
        score=negative_log_density(y,t)
        return float(score.mean())if np.isfinite(score).all()else np.inf
    answer=minimize_scalar(objective,bounds=(-bound,bound),method='bounded',options={'xatol':1e-9})
    if not answer.success or not np.isfinite(answer.fun):raise ArithmeticError('Training skew likelihood failed')
    chosen=float(answer.x)if answer.fun<normal else 0.
    fitted=min(float(answer.fun),normal)
    improvement=2*len(y)*(normal-fitted)
    return {'likelihood_t':chosen,'BIC_t':chosen if improvement>np.log(len(y))else 0.,
        'normal_mean_NLL':normal,'fitted_mean_NLL':fitted,'twice_log_likelihood_gain':improvement,
        'BIC_one_parameter_penalty':float(np.log(len(y))),'training_rows':len(y),
        'tail_delta':1.,'location_and_variance':'Analytically standardized, not changed by fitted skew'}
