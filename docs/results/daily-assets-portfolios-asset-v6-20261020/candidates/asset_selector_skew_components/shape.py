"""Training-only skew fit and actual positive/negative shock second moments."""
import numpy as np
from scipy.integrate import quad
from candidates.asset_SAS_skew.skew import fit as original_fit,transform,inverse


def partial_moments(t):
    if t==0:return .5,.5
    threshold=float(inverse(np.array(0.),t))
    positive,error=quad(lambda z:float(transform(z,t))**2*np.exp(-z*z/2)/np.sqrt(2*np.pi),threshold,np.inf,epsabs=1e-11,epsrel=1e-11)
    if error>1e-9 or not 0<positive<1:raise ArithmeticError('Skew truncated second moment unresolved')
    return positive,1.-positive


def fit(pool):
    result=original_fit(pool);positive,negative=partial_moments(result['BIC_t'])
    return {**result,'positive_second_moment':positive,'negative_second_moment':negative}
