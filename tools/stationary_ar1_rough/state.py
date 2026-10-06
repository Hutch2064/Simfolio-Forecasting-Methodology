"""Stationary Gaussian AR(1) proxy likelihood with profiled level and variance.

This matched one-state ablation follows M242's zero-memory selection on every
canonical asset history. It is not joint raw-return inference.
"""
import numpy as np
from scipy.optimize import minimize_scalar


def profile(h, rho):
    n=len(h); origin=float(np.mean(h)); x=h-origin
    # Cancel the common (1-rho) factor analytically, preserving near-unit accuracy.
    m=((1.+rho)*x[0]+np.sum(x[1:])-rho*np.sum(x[:-1]))/((1.+rho)+(n-1)*(1.-rho))
    error=x[1:]-rho*x[:-1]-(1.-rho)*m
    variance=((1.-rho)*(1.+rho)*(x[0]-m)**2+error@error)/n
    if variance<=0 or not np.isfinite(variance):return np.inf,origin+m,variance
    likelihood=.5*n*(np.log(2.*np.pi*variance)+1.)-.5*np.log1p(-rho*rho)
    return float(likelihood),origin+float(m),float(variance)


def fit(h):
    h=np.asarray(h,float)
    if np.ptp(h)==0.:
        return {'success':True,'phis':[],'coefficients':[0.],'innovation_sd':0.,
                'level':float(h[0]),'initial':[0.],'component_count':0,
                'count_selection':'exact_constant_proxy_zero_variance_boundary',
                'selected_bic':None,'tested_orders':[],'bic_parameter_count':3}
    result=minimize_scalar(lambda rho:profile(h,rho)[0],method='bounded',
                           bounds=(0.,np.nextafter(1.,0.)),
                           options={'xatol':1e-12,'maxiter':200})
    if not result.success or not np.isfinite(result.fun):
        raise ArithmeticError(f'stationary AR1 fit unresolved: {result.message}')
    rho=float(result.x)
    if profile(h,0.)[0]<=result.fun:rho=0.
    nll,level,variance=profile(h,rho);bic=2.*nll+3.*np.log(len(h))
    return {'success':True,'phis':[],'coefficients':[rho],'innovation_sd':float(np.sqrt(variance)),
            'level':level,'initial':[float(h[-1]-level)],'component_count':0,
            'negative_loglikelihood':nll,'evaluations':int(result.nfev),
            'solver':'bounded_scalar_profile_likelihood_with_exact_zero_persistence_boundary',
            'optimizer_message':str(result.message),
            'estimator':'stationary_Gaussian_AR1_QMLE_on_clipped_Student_latent_proxy',
            'support':'0<=rho<1; unrestricted level; positive innovation variance',
            'count_selection':'matched_single_state_ablation_after_all_M242_asset_histories_selected_zero_memories',
            'selected_bic':float(bic),'tested_orders':[{'count':0,'bic':float(bic)}],
            'bic_parameter_count':3,'parameter_uncertainty':False,
            'historical_initial_distribution':'Gaussian(level,innovation_variance/(1-rho^2)); included in likelihood'}
