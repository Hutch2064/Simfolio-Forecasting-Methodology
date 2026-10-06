"""Stationary Gaussian AR order selection on the unchanged Student latent proxy.

Exact stationary likelihood, analytically profiled level/variance, forward BIC.
Partial autocorrelations enforce stability without sign restrictions on lags.
This is proxy QMLE, not joint raw-return posterior inference.
"""
import importlib.util
from pathlib import Path

import numpy as np
from scipy.linalg import cho_solve, cholesky, solve_discrete_lyapunov
from scipy.optimize import minimize, minimize_scalar

spec=importlib.util.spec_from_file_location('adaptive_ar_scalar_reference',Path(__file__).parents[1]/'stationary_ar1_rough/state.py')
scalar=importlib.util.module_from_spec(spec);spec.loader.exec_module(scalar)


def coefficients(partials):
    phi=np.empty(0)
    for k in partials:
        phi=np.r_[phi-k*phi[::-1],k]
    return phi


def transition(phis, phi):
    p=len(phi);f=np.zeros((p,p));f[0]=phi
    if p>1:f[1:,:-1]=np.eye(p-1)
    return f


class Profile:
    def __init__(self,h,p):
        self.n=len(h);self.p=p;self.origin=float(np.mean(h));self.scale=float(np.std(h))
        x=(np.asarray(h)-self.origin)/self.scale
        self.first=x[:p][::-1];self.y=x[p:]
        self.x=np.column_stack([x[p-j-1:-j-1] for j in range(p)])
        self.xx=self.x.T@self.x;self.xy=self.x.T@self.y
        self.sx=self.x.sum(axis=0);self.sy=self.y.sum();self.yy=self.y@self.y

    def evaluate(self,phi):
        f=transition([],phi);q=np.zeros_like(f);q[0,0]=1.
        try:
            cov=solve_discrete_lyapunov(f,q,method='bilinear')
            l=cholesky(cov,lower=True,check_finite=False)
        except (np.linalg.LinAlgError,ValueError):return np.inf,0.,np.nan
        ones=np.ones(self.p)
        inv=cho_solve((l,True),np.column_stack((self.first,ones)),check_finite=False)
        a=self.yy-2.*(phi@self.xy)+phi@self.xx@phi+self.first@inv[:,0]
        c=1.-float(np.sum(phi))
        b=c*(self.sy-phi@self.sx)+ones@inv[:,0]
        d=(self.n-self.p)*c*c+ones@inv[:,1]
        m=b/d;s=a-b*m
        # Recompute residuals near cancellation instead of flooring variance.
        if s<=np.sqrt(np.finfo(float).eps)*max(abs(a),1.):
            errors=self.y-self.x@phi-c*m;initial=self.first-m
            s=errors@errors+initial@cho_solve((l,True),initial,check_finite=False)
        variance=s/self.n
        if not np.isfinite(variance) or variance<=0:return np.inf,0.,variance
        nll=.5*self.n*(np.log(2.*np.pi*variance)+1.)+np.log(np.diag(l)).sum()+self.n*np.log(self.scale)
        return float(nll),self.origin+self.scale*m,float(variance*self.scale*self.scale)


def fit_order(h,partial_start):
    p=len(partial_start);profile=Profile(h,p)
    start=np.arctanh(partial_start)
    def objective(z):
        partials=np.tanh(z)
        if not np.isfinite(z).all() or np.any(abs(partials)>=1.):return 1e100
        value=profile.evaluate(coefficients(partials))[0]
        return value if np.isfinite(value) else 1e100
    initial=objective(start)
    result=minimize(objective,start,method='L-BFGS-B',options={'maxiter':200,'ftol':1e-11,'gtol':1e-5})
    solver='L-BFGS-B'
    if not result.success or not np.isfinite(result.fun) or result.fun>initial+1e-7:
        result=minimize(objective,result.x if np.isfinite(result.x).all() else start,method='Powell',options={'maxiter':200,'ftol':1e-11,'xtol':1e-7})
        solver='Powell_after_unresolved_gradient_fit'
    if not result.success or result.fun>=1e100 or result.fun>initial+1e-7:
        raise ArithmeticError(f'stationary AR order {p} unresolved: {result.message}')
    partials=np.tanh(result.x);phi=coefficients(partials);nll,level,variance=profile.evaluate(phi)
    return record(h,phi,partials,nll,level,variance,int(result.nfev),solver)


def record(h,phi,partials,nll,level,variance,evaluations,solver):
    p=len(phi)
    return {'success':True,'phis':[0.]*(p-1),'coefficients':list(phi),'partial_autocorrelations':list(partials),
            'innovation_sd':float(np.sqrt(variance)),'level':float(level),'initial':(h[-p:][::-1]-level).tolist(),
            'component_count':p-1,'ar_order':p,'negative_loglikelihood':float(nll),'evaluations':evaluations,
            'solver':solver,'estimator':'stationary_Gaussian_AR_QMLE_on_clipped_Student_latent_proxy',
            'support':'stationary AR coefficients via partial autocorrelations strictly inside (-1,1)',
            'parameter_uncertainty':False,'historical_initial_distribution':'exact stationary Gaussian; included in likelihood',
            'bic_parameter_count':p+2}


def fit(h,force_scalar=False):
    h=np.asarray(h,float)
    base=scalar.fit(h)
    if force_scalar or np.ptp(h)==0.:return base
    n=len(h);level=float(np.mean(h));variance=float(np.mean((h-level)**2))
    nll=.5*n*(np.log(2.*np.pi*variance)+1.)
    best=record(h,np.array([0.]),np.array([0.]),nll,level,variance,0,'exact_white_proxy_boundary')
    best.update(ar_order=0,bic_parameter_count=2)
    score=2.*nll+2.*np.log(n);tested=[{'order':0,'bic':float(score)}]
    rho=base['coefficients'][0]
    # Preserve M243's positive AR1 arithmetic; also permit negative AR1 dynamics.
    result=minimize_scalar(lambda r:scalar.profile(h,r)[0],bounds=(np.nextafter(-1.,0.),0.),method='bounded',options={'xatol':1e-12,'maxiter':200})
    if not result.success:raise ArithmeticError('negative AR1 branch unresolved')
    if result.fun<base['negative_loglikelihood']:
        rho=float(result.x);nll,level,variance=scalar.profile(h,rho)
        base=record(h,np.array([rho]),np.array([rho]),nll,level,variance,int(result.nfev),'negative_scalar_stationary_profile')
    base.update(ar_order=1,partial_autocorrelations=[rho])
    next_score=2.*base['negative_loglikelihood']+3.*np.log(n);tested.append({'order':1,'bic':float(next_score)})
    if next_score<score:
        best=base;score=next_score;partials=np.array([rho])
        while len(partials)+3<n:
            proposed=fit_order(h,np.r_[partials,0.])
            next_score=2.*proposed['negative_loglikelihood']+proposed['bic_parameter_count']*np.log(n)
            tested.append({'order':proposed['ar_order'],'bic':float(next_score)})
            if next_score>=score:break
            best=proposed;score=next_score;partials=np.array(best['partial_autocorrelations'])
    best.update(count_selection='forward_stationary_BIC_from_white_proxy_first_nonimprovement',selected_bic=float(score),tested_orders=tested)
    return best


spec=importlib.util.spec_from_file_location('coupled_private_state',Path(__file__).parents[1]/'coupled_multiscale_rough/state.py')
import sys

core=importlib.util.module_from_spec(spec);sys.modules[spec.name]=core;spec.loader.exec_module(core)
moments=core.moments
