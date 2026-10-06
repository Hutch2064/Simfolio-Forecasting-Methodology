"""Student-return / Gaussian-state copula leverage, sparse Laplace marginal MAP.

The Gaussian rank of the current Student return is correlated with the next
AR1 state innovation. Marginal Student tails and stationary Gaussian AR1
states are retained. This is a joint copula likelihood, not an Omori mixture
sampler or full posterior integration. No rough-state likelihood is changed.
"""
import math
from functools import lru_cache

import numpy as np
import student_laplace_sv as original
from numba import njit
from scipy.integrate import quad
from scipy.special import expit, logit, ndtri, ndtri_exp, stdtr


@njit(cache=True, nogil=True)
def factor(diagonal, off):
    pivots=diagonal.copy()
    for i in range(1,len(pivots)):
        if pivots[i-1]<=0:return pivots,False
        pivots[i]-=off[i-1]*off[i-1]/pivots[i-1]
    return pivots,bool(np.all(pivots>0))


@njit(cache=True, nogil=True)
def solve(pivots, off, rhs):
    value=rhs.copy()
    for i in range(1,len(value)):value[i]-=off[i-1]*value[i-1]/pivots[i-1]
    value[-1]/=pivots[-1]
    for i in range(len(value)-2,-1,-1):value[i]=(value[i]-off[i]*value[i+1])/pivots[i]
    return value


@njit(cache=True, nogil=True)
def inverse_diagonal(pivots,off):
    variance=np.empty(len(pivots));adjacent=np.empty(len(pivots)-1);variance[-1]=1/pivots[-1]
    for i in range(len(pivots)-2,-1,-1):
        adjacent[i]=-off[i]/pivots[i]*variance[i+1]
        variance[i]=1/pivots[i]+(off[i]/pivots[i])**2*variance[i+1]
    return variance,adjacent


def scores(h,returns,u):
    """Stable lower-tail evaluation; first three exact state derivatives."""
    x=returns*np.exp(-.5*h);squared=x*x
    if u==0:return x,-.5*x,.25*x,-.125*x
    tail=stdtr(1/u,-np.abs(x)/math.sqrt(1-2*u))
    z=-np.sign(x)*ndtri(tail)
    for index in np.flatnonzero(tail<=0):
        a=abs(float(x[index]));denom=1-2*u+u*a*a
        scale=denom/((1+u)*a)
        def integrand(v,a=a,scale=scale,denom=denom):
            shift=scale*v
            return math.exp(-.5*(1+u)/u*math.log1p(u*(2*a*shift+shift*shift)/denom))
        integral,error=quad(integrand,0.,np.inf,epsabs=1e-11,epsrel=1e-11)
        if not np.isfinite(integral) or integral<=0 or error>1e-9*integral:
            raise ArithmeticError('unresolved Student log-tail quadrature')
        logpdf_a=original.normalizer(u)[0]-.5*(1+u)/u*math.log1p(u*a*a/(1-2*u))
        logtail=logpdf_a+math.log(scale)+math.log(integral)
        z[index]=-np.sign(x[index])*ndtri_exp(logtail)
    logpdf=original.normalizer(u)[0]-.5*(1+u)/u*np.log1p(u*squared/(1-2*u))
    a=x*np.exp(logpdf+.5*z*z+.5*math.log(2*math.pi))
    denom=1-2*u+u*squared
    b=1-(1+u)*squared/denom+z*a
    derivative=(1+u)*(1-2*u)*squared/(denom*denom)-.5*a*a-.5*z*a*b
    return z,-.5*a,.25*a*b,.25*a*(derivative-.5*b*b)


def quantities(h,returns,level,phi,eta,u,rho):
    obs=original.observation_terms(h,returns*returns,u)
    z,z1,z2,z3=scores(h[:-1],returns[:-1],u)
    q=eta*eta*(1-rho*rho);p0=(1-phi*phi)/(eta*eta);delta=h[0]-level
    d=h[1:]-level-phi*(h[:-1]-level)-eta*rho*z
    k=-phi-eta*rho*z1;b2=-eta*rho*z2;b3=-eta*rho*z3
    energy=obs[0].sum()+.5*p0*delta*delta+.5*np.dot(d,d)/q
    gradient=obs[1].copy();gradient[0]+=p0*delta
    gradient[:-1]+=d*k/q;gradient[1:]+=d/q
    diagonal=obs[2].copy();diagonal[0]+=p0
    diagonal[:-1]+=(k*k+d*b2)/q;diagonal[1:]+=1/q
    off=k/q
    return energy,gradient,diagonal,off,(obs,z,z1,z2,z3,q,p0,delta,d,k,b2,b3)


def latent_mode(returns,level,phi,eta,u,rho):
    h=np.full(len(returns),level);converged=False;decrement=math.inf
    current=quantities(h,returns,level,phi,eta,u,rho)
    for iteration in range(100):
        energy,g,diagonal,off,_=current
        pivots,positive=factor(diagonal,off)
        if not positive:
            # Damping locates a descent direction only. Final Laplace curvature
            # must be the undamped, positive Hessian at the accepted mode.
            damping=max(1.,float(np.max(np.abs(diagonal))))*np.finfo(float).eps**.25
            for _ in range(60):
                pivots,positive=factor(diagonal+damping,off)
                if positive:break
                damping*=2
        if not positive:break
        step=solve(pivots,off,-g);decrement=-float(np.dot(g,step))
        if decrement<=1e-14:
            _,positive=factor(diagonal,off)
            converged=positive;break
        rounding=32*np.finfo(float).eps*max(1.,abs(energy));scale=1.;accepted=False
        for _ in range(60):
            trial=h+scale*step
            try:
                proposed=quantities(trial,returns,level,phi,eta,u,rho);value=proposed[0]
            except (ArithmeticError,FloatingPointError):value=math.inf
            if np.isfinite(value) and value<=energy-1e-4*scale*decrement+rounding:
                h=trial;current=proposed;accepted=True;break
            scale*=.5
        if not accepted:break
    energy,g,diagonal,off,terms=current
    pivots,positive=factor(diagonal,off)
    return h,pivots,off,terms,energy,converged and positive,iteration+1,decrement


def likelihood_gradient(returns,level,phi,eta,u,rho):
    """Analytic level/phi/log-eta/rho derivatives; numerical CDF u partials.

    Only the Student CDF's tail-shape partial uses a three-point stencil at the
    fixed latent mode. The state solves and all mode/curvature corrections are
    analytic and linear-time; no nested optimizer or numerical Hessian.
    """
    h,pivots,off,t,energy,success,iterations,decrement=latent_mode(returns,level,phi,eta,u,rho)
    if not success:return -math.inf,np.zeros(5),h,np.zeros(len(h)),False,iterations,decrement
    obs,z,z1,z2,_z3,q,p0,delta,d,k,b2,b3=t;n=len(h)
    variance,adjacent=inverse_diagonal(pivots,off)
    logc,logc_u=original.normalizer(u)
    determinant=n*math.log(eta*eta)-math.log1p(-phi*phi)+(n-1)*math.log1p(-rho*rho)
    likelihood=n*logc-.5*determinant-energy-.5*np.log(pivots).sum()
    trace_mode=variance*obs[3]
    trace_mode[:-1]+=(variance[:-1]*(3*k*b2+d*b3)+2*adjacent*b2)/q
    trace_mode[1:]+=variance[:-1]*b2/q
    # Accuracy-scaled finite difference for CDF tail-shape derivatives.
    step=np.finfo(float).eps**(1/3)
    if u<step:
        s0=np.array([z,z1,z2]);s1=np.array(scores(h[:-1],returns[:-1],u+step)[:3]);s2=np.array(scores(h[:-1],returns[:-1],u+2*step)[:3]);du=(-3*s0+4*s1-s2)/(2*step)
    elif u>.5-2*step:
        s0=np.array([z,z1,z2]);s1=np.array(scores(h[:-1],returns[:-1],u-step)[:3]);s2=np.array(scores(h[:-1],returns[:-1],u-2*step)[:3]);du=(3*s0-4*s1+s2)/(2*step)
    else:du=(np.array(scores(h[:-1],returns[:-1],u+step)[:3])-np.array(scores(h[:-1],returns[:-1],u-step)[:3]))/(2*step)
    gradient=np.empty(5)
    for parameter in range(5):
        dg=np.zeros(n);hd=np.zeros(n);ho=np.zeros(n-1)
        if parameter==0:
            dp=-p0;ep=-p0*delta;hp=0.;dt=np.full(n-1,phi-1);kt=np.zeros(n-1);bt=kt;logq=0.;logdet=0.;lc=0.
        elif parameter==1:
            dp=-2*phi*delta/(eta*eta);ep=-phi*delta*delta/(eta*eta);hp=-2*phi/(eta*eta);dt=-(h[:-1]-level);kt=np.full(n-1,-1.);bt=np.zeros(n-1);logq=0.;logdet=2*phi/(1-phi*phi);lc=0.
        elif parameter==2:
            dp=-2*p0*delta;ep=-p0*delta*delta;hp=-2*p0;dt=-eta*rho*z;kt=-eta*rho*z1;bt=-eta*rho*z2;logq=2.;logdet=2.*n;lc=0.
        elif parameter==3:
            dp=ep=hp=logq=logdet=0.;dt=-eta*rho*du[0];kt=-eta*rho*du[1];bt=-eta*rho*du[2];lc=n*logc_u
            dg+=obs[5];hd+=obs[6];ep+=obs[4].sum()
        else:
            dp=ep=hp=0.;dt=-eta*z;kt=-eta*z1;bt=-eta*z2;logq=-2*rho/(1-rho*rho);logdet=(n-1)*logq;lc=0.
        dg[0]+=dp;hd[0]+=hp
        ep+=np.sum(d*dt-.5*d*d*logq)/q
        dg[:-1]+=(dt*k+d*kt-d*k*logq)/q;dg[1:]+=(dt-d*logq)/q
        hd[:-1]+=(2*k*kt+dt*b2+d*bt-(k*k+d*b2)*logq)/q;hd[1:]+=-logq/q
        ho+=(kt-k*logq)/q
        mode=solve(pivots,off,-dg)
        trace=np.dot(variance,hd)+2*np.dot(adjacent,ho)+np.dot(trace_mode,mode)
        gradient[parameter]=lc-.5*logdet-ep-.5*trace
    return float(likelihood),gradient,h,variance,success,iterations,decrement


def fit(returns,observed_proxy,initial,initial_u):
    from scipy.optimize import minimize
    center=float(observed_proxy.mean());lo,hi=np.quantile(observed_proxy,[.01,.99]);guard=np.sqrt(np.finfo(float).eps)
    bounds=[(lo-4,hi+4),(-7.,float(logit(np.nextafter(.999,0.)))),(math.log(.02),math.log(2.5)),(0.,.5-guard),(-1+guard,1-guard)]
    @lru_cache(maxsize=1)
    def cached_target(values):
        level,p,e,u,rho=values;phi,eta=expit(p),math.exp(e)
        likelihood,g,_,_,success,_,_=likelihood_gradient(returns,level,phi,eta,u,rho)
        if not success or not np.isfinite(likelihood):return 1e100,np.zeros(5)
        prior=(-.5*((level-center)/4)**2-.5*((phi-.94)/.20)**2-.5*(e-math.log(.35))**2+math.log(phi*(1-phi))+e)
        g[1]*=phi*(1-phi)
        g+=np.array([-(level-center)/16,-(phi-.94)/.04*phi*(1-phi)+1-2*phi,-(e-math.log(.35))+1,0.,0.])
        return -(likelihood+prior)/len(returns),-g/len(returns)
    def target(values):
        value,g=cached_target(tuple(values));return value,g.copy()
    def projected(values):
        _,g=target(values)
        for i,(low,high) in enumerate(bounds):
            if values[i]<=low and g[i]>0 or values[i]>=high and g[i]<0:g[i]=0.
        return float(np.max(np.abs(g)))
    start=[initial[0],logit(initial[1]),math.log(initial[2]),initial_u,0.]
    result=minimize(target,start,jac=True,method='L-BFGS-B',bounds=bounds,options={'ftol':1e-12,'gtol':1e-8,'maxiter':200,'maxls':100});evaluations=int(result.nfev)
    if not result.success or result.fun>=1e99 or projected(result.x)>1e-6:
        result=minimize(target,result.x if result.fun<1e99 else start,jac=True,method='SLSQP',bounds=bounds,options={'ftol':1e-12,'maxiter':200});evaluations+=int(result.nfev)
    pg=projected(result.x)
    if not result.success or result.fun>=1e99 or pg>1e-6:raise ArithmeticError(f'joint Student/copula leverage MAP unresolved: {result.message}, projected gradient {pg}')
    level,p,e,u,rho=result.x;theta=(float(level),float(expit(p)),math.exp(e),float(u),float(rho))
    ll,_,h,var,success,iterations,decrement=likelihood_gradient(returns,*theta)
    if not success:raise ArithmeticError('joint leverage state mode unresolved')
    return theta,h,var,{'success':True,'evaluations':evaluations,'objective':float(result.fun*len(returns)),'max_projected_mean_gradient':pg,'loglikelihood':ll,'latent_iterations':iterations,'newton_decrement':decrement,'latent_integration':'Laplace approximation, tridiagonal Hessian','posterior_parameter_uncertainty':False,'inverse_df':theta[3],'degrees_of_freedom':None if theta[3]==0 else 1/theta[3],'tail_prior':'uniform inverse_df on [0,1/2), Gaussian limit at zero','rho':theta[4],'leverage_prior':'uniform rho on (-1,1), machine precision guards','leverage_likelihood':'Gaussian copula Student return rank and next AR1 volatility innovation'}

def conditioned_multiscale_moments(phi,loading,common,independent,mean,variance,rho,observed_score):
    """Condition only the first future shock on the last observed return rank."""
    aggregate=np.array([loading@common,loading@independent])
    norm=np.linalg.norm(aggregate)
    if rho==0. or norm==0.:return mean,variance
    a=aggregate/norm
    direction=common*a[0]+independent*a[1]
    powers=phi[:,None]**np.arange(len(mean))[None,:]
    projection=(loading*direction)@powers
    return mean+rho*observed_score*projection,np.maximum(variance-rho*rho*projection*projection,0.)

