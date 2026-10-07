"""Scalar cDCC likelihood with a fixed estimated correlation target.

This implements Aielli's corrected recursion, not a claim that the fixed
Ledoit-Wolf target is his consistent target estimator. Parameters are conditional
Gaussian copula QMLE; no latent-parameter MCMC or posterior compression.
"""
import numpy as np
from numba import njit
from scipy.optimize import minimize
from scipy.special import expit, softmax


@njit(cache=True)
def likelihood(z,s,a,b,q0):
    q=q0.copy(); da=np.zeros_like(q);db=np.zeros_like(q)
    loss=0.;gradient=np.zeros(2);p=len(s)
    eye=np.eye(p)
    for t in range(len(z)):
        d=np.diag(q).copy();sd=np.sqrt(d);w=sd*z[t]
        root=np.linalg.cholesky(q)
        inv=np.linalg.solve(root.T,np.linalg.solve(root,eye))
        v=inv@w
        logdet=2*np.log(np.diag(root)).sum()-np.log(d).sum()
        loss+=.5*(logdet+w@v-z[t]@z[t])
        g=.5*inv-.5*np.outer(v,v)
        for j in range(p):g[j,j]+=-.5/d[j]+.5*v[j]*z[t,j]/sd[j]
        gradient[0]+=np.sum(g*da);gradient[1]+=np.sum(g*db)
        wa=.5*z[t]/sd*np.diag(da);wb=.5*z[t]/sd*np.diag(db)
        outer=np.outer(w,w)
        da=outer-s+a*(np.outer(wa,w)+np.outer(w,wa))+b*da
        db=q-s+a*(np.outer(wb,w)+np.outer(w,wb))+b*db
        q=(1-a-b)*s+a*outer+b*q
    return loss,gradient,q

def fit(z,s):
    """Conditional Gaussian copula QMLE with fixed training shrinkage target."""
    n=len(z)
    def physical(x):
        return softmax(np.r_[x,0.])[:2]
    def fun(x):
        theta=physical(x);a,b=theta
        if theta.sum()>=1.:
            return 1e100,np.zeros(2)
        try:loss,grad,_=likelihood(z,s,a,b,s)
        except np.linalg.LinAlgError:return 1e100,np.zeros(2)
        jac=np.array([[a*(1-a),-a*b],[-a*b,b*(1-b)]])
        return loss/n,jac@grad/n
    solutions=[]
    for start in ([.02,.95],[.05,.5],[.2,.2]):
        start=np.array(start);coords=np.log(start/(1-start.sum()))
        result=minimize(fun,coords,jac=True,method='L-BFGS-B',options={'ftol':1e-12,'gtol':1e-7,'maxiter':500,'maxls':40})
        theta=physical(result.x)
        if result.success and theta.sum()<1. and np.isfinite(result.fun) and np.max(np.abs(result.jac))<=1e-5:
            solutions.append((float(result.fun),theta,{'success':bool(result.success),'iterations':int(result.nit),'evaluations':int(result.nfev),'transformed_mean_gradient':result.jac.tolist()}))
    # Fit the b=0 boundary separately, rather than impose a positive floor.
    def boundary(x):
        a=float(expit(x[0]));loss,gradient,_=likelihood(z,s,a,0.,s)
        return loss/n,np.array([gradient[0]*a*(1-a)/n])
    edge=minimize(boundary,np.array([-2.]),jac=True,method='L-BFGS-B',
                  options={'ftol':1e-12,'gtol':1e-7,'maxiter':500,'maxls':40})
    if edge.success and np.max(np.abs(edge.jac))<=1e-5:
        solutions.append((float(edge.fun),np.array([float(expit(edge.x[0])),0.]),
                          {'success':True,'b_zero_boundary':True,'iterations':int(edge.nit),'evaluations':int(edge.nfev),'transformed_mean_gradient':edge.jac.tolist()}))
    if not solutions:raise ArithmeticError('no converged cDCC parameter fit')
    constant=likelihood(z,s,0.,0.,s)[0]/n
    solutions.append((constant,np.zeros(2),{'success':True,'constant_endpoint':True}))
    best=min(solutions,key=lambda x:x[0])
    loss,grad,q=likelihood(z,s,*best[1],s)
    return best[1],q,{'training_nll':float(loss),'static_training_nll':float(constant*n),'gradient':grad.tolist(),'optimizer':best[2]}

