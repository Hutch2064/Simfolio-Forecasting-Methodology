"""Finite-variance Student copula with the corrected-DCC shock recursion.

Conditional plug-in likelihood: the Gaussian-rank shrinkage target is fixed.
The Student shock is standardized to unit variance before updating Q. This is
not a Gaussian-score recursion with a Student CDF substituted afterward.
"""
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from numba import njit
from scipy.optimize import minimize
from scipy.special import gammaln, ndtri, softmax

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'corrected_dcc_innovation_copula'))
import cdcc_state as gaussian
import likelihood_kernel
from student_cdf import quantiles


def copula_constant(nu,p):
    # Gamma differences lose precision near the Gaussian limit. The expansion
    # is used only where its neglected O(nu^-4) term is below double precision.
    if nu>1e5:
        return p*(p-1)/(4*nu)-p*(p-1)*(p-2)/(12*nu**2)+p*(p*(p-2)**2-1)/(24*nu**3)
    return gammaln((nu+p)/2)+(p-1)*gammaln(nu/2)-p*gammaln((nu+1)/2)


@njit(cache=True)
def likelihood(z,s,a,b,q0,nu,constant,derivatives=True):
    q=q0.copy();da=np.zeros_like(q);db=np.zeros_like(q)
    loss=-len(z)*constant;gradient=np.zeros(2);p=len(s);eye=np.eye(p)
    for row in z:
        d=np.diag(q).copy();sd=np.sqrt(d);w=sd*row
        root=np.linalg.cholesky(q)
        inv=np.linalg.solve(root.T,np.linalg.solve(root,eye));v=inv@w
        maha=w@v;weight=(nu+p)/(nu-2+maha)
        logdet=2*np.log(np.diag(root)).sum()-np.log(d).sum()
        loss+=.5*logdet+.5*(nu+p)*np.log1p(maha/(nu-2))-.5*(nu+1)*np.log1p(row*row/(nu-2)).sum()
        outer=np.outer(w,w)
        if derivatives:
            g=.5*inv-.5*weight*np.outer(v,v)
            for j in range(p):g[j,j]+=-.5/d[j]+.5*weight*v[j]*row[j]/sd[j]
            gradient[0]+=np.sum(g*da);gradient[1]+=np.sum(g*db)
            wa=.5*row/sd*np.diag(da);wb=.5*row/sd*np.diag(db)
            da=outer-s+a*(np.outer(wa,w)+np.outer(w,wa))+b*da
            db=q-s+a*(np.outer(wb,w)+np.outer(w,wb))+b*db
        q=(1-a-b)*s+a*outer+b*q
    return loss,gradient,q


def standardized_scores(u,nu):
    return np.ascontiguousarray(quantiles(u,nu)*np.sqrt((nu-2)/nu))


def fit(u,s):
    n,p=u.shape;gtheta,gq,ginfo=gaussian.fit(np.ascontiguousarray(ndtri(u)),s,evaluator=likelihood_kernel.gaussian_likelihood)
    solutions=[];step=np.finfo(float).eps**(1/3)
    # Rank uniforms share the same grid across assets. Invert each distinct
    # probability once, then gather the identical per-observation scores.
    probabilities,indices=np.unique(u,return_inverse=True)
    @lru_cache(maxsize=32)
    def scores(nu):
        values=standardized_scores(probabilities,nu)
        return np.ascontiguousarray(values[indices].reshape(u.shape))
    def evaluation(a,b,eta,derivatives=True):
        with np.errstate(over='ignore',under='ignore',invalid='ignore'):
            nu=2+np.exp(eta)
        if not np.isfinite(nu) or nu<=2:return 1e100,np.zeros(2),None
        z=scores(nu)
        return likelihood_kernel.evaluate(z,s,a,b,s,nu,copula_constant(nu,p),derivatives)
    def joint(a,b,eta):
        with np.errstate(over='ignore',under='ignore',invalid='ignore'):
            nu=2+np.exp(eta)
        if not np.isfinite(nu) or nu<=2:return 1e100,np.zeros(3),None
        h=step*max(1.,abs(eta));plus=2+np.exp(eta+h);minus=2+np.exp(eta-h)
        z=scores(nu);z_eta=(scores(plus)-scores(minus))/(2*h)
        constant_eta=(copula_constant(plus,p)-copula_constant(minus,p))/(2*h)
        return likelihood_kernel.evaluate(z,s,a,b,s,nu,copula_constant(nu,p),True,z_eta,constant_eta)
    def objective(x,boundary=False):
        theta=softmax(np.r_[x[:-1],0.])
        a=float(theta[0]);b=0. if boundary else float(theta[1]);eta=x[-1]
        if a+b>=1:return 1e100,np.zeros_like(x)
        try:
            value,grad,_=joint(a,b,eta);gn=grad[2];grad=grad[:2]
        except np.linalg.LinAlgError:return 1e100,np.zeros_like(x)
        if boundary:result=np.array([grad[0]*a*(1-a),gn])
        else:result=np.r_[np.array([[a*(1-a),-a*b],[-a*b,b*(1-b)]])@grad,gn]
        return value/n,result/n
    start=gtheta if gtheta.min()>0 else np.array([.02,.8])
    def optimize(initial_nu):
        x=np.r_[np.log(start/(1-start.sum())),np.log(initial_nu-2)]
        result=minimize(objective,x,jac=True,method='L-BFGS-B',options={'ftol':1e-12,'gtol':1e-7,'maxiter':500,'maxls':40})
        theta=softmax(np.r_[result.x[:2],0.])[:2];nu=2+np.exp(result.x[-1])
        if result.success and np.isfinite(result.fun) and theta.sum()<1 and np.max(np.abs(result.jac))<=1e-5:
            return float(result.fun),theta,nu,{'success':True,'iterations':int(result.nit),'evaluations':int(result.nfev),'transformed_mean_gradient':result.jac.tolist()}
        return None
    if p>=16:
        from concurrent.futures import ThreadPoolExecutor
        from numba import get_num_threads
        with ThreadPoolExecutor(max_workers=min(2,get_num_threads())) as pool:
            results=list(pool.map(optimize,(8.,30.)))
    else:results=list(map(optimize,(8.,30.)))
    solutions.extend(result for result in results if result is not None)
    edge=minimize(lambda x:objective(x,True),np.array([-2.,np.log(6.)]),jac=True,method='L-BFGS-B',options={'ftol':1e-12,'gtol':1e-7,'maxiter':500,'maxls':40})
    if edge.success and np.isfinite(edge.fun) and np.max(np.abs(edge.jac))<=1e-5:
        theta=np.array([softmax([edge.x[0],0.])[0],0.]);nu=2+np.exp(edge.x[-1])
        solutions.append((float(edge.fun),theta,nu,{'success':True,'b_zero_boundary':True,'iterations':int(edge.nit),'evaluations':int(edge.nfev),'transformed_mean_gradient':edge.jac.tolist()}))
    def constant(x):
        h=step*max(1.,abs(x[0]));v=evaluation(0.,0.,x[0],False)[0]
        return v/n,np.array([(evaluation(0.,0.,x[0]+h,False)[0]-evaluation(0.,0.,x[0]-h,False)[0])/(2*h*n)])
    edge=minimize(constant,np.array([np.log(6.)]),jac=True,method='L-BFGS-B',options={'ftol':1e-12,'gtol':1e-7,'maxiter':500,'maxls':40})
    if edge.success and np.isfinite(edge.fun) and np.max(np.abs(edge.jac))<=1e-5:
        solutions.append((float(edge.fun),np.zeros(2),2+np.exp(edge.x[0]),{'success':True,'constant_endpoint':True,'transformed_mean_gradient':edge.jac.tolist()}))
    if not solutions:raise ArithmeticError('no converged Student cDCC fit')
    solutions.append((ginfo['training_nll']/n,gtheta,np.inf,{'success':True,'gaussian_endpoint':True,'gaussian_fit':ginfo}))
    value,theta,nu,info=min(solutions,key=lambda item:item[0])
    q=gq if np.isinf(nu) else evaluation(*theta,np.log(nu-2))[2]
    return theta,q,nu,{'training_nll':float(value*n),'gaussian_training_nll':ginfo['training_nll'],'optimizer':info,'nu_support':'nu>2 plus Gaussian endpoint','numerical_nu_derivative':'analytic recursive likelihood derivative; quantile and normalizer central differences in log(nu-2), cube-root machine epsilon relative step','a_b_gradient':'exact recursive analytic derivative'}
