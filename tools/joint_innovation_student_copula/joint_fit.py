"""Joint rank maximum pseudo-likelihood for Student/Gaussian copulas.

Demarta and McNeil (2005), Section 4.1. Correlation and degrees of freedom
are fitted together; margins are unchanged empirical ranks. This is local
maximum pseudo-likelihood, not Bayesian inference or a guaranteed global MLE.
"""
import numpy as np
from scipy.linalg import cho_solve
from scipy.optimize import minimize
from scipy.special import ndtri, poch, stdtr, stdtrit


def correlation_from_coordinates(coordinates, dimensions):
    """Row-normalized unit-diagonal Cholesky covers every PD correlation."""
    a=np.eye(dimensions)
    a[np.tril_indices(dimensions,-1)]=coordinates
    lengths=np.sqrt(np.sum(a*a,axis=1))
    root=a/lengths[:,None]
    return root@root.T,a,lengths


def correlation_coordinates(correlation):
    root=np.linalg.cholesky(correlation)
    a=root/np.diag(root)[:,None]
    return a[np.tril_indices(len(a),-1)]


class JointLikelihood:
    def __init__(self,uniforms):
        self.u=np.asarray(uniforms,dtype=np.float64)
        if self.u.ndim!=2 or min(self.u.shape)<1 or not np.isfinite(self.u).all() or np.any((self.u<=0)|(self.u>=1)):
            raise ValueError('finite interior copula pseudo-observations required')
        # A finite-density joint MLE does not exist for perfectly duplicated
        # or reflected margins. Fail explicitly rather than impose a rho cap
        # or silently change the statistical estimator.
        for j,column in enumerate(self.u.T):
            if np.ptp(column)==0:raise ValueError('constant copula margin is unidentified')
            for prior in self.u.T[:j]:
                if np.array_equal(column,prior) or np.allclose(column,1-prior,rtol=0,atol=4*np.finfo(float).eps):
                    raise ValueError('singular duplicate/reflected copula margins have no finite joint density MLE')
        self.n,self.d=self.u.shape
        self.gaussian_scores=ndtri(self.u)
        self.evaluations=0

    def density(self,correlation,nu,scores=None,gradient=False):
        """Mean log density and analytic correlation-matrix gradient."""
        self.evaluations+=1
        root=np.linalg.cholesky(correlation)
        inverse=cho_solve((root,True),np.eye(self.d),check_finite=False)
        if scores is None:scores=self.gaussian_scores if np.isinf(nu) else stdtrit(nu,self.u)
        if not np.isfinite(scores).all():raise ArithmeticError('unresolved Student quantile')
        v=scores@inverse
        q=np.sum(v*scores,axis=1)
        logdet=2*np.log(np.diag(root)).sum()
        if np.isinf(nu):
            value=-.5*logdet-.5*np.mean(q-np.sum(scores*scores,axis=1))
            weights=np.ones(self.n)
        else:
            if not np.isfinite(nu) or nu<=0:raise ArithmeticError('positive finite Student df required')
            # Gamma ratios avoid catastrophic subtraction of log-Gammas as
            # nu approaches the explicit Gaussian endpoint.
            constant=np.log(poch(nu/2,self.d/2))-self.d*np.log(poch(nu/2,.5))-.5*logdet
            value=constant-(nu+self.d)/2*np.mean(np.log1p(q/nu))+(nu+1)/2*np.mean(np.sum(np.log1p(scores*scores/nu),axis=1))
            weights=(nu+self.d)/(nu+q)
        if not np.isfinite(value):raise ArithmeticError('unresolved copula density')
        if not gradient:return float(value)
        g=.5*((v.T*weights)@v/self.n-inverse)
        return float(value),g

    def objective(self,coordinates,student):
        k=self.d*(self.d-1)//2
        correlation,a,lengths=correlation_from_coordinates(coordinates[:k],self.d)
        with np.errstate(over='ignore',invalid='ignore',divide='ignore'):
            nu=float(np.exp(coordinates[k])) if student else np.inf
            try:
                value,g=self.density(correlation,nu,gradient=True)
                # Chain rule through row normalization and A A'. Only strict
                # lower-triangle coordinates are free; diagonals remain one.
                gc=g/lengths[:,None]/lengths[None,:]
                gc[np.diag_indices(self.d)]-=np.sum(g*correlation,axis=1)/lengths**2
                grad=(2*gc@a)[np.tril_indices(self.d,-1)]
                if student:
                    h=np.cbrt(np.finfo(float).eps)*max(1.,abs(coordinates[k]))
                    upper=self.density(correlation,float(np.exp(coordinates[k]+h)))
                    lower=self.density(correlation,float(np.exp(coordinates[k]-h)))
                    grad=np.r_[grad,(upper-lower)/(2*h)]
            except (ArithmeticError,np.linalg.LinAlgError):return np.inf,np.zeros_like(coordinates)
        return -value,-grad


def fit_joint_copula(uniforms,initial_correlation,initial_nu):
    likelihood=JointLikelihood(uniforms)
    initial=correlation_coordinates(initial_correlation)
    if likelihood.d==1:
        return np.eye(1),np.eye(1),np.inf,{'method':'joint_rank_maximum_pseudo_likelihood','dimensions':1,'log_likelihood':0.,'gaussian_log_likelihood':0.,'evaluations':0}
    options={'gtol':1e-7,'ftol':1e-12,'maxiter':1000,'maxls':40}
    normal=minimize(likelihood.objective,initial,args=(False,),jac=True,method='L-BFGS-B',options=options)
    if not normal.success or not np.isfinite(normal.fun):raise ArithmeticError('Gaussian correlation pseudo-likelihood optimization failed: '+str(normal.message))
    gaussian_c,_,_=correlation_from_coordinates(normal.x,likelihood.d)
    # Initialization comes from the existing conditional fit. The Gaussian
    # endpoint is evaluated separately, with its correlation also optimized.
    start=np.r_[initial,np.log(initial_nu if np.isfinite(initial_nu) else likelihood.n)]
    initial_score=likelihood.density(initial_correlation,initial_nu)
    fitted=minimize(likelihood.objective,start,args=(True,),jac=True,method='L-BFGS-B',options=options)
    if not fitted.success or not np.isfinite(fitted.fun):raise ArithmeticError('Student joint pseudo-likelihood optimization failed: '+str(fitted.message))
    c,_,_=correlation_from_coordinates(fitted.x[:-1],likelihood.d)
    nu=float(np.exp(fitted.x[-1]));score=-float(fitted.fun);endpoint=False
    if -normal.fun>=score:
        c=gaussian_c;nu=np.inf;score=-float(normal.fun);endpoint=True
    if score+1e-10<initial_score:raise ArithmeticError('joint fit worsened its starting likelihood')
    error=0. if np.isinf(nu) else float(np.max(np.abs(stdtr(nu,stdtrit(nu,likelihood.u))-likelihood.u)))
    if not np.isfinite(error) or error>np.sqrt(np.finfo(float).eps):raise ArithmeticError('unresolved Student copula quantile accuracy')
    eigenvalues,eigenvectors=np.linalg.eigh(c)
    if eigenvalues.min()<=0:raise ArithmeticError('joint copula correlation is not positive definite')
    root=(eigenvectors*np.sqrt(eigenvalues))@eigenvectors.T
    return c,root,nu,{'method':'joint_rank_maximum_pseudo_likelihood','dimensions':likelihood.d,
        'log_likelihood':score*likelihood.n,'gaussian_log_likelihood':-float(normal.fun)*likelihood.n,
        'initial_conditional_log_likelihood':initial_score*likelihood.n,'evaluations':likelihood.evaluations,
        'gaussian_iterations':int(normal.nit),'student_iterations':int(fitted.nit),
        'gaussian_maximum_gradient':float(np.abs(normal.jac).max()),
        'student_maximum_gradient':float(np.abs(fitted.jac).max()),'gaussian_endpoint_selected':endpoint,
        'maximum_quantile_roundtrip_error':error,'minimum_correlation_eigenvalue':float(eigenvalues.min()),
        'statistical_shrinkage':False,'numerical_options':options}
