"""Convergence certified by convex optimality, without a total iteration cap."""
import numpy as np
from scipy.optimize import minimize
from candidates.asset_selector_stacking.energy import coefficients,objective,even_allocation

def fit(A,B):
    A=np.asarray(A,dtype=float);B=np.asarray(B,dtype=float);k=len(A)
    if B.shape!=(k,k)or k<1 or not np.isfinite(A).all()or not np.isfinite(B).all():raise ValueError('Finite mixture coefficients required')
    scale=max(float(np.max(np.abs(A))),float(np.max(np.abs(B))),np.finfo(float).tiny)
    a=A/scale;b=B/scale;vertices=a-.5*np.diag(b);best=int(np.argmin(vertices));initial=np.zeros(k);initial[best]=1.
    if k==1:return {'weights':initial,'objective':objective(initial,A,B),'best_vertex_objective':objective(initial,A,B),'KKT_gap':0.,'iterations':0}
    eps=np.finfo(float).eps;precision=64*eps*k;tolerance=np.sqrt(eps)*k;budget=1000;iterations=0;attempts=[]
    while True:
        result=minimize(lambda w:objective(w,a,b),initial,jac=lambda w:a-b@w,method='SLSQP',bounds=[(0.,1.)]*k,
            constraints=[{'type':'eq','fun':lambda w:w.sum()-1.,'jac':lambda w:np.ones(k)}],options={'ftol':32*eps,'maxiter':budget})
        if not np.isfinite(result.x).all():raise ArithmeticError('Nonfinite simplex solver state')
        w=np.maximum(result.x,0.)
        if w.sum()==0:raise ArithmeticError('Empty simplex solver state')
        w/=w.sum();gradient=a-b@w;gap=float(np.dot(w,gradient)-gradient.min());iterations+=int(result.nit)
        attempts.append({'solver_success':bool(result.success),'solver_status':int(result.status),'iterations':int(result.nit),'budget':budget,'scaled_KKT_gap':gap})
        # Empirical absolute-distance B is conditionally negative semidefinite;
        # this simplex gap bounds global suboptimality regardless of solver status.
        if gap<=tolerance and objective(w,a,b)<=vertices[best]+precision:
            return {'weights':w,'objective':objective(w,A,B),'best_vertex_objective':float(vertices[best]*scale),'KKT_gap':gap*scale,'iterations':iterations,'solver_attempts':attempts,'precision_scope':'Scaled convex simplex certificate; no total iteration cap and no accepted uncertified iterate'}
        if result.status!=9:raise ArithmeticError(f'Uncertified convex CRPS fit: {result.message}; gap={gap}')
        initial=w;budget*=2
