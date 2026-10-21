"""Exact empirical CRPS mixture coefficients and certified simplex fitting."""
import numpy as np
from numba import njit
from scipy.optimize import minimize

@njit(cache=True,fastmath=False)
def cross_absolute_sorted(x,y):
    total=0.;prefix=0.;sy=0.;j=0
    for value in y:sy+=value
    for value in x:
        while j<len(y)and y[j]<=value:prefix+=y[j];j+=1
        total+=(2*j-len(y))*value+sy-2*prefix
    return total/(len(x)*len(y))

def coefficients(samples,observed):
    x=np.asarray(samples,dtype=float)
    if x.ndim!=2 or x.shape[1]<2 or not np.isfinite(x).all()or not np.isfinite(observed):raise ValueError('Finite equal-member predictive ensembles required')
    sorted_x=np.sort(x,axis=1);A=np.mean(np.abs(x-observed),axis=1);B=np.empty((len(x),len(x)))
    for i in range(len(x)):
        for j in range(i+1):B[i,j]=B[j,i]=cross_absolute_sorted(sorted_x[i],sorted_x[j])
    return A,B

def objective(w,A,B):return float(np.dot(w,A)-.5*np.dot(w,B@w))

def fit(A,B):
    A=np.asarray(A,dtype=float);B=np.asarray(B,dtype=float);k=len(A)
    if B.shape!=(k,k)or k<1 or not np.isfinite(A).all()or not np.isfinite(B).all():raise ValueError('Finite mixture coefficients required')
    scale=max(float(np.max(np.abs(A))),float(np.max(np.abs(B))),np.finfo(float).tiny)
    a=A/scale;b=B/scale;vertices=a-.5*np.diag(b);best=int(np.argmin(vertices));initial=np.zeros(k);initial[best]=1.
    if k==1:return {'weights':initial,'objective':objective(initial,A,B),'best_vertex_objective':objective(initial,A,B),'KKT_gap':0.,'iterations':0}
    # Precision is numerical, not a materiality gate or component-search cap.
    eps=np.finfo(float).eps;precision=64*eps*k
    result=minimize(lambda w:objective(w,a,b),initial,jac=lambda w:a-b@w,method='SLSQP',bounds=[(0.,1.)]*k,
        constraints=[{'type':'eq','fun':lambda w:w.sum()-1.,'jac':lambda w:np.ones(k)}],options={'ftol':32*eps,'maxiter':1000})
    w=np.maximum(result.x,0.);w/=w.sum();gradient=a-b@w;gap=float(np.dot(w,gradient)-gradient.min())
    # The solver is not allowed to return an uncertified iteration-limited fit.
    # The gap bounds suboptimality of this convex simplex problem.
    tolerance=np.sqrt(eps)*k
    if not result.success or gap>tolerance or objective(w,a,b)>vertices[best]+precision:raise ArithmeticError(f'Uncertified convex CRPS fit: {result.message}; gap={gap}')
    return {'weights':w,'objective':objective(w,A,B),'best_vertex_objective':float(vertices[best]*scale),'KKT_gap':gap*scale,'iterations':int(result.nit),'precision_scope':'Scaled simplex KKT gap checked; no accepted solver-iteration truncation'}

def even_allocation(weights,paths):
    w=np.asarray(weights,dtype=float)
    if paths<2 or paths%2 or np.any(w<0)or not np.isclose(w.sum(),1.,rtol=0,atol=1e-12):raise ValueError('Even ensemble and simplex weights required')
    units=paths//2;ideal=units*w;allocation=np.floor(ideal).astype(int);missing=units-int(allocation.sum())
    order=np.argsort(-(ideal-allocation),kind='stable');allocation[order[:missing]]+=1
    return 2*allocation
