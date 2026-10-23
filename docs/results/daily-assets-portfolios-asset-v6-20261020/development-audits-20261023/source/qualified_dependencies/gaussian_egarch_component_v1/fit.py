"""Full-prefix Gaussian EGARCH marginal likelihood and exact sensitivities."""
import numpy as np
from numba import njit
from scipy.optimize import minimize

C=np.sqrt(2/np.pi)

@njit(cache=True,fastmath=False)
def likelihood(parameters,y):
    omega,alpha,gamma,beta=parameters
    n=len(y);logs=np.empty(n);z=np.empty(n);gradient=np.zeros(4)
    # Pre-sample logvariance backcast=0; pre-sample centered shock terms=0.
    # Thus first logvariance=omega, matching arch EGARCH backcast=0.
    level=omega;derivative=np.array([1.,0.,0.,0.]);loss=0.
    for t in range(n):
        logs[t]=level;shock=y[t]*np.exp(-.5*level);z[t]=shock
        if not np.isfinite(level)or not np.isfinite(shock):return np.inf,np.full(4,np.nan),logs,z,level
        square=shock*shock
        loss+=.5*(np.log(2*np.pi)+level+square)
        gradient+=.5*(1-square)*derivative
        sign=1. if shock>0 else(-1. if shock<0 else 0.)
        factor=beta-.5*shock*(alpha*sign+gamma)
        derivative=factor*derivative+np.array([1.,abs(shock)-C,shock,level])
        level=omega+beta*level+alpha*(abs(shock)-C)+gamma*shock
    return loss/n,gradient/n,logs,z,level


def objective(theta,y):
    parameters=np.asarray(theta,dtype=np.float64).copy();parameters[3]=np.tanh(theta[3])
    if not np.isfinite(parameters).all()or abs(parameters[3])>=1:
        return np.inf,np.full(4,np.nan)
    value,gradient,_,_,_=likelihood(parameters,y)
    gradient[3]*=1-parameters[3]**2
    return float(value),gradient


def estimate(values):
    x=np.asarray(values,dtype=np.float64)
    if x.ndim!=1 or len(x)<2 or not np.isfinite(x).all():raise ValueError('Finite full-prefix asset returns required')
    center=float(x.mean());scale=float(np.sqrt(np.mean((x-center)**2)))
    if not np.isfinite(scale)or scale<=0:raise ValueError('Positive finite training variance required')
    y=np.ascontiguousarray((x-center)/scale);solutions=[];reports=[]
    for initial in (np.zeros(4),np.array([0.,.1,0.,np.arctanh(.9)])):
        answer=minimize(objective,initial,args=(y,),method='BFGS',jac=True,options={'gtol':1e-7,'maxiter':1000})
        value,gradient=objective(answer.x,y)
        natural=answer.x.copy();natural[3]=np.tanh(natural[3]);natural_gradient=likelihood(natural,y)[1]
        eligible=bool(answer.success and np.isfinite(value)and np.isfinite(gradient).all()and np.max(np.abs(gradient))<=1e-6 and np.isfinite(natural_gradient).all()and np.max(np.abs(natural_gradient))<=1e-6)
        reports.append({'optimizer_success':bool(answer.success),'eligible':eligible,'message':str(answer.message),'objective':float(value)if np.isfinite(value)else None,'gradient_max':float(np.max(np.abs(gradient)))if np.isfinite(gradient).all()else None,'natural_gradient_max':float(np.max(np.abs(natural_gradient)))if np.isfinite(natural_gradient).all()else None,'iterations':int(answer.nit)})
        if eligible:solutions.append((value,answer.x.copy()))
    if not solutions:raise ArithmeticError('No converged finite EGARCH likelihood fit; no fixed-model fallback')
    value,theta=min(solutions,key=lambda r:r[0]);parameters=theta.copy();parameters[3]=np.tanh(theta[3])
    _,gradient,logs,z,next_logvariance=likelihood(parameters,y)
    if not np.isfinite(logs).all()or not np.isfinite(z).all()or not np.isfinite(next_logvariance):raise ArithmeticError('Nonfinite fitted EGARCH states')
    return {'parameters':parameters,'mean':center,'scale':scale,'log_variance':logs,'standardized_residuals':z,'next_log_variance':float(next_logvariance),'initial_log_backcast':0.,'training_rows':len(x),'mean_negative_loglikelihood':float(value),'original_units_mean_negative_loglikelihood':float(value+np.log(scale)),'natural_gradient_max':float(np.max(np.abs(gradient))),'transformed_gradient_max':float(np.max(np.abs(objective(theta,y)[1]))),'reports':reports,'global_optimum_claim':False,'distribution':'Gaussian','stationarity_bound':'abs(beta)<1; no sign constraints on alpha/gamma','numeric_clipping':False}
