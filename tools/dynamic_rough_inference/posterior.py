"""MAP point estimation of the existing rough quasi-posterior; no chains."""
import time
import numpy as np
from scipy.optimize import minimize


class InferenceLimit(RuntimeError):
    pass


def fit_map(target,start,bounds,max_seconds=10.):
    started=time.perf_counter()
    lower,upper=np.asarray(bounds,float).T
    width=upper-lower
    evaluations=0
    def physical(u):return lower+width*u
    def objective(u):
        nonlocal evaluations
        evaluations+=1
        if time.perf_counter()-started>max_seconds:
            raise InferenceLimit('MAP fit exceeded the resource ceiling')
        value=float(target(physical(u)))
        return -value if np.isfinite(value) else 1e100
    initial=np.clip((np.asarray(start)-lower)/width,1e-8,1-1e-8)
    mode=minimize(objective,initial,method='L-BFGS-B',bounds=[(1e-10,1-1e-10)]*len(start),
        options={'ftol':1e-10,'gtol':1e-5,'maxiter':200})
    attempts=[dict(method='L-BFGS-B',success=bool(mode.success),message=str(mode.message))]
    # A covariance-resolution boundary can interrupt the finite-difference
    # line search. Retry the same target and accuracy settings, never replace
    # a failed estimate with its starting parameters.
    if not mode.success and np.isfinite(mode.fun):
        mode=minimize(objective,mode.x,method='L-BFGS-B',bounds=[(1e-10,1-1e-10)]*len(start),
            options={'ftol':1e-10,'gtol':1e-5,'maxiter':200,'maxls':100})
        attempts.append(dict(method='L-BFGS-B_extended_line_search',success=bool(mode.success),message=str(mode.message)))
    if not mode.success and np.isfinite(mode.fun):
        mode=minimize(objective,mode.x,method='Powell',bounds=[(1e-10,1-1e-10)]*len(start),
            options={'ftol':1e-10,'xtol':1e-6,'maxiter':200})
        attempts.append(dict(method='bounded_Powell',success=bool(mode.success),message=str(mode.message)))
    if not mode.success or not np.isfinite(mode.fun):
        raise InferenceLimit(f'MAP fit did not converge: {mode.message}')
    return dict(map=physical(mode.x),estimator='MAP_point',posterior_uncertainty=False,
        evaluations=evaluations,seconds=time.perf_counter()-started,
        optimizer=dict(success=bool(mode.success),message=str(mode.message),
            iterations=int(mode.nit),ftol=1e-10,gtol=1e-5,maximum_iterations=200,attempts=attempts))
