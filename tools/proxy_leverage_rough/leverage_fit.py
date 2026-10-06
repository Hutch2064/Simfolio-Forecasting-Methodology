"""Gaussian-score proxy leverage and exact capture of existing return shocks."""
from functools import lru_cache

import numpy as np
from scipy.special import ndtri
from scipy.stats import rankdata


def fit_correlations(assets,volatility_innovations,dependency):
    if dependency is None:
        residual=np.asarray(assets,float)
    else:
        loading=dependency['loading'];centered=dependency['observations']-dependency['mean']
        scores=np.linalg.solve(loading.T@loading,loading.T@centered.T).T
        residual=centered-scores@loading.T
    innovations=np.asarray(volatility_innovations,float)
    if residual.shape[0]!=len(innovations)+1:raise ValueError('lagged history alignment')
    result=np.zeros(residual.shape[1])
    for a in range(residual.shape[1]):
        left=residual[:-1,a];right=innovations[:,a]
        if np.ptp(left)==0. or np.ptp(right)==0.:continue
        z=ndtri((rankdata(left,method='average')-.5)/len(left))
        e=ndtri((rankdata(right,method='average')-.5)/len(right))
        rho=float(np.corrcoef(z,e)[0,1])
        if not np.isfinite(rho) or abs(rho)>1.:raise ArithmeticError('invalid Gaussian proxy leverage')
        result[a]=rho
    return result


class Capture:
    def __init__(self,rng,simulations,assets,factors,horizon):
        self.rng=rng;self.simulations=simulations;self.assets=assets;self.factors=factors
        self.residual=np.empty((simulations,horizon,assets));self.position=0

    def multivariate_normal(self,*args,**kwargs):
        return self.rng.multivariate_normal(*args,**kwargs)

    def normal(self,*args,**kwargs):
        values=self.rng.normal(*args,**kwargs)
        days,width=values.shape
        if width!=self.simulations*(self.assets+self.factors):raise ValueError('Gaussian draw shape changed')
        residual=values[:,self.simulations*self.factors:].reshape(days,self.simulations,self.assets)
        self.residual[:,self.position:self.position+days]=residual.transpose(1,0,2)
        self.position+=days
        return values


@lru_cache(maxsize=2)
def _capture(shape,data,simulations,horizon,seed,backend):
    assets=np.frombuffer(data,np.float64).reshape(shape)
    dependency=backend.fit_dynamic_gaussian_factor_model(assets)
    rng=Capture(np.random.default_rng(seed),simulations,shape[1],dependency['factor_count'],horizon)
    uniforms=backend.simulate_future_gaussian_uniforms(dependency,simulations,horizon,rng)
    if rng.position!=horizon:raise ArithmeticError('incomplete Gaussian residual capture')
    return uniforms,rng.residual,dependency


def capture_uniforms(assets,simulations,horizon,seed,backend):
    assets=np.asarray(assets,np.float64)
    return _capture(assets.shape,assets.tobytes(),simulations,horizon,seed,backend)


def clear_capture_cache():
    _capture.cache_clear()
