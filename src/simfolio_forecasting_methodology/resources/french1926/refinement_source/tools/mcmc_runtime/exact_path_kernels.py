"""Exact sorted-marginal mapping and NumPy-order portfolio reductions."""
import math

import numpy as np
from numba import njit, prange


@njit(cache=True,fastmath=False)
def pairwise_sum(values):
    n=values.size
    if n<8:
        result=-0.
        for i in range(n):result+=values[i]
        return result
    if n<=128:
        r=values[:8].copy();i=8
        while i<n-(n%8):
            for k in range(8):r[k]+=values[i+k]
            i+=8
        result=((r[0]+r[1])+(r[2]+r[3]))+((r[4]+r[5])+(r[6]+r[7]))
        while i<n:result+=values[i];i+=1
        return result
    split=n//2;split-=split%8
    return pairwise_sum(values[:split])+pairwise_sum(values[split:])

@njit(cache=True,fastmath=False)
def sorted_map(source,uniforms):
    sims,horizon,assets=source.shape;out=np.empty_like(source)
    for sim in range(sims):
        for day in range(horizon):
            for asset in range(assets):
                position=min(max(uniforms[sim,day,asset],0.),1.)*float(sims-1)
                low=int(math.floor(position));high=min(low+1,sims-1);fraction=position-low  # noqa: RUF046 - retain the reference integer conversion in compiled kernels.
                out[sim,day,asset]=source[low,day,asset]*(1.-fraction)+source[high,day,asset]*fraction
    return out

@njit(cache=True,fastmath=False)
def rejoin_growth(growth,target,mask,cost):
    sims,horizon,assets=growth.shape;out=np.empty((sims,horizon))
    for sim in range(sims):
        holdings=target.copy();turnovers=np.empty(assets)
        for day in range(horizon):
            previous=pairwise_sum(holdings)
            for asset in range(assets):holdings[asset]*=growth[sim,day,asset]
            ending=pairwise_sum(holdings)
            if mask[day]:
                denominator=max(ending,1e-300)
                for asset in range(assets):turnovers[asset]=abs(target[asset]-holdings[asset]/denominator)
                turnover=pairwise_sum(turnovers)
                ending=max(ending-ending*.5*turnover*cost,0.)
                for asset in range(assets):holdings[asset]=ending*target[asset]
            out[sim,day]=math.log(max(ending,1e-300)/max(previous,1e-300))
    return out

@njit(cache=True,fastmath=False,parallel=True,nogil=True)
def parallel_rejoin_growth(growth,target,mask,cost):
    sims,horizon,assets=growth.shape;out=np.empty((sims,horizon))
    for sim in prange(sims):
        holdings=target.copy();turnovers=np.empty(assets)
        for day in range(horizon):
            previous=pairwise_sum(holdings)
            for asset in range(assets):holdings[asset]*=growth[sim,day,asset]
            ending=pairwise_sum(holdings)
            if mask[day]:
                denominator=max(ending,1e-300)
                for asset in range(assets):turnovers[asset]=abs(target[asset]-holdings[asset]/denominator)
                turnover=pairwise_sum(turnovers)
                ending=max(ending-ending*.5*turnover*cost,0.)
                for asset in range(assets):holdings[asset]=ending*target[asset]
            out[sim,day]=math.log(max(ending,1e-300)/max(previous,1e-300))
    return out

def fast_rejoin(paths,weights,mask,*,cost_per_turnover_bps=15.):
    target=np.maximum(np.asarray(weights,np.float64),0.);target/=target.sum()
    growth=np.exp(np.clip(paths,-745.,50.))
    return rejoin_growth(growth,target,np.asarray(mask,np.bool_),max(float(cost_per_turnover_bps),0.)/10000.)
