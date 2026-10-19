"""Bound temporary memory without reducing any asset/path/horizon work."""
import math
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from exact_path_kernels import parallel_rejoin_growth, rejoin_growth
from numba import njit, prange

from simfolio_forecasting_methodology.models.numerical import dynamic_gaussian as dg


@njit(cache=True,fastmath=False,parallel=True,nogil=True)
def copula_block(states,draws,loading,mean,phi,q_sd,r_sd,out,day_offset):
    days=draws.shape[0];simulations,factors=states.shape;assets=loading.shape[0]
    inv_sqrt_two=1./math.sqrt(2.);loading_t=loading.T
    for sim in prange(simulations):
        values=np.empty(assets)
        state=states[sim].copy()
        for day in range(days):
            start=sim*factors
            for factor in range(factors):state[factor]=state[factor]*phi[factor]+draws[day,start+factor]*q_sd[factor]
            offset=simulations*factors+sim*assets
            np.dot(state,loading_t,values)
            for asset in range(assets):
                v=(mean[asset]+draws[day,offset+asset]*r_sd[asset])+values[asset]
                value=.5*(1.+math.erf(v*inv_sqrt_two))
                out[sim,day_offset+day,asset]=min(max(value,1e-8),1.-1e-8)
        states[sim]=state

def uniforms(model,simulations,horizon,rng):
    posterior_mean,posterior_covariance=dg.kalman_terminal_posterior(model)
    states=rng.multivariate_normal(posterior_mean,posterior_covariance,size=simulations,check_valid='raise')
    out=np.empty((simulations,horizon,model['asset_count']))
    q_sd=np.sqrt(model['innovation_variance']);r_sd=np.sqrt(model['residual_variance'])
    width=simulations*(model['factor_count']+model['asset_count'])
    if horizon<=1024:
        draws=rng.normal(size=(horizon,width))
        copula_block(states,draws,model['loading'],model['mean'],model['phi'],q_sd,r_sd,out,0)
    else:
        # A single producer owns the Generator. Draw consumption stays ordered;
        # only the next block overlaps the independent current-path arithmetic.
        with ThreadPoolExecutor(max_workers=1) as pool:
            pending=pool.submit(rng.normal,size=(1024,width))
            for start in range(0,horizon,1024):
                stop=min(start+1024,horizon);draws=pending.result()
                if stop<horizon:pending=pool.submit(rng.normal,size=(min(1024,horizon-stop),width))
                copula_block(states,draws,model['loading'],model['mean'],model['phi'],q_sd,r_sd,out,start)
    return out

@njit(cache=True,fastmath=False)
def map_asset_inplace(uniforms,mean,sd,nodes):
    sims,horizon=uniforms.shape
    for sim in range(sims):
        for day in range(horizon):
            position=min(max(uniforms[sim,day],0.),1.)*float(sims-1)
            low=int(math.floor(position));high=min(low+1,sims-1);fraction=position-low  # noqa: RUF046 - retain the reference integer conversion in compiled kernels.
            left=min(max(mean[day]+sd[day]*nodes[low],-1.),1.)
            right=min(max(mean[day]+sd[day]*nodes[high],-1.),1.)
            uniforms[sim,day]=left*(1.-fraction)+right*fraction

def rejoin(paths,weights,mask,*,cost_per_turnover_bps=15.):
    target=np.maximum(np.asarray(weights,np.float64),0.);target/=target.sum();mask=np.asarray(mask,np.bool_)
    out=np.empty(paths.shape[:2]);cost=max(float(cost_per_turnover_bps),0.)/10000.
    for start in range(0,len(paths),8):
        stop=min(start+8,len(paths));growth=np.exp(np.clip(paths[start:stop],-745.,50.))
        kernel=parallel_rejoin_growth if paths.shape[1]>1024 else rejoin_growth
        out[start:stop]=kernel(growth,target,mask,cost)
    return out
