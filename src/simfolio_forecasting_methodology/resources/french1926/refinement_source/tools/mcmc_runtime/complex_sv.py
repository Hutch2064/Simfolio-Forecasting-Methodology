"""Research-only exact conditional latent-SV samplers; EB parameters from Frontier.
PGAS: Lindsten/Jordan/Schon (2014), bootstrap proposal + ancestor sampling.
ESS: Murray/Adams/MacKay (2010), stationary AR1 prior, centered ellipses.
No claim of sampling parameters, full INLA, or convergence from short chains.
"""
import hashlib
import math
import time
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numba import njit

from simfolio_forecasting_methodology.models.asset_level.filtered_innovation_moment_sv import (
    sorted_moment_marginals,
)
from simfolio_forecasting_methodology.models.asset_level.frontier import (
    _historical_rebalance_dates,
    _validate_calendar,
)
from simfolio_forecasting_methodology.models.numerical import bdes_fastmap as bd
from simfolio_forecasting_methodology.models.numerical import dynamic_gaussian as dg


@njit(cache=True)
def ll_one(y, h, nu):
    if nu == -1: # Gaussian measurement fixture; not production SV likelihood
        return -0.5*(y-h)**2 / 4.934802200544679
    if h < -700 or h > 700:
        return -np.inf
    if nu == 0:
        return -0.5*(h + y*y*math.exp(-h))
    return -0.5*h -0.5*(nu+1)*math.log1p(y*y*math.exp(-h)/(nu-2))

@njit(cache=True)
def normalize(logw):
    w=np.exp(logw-np.max(logw)); w/=np.sum(w)
    return w

@njit(cache=True)
def pick(w):
    value=np.random.random(); total=0.0
    for i in range(w.size):
        total+=w[i]
        if value<total: return i
    return w.size-1

@njit(cache=True)
def pgas(y, level, phi, eta, initial, particles, burn, kept, thin, nu, seed):
    np.random.seed(seed)
    length=y.size
    reference=initial.copy()
    paths=np.empty((length,particles)); ancestry=np.empty((length,particles),np.int64)
    logw=np.empty(particles); loga=np.empty(particles)
    draws=np.empty((kept,length)); terminal_ess=np.empty(kept)
    initial_sd=eta/math.sqrt(1-phi*phi)
    output=0
    for iteration in range(burn+kept*thin):
        for p in range(particles-1): paths[0,p]=level+initial_sd*np.random.normal()
        paths[0,particles-1]=reference[0]
        for p in range(particles): logw[p]=ll_one(y[0],paths[0,p],nu)
        weights=normalize(logw)
        for t in range(1,length):
            for p in range(particles-1):
                a=pick(weights); ancestry[t,p]=a
                paths[t,p]=level+phi*(paths[t-1,a]-level)+eta*np.random.normal()
            for p in range(particles):
                difference=reference[t]-level-phi*(paths[t-1,p]-level)
                loga[p]=math.log(max(weights[p],1e-300))-0.5*(difference/eta)**2
            ancestry[t,particles-1]=pick(normalize(loga))
            paths[t,particles-1]=reference[t]
            for p in range(particles): logw[p]=ll_one(y[t],paths[t,p],nu)
            weights=normalize(logw)
        index=pick(weights)
        for t in range(length-1,-1,-1):
            reference[t]=paths[t,index]
            if t>0: index=ancestry[t,index]
        if iteration>=burn and (iteration-burn)%thin==0:
            draws[output]=reference
            terminal_ess[output]=1/np.sum(weights*weights)
            output+=1
    return draws,terminal_ess

@njit(cache=True)
def total_ll(y,h,nu):
    result=0.0
    for t in range(y.size): result+=ll_one(y[t],h[t],nu)
    return result

@njit(cache=True)
def ess(y, level, phi, eta, initial, burn, kept, thin, nu, seed):
    np.random.seed(seed)
    h=initial.copy(); length=y.size
    prior=np.empty(length); proposal=np.empty(length)
    draws=np.empty((kept,length)); evaluations=np.empty(kept)
    output=0; current_ll=total_ll(y,h,nu)
    for iteration in range(burn+kept*thin):
        prior[0]=eta/math.sqrt(1-phi*phi)*np.random.normal()
        for t in range(1,length): prior[t]=phi*prior[t-1]+eta*np.random.normal()
        threshold=current_ll+math.log(max(np.random.random(),1e-300))
        angle=2*math.pi*np.random.random(); lower=angle-2*math.pi; upper=angle
        count=0
        while True:
            count+=1
            for t in range(length): proposal[t]=level+(h[t]-level)*math.cos(angle)+prior[t]*math.sin(angle)
            proposal_ll=total_ll(y,proposal,nu)
            if proposal_ll>threshold:
                h=proposal.copy(); current_ll=proposal_ll; break
            if angle<0: lower=angle
            else: upper=angle
            angle=lower+(upper-lower)*np.random.random()
            if count>1000: raise ValueError('ESS bracket failed')
        if iteration>=burn and (iteration-burn)%thin==0:
            draws[output]=h; evaluations[output]=count; output+=1
    return draws,evaluations

FIT_RECORDS=[]

@lru_cache(maxsize=4096)
def sampled_fit(data, method, particles, burn, kept, thin, nu):
    x=np.frombuffer(data,dtype=np.float64)
    started=time.perf_counter()
    original=bd._sv_kalman_rts_smoother_mean
    seed=int.from_bytes(hashlib.sha256(data).digest()[:4],'little')
    def smoother(y,level,phi,eta):
        loglik,initial,_variance=original(y,level,phi,eta)
        eps=(x-float(np.mean(x)))*100
        if method=='pgas': draws,diag=pgas(eps,level,phi,eta,initial,particles,burn,kept,thin,nu,seed)
        else: draws,diag=ess(eps,level,phi,eta,initial,burn,kept,thin,nu,seed)
        terminal=draws[:,-1]
        ac=float(np.corrcoef(terminal[:-1],terminal[1:])[0,1]) if np.std(terminal)>0 else 1.0
        FIT_RECORDS.append({'method': method,'particles': particles,'burn': burn,'kept': kept,'thin': thin,'nu': nu,'n': x.size,'terminal_lag1': ac,'diagnostic_mean': float(diag.mean()),'data_sha256': hashlib.sha256(data).hexdigest()})
        return loglik,draws.mean(axis=0),draws.var(axis=0)
    bd._sv_kalman_rts_smoother_mean=smoother
    try: result=bd.fit_bdes_fastmap(x,filtered_innovations=True,fixed_mean=True)
    finally: bd._sv_kalman_rts_smoother_mean=original
    FIT_RECORDS[-1]['fit_seconds']=time.perf_counter()-started
    return result

@dataclass(frozen=True)
class ComplexSV:
    model_id:str
    method:str='pgas'
    particles:int=32
    burn:int=64
    kept:int=64
    thin:int=1
    nu:float=0
    blend:float=1.0
    def simulate_daily_log_returns(self,training,context):
        training.validate(); past,future=_validate_calendar(training,context)
        assets=np.asarray(training.asset_log_returns,dtype=np.float64)
        sims,horizon=context.simulations,context.horizon_days
        if assets.shape[1]>1:
            dep=dg.fit_dynamic_gaussian_factor_model(assets)
            seed=bd.deterministic_seed('copula_alternatives',bd.FRONTIER_DEPENDENCE_ID,str(context.origin_date),horizon,sims)
            uniforms=dg.simulate_future_gaussian_uniforms(dep,sims,horizon,np.random.default_rng(seed))
        else:
            seed=bd.deterministic_seed('moment_sv_single_asset',str(context.origin_date),horizon,sims)
            uniforms=np.random.default_rng(seed).random((sims,horizon,1))
        marginals=np.empty_like(uniforms)
        for a in range(assets.shape[1]):
            fit=sampled_fit(assets[:,a].tobytes(),self.method,self.particles,self.burn,self.kept,self.thin,self.nu)
            values=sorted_moment_marginals(fit,sims,horizon)
            if self.blend!=1:
                baseline=bd.fit_bdes_fastmap(assets[:,a],filtered_innovations=True,fixed_mean=True)
                values=self.blend*values+(1-self.blend)*sorted_moment_marginals(baseline,sims,horizon)
            marginals[:,:,a]=values
        paths=dg.map_uniforms_to_marginal_paths(marginals,uniforms)
        dates=_historical_rebalance_dates(past.append(future),training.policy.rebalance)
        mask=np.asarray([d in dates for d in future])
        return dg.rebalanced_portfolio_log_paths(paths,training.policy.weights,mask)

CANDIDATES=(ComplexSV('pgas_gaussian_32_64_64'),ComplexSV('pgas_t8_32_64_64',nu=8),ComplexSV('ess_gaussian_256_256',method='ess',burn=256,kept=256),ComplexSV('pgas_gaussian_half_blend',blend=.5))
