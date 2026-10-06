"""Exact Gaussian-return rough SV via corrected Omori mixture proposals.

Gaussian collapse integrates the rough states during parameter proposals.
A joint parameter/history proposal is corrected with the exact log-chi-square
/ mixture likelihood ratio. The scalar volatility history has one element
per observation even when the numerical lift changes dimension. Mixture
indicators are auxiliary; no winsorization is applied.
"""
import hashlib
import math
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
import overlay
import standalone
from dynamic import configuration, selected_kernel
from mixture_kernels import (
    cached_simulation_smoother,
    exact_return_loglik,
    gaussian_geometry,
    marginal_likelihood,
    marginalized_level,
    measurement_geometry,
    mixture_terms,
    whiten,
)
from scipy.special import ndtri
from standalone import normal_interval_logmass, truncated_normal


def observed(data, free_level=True):
    x = np.frombuffer(data,np.float64)
    residual = 100*(x-x.mean())
    squared = residual*residual
    active = squared > 0
    log_squared = np.zeros(len(x))
    log_squared[active] = np.log(squared[active])
    # Zero returns stay in the exact raw-return likelihood, not an offset log proxy.
    proxy = log_squared[active]+1.2703628454614782
    center = float(proxy.mean()) if proxy.size else 0.
    lower,upper = np.quantile(proxy,[.01,.99])+np.array([-4,4]) if proxy.size else (-4.,4.)
    if not free_level:
        import models as shell
        center = float(shell.bd._sv_observed_log_variance(x,float(x.mean()))[0].mean())
    return squared,log_squared,active,center,float(lower),float(upper)


def parameter_chain(data, maximum_lag, free_level, seed, burn, kept, simulations, resume=None):
    """Corrected joint theta/history proposals, integrating states and level.

    Given indicators s, parameter MH is reversible for the Gaussian-mixture
    marginal posterior. Independently refreshing level and h conditional on
    theta makes a reversible joint proposal. Its final density-ratio correction
    targets the exact Gaussian-return law. The scalar history dimension is T.
    """
    squared,y,active,center,lower,upper = observed(data,free_level)
    columns = 4 if free_level else 3
    rng = np.random.default_rng(seed)
    reservoir_rng = np.random.default_rng(seed ^ 187931)
    theta = np.array([rng.uniform(.08,.15),math.log(1/63),math.log(.7),center])[:columns]
    root = np.diag([.01,.1,.05])
    log_scale = 0.
    prior_level_mass = normal_interval_logmass(center,4.,lower,upper) if free_level else 0.
    def geometry(point,noise,offset):
        phi,_weights,covariance = configuration(point[:3],maximum_lag)
        q = np.diag(covariance).copy()
        pws,inverse_f,log_f = measurement_geometry(phi,q,noise)
        if free_level:
            likelihood,mean,sd = marginalized_level(y-offset-center,phi,pws,inverse_f,log_f,16.)
            mean += center
            likelihood += normal_interval_logmass(mean,sd,lower,upper)-prior_level_mass
        else:
            likelihood = marginal_likelihood(y-offset-center,phi,pws,inverse_f,log_f)
            mean,sd = center,0.
        return overlay.log_prior(point[:3])+likelihood,phi,q,pws,inverse_f,mean,sd
    if resume:
        theta,root,h = resume['theta'].copy(),resume['root'].copy(),resume['h'].copy()
        log_scale = resume['log_scale']
        rng.bit_generator.state = resume['rng_state']
        reservoir_rng.bit_generator.state = resume['reservoir_rng_state']
        reservoir,seen = resume['reservoir'],resume['seen']
    else:
        h = np.full(len(y),center)
        reservoir = {'parameters':np.empty((simulations,columns)), 'histories':np.empty((simulations,len(y)))}
        seen = 0
    mean,m2 = np.zeros(3),np.zeros((3,3))
    draws = np.empty((kept,columns+5))
    latent_accepts,parameter_accepts = (resume['latent_accepts'],resume['parameter_accepts']) if resume else (0,0)
    for iteration in range(burn+kept):
        offset,r,current_ratio = mixture_terms(y,h,active,rng.random(len(y)))
        current_geometry = geometry(theta,r,offset)
        proposed = theta[:3]+math.exp(log_scale)*(root@rng.normal(size=3))
        proposed_geometry = geometry(proposed,r,offset) if np.isfinite(overlay.log_prior(proposed)) else None
        accepted_parameter = math.log(rng.random()) < ((proposed_geometry[0] if proposed_geometry else -np.inf)-current_geometry[0])
        chosen = proposed_geometry if accepted_parameter else current_geometry
        _,phi,q,pws,inverse_f,level_mean,level_sd = chosen
        next_level = truncated_normal(level_mean,level_sd,lower,upper,rng.random()) if free_level else center
        next_theta = np.r_[proposed if accepted_parameter else theta[:3],next_level] if free_level else (proposed if accepted_parameter else theta)
        proposed_h = next_level+cached_simulation_smoother(y-offset-next_level,phi,q,r,
            pws,inverse_f,rng.normal(size=(len(y),len(phi))),rng.normal(size=len(y)))
        proposal_ratio = mixture_terms(y,proposed_h,active,np.zeros(len(y)))[2]
        accepted_joint = math.log(rng.random()) < proposal_ratio-current_ratio
        effective_parameter_move = accepted_joint and accepted_parameter
        if accepted_joint:
            theta,h = next_theta,proposed_h
            latent_accepts += 1
            parameter_accepts += int(effective_parameter_move)
        level = theta[3] if free_level else center
        value = overlay.log_prior(theta[:3])+exact_return_loglik(squared,h)
        if free_level:
            value -= .5*((level-center)/4)**2
        if iteration < burn:
            # Diminishing adaptation occurs only in warmup and is then frozen.
            log_scale += (iteration+10)**(-.6)*(float(effective_parameter_move)-.234)
            delta = theta[:3]-mean
            mean += delta/(iteration+1)
            m2 += np.outer(delta,theta[:3]-mean)
            if iteration >= 63 and (iteration+1)%32 == 0:
                root = np.linalg.cholesky(2.38**2/3*(m2/iteration+np.eye(3)*1e-6))
        else:
            draws[iteration-burn,:columns] = theta
            draws[iteration-burn,columns:] = (value,float(h.mean()),float(h[-1]),
                math.exp(level+.5*math.exp(2*theta[2])),math.exp(h[-1]))
            index = seen if seen < simulations else int(reservoir_rng.integers(seen+1))
            if index < simulations:
                reservoir['parameters'][index] = theta
                reservoir['histories'][index] = h
            seen += 1
    return draws, {'theta':theta.copy(),'root':root.copy(),'h':h.copy(),'log_scale':log_scale,
        'rng_state':rng.bit_generator.state,'reservoir_rng_state':reservoir_rng.bit_generator.state,
        'reservoir':reservoir,'seen':seen,'latent_accepts':latent_accepts,
        'parameter_accepts':parameter_accepts,'iterations':burn+kept+(resume['iterations'] if resume else 0)}


def posterior_precision(trace):
    """Non-overlapping batch-means MCSE; halfwidth relative to posterior SD.

    Batch length grows as sqrt(chain length), as in consistent batch means.
    The predeclared .1 halfwidth budget is an accuracy policy, not a paper-
    mandated constant. Cross-chain Rhat and ESS are additional requirements.
    """
    chains,length,columns = trace.shape
    batch = int(math.sqrt(length))
    number = length//batch
    means = trace[:,:number*batch].reshape(chains,number,batch,columns).mean(axis=2)
    mc_variance = batch*np.var(means,axis=1,ddof=1).sum(axis=0)/(chains*chains*length)
    sd = np.std(trace.reshape(-1,columns),axis=0,ddof=1)
    return (1.96*np.sqrt(mc_variance)/np.maximum(sd,np.finfo(float).tiny)).tolist()


@lru_cache(maxsize=16)
def fit(data, maximum_lag, free_level, simulations, burn, kept, maximum):
    import models as shell
    identity = hashlib.sha256(data).hexdigest()
    sampler = 'exact-mixture-collapsed-level-v2' if free_level else 'exact-mixture-collapsed-v1'
    settings = f'{sampler}:{maximum_lag}:{free_level}:{simulations}:{burn}:{kept}:{maximum}'
    source = hashlib.sha256(b''.join(Path(__file__).with_name(name).read_bytes() for name in
        ('mixture.py','mixture_kernels.py','dynamic.py','overlay.py','standalone.py'))
        + Path(shell.bd.__file__).read_bytes()).hexdigest()
    contract = source+':'+settings
    key = hashlib.sha256((identity+contract).encode()).hexdigest()
    def build():
        started = time.perf_counter()
        chains = [parameter_chain(data,maximum_lag,free_level,
            shell.bd.deterministic_seed('rough_exact_mixture_fit',identity,settings,j),burn,kept,simulations)
            for j in range(2)]
        while True:
            trace = np.array([c[0] for c in chains])
            diagnostics = shell.diagnostics(trace)
            precision = posterior_precision(trace)
            converged = (all(d['rank_split_rhat'] < 1.01 and d['bulk_ess'] >= 400 for d in diagnostics)
                         and max(precision) <= .1)
            actual = trace.shape[1]
            if converged or actual >= maximum:
                break
            extension = min(actual,maximum-actual)
            for j,(draws,state) in enumerate(chains):
                extra,state = parameter_chain(data,maximum_lag,free_level,0,0,extension,simulations,state)
                chains[j] = (np.concatenate([draws,extra]),state)
        rng = np.random.default_rng(shell.bd.deterministic_seed('rough_exact_mixture_joint_draws',identity,settings))
        replacement = 2*actual < simulations
        number_first = int(rng.binomial(simulations,.5) if replacement else rng.hypergeometric(actual,actual,simulations))
        parameters,histories = [],[]
        for number,(_,state) in zip((number_first,simulations-number_first),chains):
            indices = rng.choice(min(actual,simulations),size=number,replace=replacement)
            parameters.extend(state['reservoir']['parameters'][indices])
            histories.extend(state['reservoir']['histories'][indices])
        order = rng.permutation(simulations)
        parameters,histories = np.asarray(parameters)[order],np.asarray(histories)[order]
        center = observed(data,free_level)[3]
        prepared,records = [],[]
        for theta,h in zip(parameters,histories):
            phi,weights,covariance = configuration(theta[:3],maximum_lag)
            q = np.diag(covariance).copy()
            gains,sd,p = gaussian_geometry(phi,q,len(h))
            level = theta[3] if free_level else center
            _,state = whiten(h,phi,gains,sd,level)
            values,vectors = np.linalg.eigh((p+p.T)/2)
            if values.min() < -1e-10:
                raise ValueError('invalid conditional rough covariance')
            initial_root = vectors*np.sqrt(np.maximum(values,0))
            root = np.diag(np.sqrt(q))
            next_p = p*phi[:,None]*phi[None,:]+covariance
            prepared.append((phi,weights,state,initial_root,root,next_p,phi*state,level))
            records.append(selected_kernel(float(theta[0]),math.exp(theta[1]),maximum_lag)[2])
        seconds = time.perf_counter()-started
        names = ['H','log_kappa','log_eta']+(['level'] if free_level else [])+['log_likelihood_plus_parameter_prior','latent_mean','latent_terminal','stationary_variance','terminal_variance']
        return {'parameters':parameters,'prepared':prepared,'trace':trace,
            'diagnostics':diagnostics,'diagnostic_columns':names,'convergence_flag':converged,
            'relative_95_percent_mc_halfwidth':precision,
            'kept_per_chain':actual,'fit_seconds':seconds,
            'ess_per_second':min(d['bulk_ess'] for d in diagnostics)/seconds,
            'data_sha256':identity,'contract':contract,
            'sampler_acceptance':[{k:s[k]/(s['iterations']) for k in ('latent_accepts','parameter_accepts')} for _,s in chains],
            'kernel':{'selection':'per_parameter_dynamic_positive_integer_order',
                'minimum_factors':min(r['factors'] for r in records),'maximum_factors':max(r['factors'] for r in records),
                'maximum_autocorrelation_error_upper_bound':max(r['autocorrelation_error_upper_bound'] for r in records),
                'tolerance':.001,'maximum_daily_lag':maximum_lag}}
    return shell.controls.cache('bayesian_exact_mixture_asset',key,build)


def _fit_asset(arguments):
    return fit(*arguments)


@lru_cache(maxsize=2)
def fitted_assets(data, maximum_lag, free_level, simulations, burn, kept, maximum, workers, cache_root):
    arguments = [(x,maximum_lag,free_level,simulations,burn,kept,maximum) for x in data]
    if workers == 1:
        return tuple(_fit_asset(a) for a in arguments)
    with ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'),
            initializer=overlay._initialize_asset_fit_worker,initargs=(cache_root,)) as pool:
        return tuple(pool.map(_fit_asset,arguments))


@dataclass(frozen=True)
class MixtureCandidate:
    model_id: str
    standalone: bool = False
    burn: int = 2048
    kept: int = 256
    max_kept: int = 65536

    def simulate_daily_log_returns(self, training, context):
        import models as shell
        training.validate()
        past,future = shell._validate_calendar(training,context)
        assets = np.asarray(training.asset_log_returns,np.float64)
        sims,horizon = context.simulations,context.horizon_days
        data = tuple(assets[:,a].tobytes() for a in range(assets.shape[1]))
        workers = min(assets.shape[1],max(1,shell.controls.STATE_WORKERS))
        posteriors = fitted_assets(data,len(assets)+horizon-1,self.standalone,sims,
            self.burn,self.kept,self.max_kept,workers,shell.controls.CACHE_ROOT)
        seed = shell.bd.deterministic_seed('copula_alternatives',shell.bd.FRONTIER_DEPENDENCE_ID,
            str(context.origin_date),horizon,sims)
        uniforms = shell.controls.gaussian_uniforms(assets.shape,assets.tobytes(),sims,horizon,seed).copy()
        for a,posterior in enumerate(posteriors):
            shell.FIT_DIAGNOSTICS[(self.model_id,posterior['data_sha256'])] = {
                name:posterior[name] for name in ('data_sha256','diagnostics','diagnostic_columns',
                    'convergence_flag','kept_per_chain','fit_seconds','ess_per_second','kernel','sampler_acceptance','relative_95_percent_mc_halfwidth')}
            shell.FIT_DIAGNOSTICS[(self.model_id,posterior['data_sha256'])].update(model_id=self.model_id,burn_per_chain=self.burn)
            rng = np.random.default_rng(shell.bd.deterministic_seed('rough_exact_mixture_prediction',
                posterior['data_sha256'],str(context.origin_date),horizon,sims))
            if self.standalone:
                shocks = ndtri(np.clip(uniforms[:,:,a],1e-12,1-1e-12))
                mean = float(assets[:,a].mean())
            else:
                production = shell.controls.asset_fit(data[a])
                mean,sd = shell.opt.mixture_curves(production,horizon)
                shocks = uniforms[:,:,a].copy()
                shell.controls.interpolate_nodes(shocks,shell.controls.asset_nodes(data[a],sims,False))
            for s,(phi,weights,state,initial_root,root,p,mean_state,level) in enumerate(posterior['prepared']):
                initial = state+initial_root@rng.normal(size=phi.size)
                if self.standalone:
                    volatility = standalone.volatility_path(phi,weights,np.diag(root),initial,
                        rng.normal(size=(horizon,phi.size)),level)
                    uniforms[s,:,a] = np.clip(mean+volatility*shocks[s],-1,1)
                else:
                    initial = phi*initial+root@rng.normal(size=phi.size)
                    path_normals = np.empty((horizon+1,phi.size))
                    path_normals[0] = mean_state
                    path_normals[1:] = rng.normal(size=(horizon,phi.size))
                    means,variances = overlay.path_normalizers(phi,weights,root,p,mean_state,horizon)
                    multiplier = overlay.multiplier_independent_prepared(phi,weights,root,initial,path_normals,means,variances)
                    uniforms[s,:,a] = np.clip(mean+sd*shocks[s]*multiplier,-1,1)
        dates = shell._historical_rebalance_dates(past.append(future),training.policy.rebalance)
        mask = np.asarray([date in dates for date in future])
        return shell.rejoin(uniforms,training.policy.weights,mask)
