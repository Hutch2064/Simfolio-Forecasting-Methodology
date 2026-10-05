"""Matched asset-level Bayesian SV: AR(1)/rough, Gaussian/standardized Student-t.

One volatility process, raw return likelihood, sampled parameters and history.
Student degrees of freedom are updated with scales marginalized, followed by
their exact Gamma conditional refresh before the corrected Gaussian block.
"""
from dataclasses import dataclass
from functools import lru_cache
import hashlib
import math
from pathlib import Path
import time

import numpy as np
from numba import njit
from scipy.special import ndtri, stdtrit

import mixture
import overlay
import standalone
from dynamic import configuration as rough_configuration, selected_kernel
from mixture_kernels import (exact_return_loglik, mixture_terms, measurement_geometry,
    marginalized_level, cached_simulation_smoother, gaussian_geometry, whiten)


def configuration(theta, kind, maximum_lag):
    if kind == 'rough':
        return rough_configuration(theta, maximum_lag)
    phi, eta = math.tanh(theta[0]), math.exp(theta[1])
    return np.array([phi]), np.ones(1), np.array([[eta*eta*(1-phi*phi)]])


def log_prior(theta, kind):
    if kind == 'rough':
        return overlay.log_prior(theta)
    phi = math.tanh(theta[0])
    if abs(phi) >= 1 or not math.log(.05) < theta[1] < math.log(3):
        return -np.inf
    # (phi+1)/2 ~ Beta(20,1.5), including the atanh(phi) Jacobian.
    return (19*math.log1p(phi)+.5*math.log1p(-phi)+math.log1p(-phi*phi)
            -.5*((theta[1]-math.log(.7))/1.5)**2)


def tail_prior(z):
    # nu-2 ~ Exponential(rate=.1), with the log transformation Jacobian.
    if z > 700 or 2+math.exp(z) == 2:
        return -np.inf
    return z-.1*math.exp(z)


@njit(cache=True, nogil=True)
def student_squared_loglik(squared, h, nu):
    constant = math.lgamma((nu+1)/2)-math.lgamma(nu/2)-.5*math.log((nu-2)*math.pi)
    total = 0.
    for t in range(h.size):
        if not np.isfinite(h[t]) or h[t] < -700:
            return -np.inf
        total += constant-.5*h[t]-.5*(nu+1)*math.log1p(squared[t]*math.exp(-h[t])/(nu-2))
    return total


@njit(cache=True, nogil=True)
def scale_rates(squared, h, nu):
    # Unit-variance t: e|h,tau ~ N(0, exp(h)*(nu-2)/(nu*tau)).
    result = np.empty(h.size)
    for t in range(h.size):
        result[t] = 2/(nu+squared[t]*math.exp(-h[t])*nu/(nu-2))
    return result


def parameter_chain(data, maximum_lag, kind, student, seed, burn, kept, simulations, resume=None):
    squared,y,active,center,lower,upper = mixture.observed(data,True)
    dimensions = 3 if kind == 'rough' else 2
    columns = dimensions+1+int(student)
    rng, reservoir_rng = np.random.default_rng(seed), np.random.default_rng(seed ^ 187931)
    if kind == 'rough':
        hyper = np.array([rng.uniform(.08,.15),math.log(1/63),math.log(.7)])
        root = np.diag([.01,.1,.05])
    else:
        hyper = np.array([math.atanh(math.exp(-1/63)),math.log(.7)])
        root = np.diag([.1,.05])
    theta = np.r_[hyper,center,math.log(8)] if student else np.r_[hyper,center]
    log_scale, tail_log_scale = 0., math.log(.15)
    prior_mass = standalone.normal_interval_logmass(center,4.,lower,upper)
    def geometry(point,noise,offset,observations):
        phi,_,covariance = configuration(point,kind,maximum_lag)
        q = np.diag(covariance).copy()
        pws,inverse_f,log_f = measurement_geometry(phi,q,noise)
        likelihood,mean,sd = marginalized_level(observations-offset-center,phi,pws,inverse_f,log_f,16.)
        mean += center
        likelihood += standalone.normal_interval_logmass(mean,sd,lower,upper)-prior_mass
        return log_prior(point,kind)+likelihood,phi,q,pws,inverse_f,mean,sd
    if resume:
        theta,root,h = resume['theta'].copy(),resume['root'].copy(),resume['h'].copy()
        log_scale,tail_log_scale = resume['log_scale'],resume['tail_log_scale']
        rng.bit_generator.state = resume['rng_state']
        reservoir_rng.bit_generator.state = resume['reservoir_rng_state']
        reservoir,seen = resume['reservoir'],resume['seen']
    else:
        h = np.full(len(y),center)
        reservoir = {'parameters':np.empty((simulations,columns)), 'histories':np.empty((simulations,len(y)))}
        seen = 0
    mean,m2 = np.zeros(dimensions),np.zeros((dimensions,dimensions))
    draws = np.empty((kept,columns+5))
    counts = {k:resume[k] if resume else 0 for k in ('latent_accepts','parameter_accepts','tail_accepts')}
    for iteration in range(burn+kept):
        observations = y
        if student:
            proposed_z = theta[-1]+math.exp(tail_log_scale)*rng.normal()
            current_tail = student_squared_loglik(squared,h,2+math.exp(theta[-1]))+tail_prior(theta[-1])
            prior = tail_prior(proposed_z)
            next_tail = (student_squared_loglik(squared,h,2+math.exp(proposed_z))+prior
                         if np.isfinite(prior) else -np.inf)
            accepted_tail = math.log(rng.random()) < next_tail-current_tail
            if accepted_tail:
                theta[-1] = proposed_z
                counts['tail_accepts'] += 1
            if iteration < burn:
                tail_log_scale += (iteration+10)**(-.6)*(float(accepted_tail)-.44)
            nu = 2+math.exp(theta[-1])
            tau = rng.gamma((nu+1)/2,scale_rates(squared,h,nu))
            observations = y+np.log(tau)-math.log((nu-2)/nu)
        offset,r,current_ratio = mixture_terms(observations,h,active,rng.random(len(y)))
        current = geometry(theta[:dimensions],r,offset,observations)
        proposed = theta[:dimensions]+math.exp(log_scale)*(root@rng.normal(size=dimensions))
        candidate = geometry(proposed,r,offset,observations) if np.isfinite(log_prior(proposed,kind)) else None
        accepted_parameter = math.log(rng.random()) < ((candidate[0] if candidate else -np.inf)-current[0])
        chosen = candidate if accepted_parameter else current
        _,phi,q,pws,inverse_f,level_mean,level_sd = chosen
        next_level = standalone.truncated_normal(level_mean,level_sd,lower,upper,rng.random())
        proposed_h = next_level+cached_simulation_smoother(observations-offset-next_level,phi,q,r,
            pws,inverse_f,rng.normal(size=(len(y),len(phi))),rng.normal(size=len(y)))
        proposal_ratio = mixture_terms(observations,proposed_h,active,np.zeros(len(y)))[2]
        accepted_joint = math.log(rng.random()) < proposal_ratio-current_ratio
        moved = accepted_joint and accepted_parameter
        if accepted_joint:
            theta[:dimensions] = proposed if accepted_parameter else theta[:dimensions]
            theta[dimensions],h = next_level,proposed_h
            counts['latent_accepts'] += 1
            counts['parameter_accepts'] += int(moved)
        level = theta[dimensions]
        likelihood = (student_squared_loglik(squared,h,2+math.exp(theta[-1])) if student
                      else exact_return_loglik(squared,h))
        value = log_prior(theta[:dimensions],kind)+likelihood-.5*((level-center)/4)**2
        if student:
            value += tail_prior(theta[-1])
        if iteration < burn:
            log_scale += (iteration+10)**(-.6)*(float(moved)-.234)
            delta = theta[:dimensions]-mean
            mean += delta/(iteration+1)
            m2 += np.outer(delta,theta[:dimensions]-mean)
            if iteration >= 63 and (iteration+1)%32 == 0:
                root = np.linalg.cholesky(2.38**2/dimensions*(m2/iteration+np.eye(dimensions)*1e-6))
        else:
            eta = math.exp(theta[dimensions-1])
            draws[iteration-burn,:columns] = theta
            draws[iteration-burn,columns:] = (value,float(h.mean()),float(h[-1]),
                math.exp(level+.5*eta*eta),math.exp(h[-1]))
            index = seen if seen < simulations else int(reservoir_rng.integers(seen+1))
            if index < simulations:
                reservoir['parameters'][index],reservoir['histories'][index] = theta,h
            seen += 1
    return draws, dict(theta=theta.copy(),root=root.copy(),h=h.copy(),log_scale=log_scale,
        tail_log_scale=tail_log_scale,rng_state=rng.bit_generator.state,
        reservoir_rng_state=reservoir_rng.bit_generator.state,reservoir=reservoir,seen=seen,
        iterations=burn+kept+(resume['iterations'] if resume else 0),**counts)


@lru_cache(maxsize=16)
def fit(data,maximum_lag,kind,student,simulations,burn,kept,maximum):
    import models as shell
    identity = hashlib.sha256(data).hexdigest()
    settings = f'coherent-v1:{maximum_lag}:{kind}:{student}:{simulations}:{burn}:{kept}:{maximum}'
    source = hashlib.sha256(b''.join(Path(__file__).with_name(name).read_bytes() for name in
        ('coherent.py','mixture.py','mixture_kernels.py','dynamic.py','overlay.py','standalone.py'))
        + Path(shell.bd.__file__).read_bytes()).hexdigest()
    contract = source+':'+settings
    key = hashlib.sha256((identity+contract).encode()).hexdigest()
    def build():
        started = time.perf_counter()
        chains = [parameter_chain(data,maximum_lag,kind,student,
            shell.bd.deterministic_seed('coherent_sv_fit',identity,settings,j),burn,kept,simulations) for j in range(2)]
        while True:
            trace = np.array([c[0] for c in chains])
            diagnostics,precision = shell.diagnostics(trace),mixture.posterior_precision(trace)
            converged = all(d['rank_split_rhat'] < 1.01 and d['bulk_ess'] >= 400 for d in diagnostics) and max(precision) <= .1
            actual = trace.shape[1]
            if converged or actual >= maximum:
                break
            extension = min(actual,maximum-actual)
            for j,(draws,state) in enumerate(chains):
                extra,state = parameter_chain(data,maximum_lag,kind,student,0,0,extension,simulations,state)
                chains[j] = np.concatenate([draws,extra]),state
        rng = np.random.default_rng(shell.bd.deterministic_seed('coherent_sv_prediction_draws',identity,settings))
        replacement = 2*actual < simulations
        first = int(rng.binomial(simulations,.5) if replacement else rng.hypergeometric(actual,actual,simulations))
        parameters,histories = [],[]
        for count,(_,state) in zip((first,simulations-first),chains):
            indices = rng.choice(min(actual,simulations),size=count,replace=replacement)
            parameters.extend(state['reservoir']['parameters'][indices])
            histories.extend(state['reservoir']['histories'][indices])
        order = rng.permutation(simulations)
        parameters,histories = np.asarray(parameters)[order],np.asarray(histories)[order]
        dimensions = 3 if kind == 'rough' else 2
        prepared,records = [],[]
        for theta,h in zip(parameters,histories):
            phi,weights,covariance = configuration(theta[:dimensions],kind,maximum_lag)
            q = np.diag(covariance).copy()
            gains,sd,p = gaussian_geometry(phi,q,len(h))
            level = theta[dimensions]
            _,state = whiten(h,phi,gains,sd,level)
            values,vectors = np.linalg.eigh((p+p.T)/2)
            if values.min() < -1e-10:
                raise ValueError('invalid conditional volatility covariance')
            initial_root = vectors*np.sqrt(np.maximum(values,0))
            prepared.append((phi,weights,state,initial_root,np.sqrt(q),level))
            if kind == 'rough':
                records.append(selected_kernel(float(theta[0]),math.exp(theta[1]),maximum_lag)[2])
        seconds = time.perf_counter()-started
        names = (['H','log_kappa','log_eta'] if kind == 'rough' else ['atanh_phi','log_stationary_eta'])+['level']
        if student:
            names.append('log_nu_minus_two')
        names += ['log_likelihood_plus_parameter_prior','latent_mean','latent_terminal','stationary_variance','terminal_variance']
        kernel = ({'selection':'per_parameter_dynamic_positive_integer_order',
            'minimum_factors':min(r['factors'] for r in records),'maximum_factors':max(r['factors'] for r in records),
            'maximum_autocorrelation_error_upper_bound':max(r['autocorrelation_error_upper_bound'] for r in records),
            'tolerance':.001,'maximum_daily_lag':maximum_lag} if kind == 'rough' else
            {'selection':'structural_AR1_log_variance','factors':1,'numerical_lift_required':False})
        return dict(parameters=parameters,prepared=prepared,trace=trace,diagnostics=diagnostics,
            diagnostic_columns=names,convergence_flag=converged,relative_95_percent_mc_halfwidth=precision,
            kept_per_chain=actual,fit_seconds=seconds,ess_per_second=min(d['bulk_ess'] for d in diagnostics)/seconds,
            data_sha256=identity,contract=contract,kernel=kernel,
            sampler_acceptance=[{k:s[k]/s['iterations'] for k in ('latent_accepts','parameter_accepts','tail_accepts')} for _,s in chains])
    return shell.controls.cache('coherent_bayesian_sv_asset',key,build)


@dataclass(frozen=True)
class CoherentCandidate:
    model_id: str
    kind: str = 'ar1'
    student: bool = False
    burn: int = 2048
    kept: int = 256
    max_kept: int = 65536

    def simulate_daily_log_returns(self,training,context):
        import models as shell
        training.validate()
        past,future = shell._validate_calendar(training,context)
        assets = np.asarray(training.asset_log_returns,np.float64)
        sims,horizon = context.simulations,context.horizon_days
        maximum_lag = len(assets)+horizon-1
        data = tuple(assets[:,a].tobytes() for a in range(assets.shape[1]))
        started = time.perf_counter()
        posteriors = [fit(x,maximum_lag,self.kind,self.student,sims,self.burn,self.kept,self.max_kept) for x in data]
        shell.controls.timed('coherent_sv_fit_phase_wall',started)
        seed = shell.bd.deterministic_seed('copula_alternatives',shell.bd.FRONTIER_DEPENDENCE_ID,
            str(context.origin_date),horizon,sims)
        paths = shell.controls.gaussian_uniforms(assets.shape,assets.tobytes(),sims,horizon,seed).copy()
        for a,posterior in enumerate(posteriors):
            shell.FIT_DIAGNOSTICS[(self.model_id,posterior['data_sha256'])] = {name:posterior[name] for name in
                ('data_sha256','diagnostics','diagnostic_columns','convergence_flag','kept_per_chain','fit_seconds',
                 'ess_per_second','kernel','sampler_acceptance','relative_95_percent_mc_halfwidth')}
            shell.FIT_DIAGNOSTICS[(self.model_id,posterior['data_sha256'])].update(model_id=self.model_id,burn_per_chain=self.burn)
            started = time.perf_counter()
            rng = np.random.default_rng(shell.bd.deterministic_seed('coherent_sv_paths',
                posterior['data_sha256'],str(context.origin_date),horizon,sims,self.kind,self.student))
            uniforms = np.clip(paths[:,:,a],1e-12,1-1e-12)
            if self.student:
                nu = 2+np.exp(posterior['parameters'][:,-1])
                shocks = stdtrit(nu[:,None],uniforms)*np.sqrt((nu[:,None]-2)/nu[:,None])
            else:
                shocks = ndtri(uniforms)
            mean = float(assets[:,a].mean())
            for s,(phi,weights,state,initial_root,innovation_sd,level) in enumerate(posterior['prepared']):
                initial = state+initial_root@rng.normal(size=phi.size)
                volatility = standalone.volatility_path(phi,weights,innovation_sd,initial,
                    rng.normal(size=(horizon,phi.size)),level)
                paths[s,:,a] = np.clip(mean+volatility*shocks[s],-1,1)
            shell.controls.timed('coherent_sv_predictive_paths',started)
        dates = shell._historical_rebalance_dates(past.append(future),training.policy.rebalance)
        mask = np.asarray([date in dates for date in future])
        return shell.rejoin(paths,training.policy.weights,mask)
