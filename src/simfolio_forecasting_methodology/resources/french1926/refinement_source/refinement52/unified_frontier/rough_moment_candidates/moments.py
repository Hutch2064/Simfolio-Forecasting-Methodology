"""Deterministic retained rough fit/filter/conditional-moment preparation only."""
import hashlib
import time
import numpy as np


def sha(x):return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()


def json_safe(value):
    if isinstance(value,dict):return {str(k):json_safe(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [json_safe(v) for v in value]
    if isinstance(value,np.ndarray):return json_safe(value.tolist())
    if isinstance(value,(float,np.floating)):
        return float(value) if np.isfinite(value) else str(float(value))
    if isinstance(value,np.generic):return value.item()
    return value


def summary(curve):
    x=np.asarray(curve,dtype=np.float64)
    return {'sha256':sha(x),'first':json_safe(x[0]),'terminal':json_safe(x[-1]),
        'min':json_safe(x.min()),'max':json_safe(x.max())}


def prepare(data,lag,horizon,refs):
    """Call captured original functions, never a future rough RNG/path kernel."""
    stages={};tick=time.perf_counter()
    fit=refs['fit'](data,lag,refs['fit_seconds'])
    stages['rough_fit_call_seconds']=time.perf_counter()-tick
    theta=np.ascontiguousarray(fit['map'],dtype=np.float64)
    if theta.shape!=(3,) or not np.isfinite(theta).all():raise ArithmeticError('invalid retained rough MAP parameters')
    status=fit.get('optimizer',{}).get('success')
    if status is not None and not bool(status):raise ArithmeticError('retained rough optimizer reports failure')
    tick=time.perf_counter()
    phi,w,root,terminal,initial_root,original_mean,original_variance=refs['prepared'](data,theta.tobytes(),lag,horizon)
    stages['rough_filter_preparation_call_seconds']=time.perf_counter()-tick
    tick=time.perf_counter()
    # EXACT conditional_normalization.py first-step convention/recurrence.
    posterior=initial_root@initial_root.T
    next_p=posterior*phi[:,None]*phi[None,:]+root@root.T
    mean,variance=refs['normalizer'](phi,w,root,next_p,phi*terminal,horizon)
    if mean.shape!=(horizon,) or variance.shape!=(horizon,) or not np.isfinite(mean).all() or not np.isfinite(variance).all():
        raise ArithmeticError('invalid retained conditional rough moments')
    # Inherited numerical roundoff tolerance, not a fitted cutoff.
    if variance.min() < -1e-12:raise ArithmeticError('negative conditional rough variance')
    negative_count=int(np.sum(variance<0));variance=np.maximum(variance,0.)
    stages['rough_conditional_normalizer_seconds']=time.perf_counter()-tick
    original_mean=np.asarray(original_mean);original_variance=np.asarray(original_variance)
    if original_mean.shape!=(horizon,) or original_variance.shape!=(horizon,) or not np.isfinite(original_mean).all() or not np.isfinite(original_variance).all():
        raise ArithmeticError('invalid original rough normalizer')
    tick=time.perf_counter()
    log_kappa=mean-original_mean+.5*(variance-original_variance)
    with np.errstate(over='ignore',under='ignore'):kappa=np.exp(log_kappa)
    d_rough=np.exp(-variance/8.)
    stages['rough_attenuation_and_diagnostic_seconds']=time.perf_counter()-tick
    record={'training_sha256':hashlib.sha256(data).hexdigest(),'lag':int(lag),'horizon':int(horizon),
        'fit_seconds_contract':float(refs['fit_seconds']),'MAP_parameters':theta.tolist(),'MAP_parameter_sha256':sha(theta),
        'fit_status':json_safe({k:v for k,v in fit.items() if k not in ('map','points','weights')}),
        'retained_function_provenance':refs.get('provenance',{}),
        'state_dimension':len(phi),'transition_phi_sha256':sha(phi),'loading_sha256':sha(w),'innovation_root_sha256':sha(root),
        'terminal_posterior_mean_sha256':sha(terminal),'terminal_posterior_covariance_sha256':sha(posterior),
        'terminal_posterior_root_sha256':sha(initial_root),
        'conditional_mean':summary(mean),'conditional_variance':summary(variance),
        'original_normalizer_mean':summary(original_mean),'original_normalizer_variance':summary(original_variance),
        'production_log_kappa':summary(log_kappa),'production_kappa':summary(kappa),
        'kappa_exponential_overflow_count':int(np.sum(np.isinf(kappa))),
        'kappa_exponential_underflow_count':int(np.sum(kappa==0.)),
        'rough_expected_multiplier':summary(d_rough),'negative_roundoff_variance_count':negative_count,
        'kappa_scope':'E[original rough multiplier squared | fixed training fit], before return clipping',
        'future_rough_states_drawn':False,'parameter_uncertainty_added':False}
    return d_rough,record,stages
