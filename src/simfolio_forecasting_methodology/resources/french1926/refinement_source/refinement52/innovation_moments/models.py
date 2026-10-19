"""Analytic pre-clipping moment normalization of M261's existing node curve."""
import hashlib
import importlib.util
import os
from pathlib import Path
import sys
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
os.environ['SIMFOLIO_TIMESCALE_VARIANT'] = 'existing_sv_rate'
spec = importlib.util.spec_from_file_location('innovation_moment_variance_base', HERE.parent/'variance/study.py')
BASE = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = BASE
spec.loader.exec_module(BASE)
FAST, REF, FROZEN = BASE.FAST, BASE.REF, BASE.FROZEN
DIAGNOSTICS = {}


def continuous_moments(nodes):
    """Uniform-probability integral of the piecewise-linear node quantile."""
    x = np.asarray(nodes, dtype=np.float64)
    left, right = x[:-1], x[1:]
    mean = float(np.mean((left+right)*.5))
    second = float(np.mean((left*left+left*right+right*right)/3.))
    return mean, second-mean*mean


def normalize(nodes):
    x = np.asarray(nodes, dtype=np.float64)
    mean, variance = continuous_moments(x)
    if not np.isfinite(variance) or variance <= 0:
        raise ArithmeticError('degenerate continuous node variance')
    return (x-mean)/np.sqrt(variance)


def node_diagnostics(data, n, original_nodes):
    x = original_nodes(data, n)
    mean, variance = continuous_moments(x)
    corrected = normalize(x)
    cm, cv = continuous_moments(corrected)
    return corrected, {'training_sha256': hashlib.sha256(data).hexdigest(), 'nodes': n,
        'original_nodes_sha256': hashlib.sha256(x.tobytes()).hexdigest(),
        'original_discrete_mean': float(x.mean()), 'original_discrete_variance': float(x.var()),
        'original_continuous_mean': mean, 'original_continuous_variance': variance,
        'corrected_continuous_mean': cm, 'corrected_continuous_variance': cv,
        'corrected_nodes_sha256': hashlib.sha256(corrected.tobytes()).hexdigest(),
        'normalization': 'analytic uniform-probability moments before existing clipping',
        'conditional_copula_uniformity_claim': False}


def forecast(task, kind, corrected=True):
    body = REF.MODELS.parent.body
    original_nodes = body.learned.nodes

    def mapped_nodes(data, n):
        result, diagnostic = node_diagnostics(data, n, original_nodes)
        DIAGNOSTICS[hashlib.sha256(data).hexdigest(), n] = diagnostic
        return result

    if corrected:
        body.learned.nodes = mapped_nodes
    try:
        if kind == 'rough_off':
            result = FAST.Candidate().simulate_daily_log_returns(task.training, BASE.context(task))
            return result, dict(FAST.TIMINGS)
        if kind == 'deterministic_RMS':
            result, stages, _ = BASE.forecast(task, kind)
            return result, stages
        raise ValueError(kind)
    finally:
        body.learned.nodes = original_nodes


initialize = BASE.initialize
