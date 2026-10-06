"""M227 without percentile clipping of rough log-square observations.

Retain the existing near-zero finite-value floor. Student conventional fitting,
log-square moments, offset predictor and all forecast components stay unchanged.
"""
import hashlib
import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
spec = importlib.util.spec_from_file_location('unclipped_proxy_private_m227', ROOT / 'tools/untruncated_rough_priors/models.py')
parent = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = parent
spec.loader.exec_module(parent)
from causal_noise import causal_predictor


@lru_cache(maxsize=64)
def observed(data):
    x = np.frombuffer(data, np.float64)
    eps = (x - float(x.mean())) * 100.0
    squared = eps * eps
    positive = squared[np.isfinite(squared) & (squared > 0.)]
    floor = float(max(np.quantile(positive, .001) * .1, 1e-10)) if positive.size else 1e-10
    fit = parent.parent.predecessor_fit(data)
    bias, noise = parent.parent.parent.log_square_moments(fit['student_return_laplace_fit']['inverse_df'])
    gaussian_bias, _ = parent.parent.parent.log_square_moments(0.)
    y = np.log(np.maximum(squared, floor)) - parent.shell.bd.SV_LOG_CHI_SQUARE_MEAN
    y = y + gaussian_bias - bias
    level, phi, eta = fit['posterior_center'][:3]
    return y - causal_predictor(y, level, phi, eta, noise), eps


parent.backend.base.base.observed = observed
parent.backend.base.noise.observed = observed
parent.SOURCE = hashlib.sha256((parent.SOURCE + ''.join(hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(HERE.glob('*.py')))).encode()).hexdigest()


class Candidate(parent.Candidate):
    def simulate_daily_log_returns(self, training, context):
        result = super().simulate_daily_log_returns(training, context)
        for x in np.asarray(training.asset_log_returns, float).T:
            data = np.ascontiguousarray(x).tobytes()
            FIT_DIAGNOSTICS[self.model_id, hashlib.sha256(data).hexdigest()]['rough_observation_preprocessing'] = 'remove 0.5/99.5 percentile clipping; existing near-zero floor retained'
        return result


initialize = parent.initialize
CANDIDATES = (Candidate(model_id='asset_map_unclipped_log_proxy_untruncated_dynamic_rough'),)
MODEL_IDS = (parent.MODEL_IDS[0], CANDIDATES[0].model_id)
clear_path_cache = parent.clear_path_cache
TIMINGS = parent.TIMINGS
FIT_DIAGNOSTICS = parent.FIT_DIAGNOSTICS
shell = parent.shell
overlay = parent.overlay
load_paths = parent.load_paths
