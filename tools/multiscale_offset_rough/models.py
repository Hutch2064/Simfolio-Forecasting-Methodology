"""M227 with rough observations offset by the existing multiscale predictor.

Gaussian proxy innovations, conditional on existing fitted parameters; no
additional parameter or state law. Original return mean and future laws remain.
"""
import hashlib
import importlib.util
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE))
from ms_causal import predict

spec = importlib.util.spec_from_file_location('multiscale_offset_private_m227', ROOT / 'tools/untruncated_rough_priors/models.py')
parent = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = parent
spec.loader.exec_module(parent)


@lru_cache(maxsize=64)
def observed(data):
    x = np.frombuffer(data, np.float64)
    y, eps = parent.shell.bd._sv_observed_log_variance(x, float(x.mean()))
    fit = parent.parent.predecessor_fit(data)
    mean, variance = parent.parent.parent.log_square_moments(fit['student_return_laplace_fit']['inverse_df'])
    gaussian_mean, _ = parent.parent.parent.log_square_moments(0.)
    y = y + gaussian_mean - mean
    phi, loading, common, independent, _, level, _, _ = parent.parent.multiscale_configuration(data, 1)
    covariance = np.outer(common, common) + np.outer(independent, independent)
    return y - predict(y, level, phi, loading, covariance, variance), eps


parent.backend.base.base.observed = observed
parent.backend.base.noise.observed = observed
parent.SOURCE = hashlib.sha256((parent.SOURCE + ''.join(hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(HERE.glob('*.py')))).encode()).hexdigest()


class Candidate(parent.Candidate):
    def simulate_daily_log_returns(self, training, context):
        result = super().simulate_daily_log_returns(training, context)
        for x in np.asarray(training.asset_log_returns, float).T:
            data = np.ascontiguousarray(x).tobytes()
            FIT_DIAGNOSTICS[self.model_id, hashlib.sha256(data).hexdigest()]['rough_observation_offset'] = 'one-step Gaussian prediction of existing four multiscale states and residual AR1; fitted Student log-square observation variance'
        return result


def initialize(cache_root=None, vine_threads=1, state_workers=1):
    parent.initialize(cache_root, vine_threads, state_workers)
    predict(np.zeros(3), 0., np.array([.8]), np.ones(1), np.array([[.1]]), 5.)


CANDIDATES = (Candidate(model_id='asset_map_multiscale_predictor_offset_untruncated_dynamic_rough'),)
MODEL_IDS = (parent.MODEL_IDS[0], CANDIDATES[0].model_id)
clear_path_cache = parent.clear_path_cache
TIMINGS = parent.TIMINGS
FIT_DIAGNOSTICS = parent.FIT_DIAGNOSTICS
shell = parent.shell
overlay = parent.overlay
load_paths = parent.load_paths
