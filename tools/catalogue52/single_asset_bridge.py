"""Exact one-dimensional copula limit for original rough research adapters.

A one-dimensional copula is C(u)=u. There is no cross-asset dependence to fit:
Gaussian factor and regime-vine marginals are both uniform. Older source
libraries rejected this dimension. Keep all original volatility/jump/mean
code and seed inputs; delegate every multidimensional call unchanged.
"""
import importlib.util
import os
import sys
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SOURCE = os.environ["SIMFOLIO_CATALOGUE_ADAPTER_SOURCE"]
if SOURCE not in {"tools/rough_jump_vine/models.py", "tools/rough_bayesian/models.py"}:
    raise ValueError("bridge source must be an explicitly retained rough adapter")
source = ROOT / SOURCE
sys.path.insert(0, str(source.parent))
# The retained rough suite uses `import models` inside its latent-state code.
# Load under the original runner's name so initialization and diagnostics reach
# the same module instead of creating an uninitialized second copy.
spec = importlib.util.spec_from_file_location("models", source)
original = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = original
spec.loader.exec_module(original)
CANDIDATES = original.CANDIDATES
TIMINGS = getattr(original, "TIMINGS", {})
FIT_DIAGNOSTICS = getattr(original, "FIT_DIAGNOSTICS", {})


def independent_uniforms(shape, data, sims, horizon, seed):
    return np.random.default_rng(seed).random((sims, horizon, 1))


def initialize(cache_root=None, vine_threads=1, state_workers=1):
    original.initialize(cache_root, vine_threads, state_workers)
    controls = getattr(original, "controls", original)
    gaussian = controls.gaussian_uniforms

    @lru_cache(maxsize=2)
    def gaussian_or_identity(shape, data, sims, horizon, seed):
        if shape[1] == 1:
            return independent_uniforms(shape, data, sims, horizon, seed)
        return gaussian(shape, data, sims, horizon, seed)

    controls.gaussian_uniforms = gaussian_or_identity
    dependence = controls.dependence_fit
    vine = controls.vine_uniforms

    def dependence_or_identity(shape, data):
        if shape[1] == 1:
            return {"dimension": 1, "single_asset_identity_copula": True}
        return dependence(shape, data)

    def vine_or_identity(fit, sims, horizon, seed, streams=1):
        if fit.get("single_asset_identity_copula"):
            rng = np.random.default_rng(seed)
            # Preserve the original ordering: regime uniforms, then return
            # uniforms for each stream. The unidentifiable regime cannot
            # change a one-dimensional identity copula.
            rng.random((sims, horizon))
            return [rng.random((sims * horizon, 1)).reshape(sims, horizon, 1)
                    for _ in range(streams)]
        return vine(fit, sims, horizon, seed, streams)

    controls.dependence_fit = dependence_or_identity
    controls.vine_uniforms = vine_or_identity


def clear_path_cache():
    original.clear_path_cache()
