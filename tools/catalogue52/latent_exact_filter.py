"""Scoped exact Kalman compiler for the original forty full-MCMC candidates.

Only the deterministic likelihood/filter recurrence is replaced. Sampling,
FFBS, fit settings, seeds and simulation remain in the original model class.
"""
from __future__ import annotations

import hashlib
import inspect
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numba import njit

from simfolio_forecasting_methodology.models.numerical import mcmc_sv as mc
from simfolio_forecasting_methodology.models.portfolio import full_mcmc

ROOT = Path(__file__).resolve().parents[2]
_SOURCE_FILTER = mc._sv_kalman_filter
_COMPILED_FILTER = None
CANDIDATES = ()


def source_version():
    paths = (Path(mc.__file__), Path(full_mcmc.__file__), Path(__file__),
             ROOT / "src/simfolio_forecasting_methodology/resources/catalogs/"
                    "canonical_40_full_mcmc_sv_specs.json")
    return {
        "files": {str(path.relative_to(ROOT)): hashlib.sha256(path.read_bytes()).hexdigest()
                  for path in paths},
        "filter_source_sha256": hashlib.sha256(
            inspect.getsource(_SOURCE_FILTER).rstrip().encode()).hexdigest(),
    }


def prepare():
    global _COMPILED_FILTER
    if _COMPILED_FILTER is None:
        _COMPILED_FILTER = njit(cache=False, fastmath=False, nogil=True)(_SOURCE_FILTER)
        observed = np.array([-8.0, -7.0, -9.0])
        for return_path in (False, True):
            _COMPILED_FILTER(observed, -8.0, 0.94, 0.35, return_path=return_path)


def _filter(observed_log_var, level, phi, eta, *, return_path=False):
    # Numba dispatchers require explicit forwarding of this keyword-only default.
    return _COMPILED_FILTER(observed_log_var, level, phi, eta, return_path=return_path)


@contextmanager
def accelerated():
    prepare()
    original = mc._sv_kalman_filter
    try:
        mc._sv_kalman_filter = _filter
        yield
    finally:
        mc._sv_kalman_filter = original


@dataclass(frozen=True)
class ExactFilterModel:
    reference: full_mcmc.FullMCMCSVModel

    @property
    def model_id(self):
        return self.reference.model_id

    def simulate_daily_log_returns(self, training, context):
        with accelerated():
            return self.reference.simulate_daily_log_returns(training, context)


def initialize(_data=None, _workers=1, _simulations=1):
    global CANDIDATES
    prepare()
    CANDIDATES = tuple(ExactFilterModel(full_mcmc.FullMCMCSVModel(model_id))
                       for model_id in mc.canonical_full_mcmc_sv_ids())
