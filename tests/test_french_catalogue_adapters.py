"""Bounded native-source parity; synthetic histories, never a panel sweep."""

import inspect
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import ClassVar

import numpy as np
import pytest

from simfolio_forecasting_methodology.french_panel import catalogue_adapters as adapters
from simfolio_forecasting_methodology.runner import (
    TERMINAL_FORECAST_SEMANTICS,
    ForecastContext,
    PortfolioPolicy,
    TrainingData,
)


@pytest.mark.parametrize("code", adapters.MODEL_IDS)
def test_explicit_factories_are_no_arg_lazy_and_exact_id(code, monkeypatch):
    def unexpected_install(*_):
        raise AssertionError("factory construction must not install numerical callbacks")

    monkeypatch.setattr(adapters, "_native", unexpected_install)
    factory = adapters.FRENCH_CATALOGUE_FACTORIES[code]
    assert not inspect.signature(factory).parameters
    model = factory()
    assert model.model_id == adapters.MODEL_IDS[code]
    terminal = callable(getattr(model, "simulate_terminal_log_returns", None))
    assert terminal is (code == "M035")
    if terminal:
        assert model.forecast_output_semantics == TERMINAL_FORECAST_SEMANTICS


def test_daily_forwards_full_history_context_and_clears_path_cache(monkeypatch):
    calls = []
    training = TrainingData(np.arange(1000.) / 100000, np.arange(1000.)[:, None] / 100000,
                            PortfolioPolicy(("series",), (1.,), "none"))
    context = ForecastContext(adapters.MODEL_IDS["M198"], "series", "origin", 19, 240, 315)
    paths = np.zeros((240, 19))

    class Native:
        def simulate_daily_log_returns(self, actual_training, actual_context):
            assert actual_training is training
            assert actual_context is context
            assert len(actual_training.portfolio_log_returns) == 1000
            calls.append("simulate")
            return paths

    class Source:
        FIT_DIAGNOSTICS: ClassVar = {"stale": "old origin"}
        TIMINGS: ClassVar = {"old": 1.0}

        @staticmethod
        def clear_path_cache():
            calls.append("clear")

    monkeypatch.setattr(adapters, "_native", lambda code: Native())
    monkeypatch.setattr(adapters, "_SOURCE_MODULE", Source)
    assert adapters.build_m198().simulate_daily_log_returns(training, context) is paths
    assert calls == ["clear", "simulate", "clear"]
    assert Source.FIT_DIAGNOSTICS == {}
    assert Source.TIMINGS == {}


def test_path_cache_clears_after_failed_origin(monkeypatch):
    calls = []

    class Native:
        def simulate_daily_log_returns(self, *_):
            raise ValueError("native fit failed")

    class Source:
        @staticmethod
        def clear_path_cache():
            calls.append("clear")

    monkeypatch.setattr(adapters, "_native", lambda code: Native())
    monkeypatch.setattr(adapters, "_SOURCE_MODULE", Source)
    with pytest.raises(ValueError, match="native fit failed"):
        adapters.build_m198().simulate_daily_log_returns(None, None)
    assert calls == ["clear", "clear"]


def test_one_source_install_per_fresh_process(monkeypatch):
    monkeypatch.setattr(adapters, "_ACTIVE", (os.getpid(), "M198"))
    monkeypatch.setattr(adapters, "_NATIVE", object())
    with pytest.raises(RuntimeError, match="fresh subprocess"):
        adapters._native("M202")
    monkeypatch.setattr(adapters, "_ACTIVE", (os.getpid() - 1, "M198"))
    with pytest.raises(RuntimeError, match="fresh subprocess"):
        adapters._native("M198")


def test_failed_source_install_cannot_be_retried(monkeypatch):
    monkeypatch.setattr(adapters, "_ACTIVE", (os.getpid(), "M198"))
    monkeypatch.setattr(adapters, "_NATIVE", None)
    with pytest.raises(RuntimeError, match="installation failed"):
        adapters._native("M198")


# Each method runs in a fresh interpreter: the retained research modules patch
# global callbacks, so importing multiple methodologies in pytest is unsafe.
NATIVE_PARITY = r'''
import hashlib
import json
import sys
import time
import numpy as np
import pandas as pd
from simfolio_forecasting_methodology.french_panel import catalogue_adapters as a
from simfolio_forecasting_methodology.runner import TrainingData, PortfolioPolicy, ForecastContext
code = sys.argv[1]
rng = np.random.default_rng(714)
h = np.zeros(384)
for t in range(1, len(h)):
    h[t] = .97*h[t-1]+.12*rng.normal()
x = .0003+.012*np.exp(h/2)*rng.standard_t(7,len(h))*np.sqrt(5/7)
dates = pd.bdate_range('1999-01-04', periods=len(x))
future = pd.bdate_range(dates[-1]+pd.offsets.BDay(1), periods=9)
model = a.FRENCH_CATALOGUE_FACTORIES[code]()
training = TrainingData(x, x[:,None], PortfolioPolicy(('synthetic',),(1.,),'none'), dates.to_numpy())
context = ForecastContext(model.model_id,'synthetic','fixture',9,240,123,
                          future.to_numpy(),str(dates[-1].date()))
started = time.perf_counter()
native = a._native(code)
assert native.model_id == model.model_id
if code in a.RESEARCH_SOURCES:
    if code == "M261":
        from simfolio_forecasting_methodology.french_panel.refinement_runtime import load_m261_source
        assert a._SOURCE_MODULE is load_m261_source()
    else:
        assert a._SOURCE_MODULE.__file__ == str(a.source_root()/a.RESEARCH_SOURCES[code])
    if code == "M176":
        assert isinstance(native, a._SOURCE_MODULE.opt.OptimizedMCMC)
    else:
        assert native in a._SOURCE_MODULE.CANDIDATES
    a._SOURCE_MODULE.clear_path_cache()
method = ('simulate_terminal_log_returns' if code == 'M035' else 'simulate_daily_log_returns')
# Execute the native candidate itself, then compare the public adapter's full
# 240-path matrix. No fake predictions, path reduction, or OOS score inputs.
expected = getattr(native, method)(training, context)
if code in a.RESEARCH_SOURCES:
    a._SOURCE_MODULE.clear_path_cache()
actual = getattr(model, method)(training, context)
assert expected.shape == actual.shape == (240, 9)
assert np.isfinite(actual).all()
assert expected.tobytes() == actual.tobytes(), (code,np.max(np.abs(expected-actual)))
print(json.dumps({'code':code,'sha256':hashlib.sha256(actual.tobytes()).hexdigest(),
                  'seconds':time.perf_counter()-started,'paths':240,'horizons':9,
                  'history_observations':len(x)}))
'''


@pytest.mark.parametrize("code", adapters.MODEL_IDS)
def test_native_candidate_full_240_path_parity(code):
    root = Path(__file__).resolve().parents[1]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root / "src")
    result = subprocess.run(
        [sys.executable, "-c", NATIVE_PARITY, code], cwd=root, env=environment,
        capture_output=True, text=True, timeout=300, check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    evidence = json.loads(result.stdout.strip().splitlines()[-1])
    assert evidence["code"] == code
    assert evidence["paths"] == 240
    assert evidence["history_observations"] == 384
    assert len(evidence["sha256"]) == 64
    print(json.dumps(evidence, sort_keys=True))
