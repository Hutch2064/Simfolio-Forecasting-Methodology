"""Adapter-level checks for the source-derived full MCMC SV family."""

from __future__ import annotations

import numpy as np
import pytest

from simfolio_forecasting_methodology.models.numerical.mcmc_sv import (
    load_canonical_full_mcmc_sv_specs,
)
from simfolio_forecasting_methodology.models.portfolio import FullMCMCSVModel
from simfolio_forecasting_methodology.models.portfolio import full_mcmc as adapter
from simfolio_forecasting_methodology.runner import ForecastContext, TrainingData
from simfolio_forecasting_methodology.seeds import forecast_oos_candidate_seed


def _context(model_id: str, *, seed: int = 987654) -> ForecastContext:
    return ForecastContext(
        model_id=model_id,
        portfolio_id="synthetic_adapter_fixture",
        origin_label="rolling_01",
        origin_date="2026-05-13",
        horizon_days=3,
        simulations=2,
        seed=seed,
    )


def test_adapter_uses_source_candidate_seed(monkeypatch):
    model_id = load_canonical_full_mcmc_sv_specs()[0]["id"]
    observed: list[int] = []

    def fake_fit(values, candidate_id):
        assert candidate_id == model_id
        return {"stub": True}

    def fake_simulate(fit, horizon_days, simulations, rng):
        observed.append(int(rng.integers(0, 2**32 - 1)))
        return np.zeros((simulations, horizon_days), dtype=np.float64)

    monkeypatch.setattr(adapter, "fit_full_mcmc_sv", fake_fit)
    monkeypatch.setattr(adapter, "simulate_full_mcmc_sv", fake_simulate)
    training = TrainingData(np.zeros(80, dtype=np.float64))
    FullMCMCSVModel(model_id).simulate_daily_log_returns(training, _context(model_id, seed=1))
    expected_seed = forecast_oos_candidate_seed("2026-05-13", (1, 2, 3), model_id, 2)
    expected_draw = int(np.random.default_rng(expected_seed).integers(0, 2**32 - 1))
    assert observed == [expected_draw]


def test_each_unblocked_full_mcmc_adapter_forecasts_and_harx_fails_closed():
    values = np.random.default_rng(404).normal(0.0002, 0.01, 420).astype(np.float64)
    training = TrainingData(values)
    entries = load_canonical_full_mcmc_sv_specs()
    harx_ids = [
        str(entry["id"])
        for entry in entries
        if entry.get("vol_anchor_model") == "ridge_harx_ff6"
        or "harx_ff6_vol_anchor" in str(entry["id"])
    ]
    assert len(harx_ids) == 1
    for entry in entries:
        model_id = str(entry["id"])
        model = FullMCMCSVModel(model_id)
        if model_id in harx_ids:
            with pytest.raises(ValueError, match="factor panel"):
                model.simulate_daily_log_returns(training, _context(model_id))
            continue
        paths = model.simulate_daily_log_returns(training, _context(model_id))
        assert paths.shape == (2, 3), model_id
        assert np.all(np.isfinite(paths)), model_id
