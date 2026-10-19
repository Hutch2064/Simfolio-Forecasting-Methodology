"""Synthetic 240-path oracles only; no model fitting or research execution."""

import json
from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from simfolio_forecasting_methodology.french_panel import runner
from simfolio_forecasting_methodology.origins import OriginDescriptor
from simfolio_forecasting_methodology.results.checkpoint import (
    CheckpointCorruptError,
    CheckpointIdentityError,
)
from simfolio_forecasting_methodology.runner import TERMINAL_FORECAST_SEMANTICS
from simfolio_forecasting_methodology.seeds import coherent_daily_seed


@pytest.fixture
def panel(monkeypatch, tmp_path):
    dates = np.array(["2001-01-02", "2001-01-03", "2001-01-04", "2001-01-05",
                      "2001-01-08", "2001-01-09", "2001-01-11", "2001-01-12"],
                     dtype="datetime64[D]")
    values = np.array([[.001, .002, .01, .02, .04, .08, .16, .32],
                       [.003, .004, .02, .04, .08, .16, .32, .64]])
    origins = (
        OriginDescriptor("early", "rolling_origin", 1, str(dates[1]), 4),
        OriginDescriptor("late", "temporal", 5, str(dates[5]), 2),
    )
    manifest = {"panel_fingerprint": "synthetic-french-panel",
                "simulations_per_origin": 240,
                "series": [{"series_id": "s0", "role": "portfolio"},
                           {"series_id": "s1", "role": "factor"}]}
    monkeypatch.setattr(runner, "load_panel", lambda _: manifest)
    monkeypatch.setattr(runner, "load_dates", lambda _: dates)
    monkeypatch.setattr(runner, "load_returns", lambda _: values)
    monkeypatch.setattr(runner, "schedule", lambda _: origins)
    monkeypatch.setattr(runner, "_source_closure", lambda: {"source": "synthetic-frozen"})
    monkeypatch.setattr(runner, "_dependency_identity", lambda *_: {"runtime": "synthetic"})
    return tmp_path / "panel", tmp_path / "output", values, dates, origins, manifest


class ZeroOracle:
    model_id = "synthetic-zero-oracle"

    def __init__(self):
        self.calls = []

    def simulate_daily_log_returns(self, training, context):
        self.calls.append((training, context))
        return np.zeros((context.simulations, context.horizon_days))


def run(model, panel, **kwargs):
    return runner.run_panel(model, panel[0], panel[1], implementation_digest="fixture-v1", **kwargs)


def test_exact_crps_matches_brute_formula_at_240_paths(panel):
    class SpreadOracle(ZeroOracle):
        def simulate_daily_log_returns(self, training, context):
            return np.arange(240 * context.horizon_days).reshape(240, -1) / 100_000

    result = run(SpreadOracle(), panel)
    assert result["status"] == "complete"
    means = []
    for series_index, row in enumerate(panel[2]):
        sums, counts = np.zeros(4), np.zeros(4)
        for origin_index, origin in enumerate(panel[4]):
            daily = np.arange(240 * origin.max_horizon).reshape(240, -1) / 100_000
            terminal = daily.cumsum(axis=1)
            actual = row[origin.position + 1: origin.position + 1 + origin.max_horizon].cumsum()
            brute = np.mean(np.abs(terminal - actual), axis=0) - 0.5 * np.mean(
                np.abs(terminal[:, None, :] - terminal[None, :, :]), axis=(0, 1)
            )
            path = panel[1] / "tasks" / f"{series_index:06d}/{origin_index:03d}.npz"
            with np.load(path) as checkpoint:
                np.testing.assert_allclose(checkpoint["losses"], brute, atol=1e-14, rtol=1e-14)
            sums[:origin.max_horizon] += brute
            counts[:origin.max_horizon] += 1
        means.extend(sums / counts)
    assert result["final_score"] == pytest.approx(np.mean(means), abs=1e-14)


def test_cell_first_weighting_and_all_roles_retained(panel):
    result = run(ZeroOracle(), panel)
    cells, origin_means = [], []
    for row in panel[2]:
        early = np.cumsum(row[2:6])
        late = np.cumsum(row[6:8])
        cells.extend(np.r_[(early[:2] + late) / 2, early[2:]])
        origin_means.extend([early.mean(), late.mean()])
    assert result["final_score"] == pytest.approx(np.mean(cells))
    assert result["final_score"] != pytest.approx(np.mean(origin_means))
    assert result["complete_series"] == ["s0", "s1"]
    assert result["expected_cells"] == 8
    assert result["completed_tasks"] == 4


def test_history_copies_no_future_leakage_and_existing_seed_calendar(panel):
    model = ZeroOracle()
    run(model, panel)
    for index, (training, context) in enumerate(model.calls):
        series_index, origin_index = divmod(index, 2)
        origin = panel[4][origin_index]
        np.testing.assert_array_equal(training.portfolio_log_returns,
                                      panel[2][series_index, :origin.position + 1])
        np.testing.assert_array_equal(training.asset_log_returns[:, 0],
                                      training.portfolio_log_returns)
        np.testing.assert_array_equal(training.training_dates, panel[3][:origin.position + 1])
        assert training.asset_log_returns.shape == (origin.position + 1, 1)
        assert training.policy.tickers == (f"s{series_index}",)
        assert training.policy.weights == (1.0,)
        assert training.policy.rebalance == "none"
        assert not np.shares_memory(training.portfolio_log_returns, panel[2])
        for array in (training.portfolio_log_returns, training.asset_log_returns,
                      training.training_dates, context.future_dates):
            assert not array.flags.writeable
            with pytest.raises(ValueError):
                array.setflags(write=True)
            base = array
            while isinstance(base, np.ndarray) and base.base is not None:
                base = base.base
            assert isinstance(base, bytes)
        assert context.seed == coherent_daily_seed(origin.origin_date, origin.max_horizon, 240)
        expected_future = pd.bdate_range(pd.Timestamp(origin.origin_date) + pd.offsets.BDay(1),
                                        periods=origin.max_horizon).to_numpy()
        np.testing.assert_array_equal(context.future_dates, expected_future)
    # Jan 10 is absent in the observation calendar, but present in policy dates.
    assert model.calls[1][1].future_dates[0] == np.datetime64("2001-01-10")


def test_resume_never_calls_model_again_and_keeps_exact_losses(panel):
    model = ZeroOracle()
    first = run(model, panel)
    assert len(model.calls) == 4
    hashes = {path: path.read_bytes() for path in (panel[1] / "tasks").rglob("*.npz")}
    second = run(model, panel)
    assert len(model.calls) == 4
    assert second["attempted_tasks_this_call"] == 0
    assert second["resumed_tasks"] == 4
    assert second["final_score"] == first["final_score"]
    assert all(path.read_bytes() == raw for path, raw in hashes.items())


def test_partial_only_complete_series_and_incomplete_never_final(panel):
    model = ZeroOracle()
    empty = run(model, panel, max_tasks=0)
    assert empty["final_score"] is None
    assert empty["partial_complete_series_score"] is None
    first = run(model, panel, max_tasks=1)
    assert first["completed_tasks"] == 1
    assert first["complete_series"] == []
    assert first["final_score"] is None
    assert first["partial_complete_series_score"] is None
    second = run(model, panel, max_tasks=1)
    assert second["resumed_tasks"] == 1
    assert second["completed_tasks"] == 2
    assert second["complete_series"] == ["s0"]
    assert second["final_score"] is None
    assert second["partial_complete_series_score"] is not None
    final = run(model, panel)
    assert final["status"] == "complete"
    assert len(model.calls) == 4


@pytest.mark.parametrize("drift", ["panel", "model", "implementation", "runtime", "source", "schedule"])
def test_identity_drift_refuses_resume(panel, monkeypatch, drift):
    model = ZeroOracle()
    run(model, panel, max_tasks=1)
    digest = "fixture-v1"
    if drift == "panel":
        panel[5]["panel_fingerprint"] = "different"
    elif drift == "model":
        model.model_id = "another-oracle"
    elif drift == "implementation":
        digest = "fixture-v2"
    elif drift == "runtime":
        monkeypatch.setattr(runner, "_dependency_identity", lambda *_: {"runtime": "changed"})
    elif drift == "source":
        monkeypatch.setattr(runner, "_source_closure", lambda: {"source": "changed"})
    else:
        monkeypatch.setattr(runner, "schedule", lambda _: (replace(panel[4][0], max_horizon=3),
                                                         panel[4][1]))
    with pytest.raises(CheckpointIdentityError):
        runner.run_panel(model, panel[0], panel[1], implementation_digest=digest)
    assert len(model.calls) == 1


@pytest.mark.parametrize("corruption", ["bytes", "losses", "metadata", "manifest"])
def test_corruption_fails_without_resimulation(panel, corruption):
    model = ZeroOracle()
    run(model, panel, max_tasks=1)
    path = panel[1] / "tasks/000000/000.npz"
    if corruption == "bytes":
        path.write_bytes(b"damaged")
    elif corruption in {"losses", "metadata"}:
        with np.load(path) as checkpoint:
            losses = checkpoint["losses"]
            metadata = str(checkpoint["metadata"].item())
        if corruption == "losses":
            losses[0] += 1
        else:
            envelope = json.loads(metadata)
            envelope["payload"]["diagnostics"]["forecast_mean_cagr"] = 100.0
            metadata = json.dumps(envelope)
        np.savez(path, losses=losses, metadata=metadata)
    else:
        path = panel[1] / "execution.json"
        payload = json.loads(path.read_text())
        payload["identity"]["model_id"] = "changed"
        path.write_text(json.dumps(payload))
    with pytest.raises(CheckpointCorruptError):
        run(model, panel)
    assert len(model.calls) == 1


def test_failed_origin_continues_then_retries_without_repeating_success(panel):
    class FailureOracle(ZeroOracle):
        fail = True

        def simulate_daily_log_returns(self, training, context):
            self.calls.append((training, context))
            if self.fail and context.portfolio_id == "s0" and context.origin_label == "early":
                raise ValueError("synthetic failed fit")
            return np.zeros((context.simulations, context.horizon_days))

    model = FailureOracle()
    first = run(model, panel)
    assert first["completed_tasks"] == 3
    assert first["failed_tasks"] == 1
    assert first["complete_series"] == ["s1"]
    assert first["final_score"] is None
    error = panel[1] / "errors/000000/000.json"
    assert json.loads(error.read_text())["message"] == "synthetic failed fit"
    model.fail = False
    second = run(model, panel)
    assert second["status"] == "complete"
    assert second["attempted_tasks_this_call"] == 1
    assert second["resumed_tasks"] == 3
    assert len(model.calls) == 5
    assert not error.exists()


@pytest.mark.parametrize("bad_output", ["shape", "nonfinite", "overflow"])
def test_invalid_outputs_fail_every_task_without_final_score(panel, bad_output):
    class InvalidOracle(ZeroOracle):
        def simulate_daily_log_returns(self, training, context):
            shape = (239, context.horizon_days) if bad_output == "shape" else (240, context.horizon_days)
            value = np.nan if bad_output == "nonfinite" else 1e308 if bad_output == "overflow" else 0
            return np.full(shape, value)

    with np.errstate(over="ignore", invalid="ignore"):
        result = run(InvalidOracle(), panel)
    assert result["failed_tasks"] == 4
    assert result["completed_tasks"] == 0
    assert result["final_score"] is None
    assert result["complete_series"] == []


def test_endpoint_cagr_diagnostic_no_exclusion_filter(panel):
    run(ZeroOracle(), panel)
    with np.load(panel[1] / "tasks/000000/000.npz") as checkpoint:
        diagnostic = json.loads(str(checkpoint["metadata"].item()))["payload"]["diagnostics"]
    assert diagnostic["horizon_days"] == 4
    assert diagnostic["forecast_mean_cagr"] == 0
    assert diagnostic["forecast_median_cagr"] == 0
    assert diagnostic["training_historical_cagr"] == pytest.approx(np.expm1(.0015 * 252))
    assert diagnostic["diagnostic_only_no_exclusions"] is True


def test_terminal_forecast_adapter_uses_existing_declared_semantics(panel):
    class TerminalOracle(ZeroOracle):
        forecast_output_semantics = TERMINAL_FORECAST_SEMANTICS

        def simulate_terminal_log_returns(self, training, context):
            return np.zeros((240, context.horizon_days))

        def simulate_daily_log_returns(self, training, context):
            raise AssertionError("terminal adapter must not call daily method")

    assert run(TerminalOracle(), panel)["status"] == "complete"


def test_worker_compute_timing_saved_and_resumed_without_forecasts(panel, monkeypatch):
    ticks = iter([1.0, 1.25, 2.0, 2.5, 3.0, 4.0, 5.0, 7.0])
    monkeypatch.setattr(runner, "perf_counter", lambda: next(ticks))
    model = ZeroOracle()
    first = run(model, panel)
    assert first["completed_forecast_and_score_seconds"] == 3.75
    with np.load(panel[1] / "tasks/000000/000.npz") as checkpoint:
        diagnostic = json.loads(str(checkpoint["metadata"].item()))["payload"]["diagnostics"]
    assert diagnostic["forecast_and_score_seconds"] == .25
    # The iterator is exhausted: consulting the timer would fail this resume.
    second = run(model, panel)
    assert second["completed_forecast_and_score_seconds"] == 3.75
    assert second["final_score"] == first["final_score"]
    assert len(model.calls) == 4


def test_runner_uses_shared_panel_protocol():
    from simfolio_forecasting_methodology.french_panel import data

    assert runner.PROTOCOL_ID == data.PROTOCOL_ID
    assert runner.SIMULATIONS == data.SIMULATIONS
