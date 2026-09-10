"""Bounded parity evidence for the checked-out current Frontier implementation.

The source-side production replay is intentionally kept outside the package:
the checked-in fixtures contain only sanitized arrays and source digests.  The
tests separate fitted factor-coordinate identity from conditional simulation:
eigenvector signs may vary by numerical backend, while seeded factor normals
remain coordinate-sensitive.  Stored source-versus-production paths therefore
use their recorded factor orientation, and the live adapter is compared with a
native pipeline run from the same current fit.
"""

from __future__ import annotations

import gzip
import hashlib
import json
from importlib import resources
from pathlib import Path

import numpy as np
import pandas as pd

from simfolio_forecasting_methodology.models.asset_level.frontier import (
    FRONTIER_MODEL_ID,
    HistoricalFrontierModel,
    _historical_rebalance_dates,
)
from simfolio_forecasting_methodology.models.numerical.bdes_fastmap import (
    FRONTIER_CANDIDATE,
    FRONTIER_DEPENDENCE_ID,
    FRONTIER_MARGINAL_ID,
    deterministic_seed,
    fit_bdes_fastmap,
    simulate_fastmap_marginal,
)
from simfolio_forecasting_methodology.models.numerical.dynamic_gaussian import (
    fit_dynamic_gaussian_factor_model,
    kalman_terminal_posterior,
    map_uniforms_to_marginal_paths,
    rebalanced_portfolio_log_paths,
    simulate_future_gaussian_uniforms,
)
from simfolio_forecasting_methodology.runner import ForecastContext, PortfolioPolicy, TrainingData

FIXTURE_NAMES = (
    "current_production_synthetic_fixture4_monthly.npz",
    "current_production_canonical80_row001_annually.npz",
    "current_production_canonical80_row007_quarterly.npz",
)
FIT_KEYS = ("posterior_centers", "bdes_coefficients", "mu", "state_loglik", "state_path_var")
DEPENDENCE_KEYS = (
    "factor_mean",
    "factor_observations",
    "factor_loading",
    "factor_residual_variance",
    "factor_phi",
    "factor_innovation_variance",
    "factor_initial_variance",
    "kalman_state",
    "kalman_covariance",
)
SIGNED_DEPENDENCE_KEYS = ("factor_loading", "kalman_state", "kalman_covariance")
INVARIANT_DEPENDENCE_KEYS = tuple(
    key for key in DEPENDENCE_KEYS if key not in SIGNED_DEPENDENCE_KEYS
)


def _fixture_path(name: str):
    return resources.files("simfolio_forecasting_methodology").joinpath(
        "resources", "test_fixtures", "frontier", name
    )


def _load(name: str) -> dict[str, np.ndarray]:
    fixture = _fixture_path(name)
    with fixture.open("rb") as handle, np.load(handle, allow_pickle=False) as data:
        return {key: np.asarray(data[key]) for key in data.files}


def _stored_dependence_model(fixture: dict[str, np.ndarray]) -> dict[str, np.ndarray | int]:
    loading = np.asarray(fixture["public_factor_loading"], dtype=np.float64)
    observations = np.asarray(fixture["public_factor_observations"], dtype=np.float64)
    return {
        "mean": np.asarray(fixture["public_factor_mean"], dtype=np.float64),
        "observations": observations,
        "loading": loading,
        "residual_variance": np.asarray(
            fixture["public_factor_residual_variance"], dtype=np.float64
        ),
        "phi": np.asarray(fixture["public_factor_phi"], dtype=np.float64),
        "innovation_variance": np.asarray(
            fixture["public_factor_innovation_variance"], dtype=np.float64
        ),
        "initial_variance": np.asarray(
            fixture["public_factor_initial_variance"], dtype=np.float64
        ),
        "factor_count": int(loading.shape[1]),
        "observation_count": int(observations.shape[0]),
        "asset_count": int(loading.shape[0]),
    }


def _factor_signs(
    reference_loading: np.ndarray, candidate_loading: np.ndarray
) -> np.ndarray:
    reference = np.asarray(reference_loading, dtype=np.float64)
    candidate = np.asarray(candidate_loading, dtype=np.float64)
    if reference.shape != candidate.shape or reference.ndim != 2:
        raise AssertionError("factor loading dimensions changed")
    overlap = np.einsum("ij,ij->j", reference, candidate)
    if not np.all(np.isfinite(overlap)) or np.any(overlap == 0.0):
        raise AssertionError("factor loading has an ambiguous sign alignment")
    return np.where(overlap < 0.0, -1.0, 1.0)


def _assert_fitted_dependence_matches(
    candidate: dict[str, np.ndarray], fixture: dict[str, np.ndarray]
) -> None:
    """Compare fit outputs modulo the diagonal sign of each factor coordinate."""
    reference = {
        key: np.asarray(fixture[f"public_{key}"], dtype=np.float64)
        for key in DEPENDENCE_KEYS
    }
    signs = _factor_signs(reference["factor_loading"], candidate["factor_loading"])

    for key in INVARIANT_DEPENDENCE_KEYS:
        np.testing.assert_allclose(
            candidate[key], reference[key], rtol=0.0, atol=2e-12
        )
    np.testing.assert_allclose(
        candidate["factor_loading"] * signs[None, :],
        reference["factor_loading"],
        rtol=0.0,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        candidate["kalman_state"] * signs,
        reference["kalman_state"],
        rtol=0.0,
        atol=2e-12,
    )
    np.testing.assert_allclose(
        candidate["kalman_covariance"] * signs[:, None] * signs[None, :],
        reference["kalman_covariance"],
        rtol=0.0,
        atol=2e-12,
    )


def _stored_conditional_pipeline(fixture: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Run conditional simulation with the fixture's frozen fit/orientation."""
    training_dates = pd.DatetimeIndex(fixture["training_dates"])
    origin = str(pd.Timestamp(training_dates[-1]).date())
    simulations, horizon, _ = fixture["public_marginal_paths"].shape
    dependence = _stored_dependence_model(fixture)
    state, covariance = kalman_terminal_posterior(dependence)
    np.testing.assert_allclose(
        state, fixture["public_kalman_state"], rtol=0.0, atol=2e-12
    )
    np.testing.assert_allclose(
        covariance, fixture["public_kalman_covariance"], rtol=0.0, atol=2e-12
    )
    dependence_seed = deterministic_seed(
        "copula_alternatives", FRONTIER_DEPENDENCE_ID, origin, horizon, simulations
    )
    uniforms = simulate_future_gaussian_uniforms(
        dependence, simulations, horizon, np.random.default_rng(dependence_seed)
    )
    mapped = map_uniforms_to_marginal_paths(fixture["public_marginal_paths"], uniforms)
    rejoined = rebalanced_portfolio_log_paths(
        mapped,
        fixture["weights"],
        fixture["public_rebalance_mask"],
        cost_per_turnover_bps=15.0,
    )
    return {"uniforms": uniforms, "mapped_paths": mapped, "rejoined": rejoined}


def _public_pipeline(fixture: dict[str, np.ndarray], tickers: tuple[str, ...], rebalance: str):
    logs = np.asarray(fixture["asset_log_returns"], dtype=np.float64)
    training_dates = pd.DatetimeIndex(fixture["training_dates"])
    future_dates = pd.DatetimeIndex(fixture["future_dates"])
    weights = np.asarray(fixture["weights"], dtype=np.float64)
    origin = str(pd.Timestamp(training_dates[-1]).date())
    simulations, horizon, asset_count = fixture["public_marginal_paths"].shape
    marginal = np.empty((simulations, horizon, asset_count), dtype=np.float64)
    for asset, ticker in enumerate(tickers):
        fit = fit_bdes_fastmap(logs[:, asset], FRONTIER_CANDIDATE)
        seed = deterministic_seed(
            "asset_level_current_engine",
            ticker,
            origin,
            FRONTIER_MARGINAL_ID,
            horizon,
            simulations,
        )
        marginal[:, :, asset] = simulate_fastmap_marginal(fit, simulations, horizon, seed)
    dependence = fit_dynamic_gaussian_factor_model(logs)
    state, covariance = kalman_terminal_posterior(dependence)
    dependence_seed = deterministic_seed(
        "copula_alternatives", FRONTIER_DEPENDENCE_ID, origin, horizon, simulations
    )
    uniforms = simulate_future_gaussian_uniforms(
        dependence, simulations, horizon, np.random.default_rng(dependence_seed)
    )
    mapped = map_uniforms_to_marginal_paths(marginal, uniforms)
    full_dates = training_dates.append(future_dates)
    # The fixtures were generated with zero offset and no drift band.  The
    # public adapter's calendar helper is exercised through its model below;
    # this mask mirrors that helper for the closure-array assertions.
    scheduled = _historical_rebalance_dates(full_dates, rebalance)
    mask = np.asarray([date in scheduled for date in future_dates], dtype=bool)
    rejoined = rebalanced_portfolio_log_paths(mapped, weights, mask, cost_per_turnover_bps=15.0)
    return {
        "marginal_paths": marginal,
        "factor_mean": np.asarray(dependence["mean"], dtype=np.float64),
        "factor_observations": np.asarray(dependence["observations"], dtype=np.float64),
        "factor_loading": np.asarray(dependence["loading"], dtype=np.float64),
        "factor_residual_variance": np.asarray(dependence["residual_variance"], dtype=np.float64),
        "factor_phi": np.asarray(dependence["phi"], dtype=np.float64),
        "factor_innovation_variance": np.asarray(dependence["innovation_variance"], dtype=np.float64),
        "factor_initial_variance": np.asarray(dependence["initial_variance"], dtype=np.float64),
        "kalman_state": np.asarray(state, dtype=np.float64),
        "kalman_covariance": np.asarray(covariance, dtype=np.float64),
        "uniforms": uniforms,
        "mapped_paths": mapped,
        "rejoined": rejoined,
        "rebalance_mask": mask,
    }


def _public_model_result(fixture: dict[str, np.ndarray], tickers: tuple[str, ...], rebalance: str):
    training_dates = np.asarray(fixture["training_dates"])
    weights = np.asarray(fixture["weights"], dtype=np.float64)
    train = TrainingData(
        portfolio_log_returns=np.einsum(
            "ij,j->i", fixture["asset_log_returns"], weights, dtype=np.float64
        ),
        asset_log_returns=np.asarray(fixture["asset_log_returns"], dtype=np.float64),
        policy=PortfolioPolicy(tuple(tickers), tuple(float(value) for value in weights), rebalance),
        training_dates=training_dates,
    )
    context = ForecastContext(
        model_id=FRONTIER_MODEL_ID,
        portfolio_id="current-production-frontier-audit",
        origin_label=str(pd.Timestamp(training_dates[-1]).date()),
        horizon_days=len(fixture["future_dates"]),
        simulations=int(fixture["public_marginal_paths"].shape[0]),
        seed=999,
        future_dates=np.asarray(fixture["future_dates"]),
        origin_date=str(pd.Timestamp(training_dates[-1]).date()),
    )
    return HistoricalFrontierModel().simulate_daily_log_returns(train, context)


def test_current_production_frontier_fixtures_preserve_source_evidence() -> None:
    report_path = _fixture_path("current_production_parity_report.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["status"] == "completed_bounded_audit"
    report_text = json.dumps(report)
    assert "/Users/" not in report_text
    assert "/tmp/" not in report_text
    assert report["source_identity"]["deployment_revision_verified"] is False

    for panel in report["panels"]:
        fixture = _load(Path(panel["fixture"]).name)
        tickers = tuple(panel["tickers"])
        if panel["source"]["kind"] == "frozen_canonical80_panel_row":
            columns = []
            for ticker in tickers:
                evidence = panel["source"]["source_series"][ticker]
                raw = resources.files("simfolio_forecasting_methodology").joinpath(
                    "resources", "data", "canonical_snapshot", "app", evidence["resource"]
                ).read_bytes()
                assert hashlib.sha256(raw).hexdigest() == evidence["sha256"]
                from io import BytesIO

                frame = pd.read_csv(
                    BytesIO(gzip.decompress(raw)), usecols=["date", "daily_return"],
                    parse_dates=["date"],
                ).set_index("date")
                series = frame["daily_return"]
                series = series[~series.index.duplicated(keep="last")]
                columns.append(series.loc[pd.DatetimeIndex(fixture["training_dates"])].to_numpy())
            np.testing.assert_array_equal(np.column_stack(columns), fixture["simple_returns"])
        public = _public_pipeline(fixture, tickers, panel["rebalance"])

        assert len(tickers) == fixture["asset_log_returns"].shape[1]
        assert fixture["asset_log_returns"].shape[0] >= 80
        assert fixture["public_rebalance_mask"].ndim == 1
        assert int(np.sum(fixture["public_rebalance_mask"])) == panel["metrics"]["rebalance_count"]
        np.testing.assert_array_equal(public["rebalance_mask"], fixture["public_rebalance_mask"])
        np.testing.assert_array_equal(
            fixture["public_rebalance_mask"], fixture["aligned_production_rebalance_mask"]
        )
        np.testing.assert_array_equal(
            fixture["public_rebalance_mask"], fixture["production_default_rebalance_mask"]
        )

        for key in FIT_KEYS:
            np.testing.assert_allclose(
                fixture[f"public_{key}"],
                fixture[f"aligned_production_{key}"],
                rtol=0.0,
                atol=2e-12,
            )
        for key in DEPENDENCE_KEYS:
            np.testing.assert_allclose(
                fixture[f"public_{key}"],
                fixture[f"aligned_production_{key}"],
                rtol=0.0,
                atol=2e-12,
            )
        np.testing.assert_allclose(
            fixture["public_uniforms"], fixture["aligned_production_uniforms"], rtol=0.0, atol=5e-14
        )
        np.testing.assert_allclose(
            fixture["public_marginal_paths"],
            fixture["aligned_production_marginal_paths"],
            rtol=0.0,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            fixture["public_mapped_paths"],
            fixture["aligned_production_mapped_paths"],
            rtol=0.0,
            atol=1e-8,
        )
        np.testing.assert_allclose(
            fixture["public_rejoined"],
            fixture["aligned_production_rejoined"],
            rtol=0.0,
            atol=1e-8,
        )
        _assert_fitted_dependence_matches(public, fixture)
        np.testing.assert_allclose(
            public["marginal_paths"], fixture["public_marginal_paths"], rtol=0.0, atol=2e-12
        )

        # Current production's root/per-asset seed contract is deliberately
        # retained as a separate persisted result.  It must not be presented
        # as the historical adapter output.
        assert not bool(panel["metrics"]["production_default_output_equals_public"])
        assert float(panel["metrics"]["production_default_output_vs_public_max_abs"]) > 0.0
        assert np.isfinite(fixture["production_default_output"]).all()
        assert np.isfinite(fixture["production_default_uniforms"]).all()


def test_current_production_frontier_conditional_paths_use_stored_orientation() -> None:
    """Keep conditional source/path parity strict after freezing factor coordinates."""
    report = json.loads(
        _fixture_path("current_production_parity_report.json").read_text(encoding="utf-8")
    )
    for panel in report["panels"]:
        fixture = _load(Path(panel["fixture"]).name)
        conditional = _stored_conditional_pipeline(fixture)

        np.testing.assert_allclose(
            conditional["uniforms"], fixture["public_uniforms"], rtol=0.0, atol=5e-14
        )
        np.testing.assert_allclose(
            conditional["mapped_paths"], fixture["public_mapped_paths"], rtol=0.0, atol=2e-12
        )
        np.testing.assert_allclose(
            conditional["rejoined"], fixture["public_rejoined"], rtol=0.0, atol=2e-12
        )

        # Match the two documented production storage boundaries. These are
        # exact comparisons, rather than float32-sized acceptance windows.
        stored_marginals = fixture["public_marginal_paths"].astype(np.float32)
        np.testing.assert_array_equal(
            stored_marginals, fixture["aligned_production_marginal_paths"]
        )
        stored_mapped = map_uniforms_to_marginal_paths(
            stored_marginals, conditional["uniforms"]
        ).astype(np.float32)
        np.testing.assert_array_equal(
            stored_mapped, fixture["aligned_production_mapped_paths"]
        )
        production_precision_rejoined = rebalanced_portfolio_log_paths(
            stored_mapped,
            fixture["weights"],
            fixture["public_rebalance_mask"],
            cost_per_turnover_bps=15.0,
        )
        np.testing.assert_array_equal(
            production_precision_rejoined, fixture["aligned_production_rejoined"]
        )


def test_current_production_frontier_adapter_matches_native_pipeline() -> None:
    """The adapter and manual pipeline must agree under the current fit/seed contract."""
    report = json.loads(
        _fixture_path("current_production_parity_report.json").read_text(encoding="utf-8")
    )
    for panel in report["panels"]:
        fixture = _load(Path(panel["fixture"]).name)
        tickers = tuple(panel["tickers"])
        public = _public_pipeline(fixture, tickers, panel["rebalance"])
        adapter = _public_model_result(fixture, tickers, panel["rebalance"])

        np.testing.assert_array_equal(
            public["rebalance_mask"], fixture["public_rebalance_mask"]
        )
        np.testing.assert_allclose(public["rejoined"], adapter, rtol=0.0, atol=2e-12)


def test_current_production_frontier_fixture_seed_identity_is_explicit() -> None:
    for name in FIXTURE_NAMES:
        fixture = _load(name)
        assert fixture["public_marginal_seeds"].shape == fixture["aligned_production_marginal_seeds"].shape
        np.testing.assert_array_equal(
            fixture["public_marginal_seeds"], fixture["aligned_production_marginal_seeds"]
        )
        assert fixture["production_root_seed"].shape == (1,)
        assert not np.array_equal(
            fixture["public_marginal_seeds"],
            np.repeat(fixture["production_root_seed"], fixture["public_marginal_seeds"].size),
        )
