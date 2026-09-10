"""Exact-Kalman asset-level model from the retained adaptive-PGAS wrapper.

The retained wrapper's exact-Kalman ID dispatches the adaptive-PGAS helper
with ``student_scale_mixture=False``.  That source branch uses exact Kalman
terminal-state sampling for the Gaussian dynamic factor model and streams the
result through the source marginal-path mapping and portfolio rejoin.  The
univariate marginal is the separately recovered full-INLA BDES candidate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...runner import ForecastContext, TrainingData
from ..numerical.adaptive_pgas import adaptive_pgas_uniform_paths
from ..numerical.bdes_fastmap import deterministic_seed
from ..numerical.dynamic_gaussian import (
    map_uniforms_to_marginal_paths,
    pseudo_observations,
    rebalanced_portfolio_log_paths,
)
from ..numerical.inla_bdes import (
    INLA_MODEL_ID,
    fit_bdes_full_inla,
    simulate_bdes_full_inla,
)
from .frontier import TURNOVER_COST_BPS, _historical_rebalance_dates, _validate_calendar

EXACT_KALMAN_MODEL_ID = "asset_level_exact_kalman_dynamic_gaussian_factor_rebalanced"
SOURCE_ASSET_WRAPPER_PATH = "tmp/oos_all_daily_exact_crps.py"
SOURCE_ASSET_WRAPPER_SHA256 = "88aaa532d8a0a63e24964305a26ee6fa45d0d9099dd5da800ab691afad583ea7"
SOURCE_DEPENDENCE_WRAPPER_PATH = (
    "tmp/asset_level_full_panel_20260823/simfolio_oos_copula_alternatives.py"
)
SOURCE_DEPENDENCE_WRAPPER_SHA256 = (
    "413d2ca7f74cda13dd228c8974f822ce23e690cc56e098babf3fd2c121fbf95e"
)
SOURCE_PGAS_WRAPPER_PATH = "tmp/asset_level_full_panel_20260823/simfolio_adaptive_pgas.py"
SOURCE_PGAS_WRAPPER_SHA256 = "f7988a6cfbdf674c1efeb1ca6836b34e5f34c1ee241e97ef44551ded7ba87c69"


def _source_stable_correlation(values: np.ndarray) -> np.ndarray:
    """Match the retained wrapper's PSD correlation repair callback."""
    observations = np.asarray(values, dtype=np.float64)
    correlation = np.asarray(np.corrcoef(observations, rowvar=False), dtype=np.float64)
    correlation = np.where(np.isfinite(correlation), correlation, 0.0)
    correlation = 0.5 * (correlation + correlation.T)
    np.fill_diagonal(correlation, 1.0)
    eigenvalues, eigenvectors = np.linalg.eigh(correlation)
    eigenvalues = np.maximum(eigenvalues, 1e-8)
    repaired = (eigenvectors * eigenvalues) @ eigenvectors.T
    diagonal = np.sqrt(np.maximum(np.diag(repaired), 1e-12))
    repaired = repaired / diagonal[:, None] / diagonal[None, :]
    repaired = 0.5 * (repaired + repaired.T)
    np.fill_diagonal(repaired, 1.0)
    return repaired


def _student_copula_not_used(_values: np.ndarray) -> tuple[np.ndarray, float]:
    raise ValueError("exact_kalman_student_scale_mixture_is_disabled_by_source_dispatch")


def _source_engine_log_returns(values: np.ndarray) -> np.ndarray:
    """Match the retained engine's simple-return input normalization."""
    simple = np.expm1(np.asarray(values, dtype=np.float64))
    return np.log1p(np.clip(simple, -0.999999, None))


@dataclass(frozen=True)
class ExactKalmanDynamicGaussianFactorModel:
    """Historical exact-Kalman factor/rejoin model with source INLA marginals."""

    model_id: str = EXACT_KALMAN_MODEL_ID

    def simulate_daily_log_returns(
        self, training: TrainingData, context: ForecastContext
    ) -> np.ndarray:
        training.validate()
        if self.model_id != EXACT_KALMAN_MODEL_ID:
            raise ValueError(f"exact_kalman_unknown_model_id:{self.model_id}")
        if training.asset_log_returns is None or training.policy is None:
            raise ValueError("exact_kalman_requires_asset_level_training_data")
        if int(context.simulations) < 2:
            raise ValueError("exact_kalman_requires_at_least_two_simulations")
        if int(context.horizon_days) < 1:
            raise ValueError("exact_kalman_requires_positive_horizon")
        train_dates, future_dates = _validate_calendar(training, context)
        assets = np.asarray(training.asset_log_returns, dtype=np.float64)
        tickers = tuple(str(value) for value in training.policy.tickers)
        if len(tickers) != assets.shape[1]:
            raise ValueError("exact_kalman_asset_order_does_not_match_policy")
        if len(train_dates) < 80:
            raise ValueError("exact_kalman_requires_80_historical_observations")
        horizon, simulations = int(context.horizon_days), int(context.simulations)

        marginal = np.empty((simulations, horizon, assets.shape[1]), dtype=np.float64)
        for asset_index, ticker in enumerate(tickers):
            # The retained wrapper stores asset history as logs, converts it
            # to simple returns, and the engine converts those simple returns
            # back to logs before fitting.  Keep this round trip explicit so
            # the adapter follows the source dispatcher bit-for-bit.
            fit = fit_bdes_full_inla(_source_engine_log_returns(assets[:, asset_index]))
            asset_seed = deterministic_seed(
                "asset_level_current_engine",
                ticker,
                str(context.origin_date),
                INLA_MODEL_ID,
                horizon,
                simulations,
            )
            marginal[:, :, asset_index] = simulate_bdes_full_inla(
                fit, simulations, horizon, asset_seed
            )

        if assets.shape[1] == 1:
            dependent_assets = marginal
        else:
            dependence_seed = deterministic_seed(
                "copula_alternatives",
                EXACT_KALMAN_MODEL_ID,
                str(context.origin_date),
                horizon,
                simulations,
            )
            mapped = np.empty_like(marginal)

            def consume_uniform_block(start: int, block_uniforms: np.ndarray) -> None:
                start_index = int(start)
                end_index = start_index + int(block_uniforms.shape[1])
                mapped[:, start_index:end_index, :] = map_uniforms_to_marginal_paths(
                    marginal[:, start_index:end_index, :], block_uniforms
                )

            uniforms, _ = adaptive_pgas_uniform_paths(
                assets,
                simulations,
                horizon,
                np.random.default_rng(dependence_seed),
                pseudo_observations=pseudo_observations,
                student_copula_fit=_student_copula_not_used,
                stable_correlation=_source_stable_correlation,
                config={"student_scale_mixture": False},
                uniform_callback=consume_uniform_block,
            )
            if uniforms is not None:
                raise ValueError("exact_kalman_source_callback_did_not_stream")
            dependent_assets = mapped

        full_dates = train_dates.append(future_dates)
        rebalance_dates = _historical_rebalance_dates(full_dates, training.policy.rebalance)
        rebalance_mask = np.asarray([date in rebalance_dates for date in future_dates], dtype=bool)
        output = rebalanced_portfolio_log_paths(
            dependent_assets,
            training.policy.weights,
            rebalance_mask,
            cost_per_turnover_bps=TURNOVER_COST_BPS,
        )
        if output.shape != (simulations, horizon) or not np.all(np.isfinite(output)):
            raise ValueError("exact_kalman_nonfinite_portfolio_paths")
        return np.asarray(output, dtype=np.float64)


# Keep the short alias alongside the descriptive class name used by the
# Frontier adapter.  Both names resolve to the same explicit implementation.
ExactKalmanAssetLevelModel = ExactKalmanDynamicGaussianFactorModel

__all__ = [
    "EXACT_KALMAN_MODEL_ID",
    "SOURCE_ASSET_WRAPPER_PATH",
    "SOURCE_ASSET_WRAPPER_SHA256",
    "SOURCE_DEPENDENCE_WRAPPER_PATH",
    "SOURCE_DEPENDENCE_WRAPPER_SHA256",
    "SOURCE_PGAS_WRAPPER_PATH",
    "SOURCE_PGAS_WRAPPER_SHA256",
    "ExactKalmanAssetLevelModel",
    "ExactKalmanDynamicGaussianFactorModel",
]
