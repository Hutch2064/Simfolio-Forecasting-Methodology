"""Portfolio adapter for the 40 canonical full latent MCMC SV overlays."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ...runner import ForecastContext, TrainingData
from ...seeds import forecast_oos_candidate_seed
from ..numerical.mcmc_sv import (
    CANONICAL_FULL_MCMC_SV_TYPE,
    fit_full_mcmc_sv,
    simulate_full_mcmc_sv,
    spec_for_full_mcmc_sv,
)


def _forecast_candidate_seed(context: ForecastContext) -> int:
    """Reproduce the source ``forecast_oos_candidate`` RNG context.

    The research harness seeds each candidate from the origin date, the
    available horizon tuple, the candidate ID, and the simulation count.  The
    current context contract carries the full daily horizon; callers that add
    an ``available_horizons`` attribute can pass the source's sparse horizon
    tuple directly.  The daily tuple is the canonical dense fallback used by
    this package's all-daily evaluation plan.
    """
    origin_date = getattr(context, "origin_date", None) or context.origin_label
    supplied_horizons = getattr(context, "available_horizons", None)
    if supplied_horizons is None:
        available_horizons = tuple(range(1, int(context.horizon_days) + 1))
    else:
        available_horizons = tuple(int(value) for value in supplied_horizons if int(value) > 0)
        if not available_horizons:
            raise ValueError("available_horizons must contain a positive horizon")
    return forecast_oos_candidate_seed(
        str(origin_date), available_horizons, context.model_id, int(context.simulations)
    )


@dataclass(frozen=True)
class FullMCMCSVModel:
    """Exact source-derived portfolio model for one canonical SV overlay ID."""

    model_id: str

    def __post_init__(self) -> None:
        spec = spec_for_full_mcmc_sv(self.model_id)
        if spec.get("type") != CANONICAL_FULL_MCMC_SV_TYPE:
            raise ValueError("model ID is not a canonical full MCMC SV overlay")

    def simulate_daily_log_returns(
        self, training: TrainingData, context: ForecastContext
    ) -> np.ndarray:
        training.validate()
        spec = spec_for_full_mcmc_sv(self.model_id)
        has_harx_anchor = (
            str(spec.get("vol_anchor_model", "none")) == "ridge_harx_ff6"
            or "harx_ff6_vol_anchor" in self.model_id
        )
        if has_harx_anchor:
            raise ValueError(
                "full MCMC SV HARX FF6 anchor requires an independently verified factor panel"
            )
        fit = fit_full_mcmc_sv(training.portfolio_log_returns, self.model_id)
        if fit is None:
            raise ValueError(f"full MCMC SV fit failed closed for {self.model_id}")
        paths = simulate_full_mcmc_sv(
            fit,
            int(context.horizon_days),
            int(context.simulations),
            np.random.default_rng(_forecast_candidate_seed(context)),
        )
        expected = (int(context.simulations), int(context.horizon_days))
        if paths.shape != expected or not np.all(np.isfinite(paths)):
            raise ValueError(
                f"full MCMC SV returned invalid path matrix {paths.shape}; expected {expected}"
            )
        return paths
