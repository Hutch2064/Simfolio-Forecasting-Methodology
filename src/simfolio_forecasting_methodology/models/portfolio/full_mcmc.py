"""Portfolio adapter for the 40 canonical full latent MCMC SV overlays."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..numerical.mcmc_sv import (
    CANONICAL_FULL_MCMC_SV_TYPE,
    fit_full_mcmc_sv,
    simulate_full_mcmc_sv,
    spec_for_full_mcmc_sv,
)
from ..runner import ForecastContext, TrainingData


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
        if str(spec_for_full_mcmc_sv(self.model_id).get("vol_anchor_model", "none")) == "ridge_harx_ff6":
            raise ValueError(
                "full MCMC SV HARX FF6 anchor requires an independently verified factor panel"
            )
        fit = fit_full_mcmc_sv(training.portfolio_log_returns, self.model_id)
        if fit is None:
            raise ValueError(
                f"full MCMC SV fit failed closed for {self.model_id}"
            )
        paths = simulate_full_mcmc_sv(
            fit,
            int(context.horizon_days),
            int(context.simulations),
            np.random.default_rng(int(context.seed)),
        )
        expected = (int(context.simulations), int(context.horizon_days))
        if paths.shape != expected or not np.all(np.isfinite(paths)):
            raise ValueError(
                f"full MCMC SV returned invalid path matrix {paths.shape}; expected {expected}"
            )
        return paths
