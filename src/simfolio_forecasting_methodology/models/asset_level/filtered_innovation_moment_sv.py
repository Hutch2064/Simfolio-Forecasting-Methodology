"""Verified experimental SV candidate using filtered historical innovations."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..numerical import bdes_fastmap as bd
from ..numerical import dynamic_gaussian as dg
from .frontier import _historical_rebalance_dates, _validate_calendar
from .sv_moment_functions import moment_return_curves

FILTERED_INNOVATION_FIXED_MEAN_MODEL_ID = (
    "experimental_filtered_innovation_moment_sv_fixed_mean"
)


def sorted_moment_marginals(fit, simulations, horizon, *, shrink=0.0):
    mean, sd = moment_return_curves(fit, horizon, shrink=shrink)
    probability = np.linspace(0.5 / simulations, 1.0 - 0.5 / simulations, simulations)
    nodes = np.quantile(fit["innovation_pool"], probability)
    nodes -= nodes.mean()
    nodes /= np.sqrt(np.mean(nodes * nodes))
    return np.clip(mean[None, :] + sd[None, :] * nodes[:, None], -1.0, 1.0)


@dataclass(frozen=True)
class FilteredInnovationFixedMeanMomentSV:
    """Moment-matched SV marginals with the scored filtered-innovation law."""

    model_id: str = FILTERED_INNOVATION_FIXED_MEAN_MODEL_ID

    def simulate_daily_log_returns(self, training, context):
        training.validate()
        train_dates, future_dates = _validate_calendar(training, context)
        assets = np.asarray(training.asset_log_returns, dtype=np.float64)
        sims, horizon = int(context.simulations), int(context.horizon_days)
        if assets.shape[1] > 1:
            model = dg.fit_dynamic_gaussian_factor_model(assets)
            seed = bd.deterministic_seed(
                "copula_alternatives",
                bd.FRONTIER_DEPENDENCE_ID,
                str(context.origin_date),
                horizon,
                sims,
            )
            uniforms = dg.simulate_future_gaussian_uniforms(
                model, sims, horizon, np.random.default_rng(seed)
            )
        else:
            seed = bd.deterministic_seed(
                "moment_sv_single_asset", str(context.origin_date), horizon, sims
            )
            uniforms = np.random.default_rng(seed).random((sims, horizon, 1))
        marginals = np.empty_like(uniforms)
        for asset in range(assets.shape[1]):
            fit = bd.fit_bdes_fastmap(
                assets[:, asset], filtered_innovations=True, fixed_mean=True
            )
            marginals[:, :, asset] = sorted_moment_marginals(fit, sims, horizon)
        dependent = dg.map_uniforms_to_marginal_paths(marginals, uniforms)
        dates = _historical_rebalance_dates(
            train_dates.append(future_dates), training.policy.rebalance
        )
        mask = np.asarray([date in dates for date in future_dates])
        return dg.rebalanced_portfolio_log_paths(
            dependent, training.policy.weights, mask
        )


__all__ = [
    "FILTERED_INNOVATION_FIXED_MEAN_MODEL_ID",
    "FilteredInnovationFixedMeanMomentSV",
    "sorted_moment_marginals",
]
