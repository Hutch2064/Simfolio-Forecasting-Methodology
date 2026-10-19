"""Exact scored FAST and original-node IID laws for the French single series.

The numerical owner is the sealed M261 source, including its native fitting,
node mapper, random streams, and holdings accounting. The retained refinement
sources in ``resources/french1926/refinement_source`` bind the transformations
and independently executable parity references. Use one model per subprocess.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from ..runner import ForecastContext, TrainingData
from .refinement_runtime import load_supplement

FAST_MODEL_ID = "R52-RMS-CONTINUOUS-VARIANCE-NODE-CORRECTION"
IID_MODEL_ID = "UF-ORIGINAL-NODES-IID-52d4c26b608ac1ac"
_ACTIVE: tuple[int, str] | None = None


@lru_cache(maxsize=1)
def _supplements():
    return (
        load_supplement("unified_frontier/iid_conditional_scale_candidates/scale.py"),
        load_supplement("unified_frontier/rough_moment_candidates/moments.py"),
        load_supplement("unified_frontier/conditional_candidates/marginal.py"),
    )


def _continuous_variance_nodes(nodes):
    # Literal innovation_mean_variance/models.py transform, including the
    # original continuous-mean preservation and float64 operation order.
    x = np.asarray(nodes, dtype=np.float64)
    left, right = x[:-1], x[1:]
    mean = float(np.mean((left + right) * .5))
    second = float(np.mean((left * left + left * right + right * right) / 3.))
    variance = second - mean * mean
    return (nodes - mean) / np.sqrt(variance) + mean


def _validate(training: TrainingData, context: ForecastContext):
    training.validate()
    if context.simulations != 240 or context.horizon_days < 1:
        raise ValueError("French refinement requires 240 paths and the complete positive horizon")
    assets = np.asarray(training.asset_log_returns, dtype=np.float64)
    policy = training.policy
    if (assets.shape != (len(training.portfolio_log_returns), 1) or policy is None
            or policy.weights != (1.0,) or policy.rebalance != "none"
            or not np.array_equal(assets[:, 0], training.portfolio_log_returns)):
        raise ValueError("French refinement requires one unit-weight un-rebalanced series")
    return assets


@dataclass(frozen=True)
class RefinementDailyAdapter:
    model_id: str

    def simulate_daily_log_returns(self, training: TrainingData, context: ForecastContext):
        global _ACTIVE
        if self.model_id not in (FAST_MODEL_ID, IID_MODEL_ID):
            raise ValueError("unknown exact French refinement")
        assets = _validate(training, context)
        identity = (os.getpid(), self.model_id)
        if _ACTIVE is not None and _ACTIVE != identity:
            raise RuntimeError("French refinements require one methodology per fresh subprocess")
        _ACTIVE = identity
        from ..models.asset_level.sv_moment_functions import moment_return_curves
        from .catalogue_adapters import initialized_m261_owner

        owner = initialized_m261_owner()
        body, shell = owner.parent.body, owner.shell
        past, future = shell._validate_calendar(training, context)
        data = assets[:, 0].tobytes()
        horizon = context.horizon_days
        scale, moments, marginal = _supplements()
        owner.clear_path_cache()
        owner.FIT_DIAGNOSTICS.clear()
        owner.TIMINGS.clear()
        try:
            body.refit(data)
            configuration = body.configuration(data, horizon)
            mean, _ = moment_return_curves(body.learned.student.original_fit(data), horizon)
            nodes = body.learned.nodes(data, context.simulations)
            if self.model_id == FAST_MODEL_ID:
                nodes = _continuous_variance_nodes(nodes)
            # Preserve the scored source's single-holding uniform stream. Its
            # seed includes origin, requested H and N; the IID stream below
            # additionally uses the panel task seed and exact history bytes.
            uniforms = owner.CANDIDATES[0]._return_uniforms(assets, context)
            from streamed_paths import map_asset_inplace

            base = np.ascontiguousarray(uniforms[:, :, 0])
            map_asset_inplace(base, mean, configuration[7], nodes)
            finished = marginal.finish(base, mean, None)
            if self.model_id == IID_MODEL_ID:
                # Capture original rough functions without installing the RMS
                # guards. This is exactly rough_moment_candidates.prepare's
                # deterministic fit/filter/first-step normalizer sequence.
                refs = {
                    "fit": body.rough.rough_fit, "prepared": body.pathwise.prepared,
                    "normalizer": owner.overlay.path_normalizers,
                    "fit_seconds": owner.CANDIDATES[0].fit_seconds,
                }
                # Observe the actual conditional curve through the retained
                # normalizer, exactly as total_mean_candidates.prepare does.
                captured = []
                normalizer = refs["normalizer"]

                def observe(*args):
                    value = normalizer(*args)
                    captured.append(np.maximum(np.asarray(value[1]), 0.))
                    return value

                refs["normalizer"] = observe
                _, record, _ = moments.prepare(data, len(assets) + horizon - 1, horizon, refs)
                rough_variance = captured[0]
                if moments.sha(rough_variance) != record["conditional_variance"]["sha256"]:
                    raise ArithmeticError("retained rough variance changed")
                multiplier, log_multiplier, _ = scale.draw_group(
                    np.asarray(configuration[6], np.float64) + rough_variance,
                    context.simulations, context.seed, context.origin_date, data,
                )
                finished, _ = scale.apply(finished, mean, multiplier, log_multiplier)
            dates = shell._historical_rebalance_dates(past.append(future), training.policy.rebalance)
            mask = np.asarray([date in dates for date in future])
            return shell.rejoin(finished[:, :, None], training.policy.weights, mask)
        finally:
            owner.clear_path_cache()


def build_fast():
    return RefinementDailyAdapter(FAST_MODEL_ID)


def build_iid():
    return RefinementDailyAdapter(IID_MODEL_ID)
