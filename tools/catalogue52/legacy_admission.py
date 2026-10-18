"""Explicit admission evidence for the retained canonical models.

Construction proves wiring only. Numerical source fixtures and campaign output
remain separate evidence; ledger flags are reported without upgrading them.
"""
from __future__ import annotations

import inspect
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RECOVERED_SOURCE_SHA256 = {
    "filtered_innovation_sv_candidates.py": "6c790cab4ee479a35a1b29a7943032f6177939928357c6e31825212c9f622150",
    "fixed_mean_sv_candidates.py": "fae19363e24d976c0d29613e397bca565bba07e3d6bf960f9631cfab3e4add9c",
}


def legacy_admission():
    from simfolio_forecasting_methodology.catalogue import load_canonical_models
    from simfolio_forecasting_methodology.models.registry import build_model, registration

    records = [row for row in load_canonical_models() if row["canonical_rank"] <= 175]
    if len(records) != 175 or {row["canonical_rank"] for row in records} != set(range(1, 176)):
        raise ValueError("legacy membership must contain exactly ranks 1 through 175")
    results = []
    for row in records:
        model_id = row["public_model_id"]
        entry = registration(model_id)
        if entry.factory is None or entry.implementation != "explicit_factory":
            raise ValueError(f"missing explicit factory: {model_id}")
        model = build_model(model_id)
        if model.model_id != model_id:
            raise ValueError(f"factory changed model identity: {model_id}")
        implementation = Path(inspect.getfile(type(model))).resolve()
        if not implementation.is_relative_to(ROOT / "src"):
            raise ValueError(f"factory escapes frozen source closure: {model_id}")
        results.append({
            "canonical_rank": row["canonical_rank"], "model_id": model_id,
            "implementation": str(implementation.relative_to(ROOT)),
            "factory_instantiated": True,
            "ledger_source_parity_checked": row["source_parity_checked"],
            "ledger_verification_status": row["verification_status"],
            "historical_score_verified": row["historical_score_verified"],
        })
    return {"scope": "legacy175_constructor_admission_only", "count": len(results),
            "implementation_counts": dict(Counter(row["implementation"] for row in results)),
            "models": results}


def initialize_parameter_mcmc():
    """Install the retained exact runtime, preserving both complete chains.

    Global numerical hooks require an isolated worker process. Call once before
    creating the selected model; do not mix legacy models into that process.
    """
    runtime = str(ROOT / "tools" / "mcmc_runtime")
    if runtime not in sys.path:
        sys.path.insert(0, runtime)
    import optimized_mcmc
    import runtime_setup

    runtime_setup.initialize(True)
    optimized_mcmc.install()
    return optimized_mcmc.OptimizedMCMC()


@dataclass
class ParameterMCMC:
    model_id: str = "sv_parameter_mcmc_twochain_sixteen_node_moment_mixture"
    _delegate: object = field(default=None, init=False, repr=False)

    def simulate_daily_log_returns(self, training, context):
        if self._delegate is None:
            self._delegate = initialize_parameter_mcmc()
        return self._delegate.simulate_daily_log_returns(training, context)


@dataclass(frozen=True)
class MomentSVVariant:
    """Original MomentSV law, with its explicitly recovered fitting branch.

    The recovered source classes install a temporary fit callback around their
    shared MomentSV path implementation. The package's verified rank-177 path
    implementation supplies that same moment, seed, copula and rejoin law.
    Native fitter flags implement the two original namespace substitutions.
    """
    model_id: str
    filtered_innovations: bool
    fixed_mean: bool

    def simulate_daily_log_returns(self, training, context):
        from simfolio_forecasting_methodology.models.asset_level.filtered_innovation_moment_sv import (
            FilteredInnovationFixedMeanMomentSV,
        )
        from simfolio_forecasting_methodology.models.numerical import bdes_fastmap as bd

        original = bd.fit_bdes_fastmap

        def fitted(values, **_kwargs):
            return original(values, filtered_innovations=self.filtered_innovations,
                            fixed_mean=self.fixed_mean)

        bd.fit_bdes_fastmap = fitted
        try:
            return FilteredInnovationFixedMeanMomentSV(self.model_id).simulate_daily_log_returns(
                training, context)
        finally:
            bd.fit_bdes_fastmap = original


CANDIDATES = ()


def clear_path_cache():
    """Reuse the retained benchmark's cold-fit cache boundary per task."""
    if "optimized_mcmc" not in sys.modules:
        return
    import optimized_mcmc as opt
    import parameter_mcmc_tuned as pm

    from simfolio_forecasting_methodology.models.numerical import bdes_fastmap as bd

    pm.parameter_fit.cache_clear()
    opt.nodes.cache_clear()
    opt.power.cache_clear()
    for cell in bd.fit_bdes_fastmap.__closure__ or ():
        if hasattr(cell.cell_contents, "cache_clear"):
            cell.cell_contents.cache_clear()
    pm.RECORDS.clear()


def initialize(_cache_root=None, _fit_threads=1, _path_threads=1):
    """Campaign adapter interface; importing or listing performs no fitting."""
    global CANDIDATES
    from simfolio_forecasting_methodology.models.asset_level.filtered_innovation_moment_sv import (
        FilteredInnovationFixedMeanMomentSV,
    )
    CANDIDATES = (
        ParameterMCMC(), FilteredInnovationFixedMeanMomentSV(),
        MomentSVVariant("experimental_filtered_innovation_moment_sv_dlm", True, False),
        MomentSVVariant("experimental_gaussian_moment_matched_sv_empirical_fixed_mean", False, True),
    )


if __name__ == "__main__":
    print(json.dumps(legacy_admission(), indent=2, sort_keys=True))
