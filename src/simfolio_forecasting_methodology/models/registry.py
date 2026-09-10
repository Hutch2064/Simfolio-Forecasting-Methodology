"""Evidence-aware registry for the canonical-175 ledger.

The code map is intentionally explicit.  A ledger record can advance its
evidence flags without becoming executable; execution requires a matching
verified factory in this module's map.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from typing import Any

from ..catalogue import canonical_model, load_canonical_models
from .asset_level.frontier import FRONTIER_MODEL_ID, HistoricalFrontierModel
from .numerical.base_models import CanonicalBaseModel
from .numerical.mcmc_sv import canonical_full_mcmc_sv_ids
from .portfolio.bayesian_vol import BAYESIAN_VOL_FACTORIES
from .portfolio.factor_residual import FACTOR_RESIDUAL_FACTORIES
from .portfolio.full_mcmc import FullMCMCSVModel
from .portfolio.gas_reference import REFERENCE_FACTORIES as GAS_FACTORIES
from .portfolio.gjr_reference import REFERENCE_FACTORIES as GJR_FACTORIES
from .portfolio.reference_families import REFERENCE_FACTORIES
from .portfolio.sv_extensions import REFERENCE_FACTORIES as SV_EXTENSION_FACTORIES
from .portfolio.sv_mcmc_reference import REFERENCE_FACTORIES as MCMC_REFERENCE_FACTORIES
from .portfolio.sv_reference import SVReferenceModel


@dataclass(frozen=True)
class ModelRegistration:
    model_id: str
    fidelity: str
    implementation: str
    factory: Callable[[], Any] | None = None


_EXPLICIT_FACTORIES: dict[str, Callable[[], Any]] = {
    FRONTIER_MODEL_ID: HistoricalFrontierModel,
    **REFERENCE_FACTORIES,
    **GAS_FACTORIES,
    **GJR_FACTORIES,
    **BAYESIAN_VOL_FACTORIES,
    **FACTOR_RESIDUAL_FACTORIES,
    **MCMC_REFERENCE_FACTORIES,
    **{model_id: partial(FullMCMCSVModel, model_id)
       for model_id in canonical_full_mcmc_sv_ids()
},
    **SV_EXTENSION_FACTORIES,
    "stochastic_volatility_ar1_student_t": partial(SVReferenceModel, "stochastic_volatility_ar1_student_t"),
    "stochastic_volatility_ar1_empirical": partial(SVReferenceModel, "stochastic_volatility_ar1_empirical"),
    "stochastic_volatility_ar1_empirical_sbb": partial(SVReferenceModel, "stochastic_volatility_ar1_empirical_sbb"),
    **{
        row["public_model_id"]: partial(CanonicalBaseModel, row["public_model_id"])
        for row in load_canonical_models()
        if row["model_family"] == "base"
    },
}


def registration(model_id: str) -> ModelRegistration:
    """Return evidence status for a known ID or reject unknown IDs."""

    record = canonical_model(model_id)
    factory = _EXPLICIT_FACTORIES.get(record["public_model_id"])
    if factory is None:
        return ModelRegistration(
            model_id=model_id,
            fidelity=record["verification_status"],
            implementation="blocked",
            factory=None,
        )
    return ModelRegistration(
        model_id=model_id,
        fidelity="verified_explicit_factory",
        implementation="explicit_factory",
        factory=factory,
    )


def build_model(model_id: str):
    """Build only a whitelisted, verified factory; fail closed otherwise."""

    item = registration(model_id)
    if item.factory is None:
        raise ValueError(
            f"canonical model '{model_id}' is not executable: no verified explicit factory is registered; "
            f"ledger verification_status={item.fidelity!r}"
        )
    return item.factory()


__all__ = ["ModelRegistration", "build_model", "registration"]
