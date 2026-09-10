"""Evidence-aware registry for the canonical-175 ledger.

The code map is intentionally explicit.  A ledger record can advance its
evidence flags without becoming executable; execution requires a matching
verified factory in this module's map.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from ..catalogue import canonical_model


@dataclass(frozen=True)
class ModelRegistration:
    model_id: str
    fidelity: str
    implementation: str
    factory: Callable[[], Any] | None = None


# Deliberately empty in Gate 1: the existing Frontier and compositional code
# are not source-parity verified, and no partial generic fallback is allowed.
_EXPLICIT_FACTORIES: dict[str, Callable[[], Any]] = {}


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
