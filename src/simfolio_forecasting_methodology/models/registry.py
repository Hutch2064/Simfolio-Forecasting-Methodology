"""Model registry shared by canonical and master OOS harnesses."""

from __future__ import annotations

from dataclasses import dataclass

from .frontier import FRONTIER_MODEL_ID, FrontierModel
from .generic import CompositionalModel, LegacyReferenceModel
from ..specifications import parse_compositional_spec


@dataclass(frozen=True)
class ModelRegistration:
    model_id: str
    fidelity: str
    implementation: str


def registration(model_id: str) -> ModelRegistration:
    if model_id == FRONTIER_MODEL_ID:
        return ModelRegistration(model_id, "production_method_reference", "FrontierModel")
    try:
        parse_compositional_spec(model_id)
    except ValueError:
        return ModelRegistration(model_id, "legacy_reference", "LegacyReferenceModel")
    return ModelRegistration(model_id, "component_reference", "CompositionalModel")


def build_model(model_id: str):
    item = registration(model_id)
    if item.implementation == "FrontierModel":
        return FrontierModel()
    if item.implementation == "CompositionalModel":
        return CompositionalModel(model_id=model_id)
    return LegacyReferenceModel(model_id=model_id)
