"""Explicit, user-approved fifteen-model French-panel selection.

Reading this registry never fits a model or starts a sweep. Factories are lazy
because several retained native implementations install process-local hooks.
"""
from __future__ import annotations

import hashlib
import importlib
import json
from pathlib import Path

from ..results.checkpoint import stable_digest
from .runner import _atomic_json
from .sweep import read_control

PACKAGE = "simfolio_forecasting_methodology.french_panel"
DEFAULT_ID = "asset_level_naive_iid_mean_shrunk_rebalanced"
PUBLIC_ID = "model_testing_joint_stationary_bootstrap_v1"
EWMA_ID = "FRENCH1926-EWMA-QMLE"

FACTORIES = {
    "R52-RMS-CONTINUOUS-VARIANCE-NODE-CORRECTION": f"{PACKAGE}.refinement_adapters:build_fast",
    "UF-ORIGINAL-NODES-IID-52d4c26b608ac1ac": f"{PACKAGE}.refinement_adapters:build_iid",
    "asset_m256_data_timescales_existing_sv_rate": f"{PACKAGE}.catalogue_adapters:build_m261",
    DEFAULT_ID: f"{PACKAGE}.selected:build_default",
    PUBLIC_ID: f"{PACKAGE}.selected:build_public",
    "asset_level_fastmap_kalman_dynamic_gaussian_factor_rebalanced": f"{PACKAGE}.catalogue_adapters:build_m001",
    "constant_mean_student_t": f"{PACKAGE}.catalogue_adapters:build_m035",
    "portfolio_gjr_garch_eb_sampler_stationary_sbb_optimal": f"{PACKAGE}.catalogue_adapters:build_m052",
    "sv_parameter_mcmc_twochain_sixteen_node_moment_mixture": f"{PACKAGE}.catalogue_adapters:build_m176",
    "asset_map_multiscale_untrimmed_conditional_residual_empirical_rough_map": f"{PACKAGE}.catalogue_adapters:build_m198",
    "asset_map_multiscale_conditional_empirical_gamma_supou_whittle_map": f"{PACKAGE}.catalogue_adapters:build_m202",
    "asset_map_single_learned_multiscale_rate_untruncated_dynamic_rough": f"{PACKAGE}.catalogue_adapters:build_m233",
    "asset_map_stationary_ar1_ou_residual_volatility": f"{PACKAGE}.catalogue_adapters:build_m244",
    "asset_map_stationary_ar1_proxy_leverage_dynamic_rough": f"{PACKAGE}.catalogue_adapters:build_m247",
    EWMA_ID: f"{PACKAGE}.selected:build_ewma",
}


def _build_control(model_id):
    from .control_adapters import build_control_model

    return build_control_model(model_id)


def build_default():
    return _build_control(DEFAULT_ID)


def build_public():
    return _build_control(PUBLIC_ID)


def build_ewma():
    return _build_control(EWMA_ID)


def source_identity() -> dict[str, str]:
    """Bind package, native research tools, and their statistical definitions."""
    package = Path(__file__).resolve().parents[1]
    repository = package.parents[1]
    paths = set(package.rglob("*.py"))
    paths.update((repository / "tools").rglob("*.py"))
    paths.update((repository / "tools").rglob("*.cpp"))
    paths.update((repository / "tools").rglob("*.h"))
    paths.update((repository / "tools").rglob("*.patch"))
    paths.add(repository / "pyproject.toml")
    for directory in ("specifications", "canonical_175"):
        paths.update((package / "resources" / directory).rglob("*.json"))
    for directory in ("control_sources", "refinement_source"):
        paths.update(path for path in (package / "resources/french1926" / directory).rglob("*")
                     if path.is_file() and path.suffix in {".py", ".cpp", ".h", ".json"})
    return {str(path.relative_to(repository)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)}


def implementation_digest(model_id: str, sources: dict | None = None) -> str:
    if model_id not in FACTORIES:
        raise ValueError(f"unselected French-panel model: {model_id}")
    return stable_digest({"model_id": model_id, "factory": FACTORIES[model_id],
                          "source_identity": source_identity() if sources is None else sources})


def build_selected(model_id: str):
    """Instantiate only an explicitly selected factory; no fallback methods."""
    if model_id not in FACTORIES:
        raise ValueError(f"unselected French-panel model: {model_id}")
    module, name = FACTORIES[model_id].split(":")
    model = getattr(importlib.import_module(module), name)()
    if getattr(model, "model_id", None) != model_id:
        raise ValueError("selected factory returned a different model identity")
    return model


def prepare_control(path: Path, *, workers: int = 12) -> dict:
    """Write a complete paused control; explicit later launch is required."""
    source = source_identity()
    control = {"models": {
        key: {"enabled": True, "factory": factory,
              "implementation_digest": implementation_digest(key, source)}
        for key, factory in FACTORIES.items()}, "workers": workers, "stop": True}
    path = Path(path)
    _atomic_json(path, control)
    read_control(path)
    return control


def selection_matches_metadata() -> bool:
    path = Path(__file__).resolve().parents[1] / "resources/french1926/shortlist_proposal.json"
    data = json.loads(path.read_text())
    selected = {row["model_id"] for row in data["selected_models"] + data["required_controls"]}
    return selected == set(FACTORIES)
