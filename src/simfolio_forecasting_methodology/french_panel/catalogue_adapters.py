"""Exact native-source adapters for the selected French-panel catalogue methods.

The retained research sources install process-global numerical callbacks. Run
one methodology in each fresh subprocess; changing methods in a live worker
fails closed. Importing this module or constructing a factory does not install
callbacks, fit a model, load historical scores, or start a panel run.
"""

from __future__ import annotations

import hashlib
import importlib.util
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType

from ..runner import TERMINAL_FORECAST_SEMANTICS, ForecastContext, TrainingData

MODEL_IDS = {
    "M001": "asset_level_fastmap_kalman_dynamic_gaussian_factor_rebalanced",
    "M035": "constant_mean_student_t",
    "M052": "portfolio_gjr_garch_eb_sampler_stationary_sbb_optimal",
    "M176": "sv_parameter_mcmc_twochain_sixteen_node_moment_mixture",
    "M198": "asset_map_multiscale_untrimmed_conditional_residual_empirical_rough_map",
    "M202": "asset_map_multiscale_conditional_empirical_gamma_supou_whittle_map",
    "M233": "asset_map_single_learned_multiscale_rate_untruncated_dynamic_rough",
    "M244": "asset_map_stationary_ar1_ou_residual_volatility",
    "M247": "asset_map_stationary_ar1_proxy_leverage_dynamic_rough",
    "M261": "asset_m256_data_timescales_existing_sv_rate",
}
RESEARCH_SOURCES = {
    "M176": "tools/rough_jump_vine/models.py",
    "M198": "tools/untrimmed_conditional_rough/models.py",
    "M202": "tools/gamma_supou_spectral/models.py",
    "M233": "tools/single_multiscale_rate_rough/models.py",
    "M244": "tools/endpoint_residual_volatility/models.py",
    "M247": "tools/proxy_leverage_rough/models.py",
    "M261": "tools/data_selected_timescales/models.py",
}
_ACTIVE: tuple[int, str] | None = None
_NATIVE = None
_SOURCE_MODULE: ModuleType | None = None


def source_root() -> Path:
    """Require the retained source checkout rather than inventing a substitute."""
    root = Path(__file__).resolve().parents[3]
    if not (root / "tools/mcmc_runtime/runtime_setup.py").is_file():
        raise RuntimeError("French catalogue research adapters require the retained tools checkout")
    return root


def source_hashes() -> dict[str, str]:
    """Bind the complete numerical source closure, including runtime compilers."""
    root = source_root()
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for folder in (root / "src", root / "tools")
        for path in sorted(folder.rglob("*"))
        if path.is_file() and path.suffix in {".py", ".cpp", ".patch"}
    }


def _load_source(code: str) -> ModuleType:
    if code == "M261":
        from .refinement_runtime import load_m261_source

        return load_m261_source()
    # Private loaders compile shared files under different module names. Numba
    # includes those names in cached environments, but not in the cache key;
    # a cache shared by M198 and M202 can therefore unpickle a missing alias.
    # Keep each exact source closure in its own cache without changing kernels.
    import numba

    closure = "".join(f"{path}:{digest}\n" for path, digest in source_hashes().items())
    identity = hashlib.sha256(closure.encode()).hexdigest()
    cache_root = Path(os.environ.get("NUMBA_CACHE_DIR") or tempfile.gettempdir())
    numba.config.CACHE_DIR = str(cache_root / "french-catalogue" / code / identity)
    root = source_root()
    path = root / RESEARCH_SOURCES[code]
    # Match the native runners' import order. Sources add their own narrow
    # dependency directories as they load private parent modules.
    sys.path[:0] = [str(path.parent), str(root / "tools/mcmc_runtime")]
    name = f"french_catalogue_native_{code.lower()}"
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load retained native source: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    # Native initializer installs the exact retained acceleration kernels.
    # No disk fit cache is requested: only the native bounded memory caches.
    module.initialize(cache_root=None, vine_threads=1, state_workers=1)
    return module


def _native(code: str):
    global _ACTIVE, _NATIVE, _SOURCE_MODULE
    identity = (os.getpid(), code)
    if _ACTIVE is not None and _ACTIVE != identity:
        raise RuntimeError(
            "French catalogue methods require one methodology per fresh subprocess; "
            f"already installed {_ACTIVE}, requested {identity}"
        )
    if _NATIVE is None:
        # Reserve before import: a partial/failed installation also must not
        # contaminate a different methodology or be retried in the same worker.
        if _ACTIVE is not None:
            raise RuntimeError("native source installation failed; start a fresh subprocess")
        _ACTIVE = identity
        if code in RESEARCH_SOURCES:
            _SOURCE_MODULE = _load_source(code)
            # This exact M176 implementation retains the source single-asset
            # RNG branch; the older rough/jump panel shell requires two assets.
            candidates = ((_SOURCE_MODULE.opt.OptimizedMCMC(fit_workers=1),)
                          if code == "M176" else _SOURCE_MODULE.CANDIDATES)
            matches = [candidate for candidate in candidates
                       if candidate.model_id == MODEL_IDS[code]]
            if len(matches) != 1:
                raise ValueError(f"retained source has no unique exact candidate for {code}")
            _NATIVE = matches[0]
        else:
            from ..models.registry import build_model

            _NATIVE = build_model(MODEL_IDS[code])
    return _NATIVE


@dataclass(frozen=True)
class NativeDailyAdapter:
    code: str

    @property
    def model_id(self) -> str:
        return MODEL_IDS[self.code]

    def simulate_daily_log_returns(self, training: TrainingData, context: ForecastContext):
        model = _native(self.code)
        clear = getattr(_SOURCE_MODULE, "clear_path_cache", None)
        if clear is not None:
            clear()
        diagnostics = getattr(_SOURCE_MODULE, "FIT_DIAGNOSTICS", None)
        if diagnostics is not None:
            diagnostics.clear()
        timings = getattr(_SOURCE_MODULE, "TIMINGS", None)
        if timings is not None:
            timings.clear()
        try:
            # Forward the complete history, calendar, requested paths/horizon,
            # and untouched seed context. Source-specific seed rules prevail.
            return model.simulate_daily_log_returns(training, context)
        finally:
            if clear is not None:
                clear()


@dataclass(frozen=True)
class NativeTerminalAdapter(NativeDailyAdapter):
    forecast_output_semantics = TERMINAL_FORECAST_SEMANTICS

    def simulate_terminal_log_returns(self, training: TrainingData, context: ForecastContext):
        return _native(self.code).simulate_terminal_log_returns(training, context)


def build_m001():
    return NativeDailyAdapter("M001")


def build_m035():
    # The verified Student-t source returns terminal cumulative ensembles.
    # Forward those directly; differencing/re-summing changes arithmetic.
    return NativeTerminalAdapter("M035")


def build_m052():
    return NativeDailyAdapter("M052")


def build_m176():
    return NativeDailyAdapter("M176")


def build_m198():
    return NativeDailyAdapter("M198")


def build_m202():
    return NativeDailyAdapter("M202")


def build_m233():
    return NativeDailyAdapter("M233")


def build_m244():
    return NativeDailyAdapter("M244")


def build_m247():
    return NativeDailyAdapter("M247")


def build_m261():
    return NativeDailyAdapter("M261")


FRENCH_CATALOGUE_FACTORIES = {
    "M001": build_m001, "M035": build_m035, "M052": build_m052,
    "M176": build_m176, "M198": build_m198, "M202": build_m202,
    "M233": build_m233, "M244": build_m244, "M247": build_m247,
    "M261": build_m261,
}


def initialized_m261_owner() -> ModuleType:
    """Share one exact source installation with the frozen M261 refinements."""
    _native("M261")
    assert _SOURCE_MODULE is not None
    return _SOURCE_MODULE
