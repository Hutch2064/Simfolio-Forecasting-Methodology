"""Catalogue-independent execution interfaces for dense OOS evaluation."""

from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
import platform
import re
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from .evaluation import CellAccumulator, empirical_crps_by_horizon
from .protocol import CANONICAL_PROTOCOL_ID

CANONICAL_EXPERIMENT_ID = "canonical-dense-oos-2026-08-23"
TERMINAL_FORECAST_SEMANTICS = "terminal_log_return_ensemble_by_horizon"


@dataclass(frozen=True)
class PortfolioPolicy:
    tickers: tuple[str, ...]
    weights: tuple[float, ...]
    rebalance: str

    def validate(self) -> None:
        if not self.tickers:
            raise ValueError("portfolio policy requires at least one ticker")
        if len(self.tickers) != len(self.weights):
            raise ValueError("ticker and weight lengths differ")
        if not np.isclose(sum(self.weights), 1.0, atol=1e-10):
            raise ValueError("portfolio weights must sum to one")


@dataclass(frozen=True)
class TrainingData:
    portfolio_log_returns: np.ndarray
    asset_log_returns: np.ndarray | None = None
    policy: PortfolioPolicy | None = None
    training_dates: np.ndarray | None = None

    def validate(self) -> None:
        portfolio = np.asarray(self.portfolio_log_returns, dtype=np.float64)
        if portfolio.ndim != 1 or portfolio.size < 1 or not np.all(np.isfinite(portfolio)):
            raise ValueError("portfolio_log_returns must be a finite one-dimensional series")
        if self.training_dates is not None:
            dates = np.asarray(self.training_dates, dtype="datetime64[ns]")
            if dates.shape != portfolio.shape or np.any(np.isnat(dates)):
                raise ValueError("training_dates must align with portfolio history")
            if np.any(dates[1:] <= dates[:-1]):
                raise ValueError("training_dates must be strictly increasing")
        if self.asset_log_returns is not None:
            assets = np.asarray(self.asset_log_returns, dtype=np.float64)
            if assets.ndim != 2 or assets.shape[0] != portfolio.size:
                raise ValueError("asset_log_returns must align row-wise with portfolio history")
            if not np.all(np.isfinite(assets)):
                raise ValueError("asset_log_returns must be finite")
            if self.policy is None:
                raise ValueError("asset-level training data requires a portfolio policy")
            self.policy.validate()
            if assets.shape[1] != len(self.policy.tickers):
                raise ValueError("asset columns must align with policy tickers")


@dataclass(frozen=True)
class ForecastContext:
    model_id: str
    portfolio_id: str
    origin_label: str
    horizon_days: int
    simulations: int
    seed: int
    future_dates: np.ndarray | None = None
    origin_date: str | None = None


@runtime_checkable
class ForecastModel(Protocol):
    model_id: str

    def simulate_daily_log_returns(
        self,
        training: TrainingData,
        context: ForecastContext,
    ) -> np.ndarray:
        """Return a matrix shaped ``(simulations, horizon_days)``."""
        ...


@dataclass(frozen=True)
class OriginTask:
    portfolio_id: str
    origin_label: str
    training: TrainingData
    realized_future_daily_log_returns: np.ndarray
    seed: int
    future_dates: np.ndarray | None = None
    origin_date: str | None = None

    @property
    def horizon_days(self) -> int:
        return int(np.asarray(self.realized_future_daily_log_returns).size)

    def validate(self) -> None:
        self.training.validate()
        realized = np.asarray(self.realized_future_daily_log_returns, dtype=np.float64)
        if realized.ndim != 1 or realized.size < 1 or not np.all(np.isfinite(realized)):
            raise ValueError("realized future returns must be a finite nonempty vector")
        if self.future_dates is not None and len(self.future_dates) != realized.size:
            raise ValueError("future_dates must align with realized future returns")


def evaluate_origin_task(
    model: ForecastModel,
    task: OriginTask,
    *,
    simulations: int,
) -> np.ndarray:
    """Generate one coherent path matrix and score every daily horizon."""
    task.validate()
    context = ForecastContext(
        model_id=model.model_id,
        portfolio_id=task.portfolio_id,
        origin_label=task.origin_label,
        horizon_days=task.horizon_days,
        simulations=int(simulations),
        seed=int(task.seed),
        future_dates=None if task.future_dates is None else np.asarray(task.future_dates),
        origin_date=task.origin_date,
    )
    expected_shape = (int(simulations), task.horizon_days)
    terminal_method = getattr(model, "simulate_terminal_log_returns", None)
    if callable(terminal_method):
        semantics = getattr(model, "forecast_output_semantics", None)
        if semantics != TERMINAL_FORECAST_SEMANTICS:
            raise ValueError(
                "terminal forecast method must declare "
                f"forecast_output_semantics={TERMINAL_FORECAST_SEMANTICS!r}"
            )
        terminal_samples = np.asarray(terminal_method(task.training, context), dtype=np.float64)
        if terminal_samples.shape != expected_shape:
            raise ValueError(
                f"model returned terminal ensemble shape {terminal_samples.shape}; "
                f"expected {expected_shape}"
            )
    else:
        daily_paths = np.asarray(
            model.simulate_daily_log_returns(task.training, context), dtype=np.float64
        )
        if daily_paths.shape != expected_shape:
            raise ValueError(
                f"model returned path shape {daily_paths.shape}; expected {expected_shape}"
            )
        terminal_samples = np.cumsum(daily_paths, axis=1, dtype=np.float64)
    if not np.all(np.isfinite(terminal_samples)):
        raise ValueError("model returned nonfinite forecast paths")
    realized_terminal = np.cumsum(
        np.asarray(task.realized_future_daily_log_returns, dtype=np.float64), dtype=np.float64
    )
    return empirical_crps_by_horizon(terminal_samples, realized_terminal)


def evaluate_model(
    model: ForecastModel,
    tasks: list[OriginTask],
    *,
    simulations: int,
) -> CellAccumulator:
    """Evaluate a model over origin tasks using streaming cell-first aggregation."""
    accumulator = CellAccumulator()
    for task in tasks:
        losses = evaluate_origin_task(model, task, simulations=simulations)
        accumulator.add_vector(task.portfolio_id, losses)
    return accumulator


def _array_digest(values: np.ndarray | None, *, dtype: str | None = None) -> str | None:
    """Hash array shape, dtype, and bytes for a deterministic task identity."""

    if values is None:
        return None
    array = np.asarray(values, dtype=dtype) if dtype is not None else np.asarray(values)
    contiguous = np.ascontiguousarray(array)
    payload = {
        "dtype": str(contiguous.dtype),
        "shape": list(contiguous.shape),
        "sha256": hashlib.sha256(contiguous.tobytes(order="C")).hexdigest(),
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _training_digest(training: TrainingData) -> str:
    policy = None
    if training.policy is not None:
        policy = {
            "tickers": list(training.policy.tickers),
            "weights": [float(item) for item in training.policy.weights],
            "rebalance": training.policy.rebalance,
        }
    payload = {
        "portfolio_log_returns": _array_digest(training.portfolio_log_returns, dtype="<f8"),
        "asset_log_returns": _array_digest(training.asset_log_returns, dtype="<f8"),
        "training_dates": _array_digest(training.training_dates, dtype="datetime64[ns]"),
        "policy": policy,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _implementation_digest(model, record: dict[str, object]) -> str | None:
    explicit = getattr(model, "implementation_digest", None)
    if isinstance(explicit, str) and explicit:
        return explicit
    source_files: set[Path] = set()
    if type(model).__module__.startswith("simfolio_forecasting_methodology."):
        # A model adapter delegates to numerical helpers in other modules.
        # Bind the installed executable closure, including scoring and seeds,
        # so changing a helper cannot silently reuse old task losses.
        package_root = Path(__file__).resolve().parent
        source_files.update(package_root.rglob("*.py"))
        source_files.update(
            path for path in (package_root / "resources").rglob("*")
            if path.is_file() and path.suffix in {".json", ".yaml", ".yml", ".csv", ".gz"}
        )
    try:
        source_file = inspect.getsourcefile(type(model))
        if source_file is not None:
            source_files.add(Path(source_file))
    except (OSError, TypeError):
        pass
    process_factory = getattr(model, "process_worker_factory", None)
    if callable(process_factory):
        try:
            source_file = inspect.getsourcefile(process_factory)
            if source_file is not None:
                source_files.add(Path(source_file))
        except (OSError, TypeError):
            pass
    if source_files:
        digests: list[str] = []
        for source_file in sorted(source_files, key=lambda item: item.as_posix()):
            try:
                with source_file.open("rb") as handle:
                    digests.append(hashlib.sha256(handle.read()).hexdigest())
            except OSError:
                continue
        if len(digests) == 1:
            return digests[0]
        if digests:
            return hashlib.sha256("".join(sorted(digests)).encode("ascii")).hexdigest()
    source_digest = record.get("source_artifact_digest", {})
    if isinstance(source_digest, dict):
        code = source_digest.get("source_code", {})
        if isinstance(code, dict) and isinstance(code.get("sha256"), str):
            return code["sha256"]
    return None


def _dependency_json_default(value):
    if isinstance(value, Mapping):
        return dict(value)
    raise TypeError(f"dependency identity contains non-JSON value: {type(value).__name__}")


_NUMERICAL_THREAD_ENVIRONMENT_KEYS = (
    "BLIS_NUM_THREADS",
    "GOTO_NUM_THREADS",
    "MKL_DYNAMIC",
    "MKL_NUM_THREADS",
    "NUMBA_NUM_THREADS",
    "NUMBA_THREADING_LAYER",
    "NUMEXPR_NUM_THREADS",
    "OMP_DYNAMIC",
    "OMP_NUM_THREADS",
    "OMP_PLACES",
    "OMP_PROC_BIND",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)
_SAFE_IDENTITY_TEXT = re.compile(r"^[A-Za-z0-9_.:+,\- ]+$")


def _safe_identity_text(value, *, max_length: int = 128) -> str | None:
    """Keep environment metadata scalar and free of host/path information."""

    if value is None:
        return None
    text = str(value).strip()
    if not text or len(text) > max_length or not _SAFE_IDENTITY_TEXT.fullmatch(text):
        return "<redacted>"
    return text


def _safe_thread_setting(value) -> str | None:
    """Serialize an allowlisted thread setting without copying arbitrary env text."""

    return _safe_identity_text(value, max_length=64)


def _build_library_identity(config_module) -> dict[str, object]:
    """Read static BLAS/LAPACK build metadata without loading runtime libraries."""

    config = getattr(config_module, "CONFIG", None)
    dependencies = config.get("Build Dependencies") if isinstance(config, Mapping) else None
    if not isinstance(dependencies, Mapping):
        dependencies = {}
    result: dict[str, object] = {}
    fields = (
        "name",
        "version",
        "detection method",
        "found",
        "has ilp64",
        "cython blas ilp64",
        "openblas configuration",
    )
    for library in ("blas", "lapack"):
        details = dependencies.get(library)
        if not isinstance(details, Mapping):
            result[library] = {"name": None, "version": None}
            continue
        normalized: dict[str, object] = {}
        for field in fields:
            if field not in details:
                continue
            value = details[field]
            if isinstance(value, (bool, int, float)) or value is None:
                normalized[field] = value
            else:
                normalized[field] = _safe_identity_text(value)
        result[library] = normalized
    return result


def _package_build_identity(package: str, config_module) -> dict[str, object]:
    """Return package version and static numerical backend metadata."""

    try:
        package_version = importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        package_version = None
    return {
        "version": package_version,
        "blas_lapack": _build_library_identity(config_module),
    }


def _numerical_environment_identity() -> dict[str, object]:
    """Return stable numerical runtime identity used by checkpoint manifests.

    This deliberately uses static package build metadata and an allowlist of
    thread settings.  Runtime library discovery (for example, threadpoolctl)
    is avoided because its answer depends on which extension modules have
    already been imported.
    """

    try:
        scipy_config = importlib.import_module("scipy.__config__")
    except ImportError:
        scipy_config = None
    scipy_identity = _package_build_identity("scipy", scipy_config)
    return {
        "platform": {
            "system": _safe_identity_text(platform.system()),
            "release": _safe_identity_text(platform.release()),
            "machine": _safe_identity_text(platform.machine()),
            "version": _safe_identity_text(platform.version()),
        },
        "interpreter": {
            "implementation": _safe_identity_text(platform.python_implementation()),
            "version": _safe_identity_text(platform.python_version()),
            "compiler": _safe_identity_text(platform.python_compiler()),
            "build": [_safe_identity_text(value) for value in platform.python_build()],
            "cache_tag": _safe_identity_text(getattr(sys.implementation, "cache_tag", None)),
        },
        "packages": {
            "numpy": _package_build_identity("numpy", np.__config__),
            "scipy": scipy_identity,
        },
        "thread_settings": {
            "environment": {
                key: _safe_thread_setting(os.environ.get(key))
                for key in _NUMERICAL_THREAD_ENVIRONMENT_KEYS
            },
            "numpy_seterr": {
                key: _safe_thread_setting(value) for key, value in sorted(np.geterr().items())
            },
        },
    }


def _dependency_identity(model, record: dict[str, object]) -> dict[str, object]:
    explicit = getattr(model, "dependency_identity", None)
    if explicit is not None:
        if not isinstance(explicit, Mapping):
            raise ValueError("model dependency_identity must be a mapping")
        identity = json.loads(json.dumps(explicit, sort_keys=True, default=_dependency_json_default))
    else:
        identity = {}
    identity["python"] = platform.python_version()
    identity["numpy"] = np.__version__
    for package in ("pandas", "scipy", "numba", "arch", "statsmodels", "scikit-learn"):
        try:
            identity[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            identity[package] = None
    identity["numerical_environment"] = _numerical_environment_identity()
    required = record.get("required_dependencies")
    if required is not None:
        identity["ledger_required_dependencies"] = required
    return identity


def task_identity(task: OriginTask, model_id: str, simulations: int):
    """Return the durable identity of one constructed forecast task."""

    from .results.checkpoint import TaskIdentity, stable_digest

    task.validate()
    future_dates_digest = _array_digest(task.future_dates, dtype="datetime64[ns]")
    payload = {
        "model_id": str(model_id),
        "simulations": int(simulations),
        "portfolio_id": task.portfolio_id,
        "origin_label": task.origin_label,
        "horizon_days": task.horizon_days,
        "seed": int(task.seed),
        "training_digest": _training_digest(task.training),
        "realized_digest": _array_digest(task.realized_future_daily_log_returns, dtype="<f8"),
        "future_dates_digest": future_dates_digest,
        "origin_date": task.origin_date,
    }
    return TaskIdentity(
        task_id=stable_digest(payload),
        model_id=str(model_id),
        portfolio_id=str(task.portfolio_id),
        origin_label=str(task.origin_label),
        horizon_days=int(task.horizon_days),
        seed=int(task.seed),
        training_digest=str(payload["training_digest"]),
        realized_digest=str(payload["realized_digest"]),
        future_dates_digest=future_dates_digest,
        origin_date=None if task.origin_date is None else str(task.origin_date),
    )


def build_execution_manifest(
    model_id: str,
    tasks: list[OriginTask] | tuple[OriginTask, ...],
    *,
    simulations: int,
    model: ForecastModel | None = None,
    execution_variant: str = "canonical",
):
    """Build a source-linked manifest from the canonical task constructor."""

    from .catalogue import EXPECTED_MEMBERSHIP_DIGEST, canonical_model, load_canonical_ledger
    from .results.checkpoint import ExecutionManifest, stable_digest

    record = canonical_model(model_id)
    ledger = load_canonical_ledger()
    identities = tuple(task_identity(task, model_id, int(simulations)) for task in tasks)
    if not identities:
        raise ValueError("an execution manifest requires at least one origin task")
    expected_caps: list[tuple[str, int]] = []
    cap_positions: dict[str, int] = {}
    for identity in identities:
        position = cap_positions.get(identity.portfolio_id)
        if position is None:
            cap_positions[identity.portfolio_id] = len(expected_caps)
            expected_caps.append((identity.portfolio_id, identity.horizon_days))
        else:
            portfolio, cap = expected_caps[position]
            expected_caps[position] = (portfolio, max(cap, identity.horizon_days))
    identity_policy = ledger.get("identity_policy", {})
    explicit_specification = getattr(model, "specification_fingerprint", None)
    specification = record.get("specification_fingerprint")
    specification_fingerprint = (
        explicit_specification
        if isinstance(explicit_specification, str) and explicit_specification
        else (
            specification.get("value")
            if isinstance(specification, dict)
            and isinstance(specification.get("value"), str)
            else None
        )
    )
    dependencies = _dependency_identity(model, record) if model is not None else {
        "ledger_required_dependencies": record.get("required_dependencies")
    }
    implementation_digest = _implementation_digest(model, record) if model is not None else None
    default_seed_contract = "origin_task.seed_to_forecast_context.seed.v1"
    requested_seed_contract = getattr(model, "seed_contract", default_seed_contract)
    seed_contract = str(requested_seed_contract or default_seed_contract)
    if execution_variant == "canonical":
        manifest_experiment_id = str(record["experiment_id"])
        manifest_protocol_id = CANONICAL_PROTOCOL_ID
    else:
        manifest_experiment_id = f"{record['experiment_id']}::{execution_variant}"
        manifest_protocol_id = f"{CANONICAL_PROTOCOL_ID}::{execution_variant}"
    dataset_fingerprint = record.get("dataset_fingerprint") or identity_policy.get("dataset_fingerprint")
    panel_fingerprint = record.get("panel_fingerprint") or identity_policy.get("panel_fingerprint")
    if execution_variant != "canonical":
        dataset_fingerprint = stable_digest({
            "kind": "noncanonical_task_inputs",
            "inputs": [(item.training_digest, item.realized_digest) for item in identities],
        })
        panel_fingerprint = stable_digest({
            "kind": "noncanonical_task_panel",
            "portfolios": [(task.portfolio_id, None if task.training.policy is None
                            else task.training.policy.__dict__) for task in tasks],
        })
    return ExecutionManifest.create(
        model_id=model_id,
        experiment_id=manifest_experiment_id,
        protocol_id=manifest_protocol_id,
        execution_variant=execution_variant,
        membership_digest=EXPECTED_MEMBERSHIP_DIGEST,
        source_revision=str(record["source_revision"]),
        simulations=int(simulations),
        tasks=identities,
        expected_cell_caps=tuple(expected_caps),
        specification_fingerprint=specification_fingerprint,
        implementation_digest=implementation_digest,
        dependency_identity=dependencies,
        seed_contract=seed_contract,
        protocol_fingerprint=record.get("protocol_fingerprint")
        or identity_policy.get("protocol_fingerprint"),
        dataset_fingerprint=dataset_fingerprint,
        panel_fingerprint=panel_fingerprint,
    )


def execute_model_checkpointed(
    model: ForecastModel,
    tasks: list[OriginTask] | tuple[OriginTask, ...],
    *,
    simulations: int,
    checkpoint_dir: Path,
    workers: int = 1,
    resume: bool = False,
    execution_variant: str = "canonical",
    progress_callback=None,
):
    """Execute a canonical model with durable task checkpoints."""

    from .results.executor import execute_checkpointed

    manifest = build_execution_manifest(
        model.model_id,
        tasks,
        simulations=int(simulations),
        model=model,
        execution_variant=execution_variant,
    )
    return execute_checkpointed(
        model,
        tasks,
        manifest=manifest,
        checkpoint_dir=Path(checkpoint_dir),
        simulations=int(simulations),
        workers=int(workers),
        resume=bool(resume),
        progress_callback=progress_callback,
    )


__all__ = [
    "CANONICAL_EXPERIMENT_ID",
    "CANONICAL_PROTOCOL_ID",
    "TERMINAL_FORECAST_SEMANTICS",
    "ForecastContext",
    "ForecastModel",
    "OriginTask",
    "PortfolioPolicy",
    "TrainingData",
    "build_execution_manifest",
    "evaluate_model",
    "evaluate_origin_task",
    "execute_model_checkpointed",
    "task_identity",
]
