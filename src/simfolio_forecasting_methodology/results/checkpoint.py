"""Atomic checkpoint storage and immutable execution identities."""

from __future__ import annotations

import hashlib
import json
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

CHECKPOINT_SCHEMA_VERSION = "canonical-175-checkpoint-v2"


class CheckpointIdentityError(RuntimeError):
    """Raised when a checkpoint is reused for a different execution identity."""


class CheckpointCorruptError(RuntimeError):
    """Raised when checkpoint metadata cannot be trusted."""


def stable_json_bytes(payload: Any) -> bytes:
    """Serialize JSON identity payloads without platform-dependent whitespace."""

    return json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def stable_digest(payload: Any) -> str:
    return hashlib.sha256(stable_json_bytes(payload)).hexdigest()


@dataclass(frozen=True)
class TaskIdentity:
    """The immutable inputs that identify one forecast/scoring task."""

    task_id: str
    model_id: str
    portfolio_id: str
    origin_label: str
    horizon_days: int
    seed: int
    training_digest: str
    realized_digest: str
    future_dates_digest: str | None
    origin_date: str | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "model_id": self.model_id,
            "portfolio_id": self.portfolio_id,
            "origin_label": self.origin_label,
            "horizon_days": self.horizon_days,
            "seed": self.seed,
            "training_digest": self.training_digest,
            "realized_digest": self.realized_digest,
            "future_dates_digest": self.future_dates_digest,
            "origin_date": self.origin_date,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TaskIdentity:
        required = {
            "task_id",
            "model_id",
            "portfolio_id",
            "origin_label",
            "horizon_days",
            "seed",
            "training_digest",
            "realized_digest",
            "future_dates_digest",
            "origin_date",
        }
        missing = sorted(required - set(payload))
        if missing:
            raise CheckpointCorruptError(f"task identity is missing fields: {missing}")
        try:
            return cls(
                task_id=str(payload["task_id"]),
                model_id=str(payload["model_id"]),
                portfolio_id=str(payload["portfolio_id"]),
                origin_label=str(payload["origin_label"]),
                horizon_days=int(payload["horizon_days"]),
                seed=int(payload["seed"]),
                training_digest=str(payload["training_digest"]),
                realized_digest=str(payload["realized_digest"]),
                future_dates_digest=(
                    None
                    if payload["future_dates_digest"] is None
                    else str(payload["future_dates_digest"])
                ),
                origin_date=None if payload["origin_date"] is None else str(payload["origin_date"]),
            )
        except (TypeError, ValueError) as exc:
            raise CheckpointCorruptError("task identity contains an invalid value") from exc


@dataclass(frozen=True)
class ExecutionManifest:
    """Execution identity shared by every task in one checkpoint."""

    schema_version: str
    model_id: str
    experiment_id: str
    protocol_id: str
    execution_variant: str
    membership_digest: str
    source_revision: str
    simulations: int
    task_count: int
    expected_cell_caps: tuple[tuple[str, int], ...]
    tasks: tuple[TaskIdentity, ...]
    specification_fingerprint: str | None
    implementation_digest: str | None
    dependency_identity: dict[str, Any]
    seed_contract: str
    protocol_fingerprint: str | None
    dataset_fingerprint: str | None
    panel_fingerprint: str | None
    manifest_fingerprint: str

    @classmethod
    def create(
        cls,
        *,
        model_id: str,
        experiment_id: str,
        protocol_id: str,
        execution_variant: str,
        membership_digest: str,
        source_revision: str,
        simulations: int,
        tasks: tuple[TaskIdentity, ...],
        expected_cell_caps: tuple[tuple[str, int], ...],
        specification_fingerprint: str | None,
        implementation_digest: str | None,
        dependency_identity: Mapping[str, Any],
        seed_contract: str,
        protocol_fingerprint: str | None,
        dataset_fingerprint: str | None,
        panel_fingerprint: str | None,
    ) -> ExecutionManifest:
        if not tasks:
            raise ValueError("an execution manifest requires at least one task")
        if len({item.task_id for item in tasks}) != len(tasks):
            raise ValueError("execution manifest contains duplicate task IDs")
        if int(simulations) < 1:
            raise ValueError("execution manifest requires at least one simulation")
        normalized_caps: list[tuple[str, int]] = []
        seen_portfolios: set[str] = set()
        for portfolio, horizon in expected_cell_caps:
            normalized_portfolio = str(portfolio)
            normalized_horizon = int(horizon)
            if not normalized_portfolio or normalized_horizon < 1:
                raise ValueError("execution manifest cell caps must be nonempty and positive")
            if normalized_portfolio in seen_portfolios:
                raise ValueError("execution manifest contains duplicate portfolio cell caps")
            seen_portfolios.add(normalized_portfolio)
            normalized_caps.append((normalized_portfolio, normalized_horizon))
        if not normalized_caps:
            raise ValueError("execution manifest requires at least one expected cell cap")
        if not isinstance(dependency_identity, Mapping):
            raise TypeError("execution manifest dependency identity must be an object")
        normalized_dependencies = json.loads(stable_json_bytes(dependency_identity))
        if not isinstance(normalized_dependencies, dict):
            raise TypeError("execution manifest dependency identity must serialize to an object")
        if not str(seed_contract).strip():
            raise ValueError("execution manifest seed contract must be nonempty")
        if not str(execution_variant).strip():
            raise ValueError("execution manifest execution variant must be nonempty")
        payload = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "model_id": model_id,
            "experiment_id": experiment_id,
            "protocol_id": protocol_id,
            "execution_variant": str(execution_variant),
            "membership_digest": membership_digest,
            "source_revision": source_revision,
            "simulations": int(simulations),
            "task_count": len(tasks),
            "expected_cell_caps": [
                [portfolio, horizon] for portfolio, horizon in normalized_caps
            ],
            "tasks": [item.to_dict() for item in tasks],
            "specification_fingerprint": specification_fingerprint,
            "implementation_digest": implementation_digest,
            "dependency_identity": normalized_dependencies,
            "seed_contract": str(seed_contract),
            "protocol_fingerprint": protocol_fingerprint,
            "dataset_fingerprint": dataset_fingerprint,
            "panel_fingerprint": panel_fingerprint,
        }
        return cls(
            schema_version=CHECKPOINT_SCHEMA_VERSION,
            model_id=str(model_id),
            experiment_id=str(experiment_id),
            protocol_id=str(protocol_id),
            execution_variant=str(execution_variant),
            membership_digest=str(membership_digest),
            source_revision=str(source_revision),
            simulations=int(simulations),
            task_count=len(tasks),
            expected_cell_caps=tuple(normalized_caps),
            tasks=tuple(tasks),
            specification_fingerprint=specification_fingerprint,
            implementation_digest=implementation_digest,
            dependency_identity=normalized_dependencies,
            seed_contract=str(seed_contract),
            protocol_fingerprint=protocol_fingerprint,
            dataset_fingerprint=dataset_fingerprint,
            panel_fingerprint=panel_fingerprint,
            manifest_fingerprint=stable_digest(payload),
        )

    def identity_payload(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "experiment_id": self.experiment_id,
            "protocol_id": self.protocol_id,
            "execution_variant": self.execution_variant,
            "membership_digest": self.membership_digest,
            "source_revision": self.source_revision,
            "simulations": self.simulations,
            "task_count": self.task_count,
            "expected_cell_caps": [
                [portfolio, horizon] for portfolio, horizon in self.expected_cell_caps
            ],
            "tasks": [item.to_dict() for item in self.tasks],
            "specification_fingerprint": self.specification_fingerprint,
            "implementation_digest": self.implementation_digest,
            "dependency_identity": self.dependency_identity,
            "seed_contract": self.seed_contract,
            "protocol_fingerprint": self.protocol_fingerprint,
            "dataset_fingerprint": self.dataset_fingerprint,
            "panel_fingerprint": self.panel_fingerprint,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["manifest_fingerprint"] = self.manifest_fingerprint
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ExecutionManifest:
        try:
            tasks = tuple(TaskIdentity.from_dict(item) for item in payload["tasks"])
            caps = tuple(
                (str(item[0]), int(item[1])) for item in payload["expected_cell_caps"]
            )
            dependencies = payload["dependency_identity"]
            if not isinstance(dependencies, dict):
                raise TypeError("dependency identity must be an object")
            manifest = cls(
                schema_version=str(payload["schema_version"]),
                model_id=str(payload["model_id"]),
                experiment_id=str(payload["experiment_id"]),
                protocol_id=str(payload["protocol_id"]),
                execution_variant=str(payload["execution_variant"]),
                membership_digest=str(payload["membership_digest"]),
                source_revision=str(payload["source_revision"]),
                simulations=int(payload["simulations"]),
                task_count=int(payload["task_count"]),
                expected_cell_caps=caps,
                tasks=tasks,
                specification_fingerprint=(
                    None
                    if payload.get("specification_fingerprint") is None
                    else str(payload["specification_fingerprint"])
                ),
                implementation_digest=(
                    None
                    if payload.get("implementation_digest") is None
                    else str(payload["implementation_digest"])
                ),
                dependency_identity=dependencies,
                seed_contract=str(payload["seed_contract"]),
                protocol_fingerprint=(
                    None
                    if payload.get("protocol_fingerprint") is None
                    else str(payload["protocol_fingerprint"])
                ),
                dataset_fingerprint=(
                    None
                    if payload.get("dataset_fingerprint") is None
                    else str(payload["dataset_fingerprint"])
                ),
                panel_fingerprint=(
                    None
                    if payload.get("panel_fingerprint") is None
                    else str(payload["panel_fingerprint"])
                ),
                manifest_fingerprint=str(payload["manifest_fingerprint"]),
            )
        except (KeyError, TypeError, ValueError, IndexError) as exc:
            raise CheckpointCorruptError("execution manifest is malformed") from exc
        if manifest.schema_version != CHECKPOINT_SCHEMA_VERSION:
            raise CheckpointCorruptError(
                f"unsupported checkpoint schema: {manifest.schema_version!r}"
            )
        if manifest.task_count != len(manifest.tasks):
            raise CheckpointCorruptError("execution manifest task count is inconsistent")
        if len({item.task_id for item in manifest.tasks}) != len(manifest.tasks):
            raise CheckpointCorruptError("execution manifest contains duplicate task IDs")
        if len({portfolio for portfolio, _ in manifest.expected_cell_caps}) != len(
            manifest.expected_cell_caps
        ):
            raise CheckpointCorruptError("execution manifest contains duplicate portfolio cell caps")
        if any(horizon < 1 for _, horizon in manifest.expected_cell_caps):
            raise CheckpointCorruptError("execution manifest contains a nonpositive cell cap")
        if stable_digest(manifest.identity_payload()) != manifest.manifest_fingerprint:
            raise CheckpointCorruptError("execution manifest fingerprint is invalid")
        return manifest

    @property
    def expected_cells(self) -> tuple[tuple[str, int], ...]:
        """Expand compact contiguous caps for callers that need exact cell keys."""

        return tuple(
            (portfolio, horizon)
            for portfolio, cap in self.expected_cell_caps
            for horizon in range(1, cap + 1)
        )


@dataclass(frozen=True)
class TaskResult:
    """One durable task outcome, including failures visible to the caller."""

    task_id: str
    status: str
    losses: tuple[float, ...] = ()
    error_type: str | None = None
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "task_id": self.task_id,
            "status": self.status,
            "losses": list(self.losses),
        }
        if self.error_type is not None:
            payload["error_type"] = self.error_type
        if self.error_message is not None:
            payload["error_message"] = self.error_message
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> TaskResult:
        try:
            result = cls(
                task_id=str(payload["task_id"]),
                status=str(payload["status"]),
                losses=tuple(float(item) for item in payload.get("losses", ())),
                error_type=(
                    None
                    if payload.get("error_type") is None
                    else str(payload["error_type"])
                ),
                error_message=(
                    None if payload.get("error_message") is None else str(payload["error_message"])
                ),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise CheckpointCorruptError("task result is malformed") from exc
        if result.status not in {"pending", "completed", "failed"}:
            raise CheckpointCorruptError(f"unknown task result status: {result.status!r}")
        if result.status == "completed" and (result.error_type or result.error_message):
            raise CheckpointCorruptError("completed task contains an error")
        return result


def _atomic_write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        json.dump(payload, handle, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        handle.write("\n")
    temporary.replace(path)


class CheckpointStore:
    """Persist one manifest and one atomic file per task."""

    def __init__(self, root: Path, manifest: ExecutionManifest, *, resume: bool) -> None:
        self.root = Path(root)
        self.manifest = manifest
        self._expected_task_ids = frozenset(item.task_id for item in manifest.tasks)
        self.manifest_path = self.root / "manifest.json"
        self.summary_path = self.root / "summary.json"
        self.tasks_path = self.root / "tasks"
        self.root.mkdir(parents=True, exist_ok=True)
        if self.manifest_path.exists():
            existing = self._load_manifest()
            if existing.manifest_fingerprint != manifest.manifest_fingerprint:
                raise CheckpointIdentityError(
                    "checkpoint manifest fingerprint differs; refusing to resume with changed "
                    "model, data, protocol, simulations, or task identity"
                )
            if not resume:
                raise CheckpointIdentityError(
                    "checkpoint already exists; pass resume=True to continue the same manifest"
                )
            self._load_task_state()
        else:
            if self.tasks_path.exists() and any(self.tasks_path.iterdir()):
                raise CheckpointCorruptError("checkpoint directory has state but no manifest")
            _atomic_write_json(self.manifest_path, manifest.to_dict())
            self.tasks_path.mkdir(parents=True, exist_ok=True)
            self._status_map = {item.task_id: "pending" for item in manifest.tasks}
            self._result_map: dict[str, TaskResult] = {}

    def _load_manifest(self) -> ExecutionManifest:
        try:
            with self.manifest_path.open(encoding="utf-8") as handle:
                return ExecutionManifest.from_dict(json.load(handle))
        except (OSError, json.JSONDecodeError) as exc:
            raise CheckpointCorruptError("cannot read checkpoint manifest") from exc

    def _load_task_result(self, path: Path, task_id: str) -> TaskResult:
        try:
            with path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise CheckpointCorruptError(f"cannot read task checkpoint {task_id}") from exc
        if not isinstance(payload, dict):
            raise CheckpointCorruptError(f"task checkpoint is not an object for {task_id}")
        stored_digest = payload.get("result_digest")
        result_payload = {key: value for key, value in payload.items() if key != "result_digest"}
        if result_payload.get("manifest_fingerprint") != self.manifest.manifest_fingerprint:
            raise CheckpointCorruptError(f"task result manifest binding mismatch for {task_id}")
        result = TaskResult.from_dict(result_payload)
        if stored_digest != stable_digest(result_payload):
            raise CheckpointCorruptError(f"task result digest mismatch for {task_id}")
        if result.task_id != task_id or result.status not in {"completed", "failed"}:
            raise CheckpointCorruptError(f"task checkpoint identity/status drifted for {task_id}")
        return result

    def _load_task_state(self) -> None:
        self.tasks_path.mkdir(parents=True, exist_ok=True)
        expected = {item.task_id for item in self.manifest.tasks}
        actual_files = {
            path.stem for path in self.tasks_path.glob("*.json") if path.is_file()
        }
        unknown = actual_files - expected
        if unknown:
            raise CheckpointCorruptError(
                f"checkpoint contains task records outside the manifest: {sorted(unknown)[:3]}"
            )
        self._status_map = {task_id: "pending" for task_id in expected}
        self._result_map = {}
        for task_id in self.manifest.tasks:
            path = self.tasks_path / f"{task_id.task_id}.json"
            if not path.is_file():
                continue
            result = self._load_task_result(path, task_id.task_id)
            self._status_map[task_id.task_id] = result.status
            self._result_map[task_id.task_id] = result

    def statuses(self) -> dict[str, str]:
        return dict(self._status_map)

    def status(self, task_id: str) -> str:
        statuses = self.statuses()
        if task_id not in statuses:
            raise CheckpointCorruptError(f"unknown task ID in checkpoint lookup: {task_id}")
        return statuses[task_id]

    def results(self) -> dict[str, TaskResult]:
        return dict(self._result_map)

    def write_result(self, result: TaskResult) -> None:
        if result.task_id not in self._expected_task_ids:
            raise CheckpointCorruptError(f"cannot store unknown task ID {result.task_id}")
        if result.status not in {"completed", "failed"}:
            raise ValueError("checkpoint results must be completed or failed")
        result_payload = result.to_dict()
        result_payload["manifest_fingerprint"] = self.manifest.manifest_fingerprint
        result_payload["result_digest"] = stable_digest(result_payload)
        _atomic_write_json(self.tasks_path / f"{result.task_id}.json", result_payload)
        self._status_map[result.task_id] = result.status
        self._result_map[result.task_id] = result

    def write_summary(self, payload: Mapping[str, Any]) -> None:
        """Persist the latest aggregate/status view after task state is durable."""

        _atomic_write_json(self.summary_path, dict(payload))

    def summary(self) -> dict[str, int]:
        counts = {"pending": 0, "completed": 0, "failed": 0}
        for status in self.statuses().values():
            if status not in counts:
                raise CheckpointCorruptError(f"unknown task status: {status}")
            counts[status] += 1
        return counts
