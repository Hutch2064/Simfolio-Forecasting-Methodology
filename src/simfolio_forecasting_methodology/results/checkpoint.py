"""Atomic checkpoint storage and immutable execution identities."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tempfile
from typing import Any, Mapping


CHECKPOINT_SCHEMA_VERSION = "canonical-175-checkpoint-v1"


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
    def from_dict(cls, payload: Mapping[str, Any]) -> "TaskIdentity":
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
    membership_digest: str
    source_revision: str
    simulations: int
    task_count: int
    expected_cells: tuple[tuple[str, int], ...]
    tasks: tuple[TaskIdentity, ...]
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
        membership_digest: str,
        source_revision: str,
        simulations: int,
        tasks: tuple[TaskIdentity, ...],
        expected_cells: tuple[tuple[str, int], ...],
        protocol_fingerprint: str | None,
        dataset_fingerprint: str | None,
        panel_fingerprint: str | None,
    ) -> "ExecutionManifest":
        if not tasks:
            raise ValueError("an execution manifest requires at least one task")
        if len({item.task_id for item in tasks}) != len(tasks):
            raise ValueError("execution manifest contains duplicate task IDs")
        if int(simulations) < 1:
            raise ValueError("execution manifest requires at least one simulation")
        normalized_cells = tuple(
            sorted({(str(portfolio), int(horizon)) for portfolio, horizon in expected_cells})
        )
        if not normalized_cells:
            raise ValueError("execution manifest requires at least one expected cell")
        payload = {
            "schema_version": CHECKPOINT_SCHEMA_VERSION,
            "model_id": model_id,
            "experiment_id": experiment_id,
            "protocol_id": protocol_id,
            "membership_digest": membership_digest,
            "source_revision": source_revision,
            "simulations": int(simulations),
            "task_count": len(tasks),
            "expected_cells": [[portfolio, horizon] for portfolio, horizon in normalized_cells],
            "tasks": [item.to_dict() for item in tasks],
            "protocol_fingerprint": protocol_fingerprint,
            "dataset_fingerprint": dataset_fingerprint,
            "panel_fingerprint": panel_fingerprint,
        }
        return cls(
            schema_version=CHECKPOINT_SCHEMA_VERSION,
            model_id=str(model_id),
            experiment_id=str(experiment_id),
            protocol_id=str(protocol_id),
            membership_digest=str(membership_digest),
            source_revision=str(source_revision),
            simulations=int(simulations),
            task_count=len(tasks),
            expected_cells=normalized_cells,
            tasks=tuple(tasks),
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
            "membership_digest": self.membership_digest,
            "source_revision": self.source_revision,
            "simulations": self.simulations,
            "task_count": self.task_count,
            "expected_cells": [[portfolio, horizon] for portfolio, horizon in self.expected_cells],
            "tasks": [item.to_dict() for item in self.tasks],
            "protocol_fingerprint": self.protocol_fingerprint,
            "dataset_fingerprint": self.dataset_fingerprint,
            "panel_fingerprint": self.panel_fingerprint,
        }

    def to_dict(self) -> dict[str, Any]:
        payload = self.identity_payload()
        payload["manifest_fingerprint"] = self.manifest_fingerprint
        return payload

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "ExecutionManifest":
        try:
            tasks = tuple(TaskIdentity.from_dict(item) for item in payload["tasks"])
            cells = tuple((str(item[0]), int(item[1])) for item in payload["expected_cells"])
            manifest = cls(
                schema_version=str(payload["schema_version"]),
                model_id=str(payload["model_id"]),
                experiment_id=str(payload["experiment_id"]),
                protocol_id=str(payload["protocol_id"]),
                membership_digest=str(payload["membership_digest"]),
                source_revision=str(payload["source_revision"]),
                simulations=int(payload["simulations"]),
                task_count=int(payload["task_count"]),
                expected_cells=cells,
                tasks=tasks,
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
        if stable_digest(manifest.identity_payload()) != manifest.manifest_fingerprint:
            raise CheckpointCorruptError("execution manifest fingerprint is invalid")
        return manifest


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
    def from_dict(cls, payload: Mapping[str, Any]) -> "TaskResult":
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
        self.manifest_path = self.root / "manifest.json"
        self.index_path = self.root / "index.json"
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
            self._load_index()
        else:
            if self.index_path.exists() or self.tasks_path.exists():
                raise CheckpointCorruptError("checkpoint directory has state but no manifest")
            _atomic_write_json(self.manifest_path, manifest.to_dict())
            _atomic_write_json(
                self.index_path,
                {
                    "schema_version": CHECKPOINT_SCHEMA_VERSION,
                    "manifest_fingerprint": manifest.manifest_fingerprint,
                    "tasks": {
                        item.task_id: {"status": "pending"} for item in manifest.tasks
                    },
                },
            )

    def _load_manifest(self) -> ExecutionManifest:
        try:
            with self.manifest_path.open(encoding="utf-8") as handle:
                return ExecutionManifest.from_dict(json.load(handle))
        except (OSError, json.JSONDecodeError) as exc:
            raise CheckpointCorruptError("cannot read checkpoint manifest") from exc

    def _load_index(self) -> dict[str, Any]:
        try:
            with self.index_path.open(encoding="utf-8") as handle:
                index = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise CheckpointCorruptError("cannot read checkpoint index") from exc
        if index.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
            raise CheckpointCorruptError("checkpoint index schema drifted")
        if index.get("manifest_fingerprint") != self.manifest.manifest_fingerprint:
            raise CheckpointIdentityError("checkpoint index fingerprint differs from manifest")
        expected = {item.task_id for item in self.manifest.tasks}
        actual = set(index.get("tasks", {}))
        if actual != expected:
            raise CheckpointCorruptError("checkpoint index task membership differs from manifest")
        return index

    def _index(self) -> dict[str, Any]:
        return self._load_index()

    def statuses(self) -> dict[str, str]:
        index = self._index()
        return {task_id: str(item["status"]) for task_id, item in index["tasks"].items()}

    def status(self, task_id: str) -> str:
        statuses = self.statuses()
        if task_id not in statuses:
            raise CheckpointCorruptError(f"unknown task ID in checkpoint lookup: {task_id}")
        return statuses[task_id]

    def results(self) -> dict[str, TaskResult]:
        records: dict[str, TaskResult] = {}
        for task_id, status in self.statuses().items():
            if status == "pending":
                continue
            path = self.tasks_path / f"{task_id}.json"
            try:
                with path.open(encoding="utf-8") as handle:
                    result = TaskResult.from_dict(json.load(handle))
            except (OSError, json.JSONDecodeError) as exc:
                raise CheckpointCorruptError(f"cannot read task checkpoint {task_id}") from exc
            if result.task_id != task_id or result.status != status:
                raise CheckpointCorruptError(f"task checkpoint status drifted for {task_id}")
            records[task_id] = result
        return records

    def write_result(self, result: TaskResult) -> None:
        expected = {item.task_id for item in self.manifest.tasks}
        if result.task_id not in expected:
            raise CheckpointCorruptError(f"cannot store unknown task ID {result.task_id}")
        if result.status not in {"completed", "failed"}:
            raise ValueError("checkpoint results must be completed or failed")
        _atomic_write_json(self.tasks_path / f"{result.task_id}.json", result.to_dict())
        index = self._index()
        index["tasks"][result.task_id] = {"status": result.status}
        _atomic_write_json(self.index_path, index)

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
