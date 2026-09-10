"""Explicit canonical and master research harness definitions.

Both harnesses share one evaluation protocol.  The canonical harness freezes the
175-paper catalogue.  The master harness is the stable union of the historical
369-ID research snapshot and every canonical model, so canonical membership can
never be lost when the broader research universe is reconstructed.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from importlib.resources.abc import Traversable
from pathlib import Path

from .catalog import CanonicalRow, _catalog_root, load_canonical_175
from .protocol import CANONICAL_DENSE_PROTOCOL, DenseDailyProtocol

CANONICAL_HARNESS_ID = "canonical-175"
MASTER_HARNESS_ID = "master-research"
HISTORICAL_MASTER_COUNT = 369


@dataclass(frozen=True)
class HarnessPlan:
    harness_id: str
    protocol: DenseDailyProtocol
    model_ids: tuple[str, ...]

    @property
    def model_count(self) -> int:
        return len(self.model_ids)

    @property
    def expected_origin_tasks(self) -> int:
        return self.protocol.total_origin_tasks


def canonical_plan(rows: Iterable[CanonicalRow] | None = None) -> HarnessPlan:
    catalogue = list(rows) if rows is not None else load_canonical_175()
    return HarnessPlan(
        harness_id=CANONICAL_HARNESS_ID,
        protocol=CANONICAL_DENSE_PROTOCOL,
        model_ids=tuple(row.model_id for row in catalogue),
    )


def _master_catalog_path(root: Path | None = None) -> Traversable:
    if root is not None:
        return Path(root) / "master_369_ids.txt"
    return _catalog_root() / "master_369_ids.txt"


def load_historical_master_ids(root: Path | None = None) -> tuple[str, ...]:
    path = _master_catalog_path(root)
    ids = tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if len(ids) != HISTORICAL_MASTER_COUNT:
        raise ValueError(
            f"historical master snapshot must contain {HISTORICAL_MASTER_COUNT} source IDs, "
            f"found {len(ids)}"
        )
    if len(ids) != len(set(ids)):
        raise ValueError("historical master snapshot contains duplicate source IDs")
    return ids


def load_master_ids(root: Path | None = None) -> tuple[str, ...]:
    historical = load_historical_master_ids(root)
    canonical = canonical_plan().model_ids
    seen: set[str] = set()
    merged: list[str] = []
    for model_id in (*canonical, *historical):
        if model_id not in seen:
            seen.add(model_id)
            merged.append(model_id)
    if not set(canonical).issubset(seen):
        raise AssertionError("master research catalogue lost canonical membership")
    return tuple(merged)


def master_plan(root: Path | None = None) -> HarnessPlan:
    return HarnessPlan(
        harness_id=MASTER_HARNESS_ID,
        protocol=CANONICAL_DENSE_PROTOCOL,
        model_ids=load_master_ids(root),
    )
