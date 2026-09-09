"""Explicit research harnesses.

The canonical and master harnesses share one evaluation protocol.  They differ
only in catalogue membership.  This module deliberately contains no web,
deployment, or service integration.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .catalog import CanonicalRow, load_canonical_175
from .protocol import CANONICAL_DENSE_PROTOCOL, DenseDailyProtocol

CANONICAL_HARNESS_ID = "canonical-175"
MASTER_HARNESS_ID = "master-research"


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


def _master_catalog_path(root: Path | None = None) -> Path:
    if root is not None:
        return Path(root) / "master_369_ids.txt"
    return Path(__file__).resolve().parents[2] / "catalogs" / "master_369_ids.txt"


def load_master_ids(root: Path | None = None) -> tuple[str, ...]:
    path = _master_catalog_path(root)
    ids = tuple(
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    )
    if len(ids) != 369:
        raise ValueError(f"master catalogue must contain 369 source IDs, found {len(ids)}")
    if len(ids) != len(set(ids)):
        raise ValueError("master catalogue contains duplicate source IDs")
    return ids


def master_plan(root: Path | None = None) -> HarnessPlan:
    return HarnessPlan(
        harness_id=MASTER_HARNESS_ID,
        protocol=CANONICAL_DENSE_PROTOCOL,
        model_ids=load_master_ids(root),
    )
