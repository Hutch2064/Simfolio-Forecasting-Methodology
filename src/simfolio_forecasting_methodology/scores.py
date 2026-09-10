"""Retained historical OOS score registry.

Scores are evidence tied to a specific experiment, not generic model metadata.
The canonical dense study is preserved so researchers do not need to rerun a
multi-day experiment merely to inspect the published ranking.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from .catalog import EXPECTED_CELLS_PER_MODEL, load_canonical_175
from .models.registry import registration
from .naming import descriptive_name

CANONICAL_EXPERIMENT_ID = "canonical-dense-oos-2026-08-23"
CANONICAL_PROTOCOL_ID = "dense-daily-crps-v1"


@dataclass(frozen=True)
class ScoreRecord:
    experiment_id: str
    protocol_id: str
    model_id: str
    model_name: str
    canonical_rank: int
    source_rank: int
    exact_empirical_crps: float
    cells: int
    implementation_fidelity: str


def canonical_scores() -> list[ScoreRecord]:
    records: list[ScoreRecord] = []
    for row in load_canonical_175():
        records.append(
            ScoreRecord(
                experiment_id=CANONICAL_EXPERIMENT_ID,
                protocol_id=CANONICAL_PROTOCOL_ID,
                model_id=row.model_id,
                model_name=descriptive_name(row.model_id),
                canonical_rank=row.canonical_rank,
                source_rank=row.source_rank,
                exact_empirical_crps=row.exact_empirical_crps,
                cells=row.cells,
                implementation_fidelity=registration(row.model_id).fidelity,
            )
        )
    validate_score_registry(records)
    return records


def validate_score_registry(records: list[ScoreRecord]) -> None:
    if len(records) != 175:
        raise ValueError("canonical retained score registry must contain 175 rows")
    ids = [record.model_id for record in records]
    if len(ids) != len(set(ids)):
        raise ValueError("retained score registry contains duplicate model IDs")
    if [record.canonical_rank for record in records] != list(range(1, 176)):
        raise ValueError("retained canonical ranks must be exactly 1..175")
    if any(record.cells != EXPECTED_CELLS_PER_MODEL for record in records):
        raise ValueError("retained score registry cell count drifted")
    if any(
        records[index].exact_empirical_crps > records[index + 1].exact_empirical_crps
        for index in range(len(records) - 1)
    ):
        raise ValueError("retained scores are not sorted by CRPS")


def score_by_model_id(model_id: str) -> ScoreRecord | None:
    return next((record for record in canonical_scores() if record.model_id == model_id), None)


def write_score_registry(path: Path) -> None:
    payload = [asdict(record) for record in canonical_scores()]
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
