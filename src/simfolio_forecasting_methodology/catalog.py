"""Compatibility loader backed solely by the canonical-175 ledger."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .catalogue import (
    EXPECTED_CANONICAL_COUNT,
    EXPECTED_CELLS_PER_MODEL,
    EXPECTED_SOURCE_RANKS,
    frontier_model_id,
    load_canonical_ledger,
    load_canonical_models,
)

FRONTIER_SOURCE_ID = frontier_model_id()


@dataclass(frozen=True)
class CanonicalRow:
    canonical_rank: int
    source_rank: int
    model_id: str
    exact_empirical_crps: float
    cells: int
    exact_empirical_crps_text: str = ""
    publication_empirical_crps_text: str = ""


def load_canonical_175(root: Path | None = None) -> list[CanonicalRow]:
    """Load canonical rows from the ledger; ``root`` is a ledger override."""

    payload = load_canonical_ledger(root)
    cells = payload["score_evidence"]["cells_per_model"]
    return [
        CanonicalRow(
            canonical_rank=model["canonical_rank"],
            source_rank=model["historical_rank"],
            model_id=model["public_model_id"],
            exact_empirical_crps=float(model["historical_score"]["exact_empirical_crps"]),
            cells=cells,
            exact_empirical_crps_text=model["historical_score"]["exact_empirical_crps"],
            publication_empirical_crps_text=model["score_precision"]["publication_token"],
        )
        for model in load_canonical_models(root)
    ]


def validate_canonical_175(rows: list[CanonicalRow]) -> None:
    """Preserve the old validation entry point for callers using ``CanonicalRow``."""

    if len(rows) != EXPECTED_CANONICAL_COUNT:
        raise ValueError(f"canonical catalogue must contain {EXPECTED_CANONICAL_COUNT} rows")
    if [row.canonical_rank for row in rows] != list(range(1, EXPECTED_CANONICAL_COUNT + 1)):
        raise ValueError("canonical ranks must be exactly 1..175")
    if tuple(row.source_rank for row in rows) != EXPECTED_SOURCE_RANKS:
        raise ValueError("source ranks must be exactly 12..186")
    ids = [row.model_id for row in rows]
    if len(ids) != len(set(ids)):
        raise ValueError("canonical catalogue contains duplicate source model IDs")
    if any(row.cells != EXPECTED_CELLS_PER_MODEL for row in rows):
        raise ValueError("canonical catalogue cell count drifted")
    if rows[0].model_id != FRONTIER_SOURCE_ID:
        raise ValueError("canonical rank one is not the Frontier source specification")
    if any(
        rows[index].exact_empirical_crps > rows[index + 1].exact_empirical_crps
        for index in range(len(rows) - 1)
    ):
        raise ValueError("canonical scores are not monotonically nondecreasing")


__all__ = [
    "EXPECTED_CANONICAL_COUNT",
    "EXPECTED_CELLS_PER_MODEL",
    "EXPECTED_SOURCE_RANKS",
    "FRONTIER_SOURCE_ID",
    "CanonicalRow",
    "load_canonical_175",
    "validate_canonical_175",
]
