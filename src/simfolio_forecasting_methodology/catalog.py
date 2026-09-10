"""Canonical catalogue loading and validation."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from importlib.resources import files
from importlib.resources.abc import Traversable
from pathlib import Path

FRONTIER_SOURCE_ID = "asset_level_fastmap_kalman_dynamic_gaussian_factor_rebalanced"
EXPECTED_CANONICAL_COUNT = 175
EXPECTED_SOURCE_RANKS = tuple(range(12, 187))
EXPECTED_CELLS_PER_MODEL = 701280


@dataclass(frozen=True)
class CanonicalRow:
    canonical_rank: int
    source_rank: int
    model_id: str
    exact_empirical_crps: float
    cells: int


def _catalog_root() -> Traversable:
    return files("simfolio_forecasting_methodology").joinpath("resources/catalogs")


def load_canonical_175(root: Path | None = None) -> list[CanonicalRow]:
    catalog_root = Path(root) if root is not None else _catalog_root()
    rows: list[CanonicalRow] = []
    for filename in ("canonical_175_part1.csv", "canonical_175_part2.csv"):
        with (catalog_root / filename).open(newline="", encoding="utf-8") as handle:
            for item in csv.DictReader(handle):
                rows.append(
                    CanonicalRow(
                        canonical_rank=int(item["canonical_rank"]),
                        source_rank=int(item["source_rank"]),
                        model_id=str(item["model_id"]),
                        exact_empirical_crps=float(item["exact_empirical_crps"]),
                        cells=int(item["cells"]),
                    )
                )
    rows.sort(key=lambda row: row.canonical_rank)
    validate_canonical_175(rows)
    return rows


def validate_canonical_175(rows: list[CanonicalRow]) -> None:
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
        raise ValueError("canonical rank 1 is not the Frontier source specification")
    if any(
        rows[index].exact_empirical_crps > rows[index + 1].exact_empirical_crps
        for index in range(len(rows) - 1)
    ):
        raise ValueError("canonical scores are not monotonically nondecreasing")
