"""Deterministic 80-portfolio panel used by the dense OOS methodology."""

from __future__ import annotations

from dataclasses import dataclass
import itertools
from typing import Iterable, Mapping, Sequence

import numpy as np

ASSET_CLASSES = ("equity", "fixed_income", "alternatives")
REBALANCE_OPTIONS = ("none", "monthly", "quarterly", "annually")
CANONICAL_SEED = 20260528
TARGET_PORTFOLIOS = 80
HOLDINGS_PER_CLASS = 2

CANONICAL_ELIGIBLE_TICKERS: Mapping[str, tuple[str, ...]] = {
    "equity": (
        "EFASIM", "IWCSIM", "IWMSIM", "SPYSIM", "URTHSIM", "VBSIM", "VBKSIM",
        "VBRSIM", "VEASIM", "VOESIM", "VOOSIM", "VOSIM", "VOTSIM", "VTSIM",
        "VTISIM", "VTVSIM", "VUGSIM", "VVSIM", "VXUSSIM", "XLBSIM", "XLCSIM",
        "XLESIM", "XLFSIM", "XLISIM", "XLKSIM", "XLPSIM", "XLUSIM", "XLVSIM",
        "XLYSIM",
    ),
    "fixed_income": ("IEFSIM", "IEISIM", "SHYSIM", "TLTSIM", "ZROZSIM"),
    "alternatives": ("GLDSIM", "GSGSIM", "REITSIM", "SLVSIM"),
}


@dataclass(frozen=True)
class PortfolioSpec:
    name: str
    rebalance: str
    equity: tuple[str, str]
    fixed_income: tuple[str, str]
    alternatives: tuple[str, str]

    @property
    def tickers(self) -> tuple[str, ...]:
        return self.equity + self.fixed_income + self.alternatives

    @property
    def weights(self) -> tuple[float, ...]:
        return (1.0 / 6.0,) * 6


def _normalize_groups(groups: Mapping[str, Sequence[str]]) -> dict[str, tuple[str, ...]]:
    return {
        asset_class: tuple(sorted({str(t).strip().upper() for t in groups[asset_class] if str(t).strip()}))
        for asset_class in ASSET_CLASSES
    }


def generate_equal_class_history_panel(
    groups: Mapping[str, Sequence[str]] = CANONICAL_ELIGIBLE_TICKERS,
    *,
    target_portfolios: int = TARGET_PORTFOLIOS,
    seed: int = CANONICAL_SEED,
    holdings_per_class: int = HOLDINGS_PER_CLASS,
    rebalance_options: Sequence[str] = REBALANCE_OPTIONS,
) -> list[PortfolioSpec]:
    holdings = max(int(holdings_per_class), 1)
    normalized = _normalize_groups(groups)
    combinations = {
        asset_class: list(itertools.combinations(normalized[asset_class], holdings))
        for asset_class in ASSET_CLASSES
    }
    rebalances = tuple(str(value) for value in rebalance_options if str(value).strip())
    capacity = len(rebalances)
    for asset_class in ASSET_CLASSES:
        capacity *= len(combinations[asset_class])
    selected_count = max(0, min(int(target_portfolios), capacity))

    usage = {
        asset_class: {ticker: 0 for ticker in normalized[asset_class]}
        for asset_class in ASSET_CLASSES
    }
    rng = np.random.default_rng(int(seed))
    candidates: list[tuple[dict[str, tuple[str, ...]], str, float]] = []
    for combo_values in itertools.product(*(combinations[c] for c in ASSET_CLASSES)):
        selection = {c: tuple(combo_values[i]) for i, c in enumerate(ASSET_CLASSES)}
        for rebalance in rebalances:
            candidates.append((selection, rebalance, float(rng.random())))

    selected_keys: set[tuple[object, ...]] = set()
    output: list[PortfolioSpec] = []
    for portfolio_index in range(1, selected_count + 1):
        best_index: int | None = None
        best_score: tuple[object, ...] | None = None
        for index, (selection, rebalance, jitter) in enumerate(candidates):
            key = tuple(selection[c] for c in ASSET_CLASSES) + (rebalance,)
            if key in selected_keys:
                continue
            projected = {c: dict(counts) for c, counts in usage.items()}
            for asset_class in ASSET_CLASSES:
                for ticker in selection[asset_class]:
                    projected[asset_class][ticker] += 1
            imbalance = tuple(
                max(projected[c].values()) - min(projected[c].values())
                for c in ASSET_CLASSES
            )
            usage_pressure = tuple(
                sum(usage[c][ticker] for ticker in selection[c])
                for c in ASSET_CLASSES
            )
            square_pressure = tuple(
                float(sum(v * v for v in projected[c].values()))
                for c in ASSET_CLASSES
            )
            score = (imbalance, usage_pressure, square_pressure, jitter, key)
            if best_score is None or score < best_score:
                best_score = score
                best_index = index
        if best_index is None:
            break
        selection, rebalance, _ = candidates[best_index]
        key = tuple(selection[c] for c in ASSET_CLASSES) + (rebalance,)
        selected_keys.add(key)
        for asset_class in ASSET_CLASSES:
            for ticker in selection[asset_class]:
                usage[asset_class][ticker] += 1
        output.append(
            PortfolioSpec(
                name=f"equal_class_history_{portfolio_index:03d}",
                rebalance=rebalance,
                equity=tuple(selection["equity"]),
                fixed_income=tuple(selection["fixed_income"]),
                alternatives=tuple(selection["alternatives"]),
            )
        )
    return output


def rebalance_counts(panel: Iterable[PortfolioSpec]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for portfolio in panel:
        counts[portfolio.rebalance] = counts.get(portfolio.rebalance, 0) + 1
    return dict(sorted(counts.items()))
