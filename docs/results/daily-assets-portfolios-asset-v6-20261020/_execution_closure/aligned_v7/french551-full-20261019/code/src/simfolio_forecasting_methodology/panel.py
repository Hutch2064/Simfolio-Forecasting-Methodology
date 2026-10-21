"""The frozen 80-portfolio panel used by the retained dense OOS study.

The active canonical harness loads the reviewed panel resource produced by the
retained 2026-08-23 score run. The previous seeded panel generator remains
available for explicitly supplied research groups, but is never used for the
canonical default: a seed is not a substitute for the exact membership file.
"""

from __future__ import annotations

import csv
import hashlib
import itertools
import json
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from importlib import resources

import numpy as np

ASSET_CLASSES = ("equity", "fixed_income", "alternatives")
REBALANCE_OPTIONS = ("none", "monthly", "quarterly", "annually")
CANONICAL_SEED = 20260528
TARGET_PORTFOLIOS = 80
HOLDINGS_PER_CLASS = 2
CANONICAL_PANEL_ID = "scored52-dense-whitepaper-20260823"
CANONICAL_PANEL_FINGERPRINT = (
    "c017963c9eb772e9bac09bb7a84ae975443cbb99cd127d666a9b0ac8103933d3"
)
CANONICAL_COMMON_RETURN_COUNT = 11_687
CANONICAL_MAX_DAILY_HORIZON = 8_766
CANONICAL_PANEL_RESOURCE_SHA256 = (
    "fc42d3a5b7b0272cab1b4fc5db998d059a3f8fed93f6275295331fdc744f2f36"
)
APPENDIX_PANEL_RESOURCE_SHA256 = (
    "6691eb1daf38cc2aae6708fe1d155aab7975f29ebc6ada197f46bd8d9a8ecf70"
)
APPENDIX_PANEL_CRLF_SHA256 = (
    "847946335fea3473e987af23f528483d66ab265e85596f5d226216091fb476b4"
)
APPENDIX_PANEL_SEMANTIC_STATUS = "provenance_only_not_executable"

CANONICAL_ELIGIBLE_TICKERS: Mapping[str, tuple[str, ...]] = {
    "equity": (
        "EFASIM", "IWCSIM", "IWMSIM", "SDSSIM", "SPMOSIM", "SPXLSIM",
        "SPXUSIM", "SPYSIM", "SSOSIM", "TECLSIM", "TMFSIM", "TNASIM",
        "TYDSIM", "UBTSIM", "UGLSIM", "UPROSIM", "URTHSIM", "URTYSIM",
        "UWMSIM", "VBKSIM", "VBRSIM", "VBSIM", "VEASIM", "VOESIM",
        "VOOSIM", "VOSIM", "VOTSIM", "VTISIM", "VTSIM", "VTVSIM",
        "VUGSIM", "VVSIM", "VXUSSIM", "XLBSIM", "XLCSIM", "XLESIM",
        "XLFSIM", "XLISIM", "XLKSIM", "XLPSIM", "XLUSIM", "XLVSIM",
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
        # The retained panel serialized class weights as 1/3 and then divided
        # each class equally, yielding this IEEE-754 value. Preserve it for a
        # stable semantic fingerprint.
        return (0.16666666666666669,) * 6


def _normalize_groups(groups: Mapping[str, Sequence[str]]) -> dict[str, tuple[str, ...]]:
    return {
        asset_class: tuple(
            sorted({str(ticker).strip().upper() for ticker in groups[asset_class] if str(ticker).strip()})
        )
        for asset_class in ASSET_CLASSES
    }


def _panel_resource_path(name: str) -> resources.abc.Traversable:
    return resources.files("simfolio_forecasting_methodology").joinpath("resources", "panels", name)


def panel_semantic_records(panel: Iterable[PortfolioSpec]) -> list[dict[str, object]]:
    """Return the stable semantic representation used for panel hashing."""
    return [
        {
            "name": portfolio.name,
            "index": index,
            "rebalance": portfolio.rebalance,
            "tickers": list(portfolio.tickers),
            "weights": list(portfolio.weights),
        }
        for index, portfolio in enumerate(panel, start=1)
    ]


def panel_semantic_fingerprint(panel: Iterable[PortfolioSpec]) -> str:
    payload = json.dumps(
        panel_semantic_records(panel), sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _validate_canonical_panel(panel: Sequence[PortfolioSpec], rows: Sequence[dict[str, str]]) -> None:
    if len(panel) != TARGET_PORTFOLIOS or len(rows) != TARGET_PORTFOLIOS:
        raise ValueError(f"canonical panel must contain exactly {TARGET_PORTFOLIOS} portfolios")
    if panel_semantic_fingerprint(panel) != CANONICAL_PANEL_FINGERPRINT:
        raise ValueError("canonical panel semantic fingerprint does not match retained evidence")
    expected = {asset_class: set(values) for asset_class, values in CANONICAL_ELIGIBLE_TICKERS.items()}
    actual = {
        "equity": {ticker for item in panel for ticker in item.equity},
        "fixed_income": {ticker for item in panel for ticker in item.fixed_income},
        "alternatives": {ticker for item in panel for ticker in item.alternatives},
    }
    if actual != expected:
        raise ValueError(f"canonical panel ticker membership drifted: {actual ^ expected}")
    if rebalance_counts(panel) != {"annually": 19, "monthly": 26, "none": 18, "quarterly": 17}:
        raise ValueError("canonical panel rebalance distribution drifted")
    for row_index, (row, item) in enumerate(zip(rows, panel), start=1):
        if row["portfolio_name"] != item.name or int(row["portfolio_index"]) != row_index:
            raise ValueError("canonical panel row identity drifted")
        if int(row["common_return_count"]) != CANONICAL_COMMON_RETURN_COUNT:
            raise ValueError("canonical panel common date count drifted")
        if int(row["max_daily_horizon"]) != CANONICAL_MAX_DAILY_HORIZON:
            raise ValueError("canonical panel maximum horizon drifted")
        if int(row["rolling_origin_count"]) != 48 or int(row["temporal_holdout_count"]) != 3:
            raise ValueError("canonical panel origin counts drifted")


def load_scored52_portfolio_panel() -> list[PortfolioSpec]:
    """Load and validate the exact 52-series panel used by retained scores."""
    resource = _panel_resource_path("scored52_portfolio_panel.csv")
    with resource.open("rb") as handle:
        if hashlib.sha256(handle.read()).hexdigest() != CANONICAL_PANEL_RESOURCE_SHA256:
            raise ValueError("canonical panel resource byte hash drifted")
    with resource.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {
        "portfolio_name", "portfolio_index", "rebalance", "equity_1", "equity_2",
        "fixed_income_1", "fixed_income_2", "alternatives_1", "alternatives_2",
        "common_return_count", "max_daily_horizon", "rolling_origin_count",
        "temporal_holdout_count",
    }
    if not rows or set(rows[0]) != required:
        raise ValueError("canonical panel resource schema drifted")
    panel = [
        PortfolioSpec(
            name=row["portfolio_name"],
            rebalance=row["rebalance"],
            equity=(row["equity_1"], row["equity_2"]),
            fixed_income=(row["fixed_income_1"], row["fixed_income_2"]),
            alternatives=(row["alternatives_1"], row["alternatives_2"]),
        )
        for row in rows
    ]
    indices = [int(row["portfolio_index"]) for row in rows]
    if indices != list(range(1, TARGET_PORTFOLIOS + 1)):
        raise ValueError("canonical panel indices must be contiguous 1..80")
    _validate_canonical_panel(panel, rows)
    return panel


def appendix_panel_provenance() -> dict[str, object]:
    """Return the historical 38-series appendix identity without activating it."""
    resource = resources.files("simfolio_forecasting_methodology").joinpath(
        "resources", "portfolios", "appendix_a_portfolio_panel_80_holdings_rebalance.csv"
    )
    with resource.open("rb") as handle:
        raw = handle.read()
    if hashlib.sha256(raw).hexdigest() != APPENDIX_PANEL_RESOURCE_SHA256:
        raise ValueError("appendix panel provenance bytes drifted")
    rows = list(csv.DictReader(raw.decode("utf-8").splitlines()))
    counts: dict[str, int] = {}
    tickers: set[str] = set()
    for row in rows:
        counts[row["rebalance"]] = counts.get(row["rebalance"], 0) + 1
        tickers.update(row[key] for key in (
            "equity_ticker_1", "equity_ticker_2",
            "fixed_income_ticker_1", "fixed_income_ticker_2",
            "alternatives_ticker_1", "alternatives_ticker_2",
        ))
    return {
        "status": APPENDIX_PANEL_SEMANTIC_STATUS,
        "row_count": len(rows),
        "ticker_count": len(tickers),
        "rebalance_counts": dict(sorted(counts.items())),
        "normalized_lf_sha256": APPENDIX_PANEL_RESOURCE_SHA256,
        "original_crlf_sha256": APPENDIX_PANEL_CRLF_SHA256,
    }


def generate_equal_class_history_panel(
    groups: Mapping[str, Sequence[str]] | None = None,
    *,
    target_portfolios: int = TARGET_PORTFOLIOS,
    seed: int = CANONICAL_SEED,
    holdings_per_class: int = HOLDINGS_PER_CLASS,
    rebalance_options: Sequence[str] = REBALANCE_OPTIONS,
) -> list[PortfolioSpec]:
    """Return the frozen canonical panel or generate an explicitly custom panel.

    ``groups=None`` is intentionally the canonical loader. Callers that need
    the balancing algorithm for a separate study must supply groups explicitly.
    """
    if groups is None:
        if (
            int(target_portfolios) == TARGET_PORTFOLIOS
            and int(seed) == CANONICAL_SEED
            and int(holdings_per_class) == HOLDINGS_PER_CLASS
            and tuple(rebalance_options) == REBALANCE_OPTIONS
        ):
            return load_scored52_portfolio_panel()
        groups = CANONICAL_ELIGIBLE_TICKERS
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
                max(projected[c].values()) - min(projected[c].values()) for c in ASSET_CLASSES
            )
            usage_pressure = tuple(
                sum(usage[c][ticker] for ticker in selection[c]) for c in ASSET_CLASSES
            )
            square_pressure = tuple(
                float(sum(value * value for value in projected[c].values()))
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
