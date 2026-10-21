"""Deterministic random-seed derivation retained by the research harnesses."""

from __future__ import annotations

import hashlib
from typing import Any


def deterministic_seed(*parts: Any) -> int:
    """Derive the exact 32-bit seed used by retained Simfolio research code.

    Each part is converted with ``str``, UTF-8 encoded, and null-delimited into
    an 8-byte BLAKE2b digest. The little-endian digest integer is reduced modulo
    ``2**32 - 1``.
    """

    digest = hashlib.blake2b(digest_size=8)
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\0")
    return int.from_bytes(digest.digest(), "little") % (2**32 - 1)


def coherent_daily_seed(origin_date: str, max_horizon: int, simulations: int) -> int:
    """Seed context used by the retained all-daily coherent benchmark."""
    return deterministic_seed(
        "naive_iid_historical_portfolio_bootstrap",
        "forecast_oos_all_daily_coherent",
        str(origin_date),
        int(max_horizon),
        int(simulations),
    )


def copula_alternatives_seed(
    candidate_id: str, origin_date: str, max_horizon: int, simulations: int
) -> int:
    """Seed context used by retained asset-level dependence candidates."""
    return deterministic_seed(
        "copula_alternatives",
        str(candidate_id),
        str(origin_date),
        int(max_horizon),
        int(simulations),
    )


def forecast_oos_candidate_seed(
    origin_date: str,
    available_horizons: tuple[int, ...] | list[int],
    candidate_id: str,
    simulations: int,
) -> int:
    """Seed context used by the fixed-horizon research candidate adapter."""
    return deterministic_seed(
        "forecast_oos_candidate",
        str(origin_date),
        tuple(int(value) for value in available_horizons),
        str(candidate_id),
        int(simulations),
    )
