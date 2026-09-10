#!/usr/bin/env python3
"""Generate the bounded Frontier source rejoin/calendar fixture.

This procedure loads only the caller-supplied, read-only historical wrapper
tree.  It disables Python and Numba caches before importing that tree and
writes the fixture into the destination package resource directory.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_SHA256 = {
    "engine.py": "702dda6c2a51111724634a5b45d258889a3a411a0b5419f2b5c87066078b0665",
    "simfolio_oos_two_candidates.py": "0577e79abee09d09735ee76f85941e007f995cc6a370ee375a0743cf440a23c2",
    "simfolio_adaptive_pgas.py": "f7988a6cfbdf674c1efeb1ca6836b34e5f34c1ee241e97ef44551ded7ba87c69",
    "simfolio_oos_copula_alternatives.py": "413d2ca7f74cda13dd228c8974f822ce23e690cc56e098babf3fd2c121fbf95e",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_source(source_root: Path, panel_root: Path) -> None:
    paths = {
        "engine.py": source_root / "app/engine.py",
        "simfolio_oos_two_candidates.py": panel_root / "simfolio_oos_two_candidates.py",
        "simfolio_adaptive_pgas.py": panel_root / "simfolio_adaptive_pgas.py",
        "simfolio_oos_copula_alternatives.py": panel_root / "simfolio_oos_copula_alternatives.py",
    }
    for name, expected in EXPECTED_SHA256.items():
        path = paths[name]
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = _sha256(path)
        if actual != expected:
            raise ValueError(f"source_hash_mismatch:{name}:{actual}")


def _load_source_alt(source_root: Path, panel_root: Path):
    os.environ["SIMFOLIO_OOS_ENGINE_ROOT"] = str(source_root)
    os.environ["NUMBA_DISABLE_CACHING"] = "1"
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(panel_root))
    path = panel_root / "simfolio_oos_copula_alternatives.py"
    spec = importlib.util.spec_from_file_location("frontier_source_rejoin", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot_load_source_wrapper:{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def build_fixture(source_root: Path, panel_root: Path, output: Path) -> None:
    _verify_source(source_root, panel_root)
    source = _load_source_alt(source_root, panel_root)
    engine = source.SimfolioEngine()
    historical_dates = pd.bdate_range("2026-06-08", periods=80)
    future_dates = pd.bdate_range(
        historical_dates[-1] + pd.offsets.BDay(1),
        periods=80,
    )
    full_dates = historical_dates.append(future_dates)
    rng = np.random.default_rng(20260823)
    log_paths = rng.normal(
        loc=0.0005,
        scale=0.012,
        size=(5, 80, 4),
    ).astype(np.float64)
    weights = np.asarray([0.2, 0.3, 0.1, 0.4], dtype=np.float64)
    payload: dict[str, np.ndarray] = {
        "log_paths": log_paths,
        "weights": weights,
        "historical_dates": historical_dates.to_numpy(dtype="datetime64[ns]"),
        "future_dates": future_dates.to_numpy(dtype="datetime64[ns]"),
        "cost_bps": np.asarray(15.0),
    }
    for policy in ("none", "monthly", "quarterly", "annually"):
        rebalance_dates = engine._get_dates_for_freq(full_dates, policy)
        rebalance_mask = np.asarray(
            [date in rebalance_dates for date in future_dates],
            dtype=bool,
        )
        holdings = np.broadcast_to(weights, (5, 4)).copy()
        output_values = source._rebalanced_portfolio_log_paths_numba.py_func(
            log_paths,
            weights,
            rebalance_mask,
            15.0,
            holdings,
        )
        payload[f"mask_{policy}"] = rebalance_mask
        payload[f"output_{policy}"] = np.asarray(output_values, dtype=np.float64)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-research-root", type=Path, required=True)
    parser.add_argument("--panel-root", type=Path, required=True)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "src/simfolio_forecasting_methodology/resources/test_fixtures/frontier/"
            "rejoin_calendar_reference.npz"
        ),
    )
    args = parser.parse_args()
    build_fixture(args.source_research_root, args.panel_root, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
