#!/usr/bin/env python3
"""Re-run the retained source wrappers against the frozen Frontier panel.

The input fixture supplies only the deterministic panel, dates, and weights;
all fitted states, marginal paths, factor/Kalman arrays, uniforms, and rejoin
returns are regenerated through the caller-supplied historical source tree.
No source-side cache or bytecode files are written.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.resources
import importlib.util
import os
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pandas as pd

FRONTIER_MARGINAL_ID = (
    "sv_live_baseline_sharpe_dlm_historical_cagr_anchor_bdes_multiscale_vol_"
    "conditional_sharpe_fast_map_laplace_sigma_points"
)
FRONTIER_DEPENDENCE_ID = "asset_level_exact_kalman_dynamic_gaussian_factor_rebalanced"

EXPECTED_SHA256 = {
    "engine.py": "702dda6c2a51111724634a5b45d258889a3a411a0b5419f2b5c87066078b0665",
    "asset_level_full_exact_crps.py": "5beb318b918ea367f7048d71a4e53e1bcc9b343faccda2ef493ddcf81bf3d465",
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


def _historical_seed(*parts: object) -> int:
    digest = hashlib.blake2b(digest_size=8)
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\0")
    return int.from_bytes(digest.digest(), "little") % (2**32 - 1)


@contextmanager
def _isolated_runtime():
    """Keep source bytecode and Numba caches inside this checkout."""

    previous_cache_dir = os.environ.get("NUMBA_CACHE_DIR")
    previous_bytecode_flag = os.environ.get("PYTHONDONTWRITEBYTECODE")
    previous_dont_write_bytecode = sys.dont_write_bytecode
    with tempfile.TemporaryDirectory(
        prefix=".frontier-numba-cache-",
        dir=Path(__file__).resolve().parents[1],
    ) as cache_dir:
        os.environ["NUMBA_CACHE_DIR"] = str(Path(cache_dir).resolve())
        os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
        sys.dont_write_bytecode = True
        try:
            yield
        finally:
            sys.dont_write_bytecode = previous_dont_write_bytecode
            if previous_cache_dir is None:
                os.environ.pop("NUMBA_CACHE_DIR", None)
            else:
                os.environ["NUMBA_CACHE_DIR"] = previous_cache_dir
            if previous_bytecode_flag is None:
                os.environ.pop("PYTHONDONTWRITEBYTECODE", None)
            else:
                os.environ["PYTHONDONTWRITEBYTECODE"] = previous_bytecode_flag


def _load_source(source_root: Path, panel_root: Path):
    paths = {
        "engine.py": source_root / "app/engine.py",
        "asset_level_full_exact_crps.py": panel_root / "asset_level_full_exact_crps.py",
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
    os.environ["SIMFOLIO_OOS_ENGINE_ROOT"] = str(source_root)
    sys.path.insert(0, str(panel_root))
    path = paths["simfolio_oos_copula_alternatives.py"]
    spec = importlib.util.spec_from_file_location("frontier_source_wrapper", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot_load_source_wrapper:{path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    import simfolio_adaptive_pgas as pgas

    # The source JIT helpers are replaced with their Python bodies for this
    # bounded, cache-free regeneration procedure. Their arithmetic is intact.
    pgas._small_factor_inverse_numba = pgas._small_factor_inverse_numba.py_func
    return module, pgas


def _default_input() -> Path:
    resource = (
        importlib.resources.files("simfolio_forecasting_methodology")
        .joinpath("resources", "test_fixtures", "frontier", "source_reference.npz")
    )
    if not isinstance(resource, Path):
        raise TypeError("pass --input-fixture when package resources are not filesystem paths")
    return resource


def _regenerate(source_root: Path, panel_root: Path, input_fixture: Path, output: Path) -> None:
    source, pgas = _load_source(source_root, panel_root)
    with np.load(input_fixture, allow_pickle=False) as data:
        panel = {name: np.asarray(data[name]) for name in data.files}
    assets = np.asarray(panel["asset_log_returns"], dtype=np.float64)
    weights = np.asarray(panel["weights"], dtype=np.float64)
    training_dates = np.asarray(panel["training_dates"], dtype="datetime64[ns]")
    future_dates = np.asarray(panel["future_dates"], dtype="datetime64[ns]")
    horizon = int(future_dates.size)
    simulations = int(panel["marginal_paths"].shape[0])
    origin = str(pd.Timestamp(training_dates[-1]).date())
    engine = source.SimfolioEngine()
    candidate = engine._bdes_cagr_fastmap_candidate()
    marginal = np.empty((simulations, horizon, assets.shape[1]), dtype=np.float64)
    posterior_centers = []
    bdes_coefficients = []
    for asset_index, ticker in enumerate(("ALPHA", "BETA", "GAMMA", "DELTA")):
        fit = engine._fit_bdes_non_mcmc_sv_forecast_base(assets[:, asset_index], candidate)
        if fit is None:
            raise RuntimeError(f"source_fit_failed:{ticker}")
        seed = _historical_seed(
            "asset_level_current_engine",
            ticker,
            origin,
            FRONTIER_MARGINAL_ID,
            horizon,
            simulations,
        )
        marginal[:, :, asset_index] = engine._simulate_full_mcmc_sv_forecast_log_paths(
            fit,
            horizon,
            simulations,
            np.random.default_rng(seed),
        )
        posterior_centers.append(np.asarray(fit["posterior_center"], dtype=np.float64))
        bdes_coefficients.append(np.asarray(fit["bdes_multiscale_vol"]["b"], dtype=np.float64))

    model = pgas._fit_dynamic_factor_model(
        assets,
        pseudo_observations=source._pseudo_observations,
        student_copula_fit=source._student_copula_fit,
        stable_correlation=source.BASE._stable_correlation,
        student_scale_mixture=False,
    )
    state, covariance = pgas._kalman_terminal_posterior_numba_information_small.py_func(
        model["observations"],
        model["loading"],
        model["mean"],
        model["phi"],
        model["initial_variance"],
        model["innovation_variance"],
        model["residual_variance"],
    )
    dependence_seed = _historical_seed(
        "copula_alternatives",
        FRONTIER_DEPENDENCE_ID,
        origin,
        horizon,
        simulations,
    )
    rng = np.random.default_rng(dependence_seed)
    factor_state = rng.multivariate_normal(state, covariance, size=simulations)
    random_draws = rng.normal(
        size=(horizon, simulations * (int(model["factor_count"]) + int(model["asset_count"])))
    )
    uniforms, _ = pgas._simulate_future_gaussian_uniforms_flat_numba.py_func(
        factor_state,
        random_draws,
        model["phi"],
        np.sqrt(model["innovation_variance"]),
        np.sqrt(model["residual_variance"]),
        model["loading"],
        model["mean"],
    )
    uniforms = np.clip(uniforms, 1e-8, 1.0 - 1e-8)
    mapped = source._map_uniforms_to_marginal_paths(marginal, uniforms)
    full_dates = pd.DatetimeIndex(training_dates).append(pd.DatetimeIndex(future_dates))
    rebalance_dates = engine._get_dates_for_freq(full_dates, "monthly")
    rebalance_mask = np.asarray([date in rebalance_dates for date in future_dates], dtype=bool)
    rejoined = source._rebalanced_portfolio_log_paths_numba.py_func(
        mapped,
        weights,
        rebalance_mask,
        15.0,
        np.broadcast_to(weights, (simulations, weights.size)).copy(),
    )
    payload = {
        "asset_log_returns": assets,
        "weights": weights,
        "training_dates": training_dates,
        "future_dates": future_dates,
        "marginal_paths": marginal,
        "posterior_centers": np.asarray(posterior_centers),
        "bdes_coefficients": np.asarray(bdes_coefficients),
        "factor_mean": np.asarray(model["mean"]),
        "factor_loading": np.asarray(model["loading"]),
        "factor_residual_variance": np.asarray(model["residual_variance"]),
        "factor_phi": np.asarray(model["phi"]),
        "factor_innovation_variance": np.asarray(model["innovation_variance"]),
        "factor_initial_variance": np.asarray(model["initial_variance"]),
        "kalman_state": np.asarray(state),
        "kalman_covariance": np.asarray(covariance),
        "uniforms": np.asarray(uniforms),
        "rejoined_log_returns": np.asarray(rejoined),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **payload)


def regenerate(source_root: Path, panel_root: Path, input_fixture: Path, output: Path) -> None:
    with _isolated_runtime():
        _regenerate(source_root, panel_root, input_fixture, output)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-research-root", type=Path, required=True)
    parser.add_argument("--panel-root", type=Path, required=True)
    parser.add_argument("--input-fixture", type=Path, default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    input_fixture = args.input_fixture or _default_input()
    regenerate(args.source_research_root, args.panel_root, input_fixture, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
