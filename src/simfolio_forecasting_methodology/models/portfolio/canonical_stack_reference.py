"""Source-backed canonical forecast and explicit distribution stacks.

The retained research gate generated these candidates as terminal ensembles by
horizon.  This adapter preserves that contract: it fits the canonical daily
model once, returns cumulative terminal samples for each dense horizon, and
mixes those samples with an independently seeded Gaussian or Student-t
terminal component for the twelve stack rows.  It does not manufacture a
daily increment matrix from terminal samples.

The factor-proxy input is loaded from the manifest-whitelisted frozen source
snapshot bundled with the package.  The loader is deliberately offline and
strict: a missing, malformed, or hash-mismatched proxy fails closed rather
than silently reducing the historical nine-factor closure.
"""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from functools import lru_cache
from importlib.resources import files
from types import MappingProxyType
from typing import Any

import numpy as np
import pandas as pd
from scipy import optimize, stats

from ...runner import TERMINAL_FORECAST_SEMANTICS, ForecastContext, TrainingData
from .gjr_reference import (
    _politis_white_block_length,
    _stationary_bootstrap_indices,
)
from .reference_families import SOURCE_FUNCTIONS_SHA256 as SOURCE_COMPONENT_FUNCTIONS_SHA256
from .reference_families import _SourceKernel

SOURCE_ARTIFACTS: Mapping[str, Mapping[str, Any]] = MappingProxyType(
    {
        "engine": MappingProxyType(
            {
                "path": "source-research/app/engine.py",
                "sha256": "702dda6c2a51111724634a5b45d258889a3a411a0b5419f2b5c87066078b0665",
            }
        ),
        "research_gate": MappingProxyType(
            {
                "path": "source-research/scripts/forecast_oos_research_gate.py",
                "sha256": "e061aba8ed259339f75a98e9ea8e0a1a275c99980ed649b92efe652d7271f997",
            }
        ),
        "pinned_source_revision": MappingProxyType(
            {
                "revision": "773bc1c325559e6bf57a567f1d8bf473a3427fbc",
                "engine_path": "source-research/app/engine.py",
            }
        ),
        "canonical_proxy_snapshot_manifest": MappingProxyType(
            {
                "path": "resources/data/canonical_snapshot_manifest.json",
                "sha256": "3c2e8218b03ca7a435d145168c97a2f302be427e862170183078bc4917699c12",
                "status": "manifest_whitelisted_frozen_source_snapshot",
            }
        ),
    }
)

SOURCE_FUNCTION_NAMES: tuple[str, ...] = (
    "SimfolioEngine._deterministic_seed",
    "SimfolioEngine._canonical_forecast_candidate",
    "SimfolioEngine._sample_mean_near_zero_shrinkage",
    "_ewma_volatility_nll",
    "_ewma_volatility_sigma_path",
    "SimfolioEngine._fit_ewma_volatility",
    "SimfolioEngine._standardize_residuals",
    "SimfolioEngine._canonical_proxy_factor_returns",
    "SimfolioEngine._fit_factor_drift_prior",
    "SimfolioEngine._fit_sticky_volatility_regime_filter",
    "_politis_white_block_length",
    "_stationary_bootstrap_indices",
    "SimfolioEngine._fit_canonical_forecast_model",
    "SimfolioEngine._evt_standardized_draws",
    "SimfolioEngine._standardize_generated_innovations",
    "SimfolioEngine._standardize_generated_innovations_inplace",
    "SimfolioEngine._simulate_canonical_forecast_log_paths",
    "forecast_oos_research_gate._terminal_logs_for_candidate[stack]",
    "SimfolioEngine._mix_terminal_log_values",
)

# Digest of the source AST fragments listed above.  It identifies the source
# closure, while the two whole-file digests in SOURCE_ARTIFACTS identify the
# pinned artifacts from which those fragments were extracted.
SOURCE_FUNCTIONS_SHA256 = "dced757f302a023eb2b2d0dba40c143330a8b9a33d3d38b16d73ce0e021cb9d5"
# The Gaussian/Student-t stack components reuse the already audited exact
# source closure instead of copying a second implementation.

SOURCE_SEED_CONTRACT = (
    "blake2b-64-little-mod-2^32-1; args=('forecast_oos_candidate', "
    "origin_date, dense_horizon_tuple, candidate_id, simulations)"
)

_CANONICAL_PROXY_MAP: tuple[tuple[str, str], ...] = (
    ("SPYSIM", "Equity Market"),
    ("BNDSIM", "Core Bonds"),
    ("TLTSIM", "Long Duration"),
    ("TIPSIM", "Inflation-Linked Bonds"),
    ("GLDSIM", "Gold"),
    ("GSGSIM", "Commodities"),
    ("UUPSIM", "US Dollar"),
    ("KMLMSIM", "Managed Futures"),
    ("CASHX", "Cash"),
)
_CANONICAL_PROXY_MANIFEST_SHA256 = (
    "3c2e8218b03ca7a435d145168c97a2f302be427e862170183078bc4917699c12"
)
_CANONICAL_PROXY_SERIES_SHA256: Mapping[str, str] = MappingProxyType(
    {
        "SPYSIM": "2b4675e69ae630e29e25cd663dc73a316de77a2c049c99b1ea408d453c8d85d0",
        "BNDSIM": "c5a194fcb557be01c3bab80c3eea655f7be7a8b7032f3205e0f3cb181f899563",
        "TLTSIM": "cee5ae06395c0d550a7614a4ebeb19c1884be22d8955387927eb89f8231e75e0",
        "TIPSIM": "0b784e011dd89d8d09b41be053126016d514e498b6e922ba21ac68b4724ebae8",
        "GLDSIM": "071805166240ec36ee79c141a93ab5f823ce993af8528d8de75b01dd5ace50dc",
        "GSGSIM": "c3a3fcff7c8e3b6a8999f3383c79941149468d79e533b77f0900a62fd335b17f",
        "UUPSIM": "ad8e85a0f34016784246d05fc0302fbaaafbff64317eac239f10f7a641214855",
        "KMLMSIM": "e5bb1d333ec599b82bcf77992c3b91ac3bc6f270eebcb734a6418392cb97361c",
        "CASHX": "9facce293da12bc9cec0b7f99327d87e2a4d19cefcef499db97b96d2093598bb",
    }
)

CANONICAL_MODEL_ID = "canonical"
STACK_MODEL_IDS: tuple[str, ...] = (
    "stack_canonical_gaussian_w0.60",
    "stack_canonical_gaussian_w0.50",
    "stack_canonical_student_t_w0.60",
    "stack_canonical_student_t_w0.50",
    "stack_canonical_gaussian_w0.70",
    "stack_canonical_student_t_w0.70",
    "stack_canonical_gaussian_w0.80",
    "stack_canonical_student_t_w0.80",
    "stack_canonical_student_t_w0.90",
    "stack_canonical_gaussian_w0.90",
    "stack_canonical_gaussian_w1.00",
    "stack_canonical_student_t_w1.00",
)
REFERENCE_MODEL_IDS: tuple[str, ...] = (CANONICAL_MODEL_ID, *STACK_MODEL_IDS)

_CANONICAL_SPEC: Mapping[str, Any] = MappingProxyType(
    {"id": CANONICAL_MODEL_ID, "type": "canonical"}
)


def _stack_spec(model_id: str) -> Mapping[str, Any]:
    """Return the sparse candidate descriptor retained by the source run."""
    return MappingProxyType({"id": model_id, "type": "stack"})


# The retained stack rows contain no ``shrinkage`` or ``canonical_weight``
# keys.  These values are the source dispatcher's explicit defaults and must
# stay separate from the public labels embedded in the later catalogue.
_STACK_SOURCE_DEFAULTS: Mapping[str, Any] = MappingProxyType(
    {"shrinkage": "student_t", "canonical_weight": 1.0}
)


SOURCE_CANDIDATE_SPECS: Mapping[str, Mapping[str, Any]] = MappingProxyType(
    {
        CANONICAL_MODEL_ID: _CANONICAL_SPEC,
        **{model_id: _stack_spec(model_id) for model_id in STACK_MODEL_IDS},
    }
)

CANONICAL_REGIME_FIT_MAX_OBS = 5040
CANONICAL_REGIME_EM_MAX_ITER = 8
CANONICAL_REGIME_EM_MIN_ITER = 4
CANONICAL_REGIME_EM_TOL = 2.5e-4
STATIONARY_BOOTSTRAP_ROW_ASSEMBLY_MIN_DAYS = 128


def _deterministic_seed(*parts: Any) -> int:
    digest = hashlib.blake2b(digest_size=8)
    for part in parts:
        digest.update(str(part).encode("utf-8"))
        digest.update(b"\0")
    return int.from_bytes(digest.digest(), "little") % (2**32 - 1)


def _ewma_volatility_nll(x: np.ndarray, base_var: float, lam: float) -> float:
    arr = np.asarray(x, dtype=np.float64)
    h = float(base_var)
    ll = 0.0
    for value in arr:
        h = float(lam * h + (1.0 - lam) * value * value)
        h = max(h, 1e-8)
        ll += math.log(h) + (value * value / h)
    return float(ll)


def _ewma_volatility_sigma_path(x: np.ndarray, base_var: float, lam: float) -> np.ndarray:
    arr = np.asarray(x, dtype=np.float64)
    sigma = np.empty_like(arr)
    h = float(base_var)
    for idx, value in enumerate(arr):
        h = float(lam * h + (1.0 - lam) * value * value)
        h = max(h, 1e-8)
        sigma[idx] = math.sqrt(h)
    return sigma


def _sample_mean_near_zero_shrinkage(log_returns: np.ndarray) -> tuple[float, dict[str, Any]]:
    arr = np.asarray(log_returns, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return 0.0, {"method": "empty_zero"}
    sample_mean = float(np.mean(arr))
    sigma = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    se = sigma / math.sqrt(float(arr.size)) if sigma > 0.0 else 0.0
    t_stat = abs(sample_mean) / se if se > 0.0 else 0.0
    shrink_weight = float((t_stat * t_stat) / (1.0 + t_stat * t_stat)) if np.isfinite(t_stat) else 0.0
    return float(sample_mean * shrink_weight), {
        "method": "sample_mean_positive_part_t_stat_shrinkage_to_zero",
        "sample_mean": sample_mean,
        "shrink_weight": shrink_weight,
        "standard_error": se,
    }


def _fit_ewma_volatility(residuals: np.ndarray) -> dict[str, Any]:
    eps = np.asarray(residuals, dtype=np.float64)
    eps = eps[np.isfinite(eps)]
    x = eps * 100.0
    if x.size < 30:
        sigma = float(np.std(x, ddof=1)) if x.size > 1 else 1.0
        sigma = float(max(sigma, 1e-6))
        return {
            "vol_model": "estimated_decay_ewma_volatility",
            "lambda": 0.94,
            "sigma_x": np.full(max(len(x), 1), sigma, dtype=np.float64),
            "last_sigma_x": sigma,
            "params": {"lambda": 0.94},
            "fit_status": "fallback_short_history",
        }
    base_var = float(np.var(x, ddof=1))
    base_var = max(base_var, 1e-8)
    try:
        opt = optimize.minimize_scalar(
            lambda lam: _ewma_volatility_nll(x, base_var, float(lam)),
            bounds=(0.80, 0.995),
            method="bounded",
            options={"xatol": 1e-4},
        )
        lam = float(opt.x) if opt.success and np.isfinite(opt.x) else 0.94
    except (ArithmeticError, RuntimeError, ValueError):
        lam = 0.94
    sigma = _ewma_volatility_sigma_path(x, base_var, lam)
    return {
        "vol_model": "estimated_decay_ewma_volatility",
        "lambda": lam,
        "sigma_x": sigma,
        "last_sigma_x": float(sigma[-1]),
        "params": {"lambda": lam},
        "fit_status": "complete",
    }


def _standardize_residuals(residuals: np.ndarray, sigma_x: np.ndarray) -> np.ndarray:
    x = np.asarray(residuals, dtype=np.float64) * 100.0
    sigma = np.asarray(sigma_x, dtype=np.float64)
    n = int(min(len(x), len(sigma)))
    if n <= 0:
        return np.array([0.0], dtype=np.float64)
    z = x[-n:] / np.maximum(sigma[-n:], 1e-8)
    z = z[np.isfinite(z)]
    if z.size == 0:
        return np.array([0.0], dtype=np.float64)
    z = z - float(np.mean(z))
    std = float(np.std(z, ddof=1)) if z.size > 1 else 1.0
    if std > 1e-8:
        z = z / std
    return np.clip(z, -12.0, 12.0)


def _resource_bytes(relative_path: str) -> bytes:
    resource = files("simfolio_forecasting_methodology").joinpath(*relative_path.split("/"))
    try:
        return resource.read_bytes()
    except (FileNotFoundError, OSError) as exc:
        raise RuntimeError(
            f"canonical proxy source snapshot is unavailable: {relative_path}"
        ) from exc


def _load_canonical_proxy_manifest() -> dict[str, Any]:
    relative_path = "resources/data/canonical_snapshot_manifest.json"
    raw = _resource_bytes(relative_path)
    actual_digest = hashlib.sha256(raw).hexdigest()
    if actual_digest != _CANONICAL_PROXY_MANIFEST_SHA256:
        raise RuntimeError(
            "canonical proxy snapshot manifest digest mismatch: "
            f"expected {_CANONICAL_PROXY_MANIFEST_SHA256}, got {actual_digest}"
        )
    try:
        manifest = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError("canonical proxy snapshot manifest is not valid UTF-8 JSON") from exc
    if not isinstance(manifest, dict) or not isinstance(manifest.get("series"), list):
        raise TypeError("canonical proxy snapshot manifest has no series list")
    manifest_rows = [*manifest["series"], *(manifest.get("supporting_series") or [])]
    entries = {
        str(item.get("ticker")): item
        for item in manifest_rows
        if isinstance(item, dict) and item.get("ticker")
    }
    for ticker, _label in _CANONICAL_PROXY_MAP:
        item = entries.get(ticker)
        expected_path = f"app/simulated_data/series/{ticker}.csv.gz"
        if item is None or item.get("path") != expected_path:
            raise RuntimeError(f"canonical proxy manifest is missing the exact {ticker} source entry")
        if item.get("sha256") != _CANONICAL_PROXY_SERIES_SHA256[ticker]:
            raise RuntimeError(f"canonical proxy manifest digest entry mismatch for {ticker}")
    return manifest


def _load_canonical_proxy_series(ticker: str, label: str) -> pd.Series:
    relative_path = f"resources/data/canonical_snapshot/app/simulated_data/series/{ticker}.csv.gz"
    raw = _resource_bytes(relative_path)
    actual_digest = hashlib.sha256(raw).hexdigest()
    expected_digest = _CANONICAL_PROXY_SERIES_SHA256[ticker]
    if actual_digest != expected_digest:
        raise RuntimeError(
            f"canonical proxy source digest mismatch for {ticker}: "
            f"expected {expected_digest}, got {actual_digest}"
        )
    try:
        frame = pd.read_csv(io.BytesIO(gzip.decompress(raw)))
    except (OSError, ValueError, gzip.BadGzipFile) as exc:
        raise RuntimeError(f"canonical proxy source is not readable for {ticker}") from exc
    required_columns = {"date", "daily_return", "price"}
    if not required_columns.issubset(frame.columns):
        raise RuntimeError(f"canonical proxy source schema mismatch for {ticker}")
    dates = pd.to_datetime(frame["date"], errors="coerce")
    if dates.isna().any():
        raise RuntimeError(f"canonical proxy source contains invalid dates for {ticker}")
    index = pd.DatetimeIndex(dates).tz_localize(None).normalize()
    if index.has_duplicates:
        raise RuntimeError(f"canonical proxy source contains duplicate dates for {ticker}")
    values = pd.to_numeric(frame["daily_return"], errors="coerce").to_numpy(dtype=np.float64)
    values = np.log1p(np.clip(values, -0.999999, None))
    series = pd.Series(values, index=index, dtype=float, name=label)
    series = series.replace([np.inf, -np.inf], np.nan).dropna()
    if len(series) < 252:
        raise RuntimeError(f"canonical proxy source has fewer than 252 usable rows for {ticker}")
    return series


@lru_cache(maxsize=1)
def _load_packaged_canonical_proxy_factor_returns() -> pd.DataFrame:
    """Load the complete nine-series source factor closure without refreshes."""
    _load_canonical_proxy_manifest()
    columns = [
        _load_canonical_proxy_series(ticker, label)
        for ticker, label in _CANONICAL_PROXY_MAP
    ]
    factor_frame = pd.concat(columns, axis=1).sort_index()
    if tuple(factor_frame.columns) != tuple(label for _ticker, label in _CANONICAL_PROXY_MAP):
        raise RuntimeError("canonical proxy factor column order drifted")
    return factor_frame


def _canonical_proxy_factor_returns() -> pd.DataFrame:
    """Return a defensive copy of the complete frozen source factor frame."""
    return _load_packaged_canonical_proxy_factor_returns().copy(deep=True)


def _fit_factor_drift_prior(
    log_returns: pd.Series,
    factor_frame: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Reproduce the source factor drift prior with an optional parity frame."""

    y = log_returns.dropna().astype(float)
    if len(y) < 252:
        return {
            "status": "skipped",
            "reason": "insufficient_portfolio_history",
            "daily_log_mean": 0.0,
            "adjusted_r_squared": 0.0,
            "factor_count": 0,
            "observation_count": len(y),
        }

    factors = _canonical_proxy_factor_returns() if factor_frame is None else factor_frame.copy(deep=True)
    if factors.empty:
        raise RuntimeError("canonical proxy factor frame is empty")

    aligned_factors = factors.reindex(y.index)
    min_obs = int(max(252, min(756, len(y) // 3)))
    usable_columns = [
        column
        for column in aligned_factors.columns
        if int(aligned_factors[column].notna().sum()) >= min_obs
    ]
    if not usable_columns:
        return {
            "status": "skipped",
            "reason": "no_factor_proxy_overlap",
            "daily_log_mean": 0.0,
            "adjusted_r_squared": 0.0,
            "factor_count": 0,
            "observation_count": len(y),
        }

    factor_slice = aligned_factors[usable_columns].copy()
    combined = pd.concat([y.rename("portfolio"), factor_slice], axis=1).dropna()
    while len(combined) < min_obs and len(usable_columns) > 1:
        valid_counts = factor_slice[usable_columns].notna().sum().sort_values()
        drop_column = str(valid_counts.index[0])
        usable_columns = [column for column in usable_columns if column != drop_column]
        combined = pd.concat([y.rename("portfolio"), factor_slice[usable_columns]], axis=1).dropna()

    if len(combined) < min_obs or len(usable_columns) == 0:
        return {
            "status": "skipped",
            "reason": "insufficient_joint_factor_overlap",
            "daily_log_mean": 0.0,
            "adjusted_r_squared": 0.0,
            "factor_count": len(usable_columns),
            "observation_count": len(combined),
        }

    y_arr = combined["portfolio"].to_numpy(dtype=np.float64)
    x_arr = combined[usable_columns].to_numpy(dtype=np.float64)
    y_center = y_arr - float(np.mean(y_arr))
    x_mean = np.mean(x_arr, axis=0)
    x_center = x_arr - x_mean
    p = int(x_center.shape[1])
    try:
        xtx = x_center.T @ x_center
        trace_scale = float(np.trace(xtx) / max(p, 1))
        ridge = float(max(trace_scale * (p / max(len(combined), 1)), 1e-12))
        beta = np.linalg.solve(xtx + ridge * np.eye(p), x_center.T @ y_center)
    except (ArithmeticError, RuntimeError, ValueError):
        beta = np.linalg.pinv(x_center) @ y_center

    fitted_center = x_center @ beta
    resid = y_center - fitted_center
    ss_res = float(np.dot(resid, resid))
    ss_tot = float(np.dot(y_center, y_center))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1e-16 else 0.0
    adj_r2 = 1.0 - (1.0 - r2) * ((len(combined) - 1.0) / max(len(combined) - p - 1.0, 1.0))
    adj_r2 = float(np.clip(adj_r2 if np.isfinite(adj_r2) else 0.0, 0.0, 0.95))

    alpha_raw = float(np.mean(y_arr) - float(x_mean @ beta))
    resid_sigma = float(np.std(resid, ddof=1)) if len(resid) > p + 1 else 0.0
    alpha_se = resid_sigma / math.sqrt(float(len(combined))) if resid_sigma > 0 else 0.0
    alpha_t = abs(alpha_raw) / alpha_se if alpha_se > 0 else 0.0
    alpha_weight = float((alpha_t * alpha_t) / (1.0 + alpha_t * alpha_t)) if np.isfinite(alpha_t) else 0.0
    alpha_shrunk = float(alpha_raw * alpha_weight)

    factor_means = []
    for column in usable_columns:
        mean_i, _ = _sample_mean_near_zero_shrinkage(combined[column].to_numpy(dtype=np.float64))
        factor_means.append(float(mean_i))
    factor_mean_arr = np.asarray(factor_means, dtype=np.float64)
    daily_log_mean = float(alpha_shrunk + float(factor_mean_arr @ beta))
    if not np.isfinite(daily_log_mean):
        daily_log_mean = 0.0

    return {
        "status": "complete",
        "daily_log_mean": daily_log_mean,
        "adjusted_r_squared": adj_r2,
        "factor_count": int(p),
        "factor_names": [str(column) for column in usable_columns],
        "observation_count": len(combined),
        "alpha_raw": alpha_raw,
        "alpha_shrink_weight": alpha_weight,
        "alpha_shrunk": alpha_shrunk,
        "ridge_penalty": float(ridge) if "ridge" in locals() else None,
        "betas": {str(column): float(value) for column, value in zip(usable_columns, beta)},
    }


def _fit_sticky_volatility_regime_filter(sigma_x: np.ndarray) -> dict[str, Any]:
    sigma = np.asarray(sigma_x, dtype=np.float64)
    sigma = sigma[np.isfinite(sigma)]
    sigma = np.maximum(sigma, 1e-6)
    n = len(sigma)
    fallback_transition = np.asarray(
        [[0.97, 0.03, 0.00], [0.015, 0.97, 0.015], [0.00, 0.03, 0.97]],
        dtype=np.float64,
    )
    if n < 30:
        states = np.ones(max(n, 1), dtype=np.int32)
        state_sigma = np.asarray(
            [
                float(np.quantile(sigma, 0.25)) if n else 1.0,
                float(np.median(sigma)) if n else 1.0,
                float(np.quantile(sigma, 0.85)) if n else 1.0,
            ],
            dtype=np.float64,
        )
        return {
            "states": states[:n],
            "state_probabilities_last": np.asarray([0.15, 0.70, 0.15], dtype=np.float64),
            "transition_matrix": fallback_transition,
            "state_sigma_x": np.maximum(state_sigma, 1e-6),
            "status": "fallback_short_history",
            "iteration_count": 0,
        }

    y = np.log(sigma)
    k = 3
    means = np.quantile(y, [0.20, 0.55, 0.85]).astype(np.float64)
    common_var = float(np.var(y, ddof=1)) if n > 1 else 0.05
    variances = np.full(k, max(common_var, 1e-4), dtype=np.float64)
    trans = fallback_transition.copy()
    start = np.asarray([0.20, 0.60, 0.20], dtype=np.float64)
    gamma = np.full((n, k), 1.0 / k, dtype=np.float64)
    iterations = 0

    max_iter = int(max(1, CANONICAL_REGIME_EM_MAX_ITER))
    min_iter = int(max(1, min(CANONICAL_REGIME_EM_MIN_ITER, max_iter)))
    tol = float(max(CANONICAL_REGIME_EM_TOL, 0.0))
    for iterations in range(1, max_iter + 1):
        previous_means = means.copy()
        previous_trans = trans.copy()
        log_emit = -0.5 * (
            np.log(2.0 * math.pi * variances)[None, :]
            + ((y[:, None] - means[None, :]) ** 2) / variances[None, :]
        )
        row_max = np.max(log_emit, axis=1, keepdims=True)
        emit = np.exp(np.clip(log_emit - row_max, -745.0, 50.0))
        emit = np.maximum(emit, 1e-300)

        fwd = np.empty((n, k), dtype=np.float64)
        scales = np.empty(n, dtype=np.float64)
        fwd[0] = start * emit[0]
        scales[0] = float(np.sum(fwd[0]))
        if scales[0] <= 0.0 or not np.isfinite(scales[0]):
            fwd[0] = 1.0 / k
            scales[0] = 1.0
        else:
            fwd[0] /= scales[0]
        for t in range(1, n):
            fwd[t] = (fwd[t - 1] @ trans) * emit[t]
            scales[t] = float(np.sum(fwd[t]))
            if scales[t] <= 0.0 or not np.isfinite(scales[t]):
                fwd[t] = 1.0 / k
                scales[t] = 1.0
            else:
                fwd[t] /= scales[t]

        bwd = np.ones((n, k), dtype=np.float64)
        for t in range(n - 2, -1, -1):
            bwd[t] = trans @ (emit[t + 1] * bwd[t + 1])
            denom = float(np.sum(bwd[t]))
            if denom <= 0.0 or not np.isfinite(denom):
                bwd[t] = 1.0
            else:
                bwd[t] /= denom

        gamma = fwd * bwd
        gamma_sum = np.sum(gamma, axis=1, keepdims=True)
        gamma = np.divide(
            gamma,
            gamma_sum,
            out=np.full_like(gamma, 1.0 / k),
            where=gamma_sum > 0.0,
        )
        xi_num = fwd[:-1, :, None] * trans[None, :, :] * (emit[1:] * bwd[1:])[:, None, :]
        xi_denom = np.sum(xi_num, axis=(1, 2), keepdims=True)
        xi_valid = (xi_denom[:, 0, 0] > 0.0) & np.isfinite(xi_denom[:, 0, 0])
        if np.any(xi_valid):
            xi_sum = np.sum(xi_num[xi_valid] / xi_denom[xi_valid], axis=0)
        else:
            xi_sum = np.zeros((k, k), dtype=np.float64)
        sticky_prior = np.full((k, k), 1.0, dtype=np.float64)
        np.fill_diagonal(sticky_prior, 25.0)
        trans = xi_sum + sticky_prior
        trans = trans / np.maximum(trans.sum(axis=1, keepdims=True), 1e-12)

        weights = np.maximum(gamma.sum(axis=0), 1e-12)
        means = (gamma.T @ y) / weights
        centered = y[:, None] - means[None, :]
        variances = np.maximum((gamma * centered * centered).sum(axis=0) / weights, 1e-4)
        order = np.argsort(means)
        means = means[order]
        variances = variances[order]
        trans = trans[order][:, order]
        gamma = gamma[:, order]
        start = gamma[0]
        start = start / max(float(np.sum(start)), 1e-12)
        if iterations >= min_iter:
            delta = max(
                float(np.max(np.abs(means - previous_means))),
                float(np.max(np.abs(trans - previous_trans))),
            )
            if np.isfinite(delta) and delta < tol:
                break

    states = np.argmax(gamma, axis=1).astype(np.int32)
    state_sigma = np.empty(k, dtype=np.float64)
    for state in range(k):
        mask = states == state
        state_sigma[state] = (
            float(np.median(sigma[mask]))
            if np.any(mask)
            else float(np.quantile(sigma, [0.25, 0.55, 0.85][state]))
        )
    state_sigma = np.maximum(np.sort(state_sigma), 1e-6)
    last_probs = gamma[-1]
    last_probs = last_probs / max(float(np.sum(last_probs)), 1e-12)
    return {
        "states": states,
        "state_probabilities_last": last_probs.astype(np.float64),
        "transition_matrix": trans.astype(np.float64),
        "state_sigma_x": state_sigma.astype(np.float64),
        "status": "complete",
        "iteration_count": int(iterations),
        "emission_log_sigma_means": means.astype(np.float64),
        "emission_log_sigma_variances": variances.astype(np.float64),
    }


def _canonical_forecast_candidate() -> Mapping[str, Any]:
    return _CANONICAL_SPEC


def _fit_canonical_forecast_model(log_returns: pd.Series, min_obs: int = 30) -> dict[str, Any] | None:
    clean = log_returns.dropna().astype(float)
    clean.index = pd.to_datetime(clean.index).normalize()
    clean = clean[~clean.index.duplicated(keep="last")]
    arr = clean.to_numpy(dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    if len(arr) < int(min_obs):
        return None

    portfolio_mu, portfolio_mean_meta = _sample_mean_near_zero_shrinkage(arr)
    sample_mu = float(np.mean(arr))
    factor_meta = _fit_factor_drift_prior(clean)
    factor_mu = float(factor_meta.get("daily_log_mean", 0.0) or 0.0)
    factor_obs = float(factor_meta.get("observation_count", 0) or 0)
    coverage_weight = float(np.clip(factor_obs / max(float(len(arr)), 1.0), 0.0, 1.0))
    factor_weight = float(
        np.clip((factor_meta.get("adjusted_r_squared", 0.0) or 0.0) * coverage_weight, 0.0, 0.50)
    )
    mu = float((1.0 - factor_weight) * float(portfolio_mu) + factor_weight * factor_mu)
    if np.isfinite(sample_mu):
        if sample_mu >= 0.0:
            mu = float(min(mu, sample_mu))
        else:
            mu = float(max(mu, sample_mu))
    if not np.isfinite(mu):
        mu = float(portfolio_mu) if np.isfinite(portfolio_mu) else 0.0

    residuals = arr - mu
    residuals = residuals[np.isfinite(residuals)]
    if len(residuals) < int(min_obs):
        return None
    vol_fit = _fit_ewma_volatility(residuals)
    x = residuals * 100.0
    neg_var = float(np.mean((x[x < 0.0]) ** 2)) if np.any(x < 0.0) else float(np.mean(x * x))
    all_var = float(np.mean(x * x)) if len(x) else 1.0
    asymmetry_multiplier = float(np.clip(neg_var / max(all_var, 1e-12), 0.75, 2.50))
    vol_fit["requested_vol_model"] = "regime_conditioned_asymmetric_gjr_shrinkage_volatility"
    vol_fit["vol_model"] = "regime_conditioned_asymmetric_gjr_shrinkage_volatility"
    vol_fit["params"] = {
        **dict(vol_fit.get("params", {}) or {}),
        "asymmetry_multiplier": asymmetry_multiplier,
    }
    sigma_x = np.asarray(vol_fit["sigma_x"], dtype=np.float64)
    z = _standardize_residuals(residuals, sigma_x)
    if z.size < 20:
        return None
    regime_sigma_x = sigma_x[-int(max(30, CANONICAL_REGIME_FIT_MAX_OBS)) :]
    regime_fit = _fit_sticky_volatility_regime_filter(regime_sigma_x)
    states = np.asarray(regime_fit.get("states"), dtype=np.int32)
    if len(states) != len(residuals):
        if len(states):
            pad = np.full(max(0, len(residuals) - len(states)), int(states[0]), dtype=np.int32)
            states = np.concatenate([pad, states])[-len(residuals) :].astype(np.int32)
        else:
            states = np.ones(len(residuals), dtype=np.int32)
    eps_sq_hist = residuals * residuals
    block_length = max(_politis_white_block_length(z), _politis_white_block_length(eps_sq_hist))
    sigma_daily = float(np.std(arr, ddof=0)) if len(arr) > 1 else 0.0
    sigma_annual = float(sigma_daily * math.sqrt(252.0))
    drift_se = sigma_daily / math.sqrt(float(len(arr))) if sigma_daily > 0 else 0.0
    drift_posterior_sd = float(drift_se * math.sqrt(max(0.0, 1.0 - factor_weight)))
    candidate = _canonical_forecast_candidate()
    return {
        **candidate,
        "mu": float(mu),
        "residuals": residuals.astype(np.float64),
        "standardized_residuals": z.astype(np.float64),
        "return_pool": pd.Series(residuals, index=clean.index[-len(residuals) :], dtype=float),
        "return_pool_size": len(residuals),
        "historical_daily_log_volatility": float(sigma_daily),
        "historical_volatility_annualized": float(sigma_annual),
        "block_length": int(block_length),
        "n_obs": len(arr),
        "history_years": float(len(arr) / 252.0),
        "scaled_log_returns": (arr * 100.0).astype(np.float64),
        "vol_fit": vol_fit,
        "regime_states": states.astype(np.int32),
        "regime_transition_matrix": np.asarray(regime_fit["transition_matrix"], dtype=np.float64),
        "regime_state_probabilities_last": np.asarray(
            regime_fit["state_probabilities_last"], dtype=np.float64
        ),
        "regime_state_sigma_x": np.asarray(regime_fit["state_sigma_x"], dtype=np.float64),
        "regime_fit_status": str(regime_fit.get("status", "complete")),
        "regime_iteration_count": int(regime_fit.get("iteration_count", 0) or 0),
        "factor_prior_daily_log_mean": float(factor_mu),
        "factor_prior_weight": factor_weight,
        "factor_prior_meta": factor_meta,
        "portfolio_mean_meta": portfolio_mean_meta,
        "drift_posterior_sd_daily_log": drift_posterior_sd,
        "asymmetry_multiplier": asymmetry_multiplier,
    }


def _evt_standardized_draws(z: np.ndarray, size: tuple[int, int], rng: np.random.Generator) -> np.ndarray:
    clean = np.asarray(z, dtype=np.float64)
    clean = clean[np.isfinite(clean)]
    if clean.size < 100:
        return rng.choice(clean if clean.size else np.array([0.0]), size=size, replace=True)
    exceedance_share = float(np.clip(math.sqrt(clean.size) / clean.size, 0.02, 0.10))
    lower_q = float(np.quantile(clean, exceedance_share))
    upper_q = float(np.quantile(clean, 1.0 - exceedance_share))
    central = clean[(clean >= lower_q) & (clean <= upper_q)]
    if central.size == 0:
        central = clean
    out = rng.choice(central, size=size, replace=True).astype(np.float64)
    uniforms = rng.random(size)
    lower_mask = uniforms < exceedance_share
    upper_mask = uniforms > (1.0 - exceedance_share)
    try:
        lower_excess = lower_q - clean[clean < lower_q]
        upper_excess = clean[clean > upper_q] - upper_q
        if lower_excess.size >= 10:
            c, _, scale = stats.genpareto.fit(lower_excess, floc=0.0)
            c = float(np.clip(c, -0.45, 0.45)) if np.isfinite(c) else 0.0
            out[lower_mask] = lower_q - stats.genpareto.rvs(
                c, loc=0.0, scale=max(scale, 1e-8), size=int(np.sum(lower_mask)), random_state=rng
            )
        if upper_excess.size >= 10:
            c, _, scale = stats.genpareto.fit(upper_excess, floc=0.0)
            c = float(np.clip(c, -0.45, 0.45)) if np.isfinite(c) else 0.0
            out[upper_mask] = upper_q + stats.genpareto.rvs(
                c, loc=0.0, scale=max(scale, 1e-8), size=int(np.sum(upper_mask)), random_state=rng
            )
    except (ArithmeticError, RuntimeError, ValueError):
        out[lower_mask | upper_mask] = rng.choice(
            clean, size=int(np.sum(lower_mask | upper_mask)), replace=True
        )
    return np.clip(out, -20.0, 20.0)


def _standardize_generated_innovations(draws: np.ndarray) -> np.ndarray:
    arr = np.array(draws, dtype=np.float64, copy=True)
    np.nan_to_num(arr, copy=False, nan=0.0, posinf=20.0, neginf=-20.0)
    np.clip(arr, -20.0, 20.0, out=arr)
    if arr.size == 0:
        return arr
    mean = float(np.mean(arr))
    std = float(np.std(arr))
    arr -= mean
    if std > 1e-12 and np.isfinite(std):
        arr /= std
    np.clip(arr, -20.0, 20.0, out=arr)
    return arr


def _simulate_canonical_forecast_log_paths(
    model: Mapping[str, Any],
    total_days: int,
    n_paths: int,
    rng: np.random.Generator,
) -> np.ndarray:
    total_days = int(total_days)
    n_paths = int(n_paths)
    if total_days <= 0 or n_paths <= 0:
        return np.empty((max(n_paths, 0), max(total_days, 0)), dtype=np.float64)
    z = np.asarray(model.get("standardized_residuals"), dtype=np.float64)
    z = z[np.isfinite(z)]
    if z.size == 0:
        z = np.array([0.0], dtype=np.float64)
    block_length = int(max(1, model.get("block_length", _politis_white_block_length(z))))
    idx = _stationary_bootstrap_indices(
        n=len(z), block_length=block_length, total_days=total_days, n_paths=n_paths, rng=rng
    )
    draws = z[idx].astype(np.float64, copy=False)
    if z.size >= 100:
        evt_draws = _evt_standardized_draws(z, (n_paths, total_days), rng)
        threshold = float(np.quantile(np.abs(z), 0.90))
        draws = np.where(np.abs(draws) >= threshold, evt_draws, draws)
    draws = _standardize_generated_innovations(draws)

    transition = np.asarray(model.get("regime_transition_matrix"), dtype=np.float64)
    if transition.shape != (3, 3) or not np.isfinite(transition).all():
        transition = np.asarray(
            [[0.97, 0.03, 0.00], [0.015, 0.97, 0.015], [0.00, 0.03, 0.97]],
            dtype=np.float64,
        )
    transition = transition / np.maximum(transition.sum(axis=1, keepdims=True), 1e-12)
    state_probs = np.asarray(model.get("regime_state_probabilities_last"), dtype=np.float64)
    if (
        state_probs.shape != (3,)
        or not np.isfinite(state_probs).all()
        or float(np.sum(state_probs)) <= 0.0
    ):
        state_probs = np.asarray([0.15, 0.70, 0.15], dtype=np.float64)
    state_probs = state_probs / float(np.sum(state_probs))
    state_sigma = np.asarray(model.get("regime_state_sigma_x"), dtype=np.float64)
    if state_sigma.shape != (3,) or not np.isfinite(state_sigma).all():
        last_sigma = float(model.get("vol_fit", {}).get("last_sigma_x", 1.0))
        state_sigma = np.asarray([0.70 * last_sigma, last_sigma, 1.60 * last_sigma], dtype=np.float64)
    state_sigma = np.maximum(state_sigma, 1e-6)

    cumulative_probs = np.cumsum(state_probs)
    states = np.searchsorted(cumulative_probs, rng.random(n_paths), side="right").astype(np.int32)
    states = np.clip(states, 0, 2)
    diag_persistence = float(np.mean(np.diag(transition)))
    regime_blend = float(np.clip(1.0 - diag_persistence, 0.005, 0.08))
    vol_fit = model.get("vol_fit", {}) or {}
    params = vol_fit.get("params", {}) or {}
    lam = float(np.clip(vol_fit.get("lambda", params.get("lambda", 0.94)), 0.80, 0.995))
    asymmetry_multiplier = float(
        np.clip(model.get("asymmetry_multiplier", params.get("asymmetry_multiplier", 1.0)), 0.75, 2.50)
    )
    sigma_x = np.full(
        n_paths,
        float(max(vol_fit.get("last_sigma_x", np.median(state_sigma)), 1e-6)),
        dtype=np.float64,
    )
    h_x = np.maximum(sigma_x * sigma_x, 1e-8)
    mu = float(model.get("mu", 0.0))
    drift_sd = float(max(model.get("drift_posterior_sd_daily_log", 0.0) or 0.0, 0.0))
    path_mu = (
        rng.normal(mu, drift_sd, size=n_paths).astype(np.float64)
        if drift_sd > 0.0
        else np.full(n_paths, mu, dtype=np.float64)
    )
    out = np.empty((n_paths, total_days), dtype=np.float64)
    for t in range(total_days):
        if t > 0:
            uniforms = rng.random(n_paths)
            p0 = transition[states, 0]
            p1 = p0 + transition[states, 1]
            states = (uniforms > p0).astype(np.int32) + (uniforms > p1).astype(np.int32)
        target_h = state_sigma[states] * state_sigma[states]
        h_x = (1.0 - regime_blend) * h_x + regime_blend * target_h
        sigma_x = np.sqrt(np.maximum(h_x, 1e-8))
        z_t = draws[:, t]
        eps_x = sigma_x * z_t
        out[:, t] = path_mu + eps_x / 100.0
        shock_scale = np.where(eps_x < 0.0, asymmetry_multiplier, 1.0)
        h_x = lam * h_x + (1.0 - lam) * eps_x * eps_x * shock_scale
        h_x = np.clip(h_x, 1e-8, 1e8)
    return np.clip(out, -1.0, 1.0)


def _mix_terminal_log_values(
    canonical_terminal_log_values: np.ndarray,
    benchmark_terminal_log_values: np.ndarray,
    canonical_weight: float,
    n_sims: int,
) -> np.ndarray:
    canonical = np.asarray(canonical_terminal_log_values, dtype=np.float64)
    canonical = canonical[np.isfinite(canonical)]
    benchmark = np.asarray(benchmark_terminal_log_values, dtype=np.float64)
    benchmark = benchmark[np.isfinite(benchmark)]
    n = int(max(1, n_sims))
    if canonical.size == 0:
        return benchmark[:n]
    if benchmark.size == 0:
        return canonical[:n]
    weight = float(np.clip(canonical_weight, 0.0, 1.0))
    canonical_count = round(weight * n)
    benchmark_count = n - canonical_count
    if canonical_count <= 0:
        return benchmark[:n]
    if benchmark_count <= 0:
        return canonical[:n]
    if canonical.size < canonical_count:
        canonical = np.resize(canonical, canonical_count)
    if benchmark.size < benchmark_count:
        benchmark = np.resize(benchmark, benchmark_count)
    return np.concatenate([canonical[:canonical_count], benchmark[:benchmark_count]]).astype(np.float64)


def _training_series(training: TrainingData) -> pd.Series:
    values = np.asarray(training.portfolio_log_returns, dtype=np.float64)
    if training.training_dates is None:
        index = pd.date_range("1900-01-01", periods=values.size, freq="D")
    else:
        index = pd.to_datetime(np.asarray(training.training_dates, dtype="datetime64[ns]"))
    return pd.Series(values, index=index, dtype=float)


@dataclass(frozen=True)
class CanonicalStackModel:
    """Exact-ID terminal adapter for canonical and stack rows."""

    model_id: str
    forecast_output_semantics: str = TERMINAL_FORECAST_SEMANTICS
    seed_contract: str = SOURCE_SEED_CONTRACT
    dependency_identity: Mapping[str, Any] = field(
        default_factory=lambda: MappingProxyType(
            {
                "source_imports": ("hashlib", "math", "numpy", "pandas", "scipy"),
                "runtime_constraints": ("numpy>=2.0,<3", "pandas>=2.2,<3", "scipy>=1.13,<2"),
                "factor_proxy_loader": MappingProxyType(
                    {
                        "mode": "offline_manifest_whitelisted_snapshot",
                        "manifest_path": "resources/data/canonical_snapshot_manifest.json",
                        "manifest_sha256": _CANONICAL_PROXY_MANIFEST_SHA256,
                        "tickers": tuple(ticker for ticker, _label in _CANONICAL_PROXY_MAP),
                        "series_sha256": dict(_CANONICAL_PROXY_SERIES_SHA256),
                    }
                ),
            }
        )
    )

    def __post_init__(self) -> None:
        if self.model_id not in REFERENCE_MODEL_IDS:
            raise ValueError(f"unknown canonical stack model: {self.model_id!r}")

    @property
    def family(self) -> str:
        return str(SOURCE_CANDIDATE_SPECS[self.model_id]["type"])

    @property
    def source_specification(self) -> Mapping[str, Any]:
        return SOURCE_CANDIDATE_SPECS[self.model_id]

    @property
    def source_function_names(self) -> tuple[str, ...]:
        return SOURCE_FUNCTION_NAMES

    @property
    def source_fragment_digest(self) -> str:
        return SOURCE_FUNCTIONS_SHA256

    def _source_rng(
        self,
        context: ForecastContext,
        candidate_id: str,
        horizon_grid: tuple[int, ...],
    ) -> np.random.Generator:
        source_origin = str(context.origin_date) if context.origin_date is not None else str(context.origin_label)
        seed = _deterministic_seed(
            "forecast_oos_candidate",
            source_origin,
            horizon_grid,
            candidate_id,
            int(context.simulations),
        )
        return np.random.default_rng(seed)

    def _candidate_terminals(
        self,
        candidate_id: str,
        candidate_type: str,
        training: TrainingData,
        context: ForecastContext,
        horizon_grid: tuple[int, ...],
    ) -> dict[int, np.ndarray]:
        values = np.asarray(training.portfolio_log_returns, dtype=np.float64)
        rng = self._source_rng(context, candidate_id, horizon_grid)
        max_horizon = int(max(horizon_grid))
        if candidate_type == "canonical":
            fit = _fit_canonical_forecast_model(_training_series(training), min_obs=30)
            if fit is None:
                raise RuntimeError(
                    f"{self.model_id} cannot execute: canonical fit requires at least 30 usable observations"
                )
            paths = _simulate_canonical_forecast_log_paths(
                fit, max_horizon, int(context.simulations), rng
            )
            if paths.size == 0:
                raise RuntimeError(f"{self.model_id} canonical simulator returned no paths")
            cumulative = np.cumsum(paths, axis=1, dtype=np.float64)
            return {
                int(horizon): cumulative[:, int(horizon) - 1].astype(np.float64)
                for horizon in horizon_grid
            }
        if candidate_type == "gaussian":
            return {
                int(horizon): _SourceKernel._constant_mean_gaussian_log_terminal_samples(
                    values, int(horizon), int(context.simulations), rng
                )
                for horizon in horizon_grid
            }
        if candidate_type == "student_t":
            paths = _SourceKernel._constant_mean_student_t_log_paths(
                values, max_horizon, int(context.simulations), rng
            )
            if paths.size == 0:
                raise RuntimeError(
                    f"{self.model_id} student-t stack component requires at least 30 usable observations"
                )
            cumulative = np.cumsum(paths, axis=1, dtype=np.float64)
            return {
                int(horizon): cumulative[:, int(horizon) - 1].astype(np.float64)
                for horizon in horizon_grid
            }
        raise AssertionError(f"unhandled source candidate type: {candidate_type}")

    def simulate_terminal_log_returns(
        self,
        training: TrainingData,
        context: ForecastContext,
    ) -> np.ndarray:
        training.validate()
        horizon = int(context.horizon_days)
        simulations = int(context.simulations)
        if horizon < 1 or simulations < 1:
            raise ValueError("canonical stack forecast requires positive horizon and simulation count")
        horizon_grid = tuple(range(1, horizon + 1))
        if self.model_id == CANONICAL_MODEL_ID:
            terminals = self._candidate_terminals(
                "canonical", "canonical", training, context, horizon_grid
            )
        else:
            base = self._candidate_terminals(
                "canonical", "canonical", training, context, horizon_grid
            )
            shrinkage = str(_STACK_SOURCE_DEFAULTS["shrinkage"])
            component = self._candidate_terminals(
                f"constant_mean_{shrinkage}", shrinkage, training, context, horizon_grid
            )
            terminals = {
                int(horizon_value): _mix_terminal_log_values(
                    base[int(horizon_value)],
                    component[int(horizon_value)],
                    canonical_weight=float(_STACK_SOURCE_DEFAULTS["canonical_weight"]),
                    n_sims=simulations,
                )
                for horizon_value in horizon_grid
            }
        result = np.column_stack([terminals[int(horizon_value)] for horizon_value in horizon_grid])
        expected_shape = (simulations, horizon)
        if result.shape != expected_shape or not np.all(np.isfinite(result)):
            raise ValueError(f"{self.model_id} returned invalid terminal ensemble shape {result.shape}")
        return np.asarray(result, dtype=np.float64)


REFERENCE_FACTORIES: Mapping[str, Any] = MappingProxyType(
    {model_id: (lambda model_id=model_id: CanonicalStackModel(model_id)) for model_id in REFERENCE_MODEL_IDS}
)


def make_canonical_stack_model(model_id: str) -> CanonicalStackModel:
    factory = REFERENCE_FACTORIES.get(str(model_id))
    if factory is None:
        raise ValueError(f"unknown canonical stack model: {model_id!r}")
    return factory()


__all__ = [
    "CANONICAL_MODEL_ID",
    "REFERENCE_FACTORIES",
    "REFERENCE_MODEL_IDS",
    "SOURCE_ARTIFACTS",
    "SOURCE_CANDIDATE_SPECS",
    "SOURCE_COMPONENT_FUNCTIONS_SHA256",
    "SOURCE_FUNCTIONS_SHA256",
    "SOURCE_FUNCTION_NAMES",
    "SOURCE_SEED_CONTRACT",
    "STACK_MODEL_IDS",
    "CanonicalStackModel",
    "make_canonical_stack_model",
]
