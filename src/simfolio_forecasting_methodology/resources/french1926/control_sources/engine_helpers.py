"""Exact extracted native engine block-rule and stationary-index closure."""
from __future__ import annotations
import math
import numpy as np
try:
    from numba import njit, prange
except Exception:
    njit = None
    prange = range

if njit is not None:
    @njit(cache=True, parallel=True)
    def _stationary_bootstrap_indices_from_draws_numba(
        n: int,
        initial_indices: np.ndarray,
        restart_flags: np.ndarray,
        restart_points: np.ndarray,
    ) -> np.ndarray:
        n_paths = initial_indices.shape[0]
        total_days = restart_flags.shape[1] + 1
        idx = np.empty((n_paths, total_days), dtype=np.int32)
        for row_idx in prange(n_paths):
            last_index = int(initial_indices[row_idx])
            idx[row_idx, 0] = last_index
            for col_idx in range(1, total_days):
                if restart_flags[row_idx, col_idx - 1]:
                    last_index = int(restart_points[row_idx, col_idx - 1])
                else:
                    last_index += 1
                    if last_index == n:
                        last_index = 0
                idx[row_idx, col_idx] = last_index
        return idx

else:
    _stationary_bootstrap_indices_from_draws_numba = None

def _politis_white_block_length(values: np.ndarray) -> int:
    """
    Politis–White optimal block length for the stationary bootstrap. Used by
    other engine bootstrap-replay paths (factor analysis, forecast median replay).
    """
    arr = np.asarray(values, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    n = len(arr)
    if n < 4:
        return 1
    arr = arr - float(np.mean(arr))
    max_lag = int(min(max(5, math.ceil((n ** (1.0 / 3.0)) * 2.0)), max(2, n // 4)))
    if max_lag < 2:
        return 1
    gamma0 = float(np.dot(arr, arr) / n)
    if not np.isfinite(gamma0) or gamma0 <= 0:
        return 1
    gamma = np.empty(max_lag + 1, dtype=np.float64)
    gamma[0] = gamma0
    for lag in range(1, max_lag + 1):
        gamma[lag] = float(np.dot(arr[lag:], arr[:-lag]) / n)
    weights = 1.0 - (np.arange(1, max_lag + 1, dtype=np.float64) / (max_lag + 1.0))
    g_hat = float(np.sum(2.0 * weights * np.arange(1, max_lag + 1, dtype=np.float64) * gamma[1:]))
    d_hat = float(gamma0 + 2.0 * np.sum(weights * gamma[1:]))
    if not np.isfinite(g_hat) or not np.isfinite(d_hat) or abs(d_hat) < 1e-12:
        return 1
    b_opt = ((2.0 * g_hat * g_hat) / (d_hat * d_hat) * n) ** (1.0 / 3.0)
    if not np.isfinite(b_opt) or b_opt < 1.0:
        return 1
    return int(max(1, round(float(b_opt))))

def _stationary_bootstrap_indices_from_draws_fast(
    *,
    n: int,
    initial_indices: np.ndarray,
    restart_flags: np.ndarray,
    restart_points: np.ndarray,
) -> np.ndarray:
    initial = np.asarray(initial_indices, dtype=np.int32).reshape(-1)
    flags = np.asarray(restart_flags, dtype=bool)
    points = np.asarray(restart_points, dtype=np.int32)
    n_paths = int(len(initial))
    total_days = int(flags.shape[1] + 1)
    if _stationary_bootstrap_indices_from_draws_numba is not None:
        return _stationary_bootstrap_indices_from_draws_numba(int(n), initial, flags, points)

    idx = np.empty((n_paths, total_days), dtype=np.int32)
    if n_paths == 0:
        return idx

    idx[:, 0] = initial
    if total_days == 1:
        return idx

    idx[:, 1:] = points
    positions = np.arange(total_days, dtype=np.int32)
    start_positions = np.empty(total_days, dtype=np.int32)
    for row_idx in range(n_paths):
        start_positions[0] = 0
        start_positions[1:] = np.where(flags[row_idx], positions[1:], 0)
        np.maximum.accumulate(start_positions, out=start_positions)
        idx[row_idx] = (idx[row_idx, start_positions] + positions - start_positions) % int(n)
    return idx

