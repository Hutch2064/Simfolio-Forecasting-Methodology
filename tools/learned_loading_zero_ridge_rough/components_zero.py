"""Existing multiscale construction with supplied decay rates; no other estimator change."""
from typing import Any

import numpy as np
from rate_fit import ewma as _bdes_ewma


def components(h_path: np.ndarray, phis: np.ndarray, half_lives=None, shrink_loadings=True, loading_scale=None) -> dict[str, Any]:
    h = np.asarray(h_path, dtype=float); finite = h[np.isfinite(h)]; fill = float(np.nanmedian(finite)) if finite.size else 0.0
    h = np.clip(np.nan_to_num(h, nan=fill, posinf=fill, neginf=fill), -18.0, 18.0); n = h.size; ell = float(np.nanmean(h)); centered = h - ell
    half_lives = -np.log(2.0) / np.log(phis) if half_lives is None else np.asarray(half_lives); q = np.empty((n, phis.size), dtype=float)
    for k, phi in enumerate(phis):
        q[:, k] = _bdes_ewma(centered, 1.0 - phi); q[:, k] -= float(np.nanmean(q[:, k]))
    b = np.linalg.lstsq(q, centered, rcond=np.finfo(np.float64).eps * max(q.shape))[0]
    if not np.isfinite(b).all(): raise ArithmeticError("nonfinite unregularized least-squares loading")
    with np.errstate(all="ignore"):
        component = q @ b
    component = component if np.all(np.isfinite(component)) else np.zeros_like(centered); resid = centered - component
    signal = np.abs(b) * np.nanstd(q, axis=0)
    if loading_scale is not None:
        b = b * loading_scale
        component = q @ b; resid = centered - component
    elif shrink_loadings and np.sum(signal) > 0.0:
        shrink = signal / (signal + np.nanmedian(signal[signal > 0.0]) + 1e-8); b = b * shrink
        with np.errstate(all="ignore"):
            component = q @ b
        component = component if np.all(np.isfinite(component)) else np.zeros_like(centered); resid = centered - component
    component_var = max(float(np.nanvar(component)), 0.0); resid_var = max(float(np.nanvar(resid)), 1e-8); reliability_weight = float(component_var / max(component_var + resid_var, 1e-8)); post_signal = np.abs(b) * np.nanstd(q, axis=0)
    dominant_half_life = float(np.sum(post_signal * half_lives) / np.sum(post_signal)) if np.sum(post_signal) > 0.0 else float(np.median(half_lives))
    inverse_mse_weight = reliability_weight
    if n > 8:
        lagged, actual = centered[:-1], centered[1:]; denom = float(np.dot(lagged, lagged)); ar_phi = float(np.clip(np.dot(lagged, actual) / denom, -0.999, 0.999)) if denom > 1e-12 else 0.0; ar_pred = ar_phi * lagged
        with np.errstate(all="ignore"):
            multiscale_pred = (q[:-1, :] * phis.reshape(1, -1)) @ b
        finite_mask = np.isfinite(actual) & np.isfinite(ar_pred) & np.isfinite(multiscale_pred)
        if np.count_nonzero(finite_mask) >= 8:
            ar_mse = max(float(np.mean((actual[finite_mask] - ar_pred[finite_mask]) ** 2)), 1e-8); multi_mse = max(float(np.mean((actual[finite_mask] - multiscale_pred[finite_mask]) ** 2)), 1e-8); inverse_mse_weight = float(ar_mse / (ar_mse + multi_mse))
    q_var = np.empty(phis.size); q_innov = np.empty((max(n - 1, 0), phis.size))
    for k, phi in enumerate(phis): q_innov[:, k] = q[1:, k] - phi * q[:-1, k]; q_var[k] = max(float(np.nanvar(q_innov[:, k])), 1e-8)
    residual = resid - float(np.nanmean(resid)); residual_lag, residual_next = residual[:-1], residual[1:]; denom = float(np.dot(residual_lag, residual_lag)); residual_phi = float(np.clip(np.dot(residual_lag, residual_next) / denom, -0.999999, 0.999999)) if denom > np.finfo(np.float64).tiny else 0.0; residual_innovations = residual_next - residual_phi * residual_lag; residual_innovation_sd = max(float(np.nanstd(residual_innovations, ddof=1)), np.finfo(np.float64).tiny); q_innovation_sd = np.maximum(np.nanstd(q_innov, axis=0, ddof=1), np.finfo(np.float64).tiny); standardized_q = q_innov / q_innovation_sd.reshape(1, -1); common_innovation = np.nanmean(standardized_q, axis=1); common_innovation -= float(np.nanmean(common_innovation)); common_sd = float(np.nanstd(common_innovation, ddof=1)); common_innovation = common_innovation / common_sd if common_sd > np.finfo(np.float64).tiny else np.zeros_like(common_innovation); standardized_residual = residual_innovations / residual_innovation_sd; common_denom = float(np.dot(common_innovation, common_innovation)); residual_common_loading = float(np.clip(np.dot(common_innovation, standardized_residual) / common_denom, -1.0, 1.0)) if common_denom > np.finfo(np.float64).tiny else 0.0
    h_q005, h_q25, h_q75, h_q995 = np.quantile(h, [0.005, 0.25, 0.75, 0.995]); h_iqr = max(float(h_q75 - h_q25), 1e-6)
    return {"ell": ell, "phis": phis, "b": b, "q_last": q[-1, :].copy(), "q_var": q_var, "half_lives": half_lives.copy(), "scale_count": int(half_lives.size), "hbar": float(np.nanmean(h)), "h_low": float(h_q005 - h_iqr), "h_high": float(h_q995 + h_iqr), "component_low": float(np.quantile(component, 0.01)), "component_high": float(np.quantile(component, 0.99)), "resid_var": resid_var, "residual_last": float(residual[-1]), "residual_phi": residual_phi, "residual_innovation_sd": residual_innovation_sd, "residual_common_loading": residual_common_loading, "component_var": component_var, "reliability_weight": reliability_weight, "inverse_mse_weight": inverse_mse_weight, "dominant_half_life_days": dominant_half_life, "max_half_life_days": float(np.max(half_lives))}
