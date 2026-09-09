"""Executable reference implementations for the broader research catalogue.

The historical catalogue spans many generations of experiments.  Pipe-delimited
IDs are implemented compositionally.  Remaining bespoke IDs resolve to an
explicit legacy reference adapter rather than failing catalogue execution; the
registry exposes the fidelity class so saved historical scores are never
mistaken for a fresh exact-port result.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
from scipy.optimize import minimize
from scipy import stats

from ..runner import ForecastContext, TrainingData
from ..seeds import deterministic_seed
from ..specifications import CompositionalSpec, parse_compositional_spec


def _stationary_bootstrap(pool: np.ndarray, rows: int, columns: int, rng: np.random.Generator) -> np.ndarray:
    values = np.asarray(pool, dtype=np.float64)
    block = max(2.0, min(63.0, float(values.size) ** (1.0 / 3.0)))
    restart = 1.0 / block
    indices = np.empty((rows, columns), dtype=np.int64)
    indices[:, 0] = rng.integers(0, values.size, size=rows)
    for column in range(1, columns):
        fresh = rng.integers(0, values.size, size=rows)
        continued = (indices[:, column - 1] + 1) % values.size
        indices[:, column] = np.where(rng.random(rows) < restart, fresh, continued)
    return values[indices]


def _mean_fit(values: np.ndarray, method: str) -> tuple[float, np.ndarray]:
    x = np.asarray(values, dtype=np.float64)
    sample = float(np.mean(x))
    if method == "expanding_sample_mean":
        mean = sample
    elif method == "factor_premium_near_zero_alpha_shrinkage":
        mean = 0.25 * sample
    else:  # BIC auto-ARMA reference: AR(1) only when BIC improves over constant mean.
        centered = x - sample
        left, right = centered[:-1], centered[1:]
        phi = float(np.clip(np.dot(left, right) / max(np.dot(left, left), 1e-12), -0.98, 0.98))
        residual = right - phi * left
        var_ar = max(float(np.var(residual)), 1e-12)
        var_mean = max(float(np.var(centered)), 1e-12)
        bic_ar = len(residual) * math.log(var_ar) + 2.0 * math.log(max(len(residual), 2))
        bic_mean = len(centered) * math.log(var_mean) + math.log(max(len(centered), 2))
        mean = sample + phi * float(centered[-1]) if bic_ar < bic_mean else sample
    return float(mean), x - float(mean)


def _garch_variance(residuals: np.ndarray, kind: str) -> tuple[np.ndarray, tuple[float, ...]]:
    e = np.asarray(residuals, dtype=np.float64)
    base_var = max(float(np.var(e, ddof=1)), 1e-10)
    if kind == "constant_sample_volatility":
        return np.full(e.size, base_var), (base_var,)

    def objective(theta: np.ndarray) -> float:
        omega = math.exp(float(theta[0]))
        alpha = 1.0 / (1.0 + math.exp(-float(theta[1]))) * 0.35
        beta = 1.0 / (1.0 + math.exp(-float(theta[2]))) * (0.995 - alpha)
        gamma = 0.0
        if kind == "gjr_tarch_1_1_volatility":
            gamma = 1.0 / (1.0 + math.exp(-float(theta[3]))) * max(0.0, 0.995 - alpha - beta)
        variance = base_var
        likelihood = 0.0
        for index in range(1, e.size):
            shock = e[index - 1] * e[index - 1]
            asymmetric = gamma * shock if e[index - 1] < 0.0 else 0.0
            variance = max(omega + alpha * shock + asymmetric + beta * variance, 1e-12)
            likelihood += math.log(variance) + e[index] * e[index] / variance
        return 0.5 * likelihood

    dimensions = 4 if kind == "gjr_tarch_1_1_volatility" else 3
    initial = np.zeros(dimensions)
    initial[0] = math.log(base_var * 0.03)
    initial[1] = -2.0
    initial[2] = 2.5
    fit = minimize(objective, initial, method="L-BFGS-B", options={"maxiter": 160})
    theta = np.asarray(fit.x, dtype=np.float64)
    omega = math.exp(float(theta[0]))
    alpha = 1.0 / (1.0 + math.exp(-float(theta[1]))) * 0.35
    beta = 1.0 / (1.0 + math.exp(-float(theta[2]))) * (0.995 - alpha)
    gamma = 0.0
    if kind == "gjr_tarch_1_1_volatility":
        gamma = 1.0 / (1.0 + math.exp(-float(theta[3]))) * max(0.0, 0.995 - alpha - beta)
    variance = np.empty(e.size, dtype=np.float64)
    variance[0] = base_var
    for index in range(1, e.size):
        shock = e[index - 1] ** 2
        variance[index] = max(
            omega + alpha * shock + (gamma * shock if e[index - 1] < 0 else 0.0) + beta * variance[index - 1],
            1e-12,
        )
    return variance, (omega, alpha, beta, gamma)


def _draw_innovations(spec: CompositionalSpec, pool: np.ndarray, rows: int, cols: int, rng: np.random.Generator) -> np.ndarray:
    if spec.innovation_model == "empirical":
        if spec.resampling == "stationary_bootstrap":
            return _stationary_bootstrap(pool, rows, cols, rng)
        return rng.choice(pool, size=(rows, cols), replace=True)
    if spec.innovation_model == "student_t_standardized_innovations":
        df, loc, scale = stats.t.fit(np.clip(pool, -20, 20))
        draws = stats.t.rvs(df, loc=loc, scale=scale, size=(rows, cols), random_state=rng)
        return (draws - np.mean(draws)) / max(float(np.std(draws)), 1e-8)
    if spec.innovation_model == "skew_t_standardized_innovations":
        shape, loc, scale = stats.skewnorm.fit(np.clip(pool, -20, 20))
        draws = stats.skewnorm.rvs(shape, loc=loc, scale=scale, size=(rows, cols), random_state=rng)
        return (draws - np.mean(draws)) / max(float(np.std(draws)), 1e-8)
    return rng.normal(size=(rows, cols))


@dataclass(frozen=True)
class CompositionalModel:
    model_id: str

    def simulate_daily_log_returns(self, training: TrainingData, context: ForecastContext) -> np.ndarray:
        spec = parse_compositional_spec(self.model_id)
        x = np.asarray(training.portfolio_log_returns, dtype=np.float64)
        mean, residuals = _mean_fit(x, spec.mean_model)
        variance, params = _garch_variance(residuals, spec.volatility_model)
        standardized = residuals / np.sqrt(np.maximum(variance, 1e-12))
        standardized = standardized[np.isfinite(standardized)]
        standardized -= float(np.mean(standardized))
        standardized /= max(float(np.std(standardized, ddof=1)), 1e-8)
        rng = np.random.default_rng(deterministic_seed(self.model_id, context.seed))
        innovations = _draw_innovations(
            spec, standardized, context.simulations, context.horizon_days, rng
        )
        if spec.volatility_model == "constant_sample_volatility":
            scale = np.full((context.simulations, context.horizon_days), math.sqrt(float(variance[-1])))
        else:
            omega, alpha, beta, gamma = params
            current = np.full(context.simulations, float(variance[-1]))
            scale = np.empty((context.simulations, context.horizon_days), dtype=np.float64)
            prior_shock = np.full(context.simulations, float(residuals[-1]))
            for day in range(context.horizon_days):
                shock2 = prior_shock * prior_shock
                current = omega + alpha * shock2 + beta * current
                if gamma:
                    current += gamma * shock2 * (prior_shock < 0.0)
                current = np.maximum(current, 1e-12)
                scale[:, day] = np.sqrt(current)
                prior_shock = scale[:, day] * innovations[:, day]
        return mean + scale * innovations


@dataclass(frozen=True)
class LegacyReferenceModel:
    """Runnable compatibility adapter for historical bespoke catalogue IDs.

    It preserves the model ID and deterministic OOS mechanics.  Historical
    saved scores remain the authoritative evidence for IDs not yet exact-ported.
    """

    model_id: str

    def simulate_daily_log_returns(self, training: TrainingData, context: ForecastContext) -> np.ndarray:
        values = np.asarray(training.portfolio_log_returns, dtype=np.float64)
        rng = np.random.default_rng(deterministic_seed("legacy_reference", self.model_id, context.seed))
        lowered = self.model_id.lower()
        if "mean_zero" in lowered or "zero_mean" in lowered:
            mean = 0.0
        else:
            mean = float(np.mean(values))
        residuals = values - mean
        std = max(float(np.std(residuals, ddof=1)), 1e-8)
        pool = residuals / std
        if "stationary" in lowered or "sbb" in lowered or "bootstrap" in lowered:
            draws = _stationary_bootstrap(pool, context.simulations, context.horizon_days, rng)
        elif "student" in lowered or "_t_" in lowered:
            draws = rng.standard_t(7.0, size=(context.simulations, context.horizon_days))
            draws /= math.sqrt(7.0 / 5.0)
        else:
            draws = rng.choice(pool, size=(context.simulations, context.horizon_days), replace=True)
        return mean + std * draws
