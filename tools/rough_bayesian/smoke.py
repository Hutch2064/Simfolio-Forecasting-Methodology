"""Independent reference recurrences and complete smoke score parity."""
from dataclasses import replace
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from scipy.signal import lfilter
import inference
import models
import overlay
from simfolio_forecasting_methodology.experiment import build_experiment_plan
from simfolio_forecasting_methodology.runner import evaluate_origin_task


def reference_gaussian(phi, innovation, weights, initial, noise, level, scale):
    path = np.zeros(noise.size)
    terminal = np.empty(phi.size)
    if not noise.size:
        return path, initial.copy()
    for j in range(phi.size):
        states, _ = lfilter([innovation[j]], [1, -phi[j]], noise, zi=[phi[j] * initial[j]])
        path += weights[j] * np.r_[initial[j], states[:-1]]
        terminal[j] = states[-1]
    return level + scale * path, terminal


def reference_heston(phi, weights, step, initial, noise, level, kappa, eta):
    state = initial.copy()
    h = np.empty(noise.size)
    theta = np.exp(level)
    total_step = np.dot(weights, step)
    for j, white in enumerate(noise):
        positive = max(theta + np.dot(weights, state), 1e-10)
        h[j] = np.log(positive)
        common = eta * np.sqrt(positive) * white
        predicted = theta + np.dot(weights, phi * state)
        next_variance = (predicted + total_step * (kappa * theta + common)) / (1 + kappa * total_step)
        state = phi * state + step * (common - kappa * (next_variance - theta))
    return h, state


def reference_predictive_fou(phi, innovation, weights, terminal, noise, shock, mean, level, scale):
    h, _ = reference_gaussian(phi, innovation, weights, terminal, noise, level, scale)
    return np.clip(mean + np.exp(h / 2) / 100 * shock, -1, 1)


def reference_predictive_heston(phi, weights, step, terminal, noise, shock, mean, level, kappa, eta):
    h, _ = reference_heston(phi, weights, step, terminal, noise, level, kappa, eta)
    return np.clip(mean + np.exp(h / 2) / 100 * shock, -1, 1)


def reference_multiplier(phi, weights, root, initial, posterior, normals):
    state = initial.copy()
    mean = normals[0].copy()
    variance = posterior.copy()
    out = np.empty(normals.shape[0] - 1)
    q = root @ root.T
    for t in range(out.size):
        out[t] = np.exp(.5 * (weights @ state - weights @ mean) - .25 * (weights @ variance @ weights))
        state = phi * state + root @ normals[t + 1]
        mean = phi * mean
        variance = variance * np.outer(phi, phi) + q
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    models.initialize(None)
    plan = build_experiment_plan(args.data)
    task = plan.tasks[0]
    training = replace(task.training,
        portfolio_log_returns=task.training.portfolio_log_returns[-256:],
        asset_log_returns=task.training.asset_log_returns[-256:],
        training_dates=task.training.training_dates[-256:])
    task = replace(task, training=training,
        realized_future_daily_log_returns=task.realized_future_daily_log_returns[:63],
        future_dates=task.future_dates[:63])
    originals = (inference.gaussian_path, inference.heston_path,
                 models.predictive_fou, models.predictive_heston, overlay.multiplier_path, overlay.filter_rough)
    cases = []
    for candidate in models.CANDIDATES[2:]:
        candidate = replace(candidate, burn=32, kept=64, max_kept=64)
        models.fit.cache_clear()
        overlay.fit.cache_clear()
        start = time.perf_counter()
        fast = evaluate_origin_task(candidate, task, simulations=240)
        fast_seconds = time.perf_counter() - start
        inference.gaussian_path = reference_gaussian
        inference.heston_path = reference_heston
        models.predictive_fou = reference_predictive_fou
        models.predictive_heston = reference_predictive_heston
        overlay.multiplier_path = reference_multiplier
        overlay.filter_rough = lambda *args: models.controls.filter_rough(*args)[:3]
        models.fit.cache_clear()
        overlay.fit.cache_clear()
        start = time.perf_counter()
        reference = evaluate_origin_task(candidate, task, simulations=240)
        reference_seconds = time.perf_counter() - start
        (inference.gaussian_path, inference.heston_path,
         models.predictive_fou, models.predictive_heston, overlay.multiplier_path, overlay.filter_rough) = originals
        difference = float(np.max(np.abs(fast - reference)))
        np.testing.assert_allclose(fast, reference, rtol=0, atol=1e-12)
        record = {'model_id': candidate.model_id, 'optimized_seconds': fast_seconds,
                  'reference_seconds': reference_seconds, 'max_absolute_crps_difference': difference,
                  'optimized_smoke_crps': float(fast.mean()), 'reference_smoke_crps': float(reference.mean()),
                  'all_horizon_losses_checked': True, 'simulations': 240,
                  'training_rows': 256, 'horizons': 63, 'burn': 32, 'kept': 64}
        cases.append(record)
        print(json.dumps(record), flush=True)
    receipt = {'scope': 'bounded_reference_parity_smoke_not_canonical_scores',
               'source_hashes': {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                 for p in sorted(Path(__file__).parent.glob('*.py'))},
               'all_parity_checks_passed': True, 'cases': cases}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(receipt, indent=2) + '\n')


if __name__ == '__main__':
    main()
