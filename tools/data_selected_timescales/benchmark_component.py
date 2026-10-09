"""Paired timing of the changed mean-scaling component, not M256 forecasts."""
import argparse
import json
import time
from pathlib import Path

import models
import numpy as np

from simfolio_forecasting_methodology.experiment import build_experiment_plan
from simfolio_forecasting_methodology.models.asset_level.sv_moment_functions import (
    moment_return_curves,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert models.VARIANT == 'existing_sv_rate'
    models.initialize(None, 1, 1)
    plan = build_experiment_plan(args.data)
    records = []
    for i in (0, 24, 48):
        task = plan.tasks[i]
        for asset, x in enumerate(np.asarray(task.training.asset_log_returns).T):
            data = np.ascontiguousarray(x).tobytes()
            fit = models.parent.body.learned.student.original_fit(data)
            level, phi, eta = fit['posterior_center'][:3]
            y, _ = models.bd._sv_observed_log_variance(x, float(x.mean()), winsorize=True)
            loglik, h, _ = models.bd._sv_kalman_rts_smoother_mean(y, level, phi, eta)
            assert np.isfinite(loglik) and np.isfinite(h).all()
            samples = {'fixed_four': [], 'existing_sv_rate': []}
            def evaluate(arm, h=h, phi=phi, fit=fit, horizon=task.horizon_days):
                if arm == 'fixed_four':
                    q = models.OPTIMIZED_COMPONENTS(h, 4)
                else:
                    q = models.data_components(h, 4, fitted_phi=phi)
                return moment_return_curves(dict(fit, bdes_multiscale_vol=q), horizon)
            for arm in samples:
                evaluate(arm)
            for repeat in range(12):
                for arm in (tuple(samples) if repeat % 2 == 0 else tuple(reversed(samples))):
                    models.selected_components.cache_clear()
                    start = time.perf_counter()
                    evaluate(arm)
                    samples[arm].append(time.perf_counter()-start)
            medians = {k: float(np.median(v)) for k, v in samples.items()}
            records.append({'task': i, 'asset_column': asset, 'training_rows': len(x),
                'horizon': task.horizon_days, 'samples_seconds': samples,
                'median_seconds': medians,
                'relative_time_saved': 1.-medians['existing_sv_rate']/medians['fixed_four']})
    total = {arm: sum(r['median_seconds'][arm] for r in records) for arm in records[0]['median_seconds']}
    result = {'scope': 'paired_changed_component_only; no M256 forecast or panel rerun',
        'cache_policy': 'selector cleared before every observation; shared fitted states held fixed',
        'records': records, 'summed_case_medians_seconds': total,
        'relative_time_saved': 1.-total['existing_sv_rate']/total['fixed_four']}
    args.output.write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({k: v for k, v in result.items() if k != 'records'}), flush=True)


if __name__ == '__main__':
    main()
