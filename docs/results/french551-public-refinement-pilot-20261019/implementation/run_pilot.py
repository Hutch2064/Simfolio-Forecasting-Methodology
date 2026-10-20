"""Predeclared small paired pilot; full original horizons, no campaign mutation."""
import os
for key in ('OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMBA_NUM_THREADS',
            'OMP_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS', 'NUMEXPR_NUM_THREADS'):
    os.environ[key] = '1'
import sys
import json
import hashlib
from pathlib import Path
from time import perf_counter
from dataclasses import asdict
from collections import defaultdict
import numpy as np
import pandas as pd
from simfolio_forecasting_methodology.runner import TrainingData, PortfolioPolicy, ForecastContext
from simfolio_forecasting_methodology.french_panel.control_adapters import TestingNaiveControl, _native_kernel
from simfolio_forecasting_methodology.evaluation import empirical_crps_by_horizon
from simfolio_forecasting_methodology.seeds import coherent_daily_seed
from candidate import native_history, public_indices, simulate

ROOT = Path(__file__).resolve().parent.parent
INPUT = ROOT.parent/'french551-full-20261019/inputs/french1926-equity551'
PANEL = json.loads((INPUT/'panel.json').read_text())
ORIGINS = json.loads((INPUT/'origins.json').read_text())


def write(path, value):
    path.parent.mkdir(exist_ok=True, parents=True)
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False,
                               default=lambda obj:obj.tolist() if isinstance(obj,np.ndarray)
                               else obj.item() if isinstance(obj,np.generic) else str(obj))+'\n')
    temp.replace(path)


def plan():
    strata = defaultdict(list)
    for row in PANEL['series']:
        strata[row['family'], row['weighting']].append(row)
    selected = []
    for key in sorted(strata):
        rows = sorted(strata[key], key=lambda r:r['series_id'])
        selected.append(rows[(len(rows)-1)//2])
    labels = ['rolling_01', 'rolling_24', 'rolling_48',
              'train_first_quarter_test_remaining']
    origins = [o for o in ORIGINS if o['origin_label'] in labels]
    return {
        'identity': 'public_conditional_residual_sbb_pilot_v1',
        'status': 'predeclared_before_any_pilot_scores',
        'panel_fingerprint': PANEL['panel_fingerprint'],
        'frozen_revision': 'b958ed46cc2b341a024abbd8a8dea70ff569c01f',
        'selection': 'lower median series_id within each family and weighting stratum; broad market only value',
        'series': selected, 'origins': origins, 'paths': 240,
        'seed': 'unchanged coherent_daily_seed(origin_date, full max_horizon, 240), shared baseline/candidate',
        'score': 'v2 q=CRPS/(training_std_ddof1*sqrt(h+h*h/n)); origin means inside daily horizon cells; log1p(1/h) weights; equal series',
        'sensitivity': 'same aggregation, denominator training_std_ddof1*sqrt(h)',
        'all_eligible_horizons': True,
        'partial_scope': 'pilot complete on only selected 11 series and four selected origins; no full-panel victory or significance claim',
        'artifact_budget_gib': 3, 'expected_growth_gib': .2,
        'candidate': 'raw mean; exact public block selection and index RNG; training-only existing EWMA QMLE; residuals centered and unit-RMS; recursive conditional variance; native simple-return clipping and wealth accounting',
        'source_sha256': {p.name:hashlib.sha256(p.read_bytes()).hexdigest()
                          for p in sorted(Path(__file__).parent.glob('*.py'))},
    }


def context(row, origin, values, dates):
    stop = origin['position']+1
    history = np.array(values[:stop], copy=True)
    history.flags.writeable = False
    training = TrainingData(history, history.reshape(-1,1),
                            PortfolioPolicy((row['series_id'],),(1.0,), 'none'),
                            dates[:stop])
    seed = coherent_daily_seed(origin['origin_date'], origin['max_horizon'], 240)
    future = pd.bdate_range(pd.Timestamp(origin['origin_date'])+pd.offsets.BDay(1),
                           periods=origin['max_horizon']).to_numpy(dtype='datetime64[ns]')
    ctx = ForecastContext('model_testing_joint_stationary_bootstrap_v1', row['series_id'],
                          origin['origin_label'], origin['max_horizon'], 240, seed,
                          future, origin['origin_date'])
    return training, ctx


def diagnose(terminal, history, horizon):
    cagr = np.expm1(terminal[:,-1]*252/horizon)
    if not np.all(np.isfinite(cagr)):
        raise ValueError('nonfinite CAGR')
    return {'mean_cagr': float(cagr.mean()), 'median_cagr': float(np.median(cagr)),
            'training_cagr':float(np.expm1(history.mean()*252)),
            'mean_terminal_log':float(terminal[:,-1].mean()),
            'median_terminal_log':float(np.median(terminal[:,-1])),
            'terminal_log_std':float(np.std(terminal[:,-1],ddof=1)),
            'max_abs_daily_cumulative_log':float(np.abs(terminal).max())}


def summary(predeclared):
    stats = {}
    for variant in ['public', 'ewma_residual_sbb']:
        series_rows = []
        for row in predeclared['series']:
            records = []
            for origin in predeclared['origins']:
                path = ROOT/'results'/row['series_id']/(origin['origin_label']+'.npz')
                if not path.exists():
                    break
                with np.load(path) as z:
                    records.append((z[variant+'_loss'].copy(),float(z['train_std']),int(z['train_n'])))
            if len(records) != len(predeclared['origins']):
                continue
            H = max(len(r[0]) for r in records)
            count = np.zeros(H); raw = np.zeros(H); q = np.zeros(H); sens = np.zeros(H)
            for loss, std, n in records:
                h = np.arange(1,len(loss)+1)
                count[:len(h)] += 1; raw[:len(h)] += loss
                q[:len(h)] += loss/(std*np.sqrt(h+h*h/n))
                sens[:len(h)] += loss/(std*np.sqrt(h))
            weight = np.log1p(1/np.arange(1,H+1))
            series_rows.append({'series_id':row['series_id'], 'family':row['family'],
                'weighting':row['weighting'], 'v2':float(np.dot(weight,q/count)/np.log1p(H)),
                'sensitivity':float(np.dot(weight,sens/count)/np.log1p(H)),
                'raw':float(np.mean(raw/count)), 'min_origins_per_horizon':int(count.min()),
                'max_origins_per_horizon':int(count.max()), 'horizons':H})
        stats[variant] = {'complete_series':len(series_rows), 'series':series_rows}
        if series_rows:
            stats[variant].update({k:float(np.mean([r[k] for r in series_rows]))
                                  for k in ['v2','sensitivity','raw']})
    if stats['public'].get('v2'):
        stats['skill_percent'] = 100*(1-stats['ewma_residual_sbb']['v2']/stats['public']['v2'])
    stats['scope'] = 'predeclared bounded paired pilot, not full-panel official scores'
    write(ROOT/'summary.json', stats)
    return stats


def run():
    predeclared = json.loads((ROOT/'plan.json').read_text())
    if predeclared != plan():
        raise ValueError('source or plan changed after declaration')
    dates = np.load(INPUT/'dates.npy').astype('datetime64[ns]')
    values = np.load(INPUT/'daily_log_returns.npy',mmap_mode='r')
    positions = {r['series_id']:i for i,r in enumerate(PANEL['series'])}
    public = TestingNaiveControl()
    for row in predeclared['series']:
        for origin in predeclared['origins']:
            output = ROOT/'results'/row['series_id']/(origin['origin_label']+'.npz')
            if output.exists():
                continue
            history_values = values[positions[row['series_id']]]
            training,ctx = context(row,origin,history_values,dates)
            start = perf_counter()
            daily, metadata = public.simulate_with_metadata(training,ctx)
            baseline_s = perf_counter()-start
            hist = native_history(training.portfolio_log_returns)
            start = perf_counter()
            indices,block = public_indices(hist,240,ctx.horizon_days,ctx.seed)
            index_s = perf_counter()-start
            # Independent exact replay verifies row selection, RNG ordering and native accounting.
            replay = _native_kernel()._fused_no_rebalance_terminal_wealth(
                np.clip(np.expm1(hist),-.999999,np.finfo(np.float64).max/10).reshape(-1,1),
                indices,np.ones(1),np.ones(1),np.ones(ctx.horizon_days))[0]
            if not np.array_equal(replay,daily):
                raise ValueError('independent public replay lacks byte parity')
            start = perf_counter()
            candidate,fit = simulate(hist,indices)
            candidate_s = index_s+perf_counter()-start
            if candidate.shape != daily.shape or not np.all(np.isfinite(candidate)):
                raise ValueError('candidate shape/nonfinite')
            stop=origin['position']+1
            target=np.cumsum(history_values[stop:stop+ctx.horizon_days])
            arrays={'train_std':np.std(training.portfolio_log_returns,ddof=1),
                    'train_n':stop}
            diagnostics={'series':row,'origin':origin,'seed':ctx.seed,
                'block_length':block,'fit':fit,'public_seconds':baseline_s,
                'candidate_seconds':candidate_s,'index_seconds':index_s,
                'public_exact_independent_replay':True,
                'public_metadata':metadata}
            for name,paths in [('public',daily),('ewma_residual_sbb',candidate)]:
                terminal=np.cumsum(paths,axis=1)
                arrays[name+'_loss']=empirical_crps_by_horizon(terminal,target)
                selected_h=sorted(set([1,min(21,ctx.horizon_days),min(252,ctx.horizon_days),ctx.horizon_days]))
                arrays[name+'_selected_terminal']=terminal[:,np.array(selected_h)-1]
                arrays['selected_horizons']=np.array(selected_h)
                diagnostics[name]=diagnose(terminal,training.portfolio_log_returns,ctx.horizon_days)
            output.parent.mkdir(parents=True,exist_ok=True)
            np.savez(output,**arrays)
            write(output.with_suffix('.json'),diagnostics)
            print(json.dumps({'series':row['series_id'],'origin':origin['origin_label'],
                'horizon':ctx.horizon_days,'public_s':baseline_s,'candidate_s':candidate_s,
                'decay':fit['decay'],'status':'saved'}),flush=True)
            summary(predeclared)
    print(json.dumps(summary(predeclared)),flush=True)


if __name__=='__main__':
    if '--declare' in sys.argv:
        if (ROOT/'plan.json').exists():
            raise ValueError('plan already exists')
        write(ROOT/'plan.json',plan())
        print('declared 11 series x four origins x240 paths with full horizons')
    else:
        run()
