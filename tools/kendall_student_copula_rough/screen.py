"""Bounded smoke through the unchanged canonical origin scorer."""
import argparse
import hashlib
import json
import pickle
import platform
import tempfile
import time
from dataclasses import replace
from pathlib import Path

import models
import numpy as np
import scipy

from simfolio_forecasting_methodology.runner import evaluate_origin_task


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--tasks',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--task-index',type=int,default=0)
    parser.add_argument('--model-id',choices=models.MODEL_IDS,required=True)
    parser.add_argument('--fit-seconds',type=float,default=10.)
    parser.add_argument('--asset-workers',type=int,default=2)
    parser.add_argument('--disable-kendall',action='store_true')
    args=parser.parse_args()
    task=pickle.load(args.tasks.open('rb'))[args.task_index]
    candidate=next(c for c in models.CANDIDATES if c.model_id==args.model_id)
    if isinstance(candidate,models.Candidate):candidate=replace(candidate,fit_seconds=args.fit_seconds)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    cache=Path(tempfile.mkdtemp(prefix=args.output.stem+'.cache-',dir=args.output.parent))
    models.select_copula(not args.disable_kendall)
    models.initialize(cache,1,args.asset_workers)
    # Compile forecast routines before the measurement clock.
    phi=np.array([.9]);w=np.ones(1);root=np.array([[.1]])
    models.overlay.path_normalizers(phi,w,root,root,np.zeros(1),2)
    models.overlay.multiplier_independent_prepared(phi,w,root,np.zeros(1),np.zeros((3,1)),np.zeros(2),np.ones(2))
    # Warm dependence kernels on unrelated synthetic history. The measured
    # task's own dependence fit remains cold; no benchmark observations enter
    # this compiler warm-up.
    dummy=np.random.default_rng(867).normal(size=(300,6))
    models.shell.controls.gaussian_uniforms(dummy.shape,dummy.tobytes(),4,2,867)
    models.shell.controls.gaussian_uniforms.cache_clear()
    records=[];failure=None;reference=None
    for state in ('fresh_fits','cached_fits'):
        models.TIMINGS.clear();models.FIT_DIAGNOSTICS.clear()
        started=time.perf_counter()
        try:losses=evaluate_origin_task(candidate,task,simulations=240)
        except Exception as error:  # noqa: BLE001 - Smoke receipts record any candidate failure.
            failure={'type': type(error).__name__,'message': str(error)}
            record={'cache_state': state,'seconds': time.perf_counter()-started,'timings': dict(models.TIMINGS),'failure': failure}
            records.append(record);print(json.dumps(record),flush=True);break
        record={'cache_state': state,'seconds': time.perf_counter()-started,'timings': dict(models.TIMINGS),
            'smoke_crps': float(losses.mean()),'horizons': len(losses),'loss_sha256': hashlib.sha256(losses.tobytes()).hexdigest(),
            'diagnostics': list(models.FIT_DIAGNOSTICS.values())}
        if reference is None:reference=losses.copy()
        elif reference.tobytes()!=losses.tobytes():raise AssertionError('fresh/cached loss-vector parity')
        np.save(args.output.with_name(args.output.stem+'.'+state+'.npy'),losses)
        records.append(record);print(json.dumps({k:v for k,v in record.items() if k!='diagnostics'}),flush=True)
    receipt={'scope': 'bounded_local_smoke_not_full_panel','model_id': args.model_id,'task_index': args.task_index,
        'portfolio_id': task.portfolio_id,'origin': str(task.origin_date),'training_rows': len(task.training.asset_log_returns),
        'assets': task.training.asset_log_returns.shape[1],'horizon': task.horizon_days,'simulations': 240,
        'runtime': {'python': platform.python_version(),'numpy': np.__version__,'scipy': scipy.__version__},
        'kendall_scatter_enabled':not args.disable_kendall,'fit_cache': str(cache),'failure': failure,'records': records,
        'source_hashes': {str(p.relative_to(models.ROOT)):hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(Path(__file__).resolve().parent.glob('*.py'))}}
    args.output.write_text(json.dumps(receipt,indent=2)+'\n')


if __name__=='__main__':main()
