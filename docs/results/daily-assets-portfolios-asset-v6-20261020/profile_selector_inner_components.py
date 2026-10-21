"""Time every original component on an actual slow-request historical fold."""
import os
for k in('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import datetime,hashlib,json,sys,time
import numpy as np
O=Path(__file__).resolve().parent;V=O/'asset_level_v6';D=V/'candidates/asset_selector_universe_precision/development1000';sys.path.insert(0,str(V))
import common,sweep_v6 as api
from inputs import AssetInputs
from candidates.asset_selector_universe.bank import BANK,model,prefix_fits
from candidates.asset_selector_universe_precision.candidate import schedule
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();write=api.original._atomic_json
plan=read(D/'plan.json');summary=read(D/'summary.json');assert summary['complete_execution']
sources=[read(p)for p in(D/'case_progress').glob('*.json')]
case=max((d for d in sources if not d['errors']and d['records']),key=lambda d:d['shared_fit_seconds'])['records'][0]['case']
c=plan['cases'][case];row=next(r for r in api.load_panel(V/'inputs',verify_files=False)['series']if r['series_id']==c['series_id']);history,w,ids=AssetInputs(V/'inputs').training(row,c['origin']['position']+1)
eligible,folds=schedule(len(history),c['origin']['max_horizon']);stop,H=folds[-1];x=history[:stop];prepared,invalid=prefix_fits(x,eligible);valid=[i for i in eligible if i not in invalid]
seed=int.from_bytes(hashlib.sha256(f'horizon_pool_inner_v1:{hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()}:{H}:1000'.encode()).digest()[:4],'little')
records=[]
for index in valid:
    begin=time.perf_counter();terminal=None;cursor=0
    for start,paths in model(index,prepared).terminal_chunks(x,w,H,1000,seed):
        assert start==cursor;cursor+=paths.shape[1];terminal=paths[:,-1].copy()
    assert cursor==H and np.isfinite(terminal).all()
    records.append({'bank_index':index,'specification':BANK[index],'simulation_seconds':time.perf_counter()-begin,'terminal_sha256':hashlib.sha256(terminal.tobytes()).hexdigest()})
total=sum(r['simulation_seconds']for r in records);raw=sum(r['simulation_seconds']for r in records if r['specification']['kind']=='blocks');filtered=sum(r['simulation_seconds']for r in records if r['specification']['kind']=='filtered')
result={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Single chronological inner fold of observed slowest successful completed request, not full-request speed benchmark, production latency or new model score','case':case,'series_id':c['series_id'],'history_rows':len(x),'assets':len(ids),'validation_horizon':H,'paths':1000,'seed':seed,'selection_rule':'Largest measured shared_fit_seconds among completed successful cases; no forecast-score criterion','total_component_simulation_seconds':total,'raw_blocks_fraction':raw/total,'filtered_fraction':filtered/total,'records':records,'source_plan_sha256':sha(D/'plan.json'),'no_new_scientific_method_admitted':True,'diagnostics_zero_score_or_selection_weight':True}
write(O/'selector_component_runtime_profile.json',result);print(json.dumps({k:v for k,v in result.items()if k!='records'},indent=2),flush=True)
