"""Actual-source short-history request qualification; not the frozen panel leaderboard."""
import os
for k in('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import concurrent.futures,datetime,fcntl,hashlib,json,multiprocessing,subprocess,sys,time
import numpy as np
O=Path(__file__).resolve().parent;V=O/'asset_level_v6';C=V/'candidates/asset_uncapped_selector';D=C/'request_smoke';sys.path.insert(0,str(V))
import common,sweep_v6 as api
from inputs import AssetInputs
from candidates.asset_uncapped_selector.candidate import prepare,CompleteHistorySelector,BANK
read=lambda p:json.loads(p.read_text());write=api.original._atomic_json;sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()

def run(task):
    panel=api.load_panel(V/'inputs',verify_files=False);metadata={r['series_id']:r for r in panel['series']};row=metadata[task['series_id']];x,w,ids=AssetInputs(V/'inputs').training(row,task['outer_stop']);x=x[-96:]
    started=time.perf_counter();bundle=prepare(x,w,task['horizon']);model=CompleteHistorySelector(bundle);digest=hashlib.sha256();cursor=0
    for start,paths in model.terminal_chunks(x,w,task['horizon'],1000,88301):assert start==cursor and paths.shape[0]==1000;cursor+=paths.shape[1];assert np.isfinite(paths).all();digest.update(paths.tobytes())
    assert cursor==task['horizon'];path=D/f"trace-{task['number']:03d}.json";write(path,bundle['trace'])
    return {**task,'constituent_ids':ids,'history_rows':len(x),'request_compute_seconds':time.perf_counter()-started,'history_sha256':hashlib.sha256(x.tobytes()).hexdigest(),'output_sha256':digest.hexdigest(),'trace_sha256':sha(path),'choice':bundle['choice'],'selected_specification':BANK[bundle['choice']],'diagnostics':bundle['diagnostics'],'invalid_internal_options':bundle['invalid_internal_options'],'outer_paths':1000,'official':False}

def main():
    slots=(O/'asset_v6_forecast_slots.lock').open('a');fcntl.flock(slots,fcntl.LOCK_EX)
    assert not(D/'active.json').exists(),'Inspect existing execution'
    subprocess.run([sys.executable,'/Users/aidanhutchison/simfolio/storage_preflight.py','--expected-gib','.1','--artifact-root',str(O),'--budget-gib','16'],check=True)
    assert 'Ran 7 tests'in(O/'uncapped_selector_qualification.log').read_text()and(O/'uncapped_selector_qualification.log').read_text().endswith('OK\n')
    metadata={r['series_id']:r for r in api.load_panel(V/'inputs',verify_files=False)['series']};prior=read(V/'candidates/asset_trained_pool/development1000/plan.json');seen=set();cases=[]
    for case in prior['cases']:
        group=metadata[case['series_id']]['asset_class']
        if group not in seen and case['origin']['max_horizon']>=252:cases.append(case);seen.add(group)
    assert len(cases)==7
    tasks=[{'number':i*2+j,'series_id':c['series_id'],'asset_class':metadata[c['series_id']]['asset_class'],'outer_stop':c['origin']['position']+1,'horizon':H}for i,c in enumerate(cases)for j,H in enumerate((21,252))]
    D.mkdir(parents=True);hashes={str(p.relative_to(O)):sha(p)for p in C.glob('*.py')};hashes['smoke_asset_uncapped_selector.py']=sha(Path(__file__));hashes['test_asset_uncapped_selector.py']=sha(O/'test_asset_uncapped_selector.py')
    # Source-bound frozen19option and transitive fitted-method definitions.
    hashes.update(read(V/'candidates/asset_fast_selector/development1000/plan.json')['source_hashes'])
    write(D/'plan.json',{'utc':now(),'tasks':tasks,'history_rows_per_request':96,'scope':'Actual-source request qualification with deliberately short histories, not frozen panel cases or canonical score evidence','workers':12,'source_hashes':hashes,'outer_paths':1000,'inner_paths':1000,'automatic_fallback':False,'origin_cap':None});write(D/'active.json',{'pid':os.getpid(),'status':'running','utc':now(),'workers':12})
    records=[];errors=[]
    with concurrent.futures.ProcessPoolExecutor(max_workers=12,mp_context=multiprocessing.get_context('spawn'))as pool:
        pending={pool.submit(run,t):t for t in tasks}
        for f in concurrent.futures.as_completed(pending):
            try:records.append(f.result())
            except Exception as exc:errors.append({'task':pending[f],'type':type(exc).__name__,'reason':str(exc)})
            write(D/'progress.json',{'utc':now(),'completed':len(records),'expected':len(tasks),'errors':errors});write(D/'records.json',records)
    assert all(sha(O/p)==h for p,h in hashes.items())
    write(D/'summary.json',{'utc':now(),'complete':True,'successful_requests':len(records),'expected_requests':len(tasks),'errors':errors,'asset_classes':sorted(seen),'all_requested_paths':1000,'all_inner_paths':1000,'automatic_parent_fallback':False,'origin_cap':None,'canonical_panel_score':None,'full_panel_catalogue_increment':0});write(D/'active.json',{'pid':os.getpid(),'status':'terminal','utc':now()})
    print(json.dumps(read(D/'summary.json')),flush=True)
if __name__=='__main__':main()
