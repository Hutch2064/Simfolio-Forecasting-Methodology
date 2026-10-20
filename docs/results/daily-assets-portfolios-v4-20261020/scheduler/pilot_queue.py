"""Use the established 12-slot native scheduler for bounded research pilot tasks."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS','NUMBA_NUM_THREADS'):
    os.environ[k]='1'
from pathlib import Path
import argparse,datetime,hashlib,json,sys,threading,time
ROOT=next(p for p in Path(__file__).resolve().parents if p.name=='daily-cross-asset-panel-20261020')
CODE=ROOT.parent/'french551-full-20261019/code'
for p in (ROOT,CODE/'src',CODE/'scripts'):sys.path.insert(0,str(p))
import sweep as api
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()


def run(spec_path):
    spec_path=Path(spec_path);spec=read(spec_path)
    assert spec['scope'] in ['development_pilot','reserved_pilot','representative_coverage','scheduler_parity']
    assert spec['runner_sha256']==sha(Path(__file__))
    for name,digest in spec['immutable_files'].items():assert sha(Path(name))==digest
    sys.path.insert(0,spec['factory_directory'])
    control=api.read_control(Path(spec['control_path']))
    assert control['workers']==12 and not control['stop']
    enabled={mid:value for mid,value in control['models'].items() if value['enabled']}
    assert enabled
    panel=api.load_panel(spec['input_root']);assert panel['panel_fingerprint']==spec['panel_fingerprint']
    tasks=read(spec['selected_tasks_path'])
    assert len(tasks)==spec['expected_tasks_per_model'] and tasks
    assert len({(t['series_id'],t['origin_label']) for t in tasks})==len(tasks)
    for mid,value in enabled.items():api.admitted_contract(mid,value,spec['registry_path'])
    output=Path(spec['output_root']);output.mkdir(parents=True,exist_ok=True)
    api.original._atomic_json(output/'pilot_active.json',{'pid':os.getpid(),'utc':now(),'status':'running',
        'scope':spec['scope'],'workers_requested':12,'expected_tasks_per_model':len(tasks),
        'full_panel_admission':False,'spec_sha256':sha(spec_path)})
    stopped=threading.Event();observed={'peak_active_workers':0,'worker_pids':set()}
    def monitor():
        while not stopped.wait(.05):
            try:
                state=read(output/'status.json')
            except (OSError,ValueError):continue
            observed['peak_active_workers']=max(observed['peak_active_workers'],state.get('active_workers',0))
            observed['worker_pids'].update(r['pid'] for r in state.get('worker_assignments',[]) if r.get('pid'))
    watcher=threading.Thread(target=monitor,daemon=True);watcher.start()
    try:
        api.run_controller(Path(spec['input_root']),output,Path(spec['control_path']),
            registry_path=Path(spec['registry_path']),poll_seconds=.05,selected_tasks=tasks)
    finally:
        stopped.set();watcher.join()
    for name,digest in spec['immutable_files'].items():assert sha(Path(name))==digest
    state=read(output/'status.json')
    assert state['completed_active_tasks']+state['failed_active_tasks']==len(tasks)*len(enabled)
    assert not state['active_workers'] and not state['ready_queue']
    receipt={'utc':now(),'scope':spec['scope'],'status':'terminal','spec_sha256':sha(spec_path),
        'workers_requested':12,'peak_active_workers_observed':observed['peak_active_workers'],
        'unique_worker_pids':sorted(observed['worker_pids']),
        'expected_tasks':len(tasks)*len(enabled),'successful_tasks':state['completed_active_tasks'],
        'native_failures':state['failed_active_tasks'],'unchanged_native_forecast_and_seed_contract':True,
        'full_panel_admission':False,'status_sha256':sha(output/'status.json')}
    api.original._atomic_json(output/'pilot_terminal_receipt.json',receipt)
    api.original._atomic_json(output/'pilot_active.json',{'pid':os.getpid(),'utc':now(),'status':'terminal'})
    print(json.dumps(receipt),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('spec');args=parser.parse_args();run(args.spec)
