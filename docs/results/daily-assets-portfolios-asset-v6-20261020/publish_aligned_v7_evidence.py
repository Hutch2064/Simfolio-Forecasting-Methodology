"""Prepare a reviewed inventory; publish/recover only on explicit operator command.

Default invocation never edits the research repository, commits, pushes or runs
forecasts. --publish requires the exact reviewed inventory digest.
"""
from pathlib import Path
import argparse, datetime, fcntl, hashlib, json, os, shutil, subprocess, sys, urllib.request

O=Path(__file__).resolve().parent; V=O/'asset_level_v6'; R=O/'repository'
TMP=O.parents[2]; REL='docs/results/daily-assets-portfolios-asset-v6-20261020'
C=V/'candidates/asset_selector_aligned_production'
INVENTORY=O/'aligned_v7_public_inventory.json'
ENV={**os.environ,'GODEBUG':'http2client=0'}
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'): ENV[k]='1'
read=lambda p:json.loads(Path(p).read_text())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
digest=lambda x:hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':')).encode()).hexdigest()
def write(p,x): Path(p).write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
def git(*args): return subprocess.check_output(['git',*args],cwd=R,env=ENV,text=True).strip()
def prepare():
    sys.path.insert(0,str(O)); import run_aligned_panel_v7 as runner
    attestation=read(C/'source_attestation_compiled_v2.json')
    execution=runner.closure()
    for group in (attestation['source_hashes'],execution):
        for p,h in group.items():
            if sha(p)!=h:raise ValueError('Frozen source changed: '+p)
    pinned={**attestation['source_hashes'],**execution}
    if len(attestation['source_hashes'])!=203: raise ValueError('Expected frozen203 reference closure')
    for p,h in pinned.items():
        if sha(p)!=h: raise ValueError('Frozen source changed: '+p)
    names=set(map(Path,pinned))
    names.update(O/n for n in (
        'publish_aligned_v7_evidence.py','ALIGNED_V7_METHOD.md',
        'aligned_compiled_v2_source_review.json','aligned_production_source_review.json',
        'aligned_production_acceptance_contract.json','score_aligned_v7_contract_review.json',
        'aligned_panel_v7_source_review.json','aligned_panel_v7_factory_binding_addendum.json',
        'aligned_panel_v7_fixed_reference_reporting_addendum.json',
        'final_scoring_protocol_v7.json','full_requested_horizon_coverage.json',
        'historical_horizon_methodology_validation.json','aligned_equal_day_horizon_sources.json',
        'test_run_aligned_panel_v7.py','profile_aligned_production.py','profile_aligned_compiled.py',
        'qualify_aligned_production.py','qualify_aligned_production36.py','qualify_aligned_compiled48.py',
        'adaptive_infrastructure_literature_review_20261021_v1.json',
        'adaptive_infrastructure_literature_review_20261021_v1.md'))
    names.update(V/n for n in ('test_score_aligned_v7.py',
        'full_pair_rescore_v7_diagnostic.json','full_pair_rescore_v7_independent_verification.json',
        'rescore_full_pair_v7_launchd_owner_receipt.json','rescore_full_pair_v7_first_task_smoke.json',
        'rescore_full_pair_v7_preflight.json','rescore_full_pair_v7.py',
        'smoke_rescore_full_pair_v7_first_task.py','verify_full_pair_rescore_v7_index.py',
        'test_rescore_full_pair_v7.py'))
    for sub in ('asset_selector_aligned_production','asset_selector_score_alignment','asset_selector_sequential_learning'):
        folder=V/'candidates'/sub
        names.update(p for p in folder.iterdir() if p.is_file() and p.suffix in ('.py','.md'))
    names.update(C/n for n in ('source_attestation_v1.json','source_attestation_compiled_v2.json',
        'verification_receipt.json','qualification36/verification_receipt.json',
        'qualification48_compiled/verification_receipt.json','timing_comparison_v2.json'))
    names.update((C/'profiles').glob('*.json'))
    acceptance=read(C/'acceptance_receipt_v2.json')
    # Public projection: retain exact private-receipt identity, omit provider-file
    # inventory and loss-file paths. The original private receipt is never edited.
    acceptance={k:v for k,v in acceptance.items() if k not in ('input_file_hashes','raw_evidence')}
    acceptance['original_private_receipt_sha256']=sha(C/'acceptance_receipt_v2.json')
    acceptance['projection_omissions']=['input_file_hashes','raw_evidence']
    projected=O/'aligned_v7_acceptance_public.json';write(projected,acceptance);names.add(projected)
    environment=O/'aligned_v7_environment.json';write(environment,runner.environment());names.add(environment)
    code=runner.common.CODE
    names.update(code/n for n in ('pyproject.toml','requirements-lock.txt','README_REPRODUCIBILITY.md'))
    files=[];destinations=set()
    for p in sorted(names):
        p=p.resolve()
        if not p.is_file() or p.suffix not in ('.py','.md','.json','.toml','.txt'):raise ValueError('Unapproved payload '+str(p))
        if p.is_relative_to(O):public=str(p.relative_to(O)).removeprefix('asset_level_v6/')
        else:public='_execution_closure/aligned_v7/'+str(p.relative_to(TMP))
        if public in destinations:raise ValueError('Publication collision '+public)
        destinations.add(public)
        files.append({'source':str(p),'public_path':public,'restore_path':str(p.relative_to(TMP)),
                      'sha256':sha(p),'bytes':p.stat().st_size,'mode':p.stat().st_mode&0o777})
    obj={'schema':'aligned-v7-reviewed-public-inventory','repository':'Hutch2064/Simfolio-Forecasting-Methodology',
         'destination':REL,'prepared_against_commit':git('rev-parse','HEAD'),
         'reference_closure_count':203,'execution_closure_count':len(runner.closure()),
         'files':files,'total_bytes':sum(f['bytes']for f in files),'environment':runner.environment(),
         'excluded':['provider inputs','NPZ loss arrays','full path arrays','pstats','active rescore outputs'],
         'status':'prepared only; full-panel admission and publication remain separate'}
    obj['fingerprint']=digest(obj);write(INVENTORY,obj)
    print(json.dumps({k:obj[k] for k in ('fingerprint','reference_closure_count','execution_closure_count','total_bytes')},indent=2))
    return obj

def verify_inventory():
    obj=read(INVENTORY)
    if digest({k:v for k,v in obj.items() if k!='fingerprint'})!=obj['fingerprint']:raise ValueError('Inventory digest changed')
    for f in obj['files']:
        if sha(f['source'])!=f['sha256']:raise ValueError('Inventory source changed '+f['source'])
    return obj

def preflight():
    subprocess.run([sys.executable,str(TMP.parent/'storage_preflight.py'),'--expected-gib','0.03',
                    '--artifact-root',str(O),'--budget-gib','16'],env=ENV,check=True)

def publish(reviewed):
    obj=verify_inventory();preflight()
    if reviewed!=obj['fingerprint']:raise ValueError('Explicit reviewed inventory fingerprint required')
    with (O/'asset_v6_publication.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if git('status','--porcelain'):raise ValueError('Research repository must be clean')
        git('fetch','origin','main');git('merge','--ff-only','origin/main')
        if git('rev-parse','HEAD')!=obj['prepared_against_commit']:raise ValueError('Repository advanced; re-review inventory')
        destination=R/REL;scoped=[]
        for f in obj['files']:
            target=destination/f['public_path'];target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(f['source'],target)
            if sha(target)!=f['sha256']:raise ValueError('Copy mismatch')
            scoped.append(str(target.relative_to(R)))
        manifest={str(p.relative_to(destination)):sha(p) for p in sorted(destination.rglob('*'))
                  if p.is_file() and p.name!='PUBLIC_FILES.json' and '__pycache__'not in p.parts}
        write(destination/'PUBLIC_FILES.json',manifest);scoped.append(REL+'/PUBLIC_FILES.json')
        # Preserve all old files; stage only this reviewed inventory and manifest.
        git('add','--',*scoped)
        changed=git('diff','--cached','--name-only').splitlines()
        if not changed or any(p not in scoped for p in changed):raise ValueError('Unexpected staged scope')
        git('commit','-m','Preserve exact aligned-score specialization, qualification and v7 protocol')
        git('push','origin','HEAD:main');commit=git('rev-parse','HEAD')
        recover(commit)

def recover(commit):
    obj=verify_inventory();preflight()
    if git('ls-remote','origin','refs/heads/main').split()[0]!=commit:raise ValueError('Remote main differs')
    # Distinct tree preserves common.py path resolution, with no live input copies.
    base=O/'aligned_v7_public_recovery'/commit[:12]
    if base.exists():raise ValueError('Fresh recovery directory required')
    restored=[]
    for f in obj['files']:
        url=f'https://raw.githubusercontent.com/{obj["repository"]}/{commit}/{REL}/{f["public_path"]}'
        with urllib.request.urlopen(url,timeout=60)as response:
            if response.status!=200:raise ValueError('Fresh remote HTTP200 required')
            data=response.read()
        if hashlib.sha256(data).hexdigest()!=f['sha256']:raise ValueError('Remote checksum mismatch')
        p=base/f['restore_path'];p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes(data);p.chmod(f['mode'])
        if sha(p)!=f['sha256'] or p.stat().st_mode&0o777!=f['mode']:raise ValueError('Disk restore mismatch')
        restored.append({'public_path':f['public_path'],'sha256':f['sha256'],'HTTP_status':200})
    restored_o=base/O.relative_to(TMP);restored_v=restored_o/'asset_level_v6'
    testenv={**ENV,'PYTHONPATH':str(restored_v)+os.pathsep+str(restored_o),'PYTHONPYCACHEPREFIX':str(base/'pycache')}
    suites=[('scorer',['test_score_aligned_v7'],4),
            ('specializations',['candidates.asset_selector_aligned_production.test_candidate',
                                'candidates.asset_selector_aligned_production.test_compiled_candidate'],7),
            ('runner',['test_run_aligned_panel_v7'],11),
            ('baseline_rescore',['test_rescore_full_pair_v7'],2)]
    tests=[]
    for name,modules,count in suites:
        log=base/(name+'.log')
        with log.open('w')as output:subprocess.run([sys.executable,'-m','unittest',*modules],cwd=restored_o,env=testenv,stdout=output,stderr=subprocess.STDOUT,check=True)
        text=log.read_text()
        if f'Ran {count} tests'not in text or not text.rstrip().endswith('OK'):raise ValueError('Unexpected test result')
        tests.append({'suite':name,'passed':count,'log_sha256':sha(log)})
    receipt={'schema':'aligned-v7-fresh-public-recovery','commit':commit,'inventory_fingerprint':obj['fingerprint'],
        'remote_main_verified':True,'HTTP200_SHA256_disk_mode_verified_files':restored,'tests':tests,
        'fresh_restore':str(base),'forecasts_run':'synthetic unit fixtures only','private_inputs_downloaded':False,
        'production_changed':False,'full_panel_launch_authorized_by_receipt':False,
        'utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
    write(O/'aligned_v7_public_recovery.json',receipt);print(json.dumps({'commit':commit,'restored_files':len(restored),'tests':tests},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--publish',metavar='REVIEWED_FINGERPRINT');parser.add_argument('--recover',metavar='COMMIT');parser.add_argument('--verify',action='store_true');args=parser.parse_args()
    if args.publish:publish(args.publish)
    elif args.recover:recover(args.recover)
    elif args.verify:print(verify_inventory()['fingerprint'])
    else:prepare()
