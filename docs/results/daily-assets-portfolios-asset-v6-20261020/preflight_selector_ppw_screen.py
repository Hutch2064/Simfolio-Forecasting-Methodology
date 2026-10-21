"""Check queued-screen source identity and every cached matched100path control."""
import datetime,hashlib,json
from pathlib import Path
import numpy as np
import screen_selector_ppw_menu as run
O=Path(__file__).resolve().parent;read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
prior=read(run.PRE/'summary.json');plan=read(run.PRE/'plan.json');audit=read(run.PRE/'verification_receipt.json');q=read(run.C/'qualification_receipt.json')
assert audit['verified']and audit['summary_sha256']==sha(run.PRE/'summary.json')
assert q['tests_passed']==9 and all(sha(O/p)==h for p,h in q['source_hashes'].items())and all(sha(O/p)==h for p,h in plan['source_hashes'].items())
fixed=read(Path(plan['reference_checkpoints']))['cases'];old={(r['case'],r['config_index']):r for r in prior['records']};verified=0
for number,case in enumerate(plan['cases']):
    for index in (0,1):
        r=old[number,index];p=run.PRE/'losses'/f'{index:03d}-{number:03d}.npz';assert r['paths']==100 and sha(p)==r['checkpoint_sha256']and r['model_id']==run.identity(run.SPECS[index])
        with np.load(p,allow_pickle=False)as z:assert len(z['losses'])==case['origin']['max_horizon']and hashlib.sha256(z['losses'].tobytes()).hexdigest()==r['loss_sha256']
        verified+=1
    for label in ('fixed_reference','current_best'):
        r=fixed[str(number)][label];assert r['paths']==100 and sha(Path(r['checkpoint']))==r['checkpoint_sha256'];verified+=1
assert len(set(run.identity(spec)for spec in run.SPECS))==4
receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'verified':True,'same_frozen_cases':72,'forecast_specifications':run.SPECS,'source_hashes_and_control_manifest_verified':True,'expected_control_and_reference_checkpoints_verified':verified,'paths':100,'inner_paths':100,'harness_sha256':sha(O/'screen_selector_ppw_menu.py'),'preflight_source_sha256':sha(Path(__file__)),'no_new_forecasts_generated':True,'admission_pending':'Current1000path72case stage must complete and pass its independent audit before this screen can acquire the12worker pool'}
run.write(run.C/'preflight_receipt.json',receipt);print(json.dumps(receipt,indent=2),flush=True)
