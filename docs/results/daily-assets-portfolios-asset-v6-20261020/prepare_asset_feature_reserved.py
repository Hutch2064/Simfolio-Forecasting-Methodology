from pathlib import Path
import json,hashlib,datetime
O=Path(__file__).resolve().parent;C=O/'asset_level_v6/candidates/asset_feature_batch';read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
s=read(C/'confirmation1000/summary.json');assert s['complete_execution'] and not s['errors'] and s['paths']==1000 and s['anchor_loss_vectors_bitwise_equal_each_case']
anchor=dict(family='GARCH',innovations='empirical',raw_fade=True,mean_shrinkage=True,mean_uncertainty=True)
selected=[x for x in s['complete_identical_coverage_screen_ranking'] if x['main_Q']<s['reference_main_Q']['current_best'] or x['config']==anchor];assert len(selected)==5
P=C/'reserved_selection.json';assert not P.exists();P.write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'confirmation_summary_sha256':sha(C/'confirmation1000/summary.json'),'selection':'All four configurations improving the unchanged incumbent on355matched1000-path development cases, plus exact incumbent. Frozen before reserved forecasts.','selected':selected,'paths':1000,'case_count':355,'official':False,'independent_validation_claim':False,'legacy_confirmation_field_note':'The confirmation file inherited same100path in two field names; actual plan, context seeds, loss generation and summary paths are1000. Values are1000-path only.'},indent=2)+'\n')
x=(O/'run_asset_feature_confirmation.py').read_text().replace("D=C/'confirmation1000'","D=C/'reserved1000'").replace("C/'confirmation_selection.json'","C/'reserved_selection.json'")
x=x.replace("CONFIGS=[x['config'] for x in json.loads((C/'reserved_selection.json').read_text())['selected']]","CONFIGS=[x['config'] for x in json.loads((C/'reserved_selection.json').read_text())['selected']]\nFAMILIES=tuple(f for f in fits.FAMILIES if any(c['family']==f for c in CONFIGS))")
x=x.replace('for family in fits.FAMILIES','for family in FAMILIES').replace('len(fits.FAMILIES)','len(FAMILIES)')
x=x.replace("cases=previous_plan['cases'];assert len(cases)==355","cases=read(V/'candidates/asset_fading_vol/plan.json')['phases']['reserved'];assert len(cases)==355")
x=x.replace("'--expected-gib','1.2'","'--expected-gib','.4'")
x=x.replace('All355 previously frozen development cases; original72-case screen is an adaptively reused subset, not independent validation.','All355 previously frozen reserved cases. Earlier research adaptively reused these cases; not independent validation.')
x=x.replace('same100path','same1000path').replace('Shared-fit12-configuration1000-path confirmation','Shared-fit five-configuration1000-path reserved comparison')
x=x.replace("pending={pool.submit(fit_case,i,c):('fit',i) for i,c in enumerate(cases)}","unprepared=iter(enumerate(cases))\n        pending={}\n        for _ in range(min(12,len(cases))):\n            i,c=next(unprepared);pending[pool.submit(fit_case,i,c)]=('fit',i)")
x=x.replace("for packet in result['packets']:pending[pool.submit(forecast_family,number,cases[number],packet)]=('forecast',number)","for packet in result['packets']:pending[pool.submit(forecast_family,number,cases[number],packet)]=('forecast',number)\n                        item=next(unprepared,None)\n                        if item is not None:\n                            i,c=item;pending[pool.submit(fit_case,i,c)]=('fit',i)")
(O/'run_asset_feature_reserved.py').write_text(x)
print('selected',[(r['config']['family'],r['config']['innovations'],r['main_Q'])for r in selected])
