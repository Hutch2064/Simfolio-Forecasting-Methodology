"""Prepare an unstarted19-law screen, waiting for the active full evaluation/promotion."""
from pathlib import Path
O=Path(__file__).resolve().parent;C=O/'asset_level_v6/candidates/asset_dynamic_correlation'
x=(O/'run_asset_feature_batch.py').read_text()
x=x.replace("C=V/'candidates/asset_feature_batch'","C=V/'candidates/asset_dynamic_correlation'")
x=x.replace('from candidates.asset_feature_batch import fits','from candidates.asset_feature_batch import fits\nfrom candidates.asset_dynamic_correlation import correlation\nFAMILIES=(\'GARCH\',\'SV_path\',\'SV_RMS\')')
x=x.replace('from candidates.asset_feature_batch.candidate import FeaturePool,CONFIGS,identity','from candidates.asset_dynamic_correlation.candidate import CorrelationPool as FeaturePool,CONFIGS,identity')
x=x.replace("ANCHOR={'family':'GARCH','innovations':'empirical','raw_fade':True,'mean_shrinkage':True,'mean_uncertainty':True}","ANCHOR=None")
x=x.replace("('raw','mean','filter')","('raw','mean','filter','correlation')").replace('for family in fits.FAMILIES','for family in FAMILIES').replace('len(fits.FAMILIES)','len(FAMILIES)')
a=x.index('def fit_case(');b=x.index('def forecast_family(',a)
x=x[:a]+'''def fit_case(number,case):
    worker_state('fit',case=number);row=ROWS[case['series_id']];history,_,_=AssetInputs(V/'inputs').training(row,case['origin']['position']+1)
    begin=time.perf_counter();base=fits.base_fit(history);base_seconds=time.perf_counter()-begin;packets=[];errors=[];sv=None;sv_dependence=None
    for family in FAMILIES:
        try:
            start=time.perf_counter()
            if family=='SV_RMS'and sv is not None:bundle={**sv,'family':family};dep=sv_dependence
            else:
                bundle=fits.prepare(history,family,base);dep=correlation.prepare(bundle['filter']['pool'])
                if family=='SV_path':sv=bundle;sv_dependence=dep
            seconds=time.perf_counter()-start;bundle['correlation']={k:v for k,v in dep.items()if k!='fit'}
            arrays={f'{section}.{name}':value for section in ('raw','mean','filter','correlation')for name,value in bundle[section].items()}
            arrays.update(family=family,history_sha256=bundle['history_sha256']);path=D/'fits'/f'{number:03d}-{family}.npz';np.savez_compressed(path,**arrays)
            packets.append({'case':number,'family':family,'sha256':sha(path),'base_fit_compute_seconds':base_seconds,'filter_and_dependence_fit_compute_seconds':seconds,'dependence_fit':dep.get('fit',{'scalar_identity':True})})
        except Exception as e:errors.append({'case':number,'family':family,'type':type(e).__name__,'message':str(e),'stage':'training_fit'})
    worker_state('idle');return {'packets':packets,'errors':errors}

''' +x[b:]
x=x.replace("    slots=(O/'asset_v6_forecast_slots.lock').open('a');fcntl.flock(slots,fcntl.LOCK_EX)","""    global ANCHOR
    # The active full controller releases forecast slots before audit/promotion. Wait for the
    # owning handoff to finish so this screen's current-best control cannot change mid-admission.
    while True:
        handoff=read(O/'asset_feature_handoff_active.json')
        if handoff['status'].startswith('terminal_'):break
        process=subprocess.run(['ps','-o','command=','-p',str(handoff['pid'])],capture_output=True,text=True)
        assert 'continue_asset_feature_batch.py'in process.stdout,'Current full handoff is missing; inspect, never restart'
        time.sleep(5)
    slots=(O/'asset_v6_forecast_slots.lock').open('a');fcntl.flock(slots,fcntl.LOCK_EX)""")
a=x.index("    source=V/'candidates/asset_mean_uncertainty/fading_vol_screen100'");b=x.index('    hashes=',a)
x=x[:a]+'''    source=V/'candidates/asset_feature_batch/screen100';old=read(source/'summary.json');previous_plan=read(source/'plan.json');assert old['complete_execution'] and not old['errors'] and old['paths']==100
    assert all(sha(O/p)==h for p,h in previous_plan['source_hashes'].items())
    official=read(V/'official_scoreboard.json');mid=official['current_best_model_id']
    base_config=dict(family='GARCH',innovations='empirical',raw_fade=True,mean_shrinkage=True,mean_uncertainty=True)if mid=='model_testing_asset_circular_raw_GARCH_fade_FHS_mean_v1'else read(V/'candidates/asset_feature_batch/full_selection.json')['config']
    if mid!='model_testing_asset_circular_raw_GARCH_fade_FHS_mean_v1':assert read(V/'candidates/asset_feature_batch/full_selection.json')['model_id']==mid
    ANCHOR={**base_config,'correlation':'sample'};assert ANCHOR in CONFIGS
    # Workers use the immutable references file, not changing official state.
    panel=api.load_panel(V/'inputs',verify_files=False);metadata={r['series_id']:r for r in panel['series']};cases=previous_plan['cases'];assert len(cases)==72
    old_config_index=previous_plan['configs'].index(base_config);old_records={(r['case'],r['config_index']):r for r in old['records']};old_references=read(source/'references.json')
    references={};ref_records={'current_best':[],'fixed_reference':[]}
    for number,case in enumerate(cases):
        record=old_records[number,old_config_index];path=source/'losses'/f'{old_config_index:03d}-{number:03d}.npz';assert sha(path)==record['checkpoint_sha256']
        with np.load(path,allow_pickle=False)as z:assert hashlib.sha256(z['losses'].tobytes()).hexdigest()==record['loss_sha256']
        fixed=old_references['cases'][str(number)]['fixed_reference'];assert sha(Path(fixed['checkpoint']))==fixed['checkpoint_sha256']
        references[str(number)]={'current_best':{**record,'checkpoint':str(path)},'fixed_reference':fixed}
        ref_records['current_best'].append(record);ref_records['fixed_reference'].append(fixed)
''' +x[b:]
x=x.replace("write(D/'references.json',{'cases':references})","write(D/'references.json',{'cases':references,'anchor_config':ANCHOR,'official_current_best_model_id':mid})")
x=x.replace("    ROWS={r['series_id']:r for r in api.load_panel(V/'inputs',verify_files=False)['series']}","    global ANCHOR\n    ROWS={r['series_id']:r for r in api.load_panel(V/'inputs',verify_files=False)['series']}\n    ANCHOR=read(D/'references.json')['anchor_config']")
x=x.replace("'--expected-gib','.8'","'--expected-gib','.2'").replace("'--budget-gib','12'","'--budget-gib','16'")
x=x.replace("remaining_units={family:24*sum(case_work) for family in FAMILIES}","remaining_units={family:sum(c['family']==family for c in CONFIGS)*sum(case_work) for family in FAMILIES}")
x=x.replace("'case_selection':'Training-position median per existing family, with the minimum predeclared asset-quarter support needed by the unchanged canonical reducer. No loss values enter selection.'","'case_selection':'The same72cases/70families frozen before the original216-law screen; no new case selection based on observed losses.'")
x=x.replace("'beats_same100path_current_best':complete and q<reference_Q['current_best']","'beats_same100path_current_best':complete and q<reference_Q['current_best'],'is_existing_static_sample_control':config['correlation']=='sample','eligible_as_new_dependence_hypothesis':config['correlation']!='sample'")
# Bind the separately sealed transitive fit modules as well as every direct new prototype source.
x=x.replace("    D.mkdir(parents=True,exist_ok=True)","    supplement=read(V/'candidates/asset_feature_batch/transitive_source_receipt.json')\n    for name,digest in supplement['source_files_relative_to_v6'].items():\n        assert sha(V/name)==digest;hashes['asset_level_v6/'+name]=digest\n    D.mkdir(parents=True,exist_ok=True)")
(O/'screen_asset_dynamic_correlation.py').write_text(x)
print('Prepared19-model100-path screen; admission waits for active full evaluation and final current-best identity')
