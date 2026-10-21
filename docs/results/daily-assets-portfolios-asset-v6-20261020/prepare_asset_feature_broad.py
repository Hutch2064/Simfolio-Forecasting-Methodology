from pathlib import Path
import json,hashlib,datetime
O=Path(__file__).resolve().parent;C=O/'asset_level_v6/candidates/asset_feature_batch';read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
s=read(C/'reserved1000/summary.json');assert s['complete_execution'] and not s['errors'] and s['paths']==1000 and s['anchor_loss_vectors_bitwise_equal_each_case']
anchor=dict(family='GARCH',innovations='empirical',raw_fade=True,mean_shrinkage=True,mean_uncertainty=True)
selected=[x for x in s['complete_identical_coverage_screen_ranking'] if x['main_Q']<s['reference_main_Q']['current_best'] or x['config']==anchor];assert len(selected)==4
P=C/'broad_selection.json';assert not P.exists();P.write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reserved_summary_sha256':sha(C/'reserved1000/summary.json'),'selection':'All three stochastic-volatility configurations improving both development and reserved overall scores, plus exact incumbent. Frozen before broad forecasts.','selected':selected,'paths':1000,'case_count':7418,'official':False,'independent_validation_claim':False},indent=2)+'\n')
x=(O/'run_asset_feature_reserved.py').read_text().replace("D=C/'reserved1000'","D=C/'broad1000'").replace("C/'reserved_selection.json'","C/'broad_selection.json'")
x=x.replace("['phases']['reserved'];assert len(cases)==355","['phases']['broad'];assert len(cases)==7418")
x=x.replace("'--expected-gib','.4'","'--expected-gib','1.2'").replace("'--budget-gib','12'","'--budget-gib','14'")
x=x.replace('All355 previously frozen reserved cases. Earlier research adaptively reused these cases; not independent validation.','All7418 previously frozen broad cases. Earlier research adaptively reused these cases; not independent validation.')
a=x.index('def fit_case(');b=x.index('def forecast_family(',a)
x=x[:a]+'''def fit_case(number,case):
    # Per-case ephemeral fit reuse avoids a persistent7418-history fit cache.
    worker_state('fit',case=number);row=ROWS[case['series_id']];history,_,_=AssetInputs(V/'inputs').training(row,case['origin']['position']+1)
    begin=time.perf_counter();base=fits.base_fit(history);base_seconds=time.perf_counter()-begin;packets=[];records=[];errors=[];sv_bundle=None
    for family in FAMILIES:
        try:
            worker_state('fit',case=number,family=family);start=time.perf_counter()
            if family in ('SV_path','SV_RMS') and sv_bundle is not None:bundle={**sv_bundle,'family':family}
            else:
                bundle=fits.prepare(history,family,base)
                if family in ('SV_path','SV_RMS'):sv_bundle=bundle
            seconds=time.perf_counter()-start
            digest=hashlib.sha256()
            for section in ('raw','mean','filter'):
                for name,value in sorted(bundle[section].items()):
                    v=np.asarray(value);digest.update((section+'.'+name+':'+v.dtype.str+':'+str(v.shape)).encode());digest.update(v.tobytes())
            packet={'case':number,'family':family,'fit_values_sha256':digest.hexdigest(),'base_fit_compute_seconds':base_seconds,'filter_fit_compute_seconds':seconds,'ephemeral_fit_reuse':True}
            packets.append(packet);result=forecast_family(number,case,packet,bundle);records.extend(result['records']);errors.extend(result['errors'])
        except Exception as e:errors.append({'case':number,'family':family,'type':type(e).__name__,'message':str(e),'stage':'training_fit'})
    worker_state('idle');return {'records':records,'packets':packets,'errors':errors}

''' +x[b:]
x=x.replace('def forecast_family(number,case,packet):','def forecast_family(number,case,packet,supplied_bundle=None):')
x=x.replace("bundle=load_bundle(number,family,packet['sha256']);","bundle=supplied_bundle if supplied_bundle is not None else load_bundle(number,family,packet['sha256']);")
a=x.index('        unprepared=iter(');b=x.index('            if time.monotonic()',a)
x=x[:a]+'''        pending={pool.submit(fit_case,i,c):('case',i) for i,c in enumerate(cases)}
        while pending:
            done,_=concurrent.futures.wait(pending,timeout=2,return_when=concurrent.futures.FIRST_COMPLETED)
            for future in done:
                kind,number=pending.pop(future)
                try:
                    result=future.result();errors.extend(result['errors']);records.extend(result['records']);packets.extend(result['packets'])
                except Exception as e:errors.append({'case':number,'stage':kind,'type':type(e).__name__,'message':str(e)})
''' +x[b:]
x=x.replace('Shared-fit five-configuration1000-path reserved comparison','Shared-fit four-configuration1000-path broad comparison')
(O/'run_asset_feature_broad.py').write_text(x)
(O/'asset_feature_storage_budget.json').write_text(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'owner':str(O),'budget_gib':14,'expected_broad_growth_gib':1.2,'archive_staging_peak_gib':.4,'reason':'Explicitly authorized216-model breadth expansion. Broad4x7418 loss vectors plus compact metadata; all fitting bundles remain ephemeral per worker. Prior12GiB declaration did not cover the new concurrent family study.','free_space_reserve_gib':100,'release_condition':'Preserve complete feature evidence remotely with fresh recovery; retire only verified closed exploratory caches/outputs. Keep current-best and baseline dependencies local.'},indent=2)+'\n')
print('selected',[(r['config']['family'],r['config']['innovations'],r['main_Q'])for r in selected])
