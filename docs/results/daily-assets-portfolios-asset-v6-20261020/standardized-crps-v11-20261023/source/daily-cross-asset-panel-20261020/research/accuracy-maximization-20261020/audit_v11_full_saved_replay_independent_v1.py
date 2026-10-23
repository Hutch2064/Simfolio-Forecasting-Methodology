"""Independent complete V11/V9 scalar replay. No runner/scorer imports or forecasts."""
import collections,hashlib,json,math,sqlite3
from pathlib import Path
O=Path(__file__).resolve().parent;D=O/'requested_horizon_crps_v11';S=O/'short_horizon_final240_v3';OUT=O/'requested_horizon_crps_v11_independent_full_replay_review.json'
WR='model_testing_asset_joint_blocks_replacement_v6';WOR='model_testing_asset_joint_blocks_history_partition_v6';MODELS=('endpoint',WR,WOR);BUCKETS=('short','medium','long')
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'))
def connect(p):
    c=sqlite3.connect(f'file:{p}?mode=ro&immutable=1',uri=True);assert c.execute('PRAGMA integrity_check').fetchone()[0]=='ok';return c
def signature(p):
    s=Path(p).stat();return(s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns)
def main():
    assert not OUT.exists();rp=D/'full_primary_rescore.json';assert sha(rp)=='1b3bc18d59f1ee72fe9b81b61159af1269751a7bae198edb5bdf69dd374b007a';r=read(rp)
    assert r['matched_request_coverage_complete']and r['fixed_reference']==WR and r['planned_requests_per_model']==320017
    assert r['score_protocol_id']=='daily_portfolio_panel_crps_v11_train_sd_sqrt_requested_h_equal_days_fixed_calendar_buckets'
    for path,h in r['source_hashes'].items():assert sha(path)==h, path
    assert sha(D/'rescore_v9_full_saved_scores.py')=='f0455a724000b6284a14e08629dfa90717d7f0694fe876e57616569fbcd65fb5'
    scale_review=O/'requested_horizon_crps_v11_independent_scale_index_review.json';assert sha(scale_review)=='33e2ad3e88bb57a36968325d6f4c18c8414163fc7fdb06eda752d0930b9e013d'and read(scale_review)['passed']
    oldp=O/'series_buckets_v9_live/independent_full_replay_review.json';old=read(oldp);assert old['status']=='verified'
    snapshot=O/'series_buckets_v9_live/final_scoreboard_snapshot_v9.json';assert sha(snapshot)=='a9cb3d217f9e6e95ab4211f98f5b0c97a4e3aaaf34b2951afb72844a2ecb574e'
    addp=O/'series_buckets_v9_live/final_scoreboard_binding_addendum.json';add=read(addp)
    plan=read(S/'plan.json');assert plan['fingerprint']==old['provenance']['scientific_plan_fingerprint'];assert sha(S/'terminal.json')==old['provenance']['current_terminal_sha256']=='15d6a513a9c5b5a29cd0c3e98feef60ece830accf7ffddceb6d962bf8ffff9a7'
    for path,h in plan['source_hashes'].items():assert sha(path)==h,path
    assert plan['baseline_factories'][WR]['replacement']is True and plan['baseline_factories'][WOR]['replacement']is False
    assert plan['baseline_factories'][WR]['method']==old['protocol']['fixed_WR_model_id']==r['fixed_reference']
    admission=read(S/'root_launch_admission.json');binding=admission['handoff'];assert sha(binding['path'])==binding['sha256']==old['provenance']['handoff_sha256'];handoff=read(binding['path']);cases=plan['cases']
    tasks={};phases={}
    for field,kind in(('saved_old_slots','original'),('inherited_short_groups','short')):
        for text,b in handoff[field].items():
            i=int(text);assert b['paths']==1000;assert b['checkpoint_path']not in tasks
            tasks[b['checkpoint_path']]=(i,kind,1000,b['checkpoint_sha256']);phases[b['checkpoint_path']]=field
    for i in range(len(cases)):
        if str(i)not in handoff['saved_old_slots']:
            p=str(S/'new_long_groups'/f'{i:05d}.npz');assert p not in tasks;tasks[p]=(i,'original',240,None);phases[p]='new_original'
    for i in plan['group_indices']:
        if str(i)not in handoff['inherited_short_groups']:
            p=str(S/'groups'/f'{i:05d}.npz');assert p not in tasks;tasks[p]=(i,'short',240,None);phases[p]='new_short'
    assert len(tasks)==128041
    c=connect(D/'scale_index.sqlite3');scales={row[0]:row[1:]for row in c.execute('SELECT request_id,training_sd,native_h,horizon_bucket,within_bucket_weight,series_id,case_index,origin_index,origin_date,requested_end_date FROM request_scales')};c.close();assert len(scales)==320017
    scale_commit=hashlib.sha256();scale_commit.update(b'[')
    for n,rid in enumerate(sorted(scales)):
        sd,h,b,w,sid,i,oi,origin,end=scales[rid];entry={'request_id':rid,'training_sd':sd,'native_h':h,'series_id':sid,'case_index':i,'origin_index':oi,'origin_date':origin,'requested_end_date':end}
        if n:scale_commit.update(b',')
        scale_commit.update(canonical(entry).encode())
    scale_commit.update(b']');commit=scale_commit.hexdigest();assert all(x['scale_data_commitment_sha256']==commit for x in r['models'].values())
    cache=O/'series_buckets_v9_live/raw_cache.sqlite3';sig=signature(cache);cache_sha=sha(cache);assert cache_sha==r['v9_saved_raw_means']['raw_cache_sha256']==old['provenance']['raw_cache_sha256']=='dfd0f22316f90eeac159e1acc66b83d9954a342afe02cd4ac2c72997f155f2c3'
    raw_terms={(m,b):[]for m in MODELS for b in BUCKETS};std_terms={(m,b):[]for m in MODELS for b in BUCKETS};mass=collections.defaultdict(list);seen=set();seen_tasks=set();path_counts=collections.Counter();phase=collections.Counter();request_counts=collections.Counter();bcount=collections.Counter();digests={m:hashlib.sha256()for m in MODELS};rawcommit=hashlib.sha256()
    c=connect(cache)
    try:
        for path,signature_text,cp_sha,items_text in c.execute('SELECT path,signature,checkpoint_sha256,items FROM raw_pairs ORDER BY path'):
            assert path in tasks and path not in seen_tasks;seen_tasks.add(path);i,kind,paths,oldsha=tasks[path];assert oldsha is None or oldsha==cp_sha
            items=json.loads(items_text);expectedH=[1,5,10,21]if kind=='short'else[cases[i]['origin']['max_horizon']];assert [item['H']for item in items]==expectedH
            assert all(isinstance(item['H'],int)and not isinstance(item['H'],bool)for item in items)
            rawcommit.update((path+'\n'+cp_sha+'\n'+signature_text+'\n'+items_text+'\n').encode());phase[phases[path]]+=1;path_counts['groups_'+str(paths)]+=1
            for item in items:
                H=item['H'];rid=canonical([kind,i,H]);assert rid not in seen and rid in scales;seen.add(rid);sd,h,b,w,sid,idx,oi,origin,end=scales[rid];case=cases[i]
                assert (idx,sid,oi,origin,h)==(i,case['series_id'],case['origin_index'],case['origin']['origin_date'],H);assert sd>0 and math.isfinite(sd);denom=sd*math.sqrt(h);assert math.isfinite(denom)and denom>0
                values=item['values'];assert isinstance(values,list)and len(values)==3;mass[b].append(w);bcount[b]+=1;request_counts[kind]+=1;path_counts['requests_'+str(paths)]+=1
                for m,value in zip(MODELS,values):
                    assert isinstance(value,(int,float))and not isinstance(value,bool)and math.isfinite(value)and value>=0
                    v=float(value);raw_terms[m,b].append(w*v);std_terms[m,b].append(w*(v/denom));digests[m].update((rid+'\t'+v.hex()+'\n').encode())
    finally:c.close()
    assert seen==set(scales)and seen_tasks==set(tasks)and len(seen)==320017;assert sig==signature(cache)and sha(cache)==cache_sha
    assert dict(bcount)==r['models']['endpoint']['request_counts_by_bucket'];assert path_counts['requests_1000']==59906 and path_counts['requests_240']==260111
    assert path_counts['requests_1000']==old['coverage']['final_path_count_requests']['1000']and path_counts['requests_240']==old['coverage']['final_path_count_requests']['240']
    assert phase=={'saved_old_slots':24574,'inherited_short_groups':8833,'new_original':39475,'new_short':55159}
    assert rawcommit.hexdigest()==r['v9_saved_raw_means']['raw_cache_checkpoint_item_digest_sha256']and {m:h.hexdigest()for m,h in digests.items()}==r['v9_saved_raw_means']['raw_mean_request_digest_sha256']
    masses={b:math.fsum(mass[b])for b in BUCKETS};assert all(v==1. for v in masses.values());Qs={};oldQs={};buckets={};errors={}
    for m in MODELS:
        buckets[m]={b:math.fsum(std_terms[m,b])/masses[b]for b in BUCKETS};Qs[m]=math.fsum(buckets[m].values())/3;oldQs[m]=math.fsum(math.fsum(raw_terms[m,b])/masses[b]for b in BUCKETS)/3
        assert Qs[m]==r['models'][m]['Q'];assert buckets[m]==r['models'][m]['bucket_Q'];assert oldQs[m]==old['results']['Q'][m];errors[m]=Qs[m]-r['models'][m]['Q']
    gain=100*(1-Qs['endpoint']/Qs[WR]);oldgain=100*(1-oldQs['endpoint']/oldQs[WR]);assert gain==r['comparisons']['endpoint_vs_fixed_WR']['percent_better_than_baseline'];assert oldgain==25.484505671234402
    comparison={m:{'Q':Qs[m],'bucket_Q':buckets[m],'gain_vs_fixed_WR_percent':100*(1-Qs[m]/Qs[WR])}for m in MODELS}
    out={'passed':True,'schema':'independent_v11_full_saved_replay_v1','review_source_sha256':sha(Path(__file__)),'adapter_static_review':'Frozen revision f0455a724000b6284a14e08629dfa90717d7f0694fe876e57616569fbcd65fb5. Exact handoff paths+old hashes and explicit current groups/new_long_groups resolve task identities; full unique planned coverage and three-value model order checked. No fallback, outcome weights or forecasts. This audit independently enumerates ALL allowed task paths and replays both score definitions without importing adapter/scorer.','full_rescore_sha256':sha(rp),'full_rescore_source_bindings_verified':len(r['source_hashes']),'scientific_source_bindings_verified':len(plan['source_hashes']),'prior_science_review_sha256':sha(oldp),'immutable_V9_snapshot_sha256':sha(snapshot),'immutable_V9_binding_addendum_sha256':sha(addp),'scale_review_sha256':sha(scale_review),'raw_cache_sha256':cache_sha,'scale_index_sha256':sha(D/'scale_index.sqlite3'),'scale_commitment_sha256':commit,'checkpoint_item_digest':rawcommit.hexdigest(),'matched_requests_per_model':len(seen),'checkpoint_groups':len(seen_tasks),'requests_by_kind':dict(request_counts),'requests_by_bucket':dict(bcount),'groups_and_requests_by_final_paths':dict(path_counts),'groups_by_generation':dict(phase),'fixed_baseline':WR,'fixed_baseline_preserved_replacement_true':True,'one_outer_third':True,'within_bucket_weight_masses':masses,'V11_replay':comparison,'V11_Q_replay_absolute_differences':errors,'V9_Q_exact_replay':oldQs,'V9_fixed_WR_gain_exact_replay':oldgain,'missing_duplicate_scale_or_date_errors':0,'financial_fits_QPs_or_forecasts':0,'canonical_or_public_edits':False,'scope':'Actual full320017 saved-forecast development-panel rescore, mixed1000/240 matched within each triple. Adaptive/posthoc scoring rule; not untouched holdout or production-model promotion. Previously verified source-bound raw-cache recovery chain retained; no bulk restore or new physical-path validation asserted.'}
    OUT.write_text(json.dumps(out,indent=2,sort_keys=True,allow_nan=False)+'\n');print(json.dumps({'passed':True,'review':str(OUT),'sha256':sha(OUT),'Q':Qs,'gain':gain,'V9_gain':oldgain}))
if __name__=='__main__':main()
