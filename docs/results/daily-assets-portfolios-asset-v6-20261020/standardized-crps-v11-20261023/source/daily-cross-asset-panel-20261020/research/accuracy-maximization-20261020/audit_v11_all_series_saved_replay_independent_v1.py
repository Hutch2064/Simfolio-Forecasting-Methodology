"""Read-only matched240 all-series V11 replay, bound to the prior full-loss audit."""
import collections,hashlib,json,math,sqlite3
from pathlib import Path
O=Path(__file__).resolve().parent;V=O/'requested_horizon_crps_v11';D=O/'finite_full_series_confirmation_v1';OUT=O/'requested_horizon_crps_v11_independent_all_series_replay_review.json'
METHODS=('finite41','finite77','mapped_ratio','WR');WR='model_testing_asset_joint_blocks_replacement_v6';BUCKETS=('short','medium','long')
def read(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def canonical(x):return json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False)
def main():
    assert not OUT.exists();rp=V/'all_series_confirmation_v11.json';report=read(rp);reportsha=sha(rp)
    assert report['requests_per_model']=={m:17395 for m in('finite41','finite77','mapped_ratio',WR)}and report['matched_request_coverage']
    for path,h in report['source_hashes'].items():assert sha(path)==h,path
    priorp=O/'finite_full_series_confirmation_v1_independent_closed_review.json';prior=read(priorp);assert sha(priorp)=='826c91d8043d2d5edaa7d5f88f0c732f06a44a31991c38a80559e7344a0f19d7'and prior['passed']
    diagp=O/'finite_full_series_confirmation_v1_independent_diagnostics.json';diag=read(diagp);assert sha(diagp)=='75c74d8747a7364914da97d8cf4b7f5be8093de427031af9e81edd7866ab8410'
    p=read(D/'plan.json');assert p['fingerprint']==prior['plan_fingerprint']==report['plan_fingerprint'];assert sha(D/'plan.json')==prior['plan_sha256']and sha(D/'terminal.json')==prior['terminal_sha256']
    for path,h in p['source_hashes'].items():assert sha(path)==h,path
    assert sha(V/'scale_index.sqlite3')=='f64d20747ef5ae8718feb7545fc7947e1a78c176f60526b2148f670731fb7278'
    c=sqlite3.connect('file:'+str(V/'scale_index.sqlite3')+'?mode=ro&immutable=1',uri=True);scales={r[0]:r[1:]for r in c.execute('SELECT request_id,training_sd,native_h,horizon_bucket,within_bucket_weight,series_id,case_index,origin_index,origin_date,requested_end_date FROM request_scales')};c.close()
    values={m:[]for m in METHODS};terms={(m,b):[]for m in METHODS for b in BUCKETS};rawterms={(m,b):[]for m in METHODS for b in BUCKETS};repterms={(m,b):[]for m in METHODS for b in BUCKETS};mass=collections.defaultdict(list);counts=collections.Counter();sources=collections.Counter();stream=hashlib.sha256();cpcommit=[];consumed={}
    for i,q in enumerate(p['requests']):
        rid=q['request_id'];sd,h,b,w,sid,case,oi,origin,end=scales[rid];assert (sid,case,origin,end,h)==(q['series_id'],q['case_index'],q['origin_date'],q['requested_end_date'],q['H']);assert w==q['fixed_weight']['weight']and b==q['fixed_weight']['bucket'];denom=sd*math.sqrt(h);assert sd>0 and math.isfinite(denom)and denom>0
        mass[b].append(w);counts[b]+=1;consumed[rid]={'request_id':rid,'series_id':sid,'origin_date':origin,'requested_end_date':end,'native_h':h,'training_sd':sd,'case_index':case,'origin_index':oi}
        for m in METHODS:
            if m=='WR'and not q['WR_missing']:
                ref=p['WR_retained'][rid];assert ref['final_paths']==240 and ref['model_id']==WR;record=ref['record'];cp_sha=ref['archive_checkpoint_sha256'];sources['WR_verified240_scalar']+=1
            elif m!='WR'and str(i)in p['reuse']:
                ref=p['reuse'][str(i)][m];sidep=Path(ref['sidecar']);assert sha(sidep)==ref['sidecar_sha256'];side=read(sidep);assert side['identity']==ref['original_identity'];assert side['identity']['final_paths']==240 and side['identity']['inner_paths']==2 and side['identity']['seed_start']==0;assert side['checkpoint_sha256']==ref['checkpoint_sha256'];record=ref['record'];assert record['score']==side['record']['score'];cp_sha=side['checkpoint_sha256'];sources[m+'_retained_physical']+=1
            else:
                sidep=D/'output'/f'{i:05d}-{m}.json';side=read(sidep);identity=side['identity'];assert identity['plan_fingerprint']==p['fingerprint']and identity['request']==q and identity['model_id']==p['model_ids'][m]and identity['final_paths']==240;record=side['record'];cp_sha=side['checkpoint_sha256'];sources[m+('_logical_H1_alias'if'alias'in side else'_own_physical')]+=1
            raw=record['score'];assert record==dict(q,score=raw)and isinstance(raw,(int,float))and math.isfinite(raw)and raw>=0
            stream.update(canonical([rid,m,record]).encode());cpcommit.append([rid,m,cp_sha]);values[m].append(raw);terms[m,b].append(w*(raw/denom));rawterms[m,b].append(w*raw);repterms[m,b].append(q['representative_weight']*(raw/denom))
        if h==1:assert values['finite41'][-1]==values['finite77'][-1]==values['mapped_ratio'][-1]
    assert stream.hexdigest()==diag['canonical_matched_record_stream_sha256'];assert hashlib.sha256(canonical(cpcommit).encode()).hexdigest()==prior['checkpoint_commitment']
    assert sources['WR_verified240_scalar']==13874 and sources['WR_own_physical']==3521
    commitment=hashlib.sha256(canonical([consumed[rid]for rid in sorted(consumed)]).encode()).hexdigest();assert commitment==report['exact_v11_scale_commitment_sha256']
    masses={b:math.fsum(mass[b])for b in BUCKETS};Qs={};bucket_Q={};oldraw={};representative={}
    for m in METHODS:
        key=WR if m=='WR'else m;bucket_Q[m]={b:math.fsum(terms[m,b])/masses[b]for b in BUCKETS};Qs[m]=math.fsum(bucket_Q[m].values())/3;oldraw[m]=math.fsum(math.fsum(rawterms[m,b])/masses[b]for b in BUCKETS)/3
        assert bucket_Q[m]==report['secondary_selected_row_conditional_diagnostic'][key]['bucket_Q']and Qs[m]==report['secondary_selected_row_conditional_diagnostic'][key]['conditional_observed_support_Q'];assert report['secondary_selected_row_conditional_diagnostic'][key]['observed_request_count_by_bucket']==dict(counts)and report['secondary_selected_row_conditional_diagnostic'][key]['observed_weight_support_by_bucket']==masses;assert oldraw[m]==prior['independent_projections'][m]['selected_row_equal_bucket_conditional_diagnostic']
    for m in METHODS:
        key=WR if m=='WR'else m;bybucket={b:sum(repterms[m,b])for b in BUCKETS};repq=sum(bybucket[b]for b in BUCKETS)/3;assert repq==report['primary_representative_mass_estimate'][key]['Q']and bybucket==report['primary_representative_mass_estimate'][key]['bucket_Q'];representative[m]={'Q':repq,'bucket_Q':bybucket,'independent_fsum_Q':math.fsum(math.fsum(repterms[m,b])for b in BUCKETS)/3}
    comparisons={m:{'conditional_Q':Qs[m],'gain_vs_fixed_WR_percent':100*(1-Qs[m]/Qs['WR'])}for m in METHODS};assert all(comparisons[m]['gain_vs_fixed_WR_percent']==report['comparisons'][m+'_vs_fixed_WR_conditional_observed_support']['percent_better_than_WR']for m in METHODS[:-1])
    for m in METHODS[:-1]:assert 100*(1-representative[m]['Q']/representative['WR']['Q'])==report['comparisons'][m+'_vs_fixed_WR_representative_mass']['percent_better_than_WR']
    paired={}
    for a,bref in(('finite77','finite41'),('mapped_ratio','finite41'),('mapped_ratio','finite77')):
        paired[a+'|'+bref]={'unweighted_wins':sum(x<y for x,y in zip(values[a],values[bref])),'unweighted_ties':sum(x==y for x,y in zip(values[a],values[bref])),'unweighted_losses':sum(x>y for x,y in zip(values[a],values[bref])),'conditional_Q_gain_percent':100*(1-Qs[a]/Qs[bref])}
    result={'passed':True,'audit_source_sha256':sha(Path(__file__)),'adapter_path':str(V/'rescore_all_series_confirmation_v11.py'),'adapter_sha256':sha(V/'rescore_all_series_confirmation_v11.py'),'rescore_output_sha256':reportsha,'source_binding_count_verified':len(report['source_hashes']),'prior_complete_rawloss_review_sha256':sha(priorp),'prior_exact_canonical_raw_record_stream_review_sha256':sha(diagp),'exact_canonical_raw_record_stream_sha256':stream.hexdigest(),'exact_prior_checkpoint_commitment':prior['checkpoint_commitment'],'cohort_source_input_hashes_reverified':len(p['source_hashes']),'scale_index_sha256':sha(V/'scale_index.sqlite3'),'same_scale_commitment_sha256':commitment,'requests_per_method':17395,'actual_final_paths_all4':240,'WR_source_counts':{k:v for k,v in sources.items()if k.startswith('WR_')},'candidate_physical_alias_retained_counts':{k:v for k,v in sources.items()if not k.startswith('WR_')},'request_count_by_bucket':dict(counts),'observed_full_V9_weight_mass_by_bucket':masses,'independent_V11_representative_mass_estimate':representative,'independent_V11_conditional_Q':Qs,'independent_bucket_conditional_Q':bucket_Q,'fixed_WR_comparisons':comparisons,'paired_candidate_diagnostics':paired,'old_V9_conditional_Q_exact_replay':oldraw,'full320017_headline':False,'scope':'17395 completed metadata-selected development requests. Conditional observed-support Q under full V9 weights; distinct from representative-mass quadrature primary and full320017actualscore. Exact cohort240WR retained13874scalar+own3521physical; NO substitute inherited1000WR. Rawrecordstream/checkpointcommit exactly matches prior independently verified allphysical-loss audit; no repeated broadpath/lossrestore.','financial_fits_QPs_or_forecasts':0,'canonical_or_public_edits':False}
    OUT.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n');print(json.dumps({'passed':True,'output':str(OUT),'sha256':sha(OUT),'conditionalQ':Qs}))
if __name__=='__main__':main()
