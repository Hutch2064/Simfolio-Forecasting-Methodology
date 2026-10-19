"""Saved-only independent audit; stdlib/source helpers only, no forecasts/native imports."""
import datetime,hashlib,importlib.util,json,math,os,statistics
from pathlib import Path
U=Path(__file__).resolve().parents[2]
HERE=U/'benchmarks/selected_fast_cdcc_literal_bound_speed'
OUT=Path(__file__).resolve().parent
K='selected_fast_literal_bound_cdcc'
MODELS=('m261','selected_fast',K)
STATES=('fresh_fit','cached_fit_1','cached_fit_2','cached_fit_3')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):p=Path(p);return {'path':str(p),'bytes':p.stat().st_size,'sha256':sha(p)}
def digest(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(",",":")).encode()).hexdigest()
def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m
# These helpers contain only stdlib definitions and have no financial/native initialization.
PARITY=module('independent_saved_gate',HERE/'parity.py')
STAGES=module('independent_saved_stages',U/'benchmarks/controlled_v3/stages.py')
PRIORITY=module('independent_saved_priority',U/'benchmarks/execution_priority_v1/priority.py')
PRIORITY.SITES[K]=PRIORITY.SITES['selected_fast']
verified={}
def exact(row):
    p=Path(row['path']);assert p.is_absolute() and p.is_file() and not any(x.is_symlink() for x in (p,*p.parents)),p
    assert sha(p)==row['sha256'],p
    if 'bytes' in row:assert p.stat().st_size==row['bytes'],p
    verified[str(p)]=ref(p);return p
def all_refs(value):
    if isinstance(value,dict):
        if {'path','sha256'}<=value.keys():exact(value)
        for key,item in value.items():
            if key in ('source_sha256','receipt_sha256','source_rows_sha256'):
                for p,h in item.items():exact({'path':p,'sha256':h})
            elif isinstance(item,(dict,list)):all_refs(item)
    elif isinstance(value,list):
        for item in value:all_refs(item)
def inventory():return {str(p):ref(p) for p in sorted(HERE.rglob('*')) if p.is_file()}
def raw_processes(raw):
    return {int(v[0]):v for line in raw.splitlines() if len(v:=line.strip().split(None,5))==6}
def quiet(row,case,epoch,parent):
    paused={v['pid']:v for v in epoch['stopped_family_allowlist']};controls={v['pid']:v for v in epoch['quiescent_controller_allowlist']}
    receipts=[]
    for name in ('background_before','background_after'):
        q=row[name];raw=raw_processes(q['raw_snapshot']);items=q['strict_other_activity_receipt'];seen=set();admin=set()
        assert parent in raw
        for item in items:
            pid=item['pid'];v=raw[pid]
            assert (int(v[1]),v[2],int(v[3]),v[4],hashlib.sha256(v[5].encode()).hexdigest())==(item['ppid'],item['state'],item['rss_kib'],item['cpu_time'],item['command_sha256'])
            if item['own_family']:continue
            if pid in paused:
                assert 'T' in item['state'];assert all(item[k]==paused[pid][k] for k in ('pid','ppid','command_sha256','pause_context'));seen.add(pid)
            elif pid in controls:
                assert all(item[k]==controls[pid][k] for k in ('pid','ppid','command_sha256','role','admin_context'))
                assert item['state'][0] in 'SRIU'
                c=item['quiescent_campaign'];assert c['eligible_complete_archived']==241 and c['pending_archive_models']==0 and not c['queue_pending']
                assert c['controller']['allocated_cpu_slots']==c['controller']['live_models']==0;admin.add(pid)
            else:
                assert item['administrative_controller_pid'] in controls
        assert seen==set(paused) and admin==set(controls)
        for reuse in q['retired_PID_reused_ps_identity']:
            assert reuse['identity_resolved'] and reuse['pid']==65477 and reuse['literal_command']=='/usr/libexec/gamecontrollerd'
            assert raw[65477][-1]==reuse['literal_command']
        receipts.append({v['pid']:v for v in items})
    for pid in paused:assert receipts[0][pid]['cpu_time']==receipts[1][pid]['cpu_time']
    for item in row['administrative_CPU']:
        assert item['pid'] in controls and item['cpu_seconds_delta']>=0

def check_row(row,case,source,source_id,expected,rep,kind,parent,epoch):
    h=case['identity']['horizon_days']
    assert (row['case'],row['model'],row['rep'],row['execution_id'])==(case['name'],kind,rep,source_id)
    assert row['nice']==10 and row['disk_fit_cache'] is None and set(row['thread_environment'].values())=={'1'}
    assert row['initialization']['private_reference_initialize_calls']==1
    PRIORITY.validate(row['initialization']['execution_priority'],kind)
    assert row['numerical_calls']==dict(warmup=1,timed=4,untimed_original_core_guard=1)
    assert row['oracle']==expected and row['path_values_compared']==240*h and row['CRPS_values_compared']==h
    assert row['complete_cache_state_and_uninstrumented_original_core_paths_equal']
    if 'paths_sha256' in expected:assert row['plain_complete_path_sha256']==expected['paths_sha256']
    assert row['plain_complete_daily_CRPS_sha256']==row['complete_daily_CRPS_sha256']
    if 'CRPS_sha256' in expected:assert row['complete_daily_CRPS_sha256']==expected['CRPS_sha256']
    assert tuple(s['state'] for s in row['samples'])==STATES
    for s in row['samples']:
        assert math.isfinite(s['seconds']) and s['seconds']>0
        assert s['full_path_shape']==[240,h] and s['full_path_sha256']==row['plain_complete_path_sha256']
        assert STAGES.decompose(s['seconds'],s['stages']['raw_candidate_nested_timers'],s['stages']['observed_inclusive_timers'])==s['stages']
        assert s['stages']['observed_inclusive_timers']['return_uniforms_calls']==1
    assert all(c['after']['currsize']==0 for c in row['cleared_fit_caches_after_unrelated_warmup'])
    assert row['warmup_return_uniforms_calls'] and all(c['shape'][1]==case['asset_count'] for c in row['warmup_return_uniforms_calls'])
    assert row['compiled_cache_directory']==source['shared_native_cache_and_tmp']['NUMBA_CACHE_DIR']
    if kind==K:
        setup=row['initialization']['native_setup'];all_refs(setup)
        assert setup['persistent_getter_installed_after_sole_init_before_stage_observe'] and setup['restored_before_untimed_original_plain'] and setup['prototype_native_builds']==0
        assert setup['candidate']['binary']['sha256']=='a2d609468ba5e197ed8599c99f5a4d04d80f59cba4c40eeece3ef4e61a7a7094'
        assert setup['reference']['binary']['sha256']=='0816cc9194d8f8e433a5a26a9bd3a0e1b54fd9bbd520ad2a1f4ddc50ac058135'
        if rep==0:
            proof=row['untimed_complete_output_parity'];old=read(exact(proof['actual_paired_phase']))
            assert proof['passed'] and proof['one_private_owner'] and proof['additional_forecasts']==0
            assert old['passed'] and old['reference']==old['prototype']==proof['complete_scientific_evidence']
            assert proof['full_portfolio_paths_sha256']==expected['paths_sha256'] and proof['complete_CRPS_sha256']==expected['CRPS_sha256']
            assert proof['path_values']==240*h and proof['CRPS_values']==h
            assert proof['boundary']['all_hooks_restored'] and proof['boundary']['rejoin']==[{'shape':[240,h,case['asset_count']]}]
    quiet(row,case,epoch,parent)

def summaries(rows):
    out={}
    for case in sorted({r['case'] for r in rows}):
        out[case]={}
        for group in ('fresh_fit','cached_fit'):
            samples={m:[s for r in rows if r['case']==case and r['model']==m for s in r['samples'] if (s['state']=='fresh_fit')==(group=='fresh_fit')] for m in MODELS}
            assert all(len(v)==(3 if group=='fresh_fit' else 9) for v in samples.values())
            med={m:statistics.median(s['seconds'] for s in values) for m,values in samples.items()}
            parts={m:{k:statistics.median(s['stages']['partition_seconds'][k] for s in values) for k in values[0]['stages']['partition_seconds']} for m,values in samples.items()}
            out[case][group]={'median_seconds':med,'candidate_time_multiple_FAST':med[K]/med['selected_fast'],'partition_medians_seconds':parts}
    return out

def main():
    before=inventory();binding,kernel=PARITY.proof_gate();all_refs(binding)
    assert sum(v['bytes'] for v in before.values())<=161061273
    cache_bytes=sum(p.stat().st_size for p in (HERE/'results/numba').rglob('*') if p.is_file());assert cache_bytes<=107374182
    expected_hashes={'parent-18-terminal.json':'4124dec452c4a39121e869af24539c50268025f79d2698fa0b0e2676d8f93492','parent-9-terminal.json':'67d9d947bc41fd3d272b5b663a4bb1a418dcb8266609603946531fc381293fdd','results/measured18-parent.json':'92fa77f46c87b23ca7330479c8607a437ecb8b594adbe999c94af714403311ee','results-long/measured9-parent.json':'0c79686ad8e4ddec8e345ddd0af0f8479850cf7e533e564dafd61768ab97418c'}
    for p,h in expected_hashes.items():exact({'path':str(HERE/p),'sha256':h})
    seal=read(HERE/'seal.json');all_refs(seal);assert sha(HERE/'seal.json')=='6c9b7b4a8b8f6ceed9923130998dfa27be8537dd7318a7df93971391116b0a8c'
    epoch=read(U/'benchmarks/conventional_on_timing_execution_v1/epoch-c16cd261d4341877/manifest.json')
    rows=[];pids=[];panels=[]
    for suffix,n,terminal_name,summary_name in (('results',18,'parent-18-terminal.json','measured18-parent.json'),('results-long',9,'parent-9-terminal.json','measured9-parent.json')):
        root=HERE/suffix;source=read(root/'source.json');completion=read(root/'completion.json');terminal=read(HERE/terminal_name)
        all_refs(source);all_refs(completion);all_refs(terminal)
        assert completion['passed'] and completion['trials']==n and completion['timed_calls']==n*4 and completion['all_forecasts']==n*6 and completion['extra_untimed_boundary_forecasts']==0
        assert source['source_sha256']==completion['source_sha256']
        assert source['admission']['workers']==1 and source['admission']['parent_readonly_release'] and source['admission']['resource_source_and_all_family_closure_verified']
        assert source['cache_cap_bytes']==107374182 and source['component_cap_bytes']==161061273 and source['admission']['root_budget_gib']==7.5
        assert terminal['actual_exit_code']==0 and terminal['all_observed_pids_absent']
        group=[];sid=digest(source)
        for case in source['cases']:
            assert case==next(c for c in epoch['cases'] if c['name']==case['name'])
            for rep in range(3):
                for kind in MODELS:
                    path=root/f'{rep}-{case["name"]}-{kind}.json';row=read(path);expected=epoch['oracles'][case['name']]['selected_fast' if kind==K else kind]
                    check_row(row,case,source,sid,expected,rep,kind,terminal['actual_parent_pid'],epoch)
                    assert row['pid'] in terminal['all_observed_pids'];pids.append(row['pid']);group.append(row)
        assert len(group)==n
        result=summaries(group);parent=read(root/summary_name);all_refs(parent)
        parent_cases=parent['cases'] if 'cases' in parent else {source['cases'][0]['name']:parent['states']}
        assert result==parent_cases and parent['trials']==n and parent['timed_calls']==4*n and parent['forecasts']==6*n and parent['extra_forecasts']==0
        rows+=group;panels.append({'terminal':ref(HERE/terminal_name),'summary':ref(root/summary_name),'source':ref(root/'source.json'),'completion':ref(root/'completion.json'),'cases':result})
        for pid in terminal['all_observed_pids']:
            try:os.kill(pid,0)
            except ProcessLookupError:pass
            else:raise AssertionError(('timing PID remains live',pid))
    members=read(U/'archive_factorial/pinned-snapshot-members.json')['members'];lab=U.parents[2]
    assert len(members)==590
    for rel,member in members.items():
        path=str(lab/rel);assert path in verified and verified[path]['sha256']==member['sha256']
    assert len(rows)==27 and len(set(pids))==27
    for case,kind in {(r['case'],r['model']) for r in rows}:
        assert len({(r['plain_complete_path_sha256'],r['complete_daily_CRPS_sha256']) for r in rows if (r['case'],r['model'])==(case,kind)})==1
    family=sorted({v['pid'] for r in rows for name in ('background_before','background_after') for v in r[name]['strict_other_activity_receipt'] if v['own_family']})
    for pid in family:
        try:os.kill(pid,0)
        except ProcessLookupError:pass
        else:raise AssertionError(('observed numerical/support family remains live',pid))
    assert inventory()==before,'original artifacts changed during saved-only review'
    report={'schema':'independent-native-unpaired-27-saved-actual-audit-v1','passed':True,'timestamp':datetime.datetime.now(datetime.timezone.utc).isoformat(),'new_trials':27,'new_timed_calls':108,'new_forecasts':162,'extra_boundary_forecasts':0,'prior_paired_forecasts_reused_as_evidence':24,'prior_pair_forecasts_newly_executed':0,'cases':{k:v for group in panels for k,v in group['cases'].items()},'panels':panels,'verified_unique_refs':len(verified),'snapshot_resources_verified':590,'component_and_owned_cache_actual_bytes_within_caps':True,'owned_cache_bytes':cache_bytes,'original_files':len(before),'original_bytes':sum(v['bytes'] for v in before.values()),'original_mutations':0,'full_paths_all_samples_and_original_plain_guard_pass':True,'rows_with_direct_saved_path_oracle':sum('paths_sha256' in r['oracle'] for r in rows),'daily_CRPS_all_rows_original_plain_guard_and_cross_rep_stability_pass':True,'candidate_selected_all_three_contexts_direct_saved_CRPS_oracles_pass':True,'rows_with_direct_saved_CRPS_oracle':sum('CRPS_sha256' in r['oracle'] for r in rows),'M261_oracle_scope':'SPYSIM M261 retains a prior full-path hash but no independent daily CRPS oracle. Multiasset M261 has pinned original law/input plus fresh original plain full-path/CRPS and cross-rep guards; no invented historical path/CRPS oracle.','three_rep0_original_plain_native_Q_CDF_RNG_non_timing_checkpoint_equality_pass':True,'restored_before_original_plain_and_failure_source_gate_pass':True,'exact_tested_ABI_and_binary_gate_pass':True,'one_worker_one_initializer_NI10_thread1_caps_and_recorded_quiet_scope_pass':True,'unique_worker_pids':pids,'all_observed_timing_PIDs_currently_absent':True,'observed_numerical_and_support_family':family,'all31_observed_financial_and_support_PIDs_currently_absent':len(family)==31,'pure_saved_fixtures_passed':5,'cached_median_policy':'Pool cached_fit_1/2/3 (9 observations/model/context); fresh_fit has3. Use raw wall samples; nested timers never summed twice.','unused_frozen_summarizer_issue':'summarize.py expects obsolete cached_fit labels and incorrectly adds2 boundary forecasts/context; not executed and unchanged. Parent stdlib measured summaries independently reconstructed exactly.','scope':'Bounded local compiler literal-bound p6 specialization; p1 native path bypass is control. Not all dimensions, full80, all-work parity, score improvement, production or authenticated website performance.','reviewer_calls':{'financial_initializers':0,'forecasts':0,'native_imports':0,'native_builds':0,'network':0,'timings':0}}
    (OUT/'original-inventory.json').write_text(json.dumps(before,sort_keys=True,indent=2)+'\n')
    (OUT/'verified-refs.json').write_text(json.dumps(verified,sort_keys=True,indent=2)+'\n')
    (OUT/'review.json').write_text(json.dumps(report,sort_keys=True,indent=2,allow_nan=False)+'\n')
    print(json.dumps({'passed':True,'report':ref(OUT/'review.json'),'original_files':len(before),'verified_refs':len(verified),'cases':report['cases']}))
if __name__=='__main__':main()
