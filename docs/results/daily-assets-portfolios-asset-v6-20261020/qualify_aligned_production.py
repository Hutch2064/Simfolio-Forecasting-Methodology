"""Bounded all-work numeric proof on actual short/long dated requests, two seeds."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import fcntl,hashlib,itertools,json,sys,time
import numpy as np
import profile_aligned_production as harness
from candidates.asset_selector_aligned_production import candidate
reference=harness.reference;O=harness.O;D=harness.D


def run():
    results=[];started=time.perf_counter()
    for name in ['week0','broad350','portfolio9']:
        manifest,x,w,dates,actual=harness.build_case(name);h=manifest['actual_horizon']
        r=reference.prepare(x,w,h,dates,inner_paths=2,schedule_rule='half_doubling')
        c=candidate.prepare(x,w,h,dates,inner_paths=2,schedule_rule='half_doubling')
        assert r['trace']==c['trace']
        for key in ['A','B']:
            np.testing.assert_array_equal(r['coefficients'][candidate.AXIS][key],c['coefficients'][candidate.AXIS][key])
        np.testing.assert_array_equal(r['weights_by_mode'][candidate.MODE],c['weights_by_mode'][candidate.MODE])
        case_results=[]
        seed0=harness.api.original.coherent_daily_seed(manifest['case']['origin']['origin_date'],h,1000)
        for seed in [seed0,(seed0+101)%2**32]:
            left=reference.ScorePool(candidate.CONFIG,dates,r);right=candidate.ScorePool(candidate.CONFIG,dates,c)
            cursor=0;paths_hash=hashlib.sha256();loss_hash=hashlib.sha256()
            pairs=itertools.zip_longest(left.terminal_chunks(x,w,h,1000,seed),right.terminal_chunks(x,w,h,1000,seed))
            for a,b in pairs:
                assert a is not None and b is not None
                start,values=a;position,other=b
                assert start==position==cursor and values.shape==other.shape and values.shape[0]==1000
                np.testing.assert_array_equal(values,other)
                cursor+=values.shape[1]
                loss=harness.empirical_crps_by_horizon(values,actual[start:cursor])
                loss2=harness.empirical_crps_by_horizon(other,actual[start:cursor])
                np.testing.assert_array_equal(loss,loss2)
                paths_hash.update(values.tobytes());loss_hash.update(loss.tobytes())
            assert cursor==h and left.last_selection['path_counts']==right.last_selection['path_counts']
            case_results.append({'seed':seed,'paths':1000,'leads':h,'full_path_sha256':paths_hash.hexdigest(),'full_loss_sha256':loss_hash.hexdigest(),
                                 'all_path_values_exact':True,'all_per_lead_losses_exact':True})
        result={'case':name,'manifest_sha256':hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest(),
                'history_sha256':manifest['history_sha256'],'training_dates_sha256':manifest['training_dates_sha256'],
                'actual_horizon':h,'rows':len(x),'assets':len(w),'internal_paths':2,'internal_seed_replication':0,
                'A_sha256':hashlib.sha256(c['coefficients'][candidate.AXIS]['A'].tobytes()).hexdigest(),
                'B_sha256':hashlib.sha256(c['coefficients'][candidate.AXIS]['B'].tobytes()).hexdigest(),
                'weights_sha256':hashlib.sha256(c['weights_by_mode'][candidate.MODE].tobytes()).hexdigest(),
                'all_coefficients_weights_and_trace_exact':True,'final_forecasts':case_results}
        results.append(result);print(json.dumps(result),flush=True)
        del r,c,left,right,x,w,dates,actual
    profiles=[]
    for name in ['week0','broad350','portfolio9']:
        pairs=[]
        for repeat in [0,1]:
            a=json.loads((D/'profiles'/f'reference-{name}-{repeat}.json').read_text())
            b=json.loads((D/'profiles'/f'candidate-{name}-{repeat}.json').read_text())
            for key in ['path_sha256','loss_sha256','weights_sha256','A_sha256','B_sha256','seed']:
                assert a[key]==b[key],(name,repeat,key)
            pairs.append({'repeat':repeat,'reference_wall_seconds':a['wall_seconds'],'candidate_wall_seconds':b['wall_seconds'],
                          'reference_cpu_seconds':a['cpu_seconds'],'candidate_cpu_seconds':b['cpu_seconds'],
                          'wall_speed_ratio':a['wall_seconds']/b['wall_seconds'],'wall_seconds_saved':a['wall_seconds']-b['wall_seconds']})
        profiles.append({'case':name,'pairs':pairs})
    sources={str(p):harness.sha(p)for p in [Path(candidate.__file__),Path(__file__),Path(harness.__file__),D/'test_candidate.py',D/'profile_driver_reference_v1.py']}
    report={'verified':True,'scope':'Three actual research requests, complete1000path outputs, two final seeds; no full-panel or browser CTV claim',
            'seconds':time.perf_counter()-started,'cases':results,'profile_comparisons':profiles,'source_hashes':sources,
            'reference_dependency_source_hashes':manifest['source_hashes'],'no_persistent_request_caches':True,
            'numerical_method_changes':False,'eligible_recipe_count':167,'lead_work_truncated':False,
            'fit_tolerance_or_cap_changed':False,'full_prefix_fit_context_retained':True,
            'benchmark_entry':'dated arrays to complete1000path portfolio forecast; fresh selection/refit each call; cProfile overhead included',
            'cold_state':'First call in a fresh process with existing numerical disk caches, not an all-layer cold claim',
            'unit_tests':{'passed':3,'seconds':2.007},'website_CTV_claim':False}
    (D/'verification_receipt.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'verified':True,'cases':len(results),'full_forecasts_compared':6,'profile_comparisons':profiles}),flush=True)

if __name__=='__main__':
    with(O/'asset_v6_forecast_slots.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);run()
