"""Small contract tests; no real forecasts or launch."""
import concurrent.futures, copy, tempfile, unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import run_aligned_panel_v7 as r

class Contracts(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup);self.d=Path(self.tmp.name)
        cases=[{'series_index':0,'series_id':'fixture','origin_index':i,'origin':{'position':9,
            'origin_date':'2020-01-01','max_horizon':3,'origin_label':str(i)}} for i in range(3)]
        self.p={'fingerprint':'fixture','cases':cases,'models':copy.deepcopy(r.MODELS),'source_hashes':{},
            'expected_tasks':6,'expected_per_model':3,'gate_receipts':{n:{'path':n,'sha256':n} for n in r.GATES}}
        for n in self.p['models']:self.p['models'][n]['model_id']=n
    def record(self):
        cp,rp,_=r.paths(self.d,'endpoint',0);cp.parent.mkdir(parents=True,exist_ok=True)
        loss=np.array([.1,.2,.3]);np.savez_compressed(cp,losses=loss)
        rec={'identity':r.identity(self.p,'endpoint',self.p['cases'][0]),'checkpoint_sha256':r.sha(cp),
            'loss_sha256':r.hashlib.sha256(loss.tobytes()).hexdigest(),'source_closure_digest':r.digest({}),'score':.2}
        rec['record_digest']=r.digest(rec);r.write(rp,rec);return rec
    def test_admission_requires_exact_plan_and_all_gates(self):
        receipt={'schema':'aligned-panel-v7-root-admission','plan_fingerprint':'fixture','root_cleared_launch':True,
            'passed':True,'passed_gates':sorted(r.GATES),'gate_receipts':self.p['gate_receipts']}
        r.admit(self.p,receipt)
        for key,value in [('root_cleared_launch',False),('plan_fingerprint','old'),('passed_gates',[]),('gate_receipts',{})]:
            with self.assertRaises(ValueError):r.admit(self.p,{**receipt,key:value})
    def test_live_report_keeps_full_panel_reference_when_subset_ranking_flips(self):
        live=object.__new__(r.LiveComparison)
        fixed,other=r.LiveComparison.BASELINES
        live.plan={'fixed_reference_model_id':fixed}
        live.counts={'endpoint':1}
        class Value:
            def __init__(self,value):self.value=value
            def score(self):return self.value
        live.reducers={'endpoint':{'candidate':Value(.9),fixed:Value(1.),other:Value(2.)}}
        result=live.report()['endpoint']
        self.assertEqual(result['fixed_full_panel_reference_model_id'],fixed)
        self.assertAlmostEqual(result['percent_better_vs_fixed_reference'],10.)
    def test_resume_matching_loss_and_record_only(self):
        rec=self.record();self.assertEqual(r.verified_checkpoint(self.d,self.p,'endpoint',0),rec)
        cp,_,_=r.paths(self.d,'endpoint',0);cp.write_bytes(b'corrupt')
        with self.assertRaises(ValueError):r.verified_checkpoint(self.d,self.p,'endpoint',0)
    def test_resume_wrong_origin_and_metadata_tampering(self):
        self.record();p=copy.deepcopy(self.p);p['cases'][0]['origin']['max_horizon']=4
        with self.assertRaises(ValueError):r.verified_checkpoint(self.d,p,'endpoint',0)
        _,rp,_=r.paths(self.d,'endpoint',0);rec=r.read(rp);rec['score']=123;r.write(rp,rec)
        with self.assertRaises(ValueError):r.verified_checkpoint(self.d,self.p,'endpoint',0)
    def test_orphan_failure_and_interrupted_attempt_fail_closed(self):
        cp,rp,fp=r.paths(self.d,'endpoint',0)
        self.assertIsNone(r.verified_checkpoint(self.d,self.p,'endpoint',0))
        for path in [cp.with_suffix('.attempt.json'),fp,rp]:
            r.write(path,{'incomplete':True})
            with self.assertRaises(ValueError):r.verified_checkpoint(self.d,self.p,'endpoint',0)
            path.unlink()
    def test_projection_excludes_warmup_and_unseen_work(self):
        t={'forecast_selection_score_cpu_seconds':12.,'worker_first_request':False,'fit_repair':False}
        recs=[('endpoint',0,{'timing':t}),('endpoint',1,{'timing':{**t,'worker_first_request':True,'forecast_selection_score_cpu_seconds':999}})]
        out=r.projection(recs,[('endpoint',2),('aligned',0)],self.p)
        self.assertEqual(out['mean_seconds_lower_bound'],1.);self.assertEqual(out['uncovered_remaining_tasks'],1)
        self.assertIsNone(r.projection([], [('aligned',0)],self.p)['mean_seconds_lower_bound'])
    def test_finite_controller_and_reporter_failure_terminal(self):
        calls=[]
        class Pool:
            def __init__(self,**kwargs):pass
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def submit(self,fn,task):
                calls.append(task);future=concurrent.futures.Future()
                future.set_result({'task':task,'record':{'timing':{'forecast_selection_score_cpu_seconds':.01,
                    'worker_first_request':False,'fit_repair':False},'score':1.,'series_id':'fixture','origin_date':'2020-01-01'}})
                return future
        r.write(self.d/'plan.json',self.p);r.write(self.d/'admission.json',{})
        with patch.object(r,'O',self.d),patch.object(r,'verify_plan',return_value={'series':[{'series_id':'fixture'}]}), \
             patch.object(r,'admit'),patch.object(r,'LiveComparison',return_value=type('Live',(),{'add':lambda *a:None,'report':lambda *a:{}})()),patch.object(r,'verified_checkpoint',return_value=None), \
             patch.object(r.subprocess,'run') as preflight,patch.object(r.futures,'ProcessPoolExecutor',Pool), \
             patch.object(r.scoring,'aggregate_records',side_effect=RuntimeError('report broke')):
            with self.assertRaisesRegex(RuntimeError,'report broke'):r.run(self.d,self.d/'admission.json')
        self.assertEqual(len(calls),6);self.assertEqual(len(set(calls)),6)
        terminal=r.read(self.d/'terminal.json')
        self.assertEqual(terminal['status'],'complete');self.assertEqual(terminal['reporter_status'],'failed')
        self.assertTrue(terminal['all_processes_drained']);self.assertTrue(terminal['forecast_workers_terminal'])
        self.assertIn('7',preflight.call_args.args[0]);self.assertIn('16',preflight.call_args.args[0])
    def test_incremental_hierarchy_matches_original_on_available_support(self):
        metadata={'a':{'panel_branch':'assets','asset_class':'equity','family':'one'},
            'b':{'panel_branch':'assets','asset_class':'equity','family':'two'},
            'c':{'panel_branch':'assets','asset_class':'bond','family':'three'},
            'p':{'panel_branch':'portfolios','asset_class':'portfolio','family':'four'}}
        records=[];reducer=r.Hierarchy(metadata)
        for sid,date,score in [('a','2020-01-01',1.),('a','2020-01-02',2.),('b','2020-01-01',4.),
                ('c','2020-01-01',5.),('p','2020-01-01',6.),('a','2020-04-01',7.)]:
            record={'series_id':sid,'origin_date':date,'score':score};records.append(record);reducer.add(record)
            self.assertAlmostEqual(reducer.score(),r.scoring.aggregate_records(records,metadata)['score'],places=14)
    def test_resume_preflight_only_reserves_missing_work(self):
        record={'timing':{'forecast_selection_score_cpu_seconds':.01,'worker_first_request':False,'fit_repair':False},
            'score':1.,'series_id':'fixture','origin_date':'2020-01-01'}
        class Pool:
            def __init__(self,**kwargs):pass
            def __enter__(self):return self
            def __exit__(self,*args):pass
            def submit(self,fn,task):
                f=concurrent.futures.Future();f.set_result({'task':task,'record':record});return f
        live=type('Live',(),{'add':lambda *a:None,'report':lambda *a:{}})()
        r.write(self.d/'plan.json',self.p);r.write(self.d/'admission.json',{})
        with patch.object(r,'O',self.d),patch.object(r,'verify_plan',return_value={'series':[{'series_id':'fixture'}]}), \
             patch.object(r,'admit'),patch.object(r,'LiveComparison',return_value=live), \
             patch.object(r,'verified_checkpoint',side_effect=lambda d,p,n,i:record if n=='endpoint' else None), \
             patch.object(r.subprocess,'run') as preflight,patch.object(r.futures,'ProcessPoolExecutor',Pool), \
             patch.object(r.scoring,'aggregate_records',return_value={'score':1.}):
            terminal=r.run(self.d,self.d/'admission.json')
        command=preflight.call_args.args[0];growth=float(command[command.index('--expected-gib')+1])
        self.assertAlmostEqual(growth,3*(3*8+41500)/(1024**3))
        self.assertTrue(terminal['qualification_valid'])
    def test_worker_nice_does_not_compound_inherited_priority(self):
        r.write(self.d/'plan.json',self.p)
        with patch.object(r.os,'getpriority',return_value=15),patch.object(r.os,'nice') as nice, \
             patch.object(r.api,'load_panel',return_value={'series':[]}),patch.object(r,'AssetInputs'):
            r.initialize(self.d)
        nice.assert_called_once_with(0)
    def test_prepare_requires_gates_and_preserves_existing_plan(self):
        with self.assertRaises(ValueError):r.prepare({},self.d/'new')
        self.assertFalse((self.d/'new').exists())
        with self.assertRaises(ValueError):r.prepare({},self.d)
if __name__=='__main__':unittest.main()
