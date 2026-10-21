"""Advance only a complete matched broad winner; the wider research goal stays active."""
from pathlib import Path
import os,json,subprocess,time,datetime,hashlib
O=Path(__file__).resolve().parent;V=O/'asset_level_v6';C=V/'candidates/asset_feature_batch';B=C/'broad1000';F=C/'full';PY='/Users/aidanhutchison/simfolio/tmp/catalogue52-20261018/venv/bin/python';A=O/'asset_feature_handoff_active.json'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
def write(p,d):
 t=p.with_suffix(p.suffix+'.new');t.write_text(json.dumps(d,indent=2,sort_keys=True)+'\n');os.replace(t,p)
def state(status,**extra):write(A,{'utc':now(),'pid':os.getpid(),'status':status,'goal_remains_active':True,**extra})
def command(pid):return subprocess.run(['ps','-o','command=','-p',str(pid)],capture_output=True,text=True).stdout

def main():
 assert not A.exists(),'Inspect existing handoff state before retrying'
 state('waiting_for_broad')
 while not (B/'summary.json').exists():
  owner=read(B/'active.json');assert 'run_asset_feature_broad_v2.py' in command(owner['pid']),'Broad owner exited without its final evidence'
  time.sleep(10)
 s=read(B/'summary.json');plan=read(B/'plan.json');assert s['complete_execution'] and not s['errors'] and s['native_failures']==s['scoring_integrity_errors']==0 and s['paths']==1000 and s['cases_per_model']==7418 and s['anchor_loss_vectors_bitwise_equal_each_case']
 assert all(sha(O/k)==v for k,v in plan['source_hashes'].items())
 candidates=[x for x in s['complete_identical_coverage_screen_ranking']if x['main_Q']<s['reference_main_Q']['current_best']]
 if not candidates:
  state('terminal_no_broad_advancement',reason='No candidate improved current best across broad matched1000path coverage. Further credible research remains; no goal completion.')
  return
 chosen=min(candidates,key=lambda x:x['main_Q']);assert chosen['complete']
 for stage in ('confirmation1000','reserved1000'):
  q=read(C/stage/'summary.json');assert q['models'][chosen['model_id']]['main_Q']<q['reference_main_Q']['current_best'] and not q['errors']
 assert not (C/'full_selection.json').exists()
 write(C/'full_selection.json',{'utc':now(),'model_id':chosen['model_id'],'config':chosen['config'],'broad_main_Q':chosen['main_Q'],'broad_current_best_main_Q':s['reference_main_Q']['current_best'],'broad_summary_sha256':sha(B/'summary.json'),'rule':'Lowest overall broad main Q among configurations improving both frozen development and reserved cases; no horizon/family routing or diagnostic selection weight.','official_promotion_requires_complete_full_verification':True,'wider_goal_active':True})
 write(O/'asset_feature_full_storage_budget.json',{'utc':now(),'owner':str(O),'budget_gib':16,'expected_full_growth_gib':1.8,'expected_models_in_full':1,'reason':'One fixed qualified full64049-forecast challenger with8128verified pilot vectors reused, alongside active fixed baseline/current-best evidence. Bounded outputs; no unbounded grid of full runs.','free_space_reserve_gib':100,'release_condition':'Privately archive/recover completed full challenger and catalogue it; retire only verified closed copies, keeping current scientific dependencies.'})
 log=(O/'asset_feature_full.log').open('a');owner=subprocess.Popen(['nice','-n','15',PY,'-u',str(O/'run_asset_feature_full.py')],stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
 launch={'utc':now(),'pid':owner.pid,'log':str(O/'asset_feature_full.log'),'model_id':chosen['model_id'],'paths':1000,'workers':12,'max_tasks':64049,'goal_remains_active':True};write(O/'asset_feature_full_launch.json',launch)
 rlog=(O/'asset_feature_full_reporter.log').open('a');reporter=subprocess.Popen(['nice','-n','19',PY,'-u',str(O/'report_asset_feature_full.py')],stdout=rlog,stderr=subprocess.STDOUT,start_new_session=True)
 state('full_candidate_running',owner_pid=owner.pid,reporter_pid=reporter.pid,model_id=chosen['model_id'])
 g=read(O/'goal_progress.json');g.update(stage='asset_feature_full_candidate',candidate_model_id=chosen['model_id'],actual_goal_tool_status='active',stopping_condition_met=False,previous_goal_turn_classification='progress');g['active_pilot']={'phase':'full_feature_candidate','pid':owner.pid,'reporter_pid':reporter.pid,'paths':1000,'workers_requested':12,'expected_tasks':64049,'status_path':str(F/'results/status.json'),'scoreboard_path':str(F/'scoreboard.json'),'owner_path':str(F/'active.json'),'official':False};write(O/'goal_progress.json',g)
 owner.wait();assert owner.returncode==0,'Full owner failed: inspect saved log/checkpoints, do not restart automatically'
 result=read(F/'verification_receipt.json');assert result['complete'] and result['successful_tasks']==64049 and result['native_failures']==result['scoring_integrity_errors']==0
 while reporter.poll() is None:time.sleep(5)
 assert reporter.returncode==0,'Full reporter failed: inspect independently; no automatic promotion'
 if result['eligible_for_current_best_promotion']:
  subprocess.run([PY,str(O/'promote_asset_feature.py')],check=True)
 g=read(O/'goal_progress.json');g['stage']='feature_full_finished_preservation_and_further_research';g['active_pilot']['status']='terminal_full_evaluation';g['further_credible_tests_remain']=True;g['stopping_condition_met']=False;write(O/'goal_progress.json',g)
 state('terminal_full_evaluation_complete_preservation_pending',model_id=chosen['model_id'],promotion_eligible=result['eligible_for_current_best_promotion'],verification_receipt=str(F/'verification_receipt.json'))
if __name__=='__main__':main()
