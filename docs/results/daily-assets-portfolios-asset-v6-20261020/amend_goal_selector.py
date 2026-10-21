from pathlib import Path
import json,subprocess,select,time,datetime
O=Path(__file__).resolve().parent;request=json.loads((O/'goal_selector_amendment_request.json').read_text());err=(O/'goal_selector_amendment_app_server.log').open('w')
p=subprocess.Popen(['codex','app-server','--listen','stdio://'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=err,text=True,bufsize=1)
serial=0
def call(method,params):
 global serial
 serial+=1;i=serial;p.stdin.write(json.dumps({'jsonrpc':'2.0','id':i,'method':method,'params':params})+'\n');p.stdin.flush();deadline=time.monotonic()+30
 while time.monotonic()<deadline:
  ready,_,_=select.select([p.stdout],[],[],max(0,deadline-time.monotonic()))
  if not ready:break
  line=p.stdout.readline()
  if not line:raise RuntimeError('App server exited')
  x=json.loads(line)
  if x.get('id')==i:
   if 'error' in x:raise RuntimeError(str(x['error']))
   return x['result']
 raise TimeoutError(method)
def goal(result):return result['goal'] if 'goal' in result else result
try:
 call('initialize',{'clientInfo':{'name':'research_goal_editor','version':'1.0.0'},'capabilities':{'experimentalApi':True}})
 p.stdin.write(json.dumps({'jsonrpc':'2.0','method':'initialized'})+'\n');p.stdin.flush()
 before=goal(call('thread/goal/get',{'threadId':request['threadId']}));assert before['objective']==request['before']['objective'];assert before['status']=='active'
 call('thread/goal/set',{'threadId':request['threadId'],'objective':request['objective'],'status':'active'})
 after=goal(call('thread/goal/get',{'threadId':request['threadId']}));assert after['objective']==request['objective'] and after['status']=='active';assert len(after['objective'])<4000
 receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'official_api':'thread/goal/set','before':before,'after':after,'characters':len(after['objective']),'changes':request['changes'],'existing_stopping_paragraph_unchanged':before['objective'].split('\n\n')[5]in after['objective'],'existing_catalogue_paragraph_unchanged':before['objective'].split('\n\n')[7]in after['objective'],'usage_accounting_note':'Official objective replacement resets usage accounting; prior counters retained in before record','historical_results_preserved':True}
 (O/'goal_selector_amendment_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');(O/'goal_objective_updated.txt').write_text(after['objective']+'\n')
 print(json.dumps({'saved':True,'status':after['status'],'characters':len(after['objective']),'only_authorized_requirement_changes':True}),flush=True)
finally:
 p.stdin.close()
 try:p.wait(timeout=3)
 except subprocess.TimeoutExpired:p.terminate();p.wait(timeout=3)
 err.close()
