"""Read-only incremental v8 headline over every committed same-count triple."""
import os
for key in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[key]='1'
import argparse,collections,datetime,fcntl,hashlib,json,sqlite3,sys,time,zlib
from pathlib import Path
import numpy as np
O=Path(__file__).resolve().parent;V=O/'asset_level_v6';D=O/'short_horizon_final240_v3';OUT=O/'horizon_buckets_v8';sys.path[:0]=[str(O),str(V),str(O/'repository/src')]
import score_horizon_buckets_v8 as scorer
WR='model_testing_asset_joint_blocks_replacement_v6';WOR='model_testing_asset_joint_blocks_history_partition_v6';MODELS=('endpoint',WR,WOR);HORIZONS=(1,5,10,21)
read=lambda p:json.loads(Path(p).read_text())
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb')as f:
        for block in iter(lambda:f.read(1024*1024),b''):h.update(block)
    return h.hexdigest()
def atomic(path,value):
    path=Path(path);temp=path.with_suffix(path.suffix+'.tmp');temp.write_text(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n');temp.replace(path)
def signature(path):
    s=Path(path).stat();return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,s.st_ctime_ns]
def canonical(value):return json.dumps(value,sort_keys=True,separators=(',',':'))
def digest(value):return hashlib.sha256(canonical(value).encode()).hexdigest()

class Reporter:
    def __init__(self):
        OUT.mkdir(exist_ok=True);self.plan=read(D/'plan.json');self.cases=self.plan['cases'];self.audit=read(OUT/'audit.json');self.mapping=OUT/'requested_calendar_mapping.npz'
        if sha(self.mapping)!=self.audit['mapping']['sha256']or self.audit['plan_fingerprint']!=self.plan['fingerprint']:raise ValueError('Calendar mapping identity differs')
        with np.load(self.mapping,allow_pickle=False)as z:
            self.original=z['original'].copy();self.shortends={int(i):ends.copy()for i,ends in zip(z['short_group_case_indices'],z['short_requested_end_days'])}
        if len(self.cases)!=64049 or len(self.shortends)!=63992:raise ValueError('Full case catalogue differs')
        panel=read(V/'inputs/panel.json');self.metadata={row['series_id']:row for row in panel['series']}
        self.parents={str(D):self.plan};parent=D
        while 'parent_supplement_directory'in self.parents[str(parent)]:
            parent=Path(self.parents[str(parent)]['parent_supplement_directory']);self.parents[str(parent)]=read(parent/'plan.json')
        self.source_hashes={str(Path(__file__)):sha(Path(__file__)),str(Path(scorer.__file__)):sha(scorer.__file__),str(self.mapping):sha(self.mapping),str(D/'plan.json'):sha(D/'plan.json')}
        for module in ('score_aligned_v7.py','score_main.py'):
            self.source_hashes[str(V/module)]=sha(V/module)
        cachepath=O/'short_horizon_live_paired_snapshot_20261022T035838Z.json'
        if sha(cachepath)!='278dd60cbfd7a2ebbeeb2d3b77d668fea5972f56e770fa2384da09fe4b59f12d':raise ValueError('Verified original score cache differs')
        self.original_cache={int(row['case_index']):row for row in read(cachepath)['original_request_scores']}
        admission=read(D/'root_launch_admission.json');binding=admission['handoff']
        if sha(binding['path'])!=binding['sha256']:raise ValueError('Root cutoff inventory differs')
        self.handoff=read(binding['path']);self.source_hashes[str(Path(binding['path']))]=binding['sha256']
        database=Path(self.plan['baseline_score_index']['path'])
        if sha(database)!=self.plan['baseline_score_index']['sha256']:raise ValueError('Immutable1000 baseline index differs')
        self.input_definition=read(OUT/'input_definition.json')
        expected={'scientific_plan_sha256':sha(D/'plan.json'),'calendar_mapping_sha256':sha(self.mapping),'baseline_index_sha256':self.plan['baseline_score_index']['sha256'],'legacy_original_cache_sha256':sha(cachepath)}
        if self.input_definition['bindings']!=expected or self.input_definition['input_definition_digest']!=digest(expected):raise ValueError('Input definition differs')
        for path in (OUT/'protocol.json',OUT/'input_definition.json'):self.source_hashes[str(path)]=sha(path)
        protocol=read(OUT/'protocol.json')
        if protocol['protocol_id']!=scorer.SCORE_PROTOCOL_ID or protocol['scientific_plan_fingerprint']!=self.plan['fingerprint']:raise ValueError('External protocol differs')
        lookup={(c['series_index'],c['origin_index']):i for i,c in enumerate(self.cases)};self.baselines={};needed={int(i)for i in self.handoff['saved_old_slots']}
        with sqlite3.connect(database.as_uri()+'?mode=ro&immutable=1',uri=True)as db:
            for model,si,oi,task,protocol,panel_id,score in db.execute('SELECT model_id,series_index,origin_index,task_key,protocol_id,panel_fingerprint,score_v7 FROM task_scores_v7'):
                index=lookup[si,oi]
                if index not in needed:continue
                case=self.cases[index]
                if model not in (WR,WOR)or protocol!=self.plan['score_protocol']or panel_id!=self.plan['panel_fingerprint']or json.loads(task)!=[case['series_id'],case['origin']]:raise ValueError('Baseline exact task/protocol differs')
                self.baselines[model,index]=float(score)
        if len(self.baselines)!=2*len(needed):raise ValueError('Missing matched original baseline score')
        self.cache=sqlite3.connect(OUT/'verified_live_cache.sqlite3');self.cache.execute('CREATE TABLE IF NOT EXISTS pairs(path TEXT PRIMARY KEY, binding TEXT NOT NULL, signature TEXT NOT NULL, payload TEXT NOT NULL)')
        self.cache_hits=0;self.hash_reads=0

    def load_pair(self,cp,rp,kind,index,count,bound=None):
        cp=Path(cp);rp=Path(rp);key=str(cp.resolve());stat=[signature(cp),signature(rp)];binding={'kind':kind,'index':index,'count':count,'bound':bound,'plan':self.plan['fingerprint']};binding=canonical(binding)
        found=self.cache.execute('SELECT binding,signature,payload FROM pairs WHERE path=?',(key,)).fetchone()
        if found:
            if found[0]!=binding or json.loads(found[1])!=stat:raise ValueError('Previously verified immutable checkpoint changed: '+key)
            self.cache_hits+=1;return json.loads(found[2])
        cp_hash=sha(cp);rp_hash=sha(rp);self.hash_reads+=1
        if bound and(cp_hash!=bound['checkpoint_sha256']or rp_hash!=bound['record_sha256']):raise ValueError('Inherited checkpoint hash differs')
        side=read(rp);case=self.cases[index]
        if kind=='original':
            cached=self.original_cache[index]
            if cp_hash!=cached['checkpoint_sha256']or rp_hash!=cached['record_sha256']or cached['origin']!=case['origin']or cached['series_id']!=case['series_id']:raise ValueError('Verified original cache binding differs')
            if side['identity']['outer_paths']!=1000:raise ValueError('Original count differs')
            items=[{'H':case['origin']['max_horizon'],'scores':{'endpoint':cached['candidate_q_v7'],WR:self.baselines[WR,index],WOR:self.baselines[WOR,index]}}]
            if cached['fixed_WR_q_v7']!=self.baselines[WR,index]:raise ValueError('Cached WR score differs')
        elif kind=='short':
            physical=cp.parent.parent.resolve();owner=self.parents.get(str(physical))
            if owner is None or side['plan']!=owner['fingerprint']or side['index']!=index or side['checkpoint_sha256']!=cp_hash:raise ValueError('Short checkpoint source identity differs')
            with np.load(cp,allow_pickle=False)as z:
                raw=zlib.decompress(z['metadata_zlib'].tobytes());meta=json.loads(raw)
                if hashlib.sha256(raw).hexdigest()!=side['metadata_sha256']:raise ValueError('Short metadata hash differs')
                if set(z.files)!={'metadata_zlib',*(f'{m}_{H}'for m in MODELS for H in HORIZONS)}:raise ValueError('Short rawloss coverage differs')
                for name in z.files:
                    if name=='metadata_zlib':continue
                    loss=z[name];H=int(name.rsplit('_',1)[1])
                    if loss.shape!=(H,)or loss.dtype!=np.dtype('float64')or not np.isfinite(loss).all()or np.any(loss<0):raise ValueError('Short rawloss invalid')
            if meta['errors']or meta['identity']['plan']!=owner['fingerprint']or meta['identity']['case']!=case or meta['identity']['paths']!=count or meta['identity']['inner_paths']!=2:raise ValueError('Short full request identity differs')
            results=meta['results']
            if len(results)!=12 or {(r['model'],r['H'])for r in results}!={(m,H)for m in MODELS for H in HORIZONS}:raise ValueError('Short paired model results differ')
            items=[]
            for H in HORIZONS:
                chosen=[r for r in results if r['H']==H]
                if any(r.get('final_paths',count)!=count for r in chosen):raise ValueError('Short paths unpaired')
                items.append({'H':H,'scores':{r['model']:r['score']for r in chosen}})
        else:
            owner=self.parents[str(cp.parent.parent.resolve())]
            if side['checkpoint_sha256']!=cp_hash or side['record_digest']!=digest({k:v for k,v in side.items()if k!='record_digest'})or side['identity']['implementation_plan']!=owner['fingerprint']or side['identity']['case']!=case or side['identity']['paths']!=count:raise ValueError('Long checkpoint identity differs')
            if kind=='parent_long':items=[{'H':case['origin']['max_horizon'],'scores':{'endpoint':side['score'],WR:self.baselines[WR,index],WOR:self.baselines[WOR,index]}}]
            else:
                if set(side['results'])!=set(MODELS)or any(r['final_paths']!=240 for r in side['results'].values()):raise ValueError('New originalH count/models differ')
                with np.load(cp,allow_pickle=False)as z:
                    if set(z.files)!=set(MODELS):raise ValueError('New originalH rawloss models differ')
                    for model in MODELS:
                        loss=z[model]
                        if loss.shape!=(case['origin']['max_horizon'],)or loss.dtype!=np.dtype('float64')or not np.isfinite(loss).all()or np.any(loss<0)or hashlib.sha256(loss.tobytes()).hexdigest()!=side['results'][model]['loss_sha256']:raise ValueError('New originalH rawloss differs')
                items=[{'H':case['origin']['max_horizon'],'scores':{m:side['results'][m]['score']for m in MODELS}}]
        payload={'index':index,'paths':count,'kind':kind,'items':items,'checkpoint_sha256':cp_hash,'record_sha256':rp_hash}
        if [signature(cp),signature(rp)]!=stat:raise ValueError('Checkpoint changed while read')
        self.cache.execute('INSERT INTO pairs VALUES(?,?,?,?)',(key,binding,canonical(stat),canonical(payload)));return payload

    def snapshot(self):
        cutoff=time.time_ns();self.cache_hits=0;self.hash_reads=0;payloads=[];known_short=set();known_old=set()
        for text,bound in self.handoff['saved_old_slots'].items():
            index=int(text);payloads.append(self.load_pair(bound['checkpoint_path'],bound['record_path'],bound['kind'],index,1000,bound));known_old.add(index)
        for text,bound in self.handoff['inherited_short_groups'].items():
            index=int(text);payloads.append(self.load_pair(bound['checkpoint_path'],bound['record_path'],'short',index,1000,bound));known_short.add(index)
        for folder,kind,known in ((D/'groups','short',known_short),(D/'new_long_groups','new_long',known_old)):
            for rp in sorted(folder.glob('*.json')):
                if not rp.stem.isdecimal():continue
                cp=rp.with_suffix('.npz')
                if not cp.exists()or max(cp.stat().st_mtime_ns,rp.stat().st_mtime_ns)>cutoff:continue
                index=int(rp.stem)
                if index in known:raise ValueError('Saved request recomputed in new generation')
                known.add(index);payloads.append(self.load_pair(cp,rp,kind,index,240))
        self.cache.commit();records={m:[]for m in MODELS};seen=set();count_audit=collections.Counter();bucket_counts=collections.Counter();generation=collections.Counter();commit=hashlib.sha256()
        commit.update((self.input_definition['bindings']['legacy_original_cache_sha256']+'\n').encode())
        for payload in payloads:
            index=payload['index'];case=self.cases[index];short=payload['kind']=='short';count=payload['paths'];generation[payload['kind']]+=1
            for item in payload['items']:
                H=item['H'];key=('short'if short else'original',index,H)
                if key in seen:raise ValueError('Duplicate scored request')
                seen.add(key)
                if short:
                    column=HORIZONS.index(H);end=int(self.shortends[index][column]);bucket='short'
                else:
                    mapped=self.original[index]
                    if int(mapped[0])!=index or int(mapped[4])!=H:raise ValueError('Calendar mapping original case differs')
                    end=int(mapped[6]);bucket=scorer.BUCKETS[int(mapped[7])]
                if set(item['scores'])!=set(MODELS):raise ValueError('Incomplete matched model triple')
                count_audit[str(count)]+=1;bucket_counts[bucket]+=1
                for model,value in item['scores'].items():
                    if not np.isfinite(value)or value<0:raise ValueError('Invalid matched score')
                    records[model].append({'series_id':case['series_id'],'origin_date':case['origin']['origin_date'],'requested_end_date':str(np.datetime64(end,'D')),'horizon_bucket':bucket,'score':value})
            commit.update((canonical(payload)+'\n').encode())
        results={m:scorer.aggregate_records(rows,self.metadata)for m,rows in records.items()};scores={m:r['score']for m,r in results.items()};present=all(r['complete']for r in results.values());reference=scores[WR]
        out={'score_protocol_id':scorer.SCORE_PROTOCOL_ID,'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cutoff_utc':datetime.datetime.fromtimestamp(cutoff/1e9,datetime.timezone.utc).isoformat(),'complete':len(seen)==self.plan['combined_requests_per_model']and present,'headline_available':present,'preliminary':True,'official':False,'scope':'All committed completed request triples at this cutoff, no completed-original-prefix restriction;240and1000 mixed with equal count within each request. External v8 only; forecasts unchanged.','paired_requests_per_model':len(seen),'expected_requests_per_model':self.plan['combined_requests_per_model'],'requests_by_final_paths':dict(count_audit),'requests_by_bucket':dict(bucket_counts),'physical_generation_counts':dict(generation),'models':results,'scores':scores,'percent_better_than_fixed_WR':100*(1-scores['endpoint']/reference)if present and reference else None,'same_count_within_every_pair':True,'models_in_every_request':list(MODELS),'mapping_sha256':self.audit['mapping']['sha256'],'checkpoint_commitment':commit.hexdigest(),'source_hashes':self.source_hashes,'verified_cache':{'hits':self.cache_hits,'new_physical_pairs_hashed':self.hash_reads,'path':str(OUT/'verified_live_cache.sqlite3')},'missing_buckets':{m:r['missing_buckets']for m,r in results.items()}}
        terminal_path=D/'terminal.json';terminal=read(terminal_path)if terminal_path.exists()else{}
        out.update(plan_fingerprint=self.plan['fingerprint'],terminal_sha256=sha(terminal_path)if terminal_path.exists()else None,input_definition_digest=self.input_definition['input_definition_digest'],input_record_digest=digest({'input_definition_digest':self.input_definition['input_definition_digest'],'checkpoint_commitment':out['checkpoint_commitment'],'requests_per_model':len(seen)}),scorer_sha256=self.source_hashes[str(Path(scorer.__file__))],protocol_sha256=self.source_hashes[str(OUT/'protocol.json')],requests_by_model={m:len(rows)for m,rows in records.items()},forecast_coverage_complete=out['complete'])
        out['complete']=out['complete']and terminal.get('status')=='complete'and terminal.get('workers_terminal')is True
        out.update(legacy_original_cache_sha256=self.input_definition['bindings']['legacy_original_cache_sha256'],input_definition_components=self.input_definition['bindings'],preliminary=not out['complete'],official=out['complete'],incomplete_buckets={m:r['incomplete_buckets']for m,r in results.items()},adaptive_development_disclosure='Full coverage is a completed descriptive research score, not an untouched independent holdout or automatic production-model promotion')
        for path,expected in self.source_hashes.items():
            if sha(path)!=expected:raise ValueError('Reporter/scorer/mapping/plan binding changed')
        atomic(OUT/'scoreboard.json',out);atomic(O/'authoritative_current_status_latest.json',out);atomic(O/'final240_current_status_latest.json',out);print(json.dumps({'paired_requests_per_model':len(seen),'headline_available':present,'scores':scores,'percent_better_than_fixed_WR':out['percent_better_than_fixed_WR'],'complete':out['complete'],'cache':out['verified_cache']}),flush=True);return out

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--once',action='store_true');parser.add_argument('--watch',type=float,metavar='SECONDS');args=parser.parse_args()
    if args.watch is not None and args.watch<=0:parser.error('--watch interval must be positive')
    os.nice(max(0,15-os.getpriority(os.PRIO_PROCESS,0)))
    OUT.mkdir(exist_ok=True)
    with (OUT/'reporter.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);reporter=Reporter();fcntl.flock(lock,fcntl.LOCK_UN)
    while True:
        with (OUT/'reporter.lock').open('a')as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);result=reporter.snapshot()
        if args.watch is None or result['complete']:return
        time.sleep(args.watch)
if __name__=='__main__':main()
