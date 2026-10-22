"""Read-only V9 raw-CRPS sidecar; preparing this file does not launch a watch."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[k]='1'
import argparse,collections,datetime,fcntl,json,sqlite3,time
from pathlib import Path
import numpy as np
import horizon_buckets_v8_reporter as verified
import score_series_buckets_v9 as scorer
O=Path(__file__).resolve().parent;OUT=O/'series_buckets_v9_live';RAW=O/'external_raw_bucket_diagnostic_v1'
MODELS=verified.MODELS
PROTOCOL=O/'series_buckets_v9/protocol.json'
PROTOCOL_SHA='a1afd47e29015997c541b8a1d2455173e2353471f49467fed689abdd44dcd5be'
LEDGER_SHA={verified.WR:'baf29c83bf04b22714f4f21ecc1aec5dec3b840ee6947a3b5c5ee7cb76dcf334',verified.WOR:'dc7d692a567ac401190b41891fa2dcc801070c97d769de64b0a0e847e00de587'}

def validate_ledger(path,receipt,model):
    actual=verified.sha(path)
    if actual!=LEDGER_SHA[model] or receipt['ledger_sha256']!=actual or receipt['model_id']!=model or receipt['request_count']!=24574:
        raise ValueError('Recovered ledger hash/model/count differs')
    return actual


def publish_admitted(out):
    marker=OUT/'main_score_admission.json'
    if not marker.exists():return False
    admission=verified.read(marker)
    expected={'schema':'headline-score-amendment-v9','required':True,
              'expected_requests_per_model':320017,'score_protocol_id':scorer.SCORE_PROTOCOL_ID,
              'protocol_sha256':out['protocol_sha256'],'scorer_sha256':out['scorer_sha256'],
              'reporter_sha256':verified.sha(__file__),
              'plan_fingerprint':out['scientific_plan_fingerprint'],
              'weight_plan_sha256':out['weight_plan_sha256']}
    if admission.get('required') is not True or any(admission.get(k)!=v for k,v in expected.items()):
        raise ValueError('V9 main-score admission differs from exact source/protocol/plan binding')
    verified.atomic(O/'authoritative_current_status_latest.json',out)
    verified.atomic(O/'final240_current_status_latest.json',out)
    return True


class Reporter:
    def __init__(self):
        OUT.mkdir(exist_ok=True)
        self.loader=verified.Reporter();self.plan=self.loader.plan
        if verified.sha(PROTOCOL)!=PROTOCOL_SHA:raise ValueError('External V9 protocol differs')
        protocol=verified.read(PROTOCOL);panel=verified.V/'inputs/panel.json'
        if (protocol['protocol_id']!=scorer.SCORE_PROTOCOL_ID or protocol['scientific_plan_sha256']!=verified.sha(verified.D/'plan.json') or protocol['mapping_sha256']!=verified.sha(self.loader.mapping) or protocol['panel_sha256']!=verified.sha(panel)):
            raise ValueError('Protocol plan/mapping/panel differs')
        self.bindings={str(p):verified.sha(p) for p in (Path(__file__),Path(scorer.__file__),Path(verified.__file__),RAW/'rows_manifest.json',RAW/'freeze.json',RAW/'frozen_pairs.jsonl.gz',PROTOCOL,panel)}
        manifest=verified.read(RAW/'rows_manifest.json')
        if verified.sha(RAW/'raw_request_rows.npz')!=manifest['rows_sha256']:raise ValueError('Frozen raw seed rows differ')
        self.bindings[str(RAW/'raw_request_rows.npz')]=manifest['rows_sha256']
        self.seed={}
        with np.load(RAW/'raw_request_rows.npz',allow_pickle=False) as z:
            for row,values in zip(z['requests'],z['raw_mean_crps']):
                i,H,N,b,end,short=map(int,row);self.seed[bool(short),i,H,N]=values.copy()
        self.ledgers={}
        for model,prefix in zip(MODELS[1:],('wr','wor')):
            path=RAW/'recovery'/f'{prefix}_original_raw_means.jsonl'
            receipt=path.with_name(path.stem+'.recovery_receipt.json')
            if not path.exists() or not receipt.exists():raise ValueError('Original raw recovery not ready: '+str(path))
            self.bindings[str(path)]=validate_ledger(path,verified.read(receipt),model);self.bindings[str(receipt)]=verified.sha(receipt)
            rows=[json.loads(line)for line in path.read_text().splitlines()]
            lookup={int(r['case_index']):r for r in rows}
            if len(lookup)!=24574 or len(lookup)!=len(rows):raise ValueError('Recovered raw task coverage differs')
            for index,bound in self.loader.handoff['saved_old_slots'].items():
                i=int(index);row=lookup[i];case=self.loader.cases[i]
                if row['model_id']!=model or row['series_id']!=case['series_id'] or row['origin_index']!=case['origin_index'] or row['H']!=case['origin']['max_horizon']:raise ValueError('Recovered baseline task identity differs')
            self.ledgers[model]=lookup
        planned=[]
        for i,case in enumerate(self.loader.cases):planned.append(self.row(i,False,int(self.loader.original[i,4])))
        for i in self.loader.shortends:
            for H in verified.HORIZONS:planned.append(self.row(i,True,H))
        if len(planned)!=self.plan['combined_requests_per_model']:raise ValueError('Full planned request count differs')
        self.weights=scorer.build_weights(planned,self.loader.metadata)
        self.cache=sqlite3.connect(OUT/'raw_cache.sqlite3')
        self.cache.execute('CREATE TABLE IF NOT EXISTS raw_pairs(path TEXT PRIMARY KEY, signature TEXT NOT NULL, checkpoint_sha256 TEXT NOT NULL, items TEXT NOT NULL)')
        self.previous=None

    def row(self,index,short,H):
        case=self.loader.cases[index]
        end=int(self.loader.shortends[index][verified.HORIZONS.index(H)])if short else int(self.loader.original[index,6])
        return {'request_id':json.dumps(['short'if short else'original',index,H],separators=(',',':')),
                'series_id':case['series_id'],'origin_date':case['origin']['origin_date'],
                'requested_end_date':str(np.datetime64(end,'D'))}

    def collect(self):
        loader=self.loader;cutoff=time.time_ns();self.cutoff_ns=cutoff;payloads=[];known={'short':set(),'original':set()}
        for field,kind in (('saved_old_slots',None),('inherited_short_groups','short')):
            for text,bound in loader.handoff[field].items():
                i=int(text);k=kind or bound['kind'];payloads.append((Path(bound['checkpoint_path']),Path(bound['record_path']),loader.load_pair(bound['checkpoint_path'],bound['record_path'],k,i,1000,bound)))
                known['short'if k=='short'else'original'].add(i)
        for folder,kind in ((verified.D/'groups','short'),(verified.D/'new_long_groups','new_long')):
            for rp in sorted(folder.glob('*.json')):
                if not rp.stem.isdecimal():continue
                cp=rp.with_suffix('.npz')
                if not cp.exists()or max(cp.stat().st_mtime_ns,rp.stat().st_mtime_ns)>cutoff:continue
                i=int(rp.stem);group='short'if kind=='short'else'original'
                if i in known[group]:raise ValueError('Duplicate scientific request generation')
                known[group].add(i);payloads.append((cp,rp,loader.load_pair(cp,rp,kind,i,240)))
        loader.cache.commit();return payloads

    def raw_items(self,cp,rp,payload):
        stat=verified.canonical([verified.signature(cp),verified.signature(rp)])
        found=self.cache.execute('SELECT signature,checkpoint_sha256,items FROM raw_pairs WHERE path=?',(str(cp),)).fetchone()
        if found:
            if found[:2]!=(stat,payload['checkpoint_sha256']):raise ValueError('Raw-cache source changed')
            return json.loads(found[2])
        i=payload['index'];short=payload['kind']=='short';N=payload['paths'];items=[]
        with np.load(cp,allow_pickle=False)as z:
            for item in payload['items']:
                H=item['H'];seed=self.seed.get((short,i,H,N));values=[]
                for j,model in enumerate(MODELS):
                    value=float(seed[j])if seed is not None else float('nan')
                    if not np.isfinite(value):
                        if not short and N==1000 and model in self.ledgers:value=float(self.ledgers[model][i]['raw_mean_crps'])
                        else:
                            key=f'{model}_{H}'if short else ('losses'if payload['kind'] in ('original','parent_long')else model)
                            loss=z[key]
                            if loss.dtype!=np.float64 or loss.shape!=(H,)or not np.isfinite(loss).all()or np.any(loss<0):raise ValueError('Invalid raw loss')
                            value=float(np.mean(loss))
                    if not np.isfinite(value)or value<0:raise ValueError('Missing/invalid raw request score; no cohort mask allowed')
                    values.append(value)
                items.append({'H':H,'values':values})
        if stat!=verified.canonical([verified.signature(cp),verified.signature(rp)]):raise ValueError('Raw checkpoint changed during read')
        self.cache.execute('INSERT INTO raw_pairs VALUES(?,?,?,?)',(str(cp),stat,payload['checkpoint_sha256'],verified.canonical(items)))
        return items

    def snapshot(self):
        begin=time.perf_counter()
        with (verified.OUT/'reporter.lock').open('a')as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);payloads=self.collect()
        rows={m:[]for m in MODELS};commit=[];counts=collections.Counter();branches=collections.defaultdict(float)
        for cp,rp,payload in payloads:
            i=payload['index'];short=payload['kind']=='short'
            for item in self.raw_items(cp,rp,payload):
                row=self.row(i,short,item['H']);counts[str(payload['paths'])]+=1
                for m,value in zip(MODELS,item['values']):
                    if value is not None:rows[m].append(dict(row,score=value))
            commit.append([str(cp),payload['checkpoint_sha256'],payload['record_sha256']])
        self.cache.commit()
        if any([r['request_id']for r in rows[m]]!=[r['request_id']for r in rows[MODELS[0]]]for m in MODELS):raise ValueError('Three-model request coverage differs')
        if (OUT/'raw_cache.sqlite3').stat().st_size>.25*1024**3:raise ValueError('Scoped raw cache exceeds .25GiB task budget')
        results={m:scorer.aggregate_records(r,self.loader.metadata,weights=self.weights)for m,r in rows.items()}
        for r in rows[MODELS[0]]:
            e=self.weights['requests'][r['request_id']];branch=self.loader.metadata[r['series_id']]['panel_branch'];branches[e['bucket']+':'+branch]+=e['weight']
        scores={m:r['partial_score']for m,r in results.items()};ref=scores[verified.WR]
        terminal=verified.D/'terminal.json';term=verified.read(terminal)if terminal.exists()else{}
        coverage=all(r['complete']for r in results.values());complete=coverage and term.get('status')=='complete'and term.get('workers_terminal')is True
        gain=100*(1-scores['endpoint']/ref)if ref and scores['endpoint']is not None else None
        out={'score_protocol_id':scorer.SCORE_PROTOCOL_ID,'normalization':'raw daily mean CRPS; no train-SD/native-lead normalization','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cutoff_utc':datetime.datetime.fromtimestamp(self.cutoff_ns/1e9,datetime.timezone.utc).isoformat(),'fixed_reference':verified.WR,'scorer_sha256':verified.sha(scorer.__file__),'protocol_sha256':PROTOCOL_SHA,'quarter_weights':False,'same_count_within_every_pair':True,'requests_by_bucket':{b:results[MODELS[0]]['buckets'][b]['observed_requests']for b in scorer.BUCKETS},'official':complete,'provisional':not complete,'cache_policy':{'budget_gib':.25,'expiry':'Remove derived raw cache after owning full-sweep task closure; preserve sole-copy source evidence'},'complete':complete,'forecast_coverage_complete':coverage,'final_scores':{m:scores[m]if complete else None for m in MODELS},'partial_scores':scores,'final_percent_better_than_fixed_WR':gain if complete else None,'partial_percent_better_than_fixed_WR':gain,'same_three_model_coverage':len({len(r)for r in rows.values()})==1,'requests_by_model':{m:len(r)for m,r in rows.items()},'pending_original_WOR_recovery':verified.WOR not in self.ledgers,'planned_requests_per_model':self.plan['combined_requests_per_model'],'paired_requests_per_model':len(rows[MODELS[0]]),'requests_by_final_paths':dict(counts),'observed_weight_support_by_bucket_branch':dict(branches),'models':{m:{k:v for k,v in r.items()if k!='observed_request_ids'}for m,r in results.items()},'weight_plan_sha256':self.weights['weight_plan_sha256'],'source_hashes':dict(self.bindings,**self.loader.source_hashes),'input_record_digest':verified.digest({'records':commit,'source_hashes':dict(self.bindings,**self.loader.source_hashes),'weight_plan':self.weights['weight_plan_sha256']}),'terminal_sha256':verified.sha(terminal)if terminal.exists()else None,'scientific_plan_fingerprint':self.plan['fingerprint'],'snapshot_elapsed_seconds':time.perf_counter()-begin,'posthoc_selection_disclosure':'Raw CRPS and no-quarter fixed hierarchy chosen posthoc after inspecting alternatives; adaptive development, not untouched holdout. Partial support changes can move the conditional headline; final result unknown.','partial_score_change_since_previous_snapshot':None if self.previous is None else {m:scores[m]-self.previous[m]if scores[m]is not None and self.previous[m]is not None else None for m in MODELS}}
        for path,expected in self.bindings.items():
            if verified.sha(path)!=expected:raise ValueError('Frozen reporter input/source changed')
        publish_admitted(out)
        verified.atomic(OUT/'scoreboard.json',out);self.previous=scores
        print(json.dumps({k:out[k]for k in ('paired_requests_per_model','partial_scores','partial_percent_better_than_fixed_WR','complete','snapshot_elapsed_seconds')}),flush=True)
        return out


def main():
    parser=argparse.ArgumentParser(description=__doc__);g=parser.add_mutually_exclusive_group();g.add_argument('--once',action='store_true');g.add_argument('--watch',type=float,metavar='SECONDS');args=parser.parse_args()
    if args.watch is not None and args.watch<=0:parser.error('Positive watch interval required')
    if os.getpriority(os.PRIO_PROCESS,0)!=0:raise ValueError('Launch at normal nice 0; Background/niced watcher unsupported')
    OUT.mkdir(exist_ok=True)
    with (OUT/'reporter.lock').open('a')as lock:
        fcntl.flock(lock,fcntl.LOCK_EX);reporter=Reporter()
    while True:
        with (OUT/'reporter.lock').open('a')as lock:
            fcntl.flock(lock,fcntl.LOCK_EX);result=reporter.snapshot()
        if args.watch is None or result['complete']:return
        time.sleep(args.watch)
if __name__=='__main__':main()
