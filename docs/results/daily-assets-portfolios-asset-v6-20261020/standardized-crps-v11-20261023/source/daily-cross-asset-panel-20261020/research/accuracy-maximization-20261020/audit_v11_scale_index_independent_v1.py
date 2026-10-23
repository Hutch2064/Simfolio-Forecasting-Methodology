"""Read-only V11 scale-index identity/weight audit; no forecasts or comparative losses."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
import collections,datetime,hashlib,json,math,sqlite3
from pathlib import Path
import numpy as np
O=Path(__file__).resolve().parent;D=O/'requested_horizon_crps_v11';OUT=O/'requested_horizon_crps_v11_independent_scale_index_review.json'
def load(p):return json.loads(Path(p).read_text())
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb')as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def bucket(origin,end):
    d=datetime.date.fromisoformat(origin);e=datetime.date.fromisoformat(end)
    def anniversary(y):
        try:return d.replace(year=d.year+y)
        except ValueError:return d.replace(year=d.year+y,day=28)
    assert e>d
    return 'short'if e<=anniversary(3)else'long'if e>=anniversary(20)else'medium'
def main():
    assert not OUT.exists()
    db=D/'scale_index.sqlite3';rr=D/'scale_index_receipt.json';assert sha(db)=='f64d20747ef5ae8718feb7545fc7947e1a78c176f60526b2148f670731fb7278';assert sha(rr)=='c06ed725929778e6b4c887953c018345dfe339392e1d7cf11b69f07916dd63e4'
    receipt=load(rr);assert receipt['status']=='complete'and receipt['index_sha256']==sha(db)and receipt['all_scales_positive_finite']is True
    assert receipt['protocol_sha256']==sha(D/'protocol.json')=='0472072ef517368d73bd581f11038de9f834e49efb39a5d0937bfe936ac27760'
    for path,h in receipt['source_hashes'].items():assert sha(path)==h
    qpath=O/'requested_horizon_crps_v11_independent_source_qualification.json';qproof=load(qpath);assert sha(qpath)=='2663d329f6abb7b402bd97f514ad87104580caa745cedacccde6de78dd01deeb'and qproof['passed']
    plan=load(O/'short_horizon_final240_v3/plan.json');panel=load(O/'asset_level_v6/inputs/panel.json');meta={r['series_id']:r for r in panel['series']};cases=plan['cases']
    with np.load(O/'horizon_buckets_v8/requested_calendar_mapping.npz',allow_pickle=False)as m:
        originals=m['original'].copy();shortcase=m['short_group_case_indices'].copy();shortH=m['short_horizons'].copy();shortends=m['short_requested_end_days'].copy()
    short_lookup={int(c):i for i,c in enumerate(shortcase)};hlookup={int(h):j for j,h in enumerate(shortH)}
    con=sqlite3.connect(f'file:{db}?mode=ro&immutable=1',uri=True);con.row_factory=sqlite3.Row
    try:
        assert con.execute('PRAGMA integrity_check').fetchone()[0]=='ok';prefix={r['case_index']:dict(r)for r in con.execute('SELECT * FROM prefix_scales')};assert len(prefix)==64049
        for i,r in prefix.items():
            c=cases[i];assert (r['series_id'],r['series_index'],r['origin_index'],r['origin_date'],r['origin_position'],r['training_n'])==(c['series_id'],c['series_index'],c['origin_index'],c['origin']['origin_date'],c['origin']['position'],c['origin']['position']+1)
            assert math.isfinite(r['training_sd'])and r['training_sd']>0 and r['exact_constant']==0
        for r in qproof['domain_prefix_replays']:
            assert prefix[r['case_index']]['training_sd']==r['training_portfolio_sd']
        trees={b:{}for b in ('short','medium','long')};counts=collections.Counter();kindcount=collections.Counter();bcount=collections.Counter();weight_terms=collections.defaultdict(list);request_rows=0;scalemin=math.inf
        for raw in con.execute('SELECT * FROM request_scales'):
            r=dict(raw);i=r['case_index'];c=cases[i];kind,idx,h=json.loads(r['request_id']);assert i==idx and h==r['native_h']and kind==r['request_kind']
            assert (r['series_id'],r['origin_index'],r['origin_date'])==(c['series_id'],c['origin_index'],c['origin']['origin_date'])
            assert r['training_sd']==prefix[i]['training_sd']and math.isfinite(r['requested_scale'])and r['requested_scale']>0 and r['requested_scale']==r['training_sd']*math.sqrt(h)
            if kind=='original':assert h==c['origin']['max_horizon'];day=int(originals[i,6])
            else:assert kind=='short';day=int(shortends[short_lookup[i],hlookup[h]])
            assert r['requested_end_date']==str(np.datetime64(day,'D'));b=bucket(r['origin_date'],r['requested_end_date']);assert b==r['horizon_bucket']
            assert r['global_weight']==r['within_bucket_weight']/3 and r['within_bucket_weight']>0
            label=meta[r['series_id']];trees[b].setdefault(label['panel_branch'],{}).setdefault(label['asset_class'],{}).setdefault(label['family'],set()).add(r['series_id'])
            counts[b,r['series_id']]+=1;kindcount[kind]+=1;bcount[b]+=1;weight_terms[b].append(r['global_weight']);request_rows+=1;scalemin=min(scalemin,r['requested_scale'])
        assert request_rows==320017 and kindcount=={'original':64049,'short':255968}and dict(bcount)==receipt['request_count_by_bucket']
        max_weight_relative_difference=0.;weight_map={}
        for r in con.execute('SELECT request_id,series_id,horizon_bucket,within_bucket_weight FROM request_scales'):
            b,sid=r['horizon_bucket'],r['series_id'];m=meta[sid];classes=trees[b][m['panel_branch']];families=classes[m['asset_class']];members=families[m['family']]
            expected=1/len(trees[b])/len(classes)/len(families)/len(members)/counts[b,sid];observed=r['within_bucket_weight'];err=abs(expected-observed)/expected;max_weight_relative_difference=max(err,max_weight_relative_difference);assert err<=16*np.finfo(float).eps
            weight_map[r['request_id']]={'bucket':b,'series_id':sid,'weight':observed}
        weightsha=hashlib.sha256(json.dumps(weight_map,sort_keys=True,separators=(',',':')).encode()).hexdigest();assert weightsha==receipt['weight_plan_sha256']=='207ee6316df2a0d622062e56c0ba5e9697b38f387458015db031250bc20b1132'
        masses={b:math.fsum(v)for b,v in weight_terms.items()};assert all(abs(m-1/3)<=16*np.finfo(float).eps for m in masses.values())
    finally:con.close()
    result={'passed':True,'audit_source_sha256':sha(Path(__file__)),'scale_index_path':str(db),'scale_index_sha256':sha(db),'scale_index_receipt_sha256':sha(rr),'protocol_sha256':sha(D/'protocol.json'),'prior_independent_source_qualification_sha256':sha(qpath),'all_prefix_identity_positive_scales_verified':len(prefix),'all_request_identity_calendar_native_H_scale_and_weight_bindings_verified':request_rows,'request_counts_by_kind':dict(kindcount),'request_counts_by_bucket':dict(bcount),'three_independent_Decimal_prefix_replays_match_index_exactly':True,'weights_independently_reconstructed_from_full_metadata_tree':True,'weight_plan_sha256':weightsha,'maximum_weight_relative_difference':max_weight_relative_difference,'global_weight_mass_by_bucket':masses,'minimum_requested_scale':scalemin,'sqlite_integrity_check':'ok','comparative_scores_read':False,'financial_fits_QPs_or_forecasts':0,'canonical_or_public_edits':False,'scope':'Independent full scale-index/source/identity/weight review, not completed financial rescore review.'}
    OUT.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n');print(json.dumps({'passed':True,'output':str(OUT),'sha256':sha(OUT),'prefixes':len(prefix),'requests':request_rows}))
if __name__=='__main__':main()
