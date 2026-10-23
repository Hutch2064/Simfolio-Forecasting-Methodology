"""Independent prescribed V11 checks and bounded saved-loss/past-prefix replay."""
import os
for name in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','NUMBA_NUM_THREADS','VECLIB_MAXIMUM_THREADS','NUMEXPR_NUM_THREADS'):os.environ[name]='1'
import copy,decimal,hashlib,importlib.util,json,math,sys,unittest
from pathlib import Path
import numpy as np
O=Path(__file__).resolve().parent;D=O/'requested_horizon_crps_v11';OUT=O/'requested_horizon_crps_v11_independent_source_qualification.json'
sys.path[:0]=[str(D),str(O),str(O/'asset_level_v6')]
import score_requested_horizon_crps_v11 as v
from inputs import AssetInputs

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load(p):return json.loads(Path(p).read_text())
def module(p,name):
    s=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m

def fixture():
    rows=[];scales={};meta={};planned=[]
    for bi,years in enumerate((1,10,30)):
        factor=(1,2,4)[bi]
        for j,(sid,branch,cls,z,sd,h)in enumerate((('a1','assets','A',2,.5,4),('a1','assets','A',4,.25,9),('a2','assets','A',8,.125,16),('a3','assets','B',16,.2,25),('p1','portfolios','P',32,.1,36))):
            rid=f'{bi}:{j}';origin=f'{2000+j}-01-01';end=f'{2000+j+years}-01-01';s=sd*math.sqrt(h)
            r={'request_id':rid,'series_id':sid,'origin_date':origin,'requested_end_date':end,'case_index':bi*10+j,'origin_index':j,'requested_native_h':h,'H':h,'raw_mean_crps':factor*z*s}
            rows.append(r);planned.append({k:x for k,x in r.items()if k!='raw_mean_crps'})
            scales[rid]={'request_id':rid,'series_id':sid,'origin_date':origin,'requested_end_date':end,'case_index':r['case_index'],'origin_index':j,'native_h':h,'training_sd':sd}
            meta[sid]={'panel_branch':branch,'asset_class':cls,'family':'fixture'}
    weights=v._v9.build_weights(planned,meta)
    return rows,scales,meta,weights

def aggregate(rows,scales,meta,weights):return v.aggregate_records(rows,meta,weights=weights,scales_by_request=scales)

class IndependentFixtures(unittest.TestCase):
    def test_independent_unequal_hierarchy_request_weights(self):
        r,s,m,w=fixture();a=aggregate(r,s,m,w)
        # Per bucket: a1 averages2,4 at mass1/8; a2=8 at1/8;
        # a3=16 at1/4; p1=32 at1/2 =>21.375. Buckets multiply1,2,4.
        self.assertAlmostEqual(a['score'],49.875,places=13)
        self.assertAlmostEqual(a['buckets']['short']['partial_score'],21.375,places=13)
        self.assertAlmostEqual(a['buckets']['medium']['partial_score'],42.75,places=13)
        self.assertAlmostEqual(a['buckets']['long']['partial_score'],85.5,places=13)
    def test_order_invariance_and_scalar_vector_formula(self):
        r,s,m,w=fixture();a=aggregate(r,s,m,w);b=aggregate(list(reversed(r)),s,m,w)
        self.assertEqual(a['score'],b['score']);self.assertEqual(a['scale_data_commitment_sha256'],b['scale_data_commitment_sha256'])
        losses=[.25,1.,3.,4.];sd=.125;H=4
        self.assertEqual(v.standardized_request_crps(math.fsum(losses)/H,sd,H),math.fsum(x/(sd*math.sqrt(H))for x in losses)/H)
        self.assertNotEqual(v.standardized_request_crps(math.fsum(losses)/H,sd,H),math.fsum(x/(sd*math.sqrt(i+1))for i,x in enumerate(losses))/H)
    def test_different_scale_map_cannot_compare(self):
        r,s,m,w=fixture();a=aggregate(r,s,m,w);changed=copy.deepcopy(s);changed['0:0']['training_sd']*=2;b=aggregate(r,changed,m,w)
        with self.assertRaises(ValueError):v.compare(a,b)
    def test_wrong_end_native_H_fractional_indices_rejected(self):
        r,s,m,w=fixture()
        for field,value in (('requested_end_date','2001-02-01'),('requested_native_h',4.1),('requested_native_h',True),('H',5),('case_index',.9),('origin_index',False)):
            rr=copy.deepcopy(r);rr[0][field]=value
            with self.subTest(field=field,value=value),self.assertRaises((ValueError,TypeError)):aggregate(rr,s,m,w)
    def test_changed_origin_series_and_missing_request_rejected(self):
        r,s,m,w=fixture()
        for field,value in (('origin_date','1999-01-01'),('series_id','wrong')):
            rr=copy.deepcopy(r);rr[0][field]=value
            with self.assertRaises(ValueError):aggregate(rr,s,m,w)
        partial=aggregate(r[:-1],s,m,w)
        self.assertIsNone(partial['score'])
        with self.assertRaises(ValueError):v.compare(partial,partial)
        with self.assertRaises(ValueError):aggregate(r+[r[0]],s,m,w)
    def test_zero_rawloss_baseline_and_constant_scale(self):
        r,s,m,w=fixture();zero=[dict(x,raw_mean_crps=0.)for x in r];z=aggregate(zero,s,m,w);a=aggregate(r,s,m,w)
        self.assertEqual(z['score'],0.);self.assertIsNone(v.compare(a,z)['percent_better_than_baseline'])
        for sd in (0.,-1.,math.inf,math.nan):
            with self.assertRaises(ValueError):v.standardized_request_crps(0.,sd,1)
    def test_scale_units_horizon_and_self_reference(self):
        self.assertEqual(v.standardized_request_crps(3.,.5,4),3.)
        self.assertEqual(v.standardized_request_crps(3.,.5,16),1.5)
        self.assertAlmostEqual(v.standardized_request_crps(.3,.02,21),v.standardized_request_crps(30.,2.,21),places=14)
        r,s,m,w=fixture();a=aggregate(r,s,m,w);self.assertEqual(v.compare(a,a)['percent_better_than_baseline'],0.)

def replays():
    scanpath=O/'training_sd_domain_scan_v1/scan_training_sd_domain.py';scan=module(scanpath,'independent_scan_readonly')
    plan=load(O/'short_horizon_final240_v3/plan.json');inputs=AssetInputs(O/'asset_level_v6/inputs');report=[]
    for i in(35191,63540,61471):
        case=plan['cases'][i];row=inputs.by_id[case['series_id']];n=case['origin']['position']+1;x,w,_=inputs.training(row,n)
        values=x[:,0]if x.shape[1]==1 else np.log1p(np.sum(np.expm1(x)*w[None,:],axis=1))
        stats=scan.prefix_sd_by_stop(values,[n]);sd=stats[n][1]
        with decimal.localcontext()as ctx:
            ctx.prec=60;z=[decimal.Decimal.from_float(float(a))for a in values];avg=sum(z)/len(z);reference=(sum((a-avg)**2 for a in z)/(len(z)-1)).sqrt();exact=float(reference)
        rel=abs(sd-exact)/exact;bound=64*n*np.finfo(float).eps
        assert rel<=bound
        # Changing or appending strictly future data cannot alter a stored prefix.
        padded=np.concatenate((values,[1e50,-1e50]));assert scan.prefix_sd_by_stop(padded,[n])[n]==stats[n]
        report.append({'case_index':i,'series_id':case['series_id'],'origin_date':case['origin']['origin_date'],'training_rows':n,'assets':len(w),'training_portfolio_sd':sd,'decimal60_reference_sd':exact,'relative_difference':rel,'roundoff_comparison_bound':bound,'prefix_sha256':hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest(),'weights_sha256':hashlib.sha256(w.tobytes()).hexdigest(),'future_append_preserves_prefix_bytes':True})
    study=O/'finite_full_series_confirmation_v1';p=load(study/'plan.json');actual=[]
    for i in(0,1):
        q=p['requests'][i];case=plan['cases'][q['case_index']];row=inputs.by_id[q['series_id']];n=case['origin']['position']+1;x,w,_=inputs.training(row,n);values=x[:,0]if len(w)==1 else np.log1p(np.sum(np.expm1(x)*w[None,:],axis=1));sd=scan.prefix_sd_by_stop(values,[n])[n][1]
        if str(i)in p['reuse']:ref=p['reuse'][str(i)]['finite41'];cp=Path(ref['checkpoint']);rp=Path(ref['sidecar']);assert sha(cp)==ref['checkpoint_sha256']and sha(rp)==ref['sidecar_sha256']
        else:cp=study/'output'/f'{i:05d}-finite41.npz';rp=cp.with_suffix('.json')
        side=load(rp);assert sha(cp)==side['checkpoint_sha256']
        with np.load(cp,allow_pickle=False)as z:loss=z['loss'].copy()
        assert loss.shape==(q['H'],)and np.isfinite(loss).all()and np.all(loss>=0)and hashlib.sha256(loss.tobytes()).hexdigest()==side['loss_sha256']
        raw=float(np.mean(loss));assert raw==side['record']['score'];scalar=v.standardized_request_crps(raw,sd,q['H']);direct=math.fsum(float(a)/(sd*math.sqrt(q['H']))for a in loss)/q['H'];rel=abs(scalar-direct)/max(abs(scalar),abs(direct));assert rel<=32*q['H']*np.finfo(float).eps
        actual.append({'request':q,'checkpoint':str(cp),'checkpoint_sha256':sha(cp),'loss_sha256':side['loss_sha256'],'training_sd':sd,'raw_saved_mean':raw,'standardized_scalar':scalar,'independent_daily_vector_standardized_mean':direct,'relative_roundoff_difference':rel})
    return report,actual

def main():
    assert not OUT.exists()
    expected={'score_requested_horizon_crps_v11.py':'6fe6345747292142bb0599c1c3ec0a98b78cc39a54d7cbeb809ef7741218b52a','build_scale_index_v11.py':'659d0b259a60d513614c87c0ec6d0116e35e45c4e43578fe864f50b7d200e56b','protocol.json':'0472072ef517368d73bd581f11038de9f834e49efb39a5d0937bfe936ac27760'}
    for name,h in expected.items():assert sha(D/name)==h
    suite=unittest.defaultTestLoader.loadTestsFromTestCase(IndependentFixtures);result=unittest.TextTestRunner(verbosity=2).run(suite);assert result.wasSuccessful()
    prefixes,saved=replays();sources={str(D/name):h for name,h in expected.items()};sources[str(Path(__file__))]=sha(Path(__file__))
    result={'passed':True,'independent_prescribed_tests':result.testsRun,'domain_prefix_replays':prefixes,'actual_saved_loss_replays':saved,'source_hashes':sources,'domain_scan_sha256':sha(O/'training_sd_domain_scan_v1/scan_training_sd_domain.py'),'domain_receipt_sha256':sha(O/'training_sd_domain_scan_v1/domain_scan.json'),'source_review':'Exact training input-reader rational-constituent accounting, causal Welford prefixes, original/native-short mapped dates/H, unchanged frozen V9 hierarchy and outer1/3, shared consumed scale commitment and identity guards reviewed. Positive current inventory; zeroSD rejected without fallback. Historical files unchanged.','scientific_fits_QPs_or_forecasts_executed':0,'canonical_or_public_edits':False,'scope':'Independent source/formula/prescribed/three-prefix/two-saved-loss qualification, not full rescore audit or new financial accuracy admission.'}
    OUT.write_text(json.dumps(result,indent=2,sort_keys=True,allow_nan=False)+'\n');print(json.dumps({'passed':True,'receipt':str(OUT),'sha256':sha(OUT),'fixtures':result['independent_prescribed_tests'],'prefixes':len(prefixes),'saved_losses':len(saved)}))
if __name__=='__main__':main()
