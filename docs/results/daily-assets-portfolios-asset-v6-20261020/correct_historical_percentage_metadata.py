"""Correct derived public percentage metadata; keep frozen scientific inputs intact."""
from pathlib import Path
import concurrent.futures,copy,datetime,fcntl,hashlib,json,math,os,shutil,subprocess,urllib.request
O=Path(__file__).resolve().parent;R=O/'repository';REL='docs/results/daily-assets-portfolios-asset-v6-20261020';D=R/REL
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def normalize(summary):
    result=copy.deepcopy(summary);reference=summary['reference_main_Q']['fixed_reference'];changes=[]
    for mid,model in result['models'].items():
        q=model.get('main_Q')
        if q is None:continue
        expected=100*(1-q/reference)
        for key,value in list(model.items()):
            if key.startswith('percent_better_vs_')and key.endswith('fixed_reference')and isinstance(value,(int,float))and not math.isclose(value,expected,rel_tol=1e-10,abs_tol=1e-10):
                model[key]=expected;changes.append({'model_id':mid,'field':key,'original':value,'corrected':expected,'main_Q':q,'reference_main_Q':reference})
    for key,value in result.items():
        if isinstance(value,list)and key.endswith('ranking'):
            for model in value:
                for change in changes:
                    if model.get('model_id')==change['model_id']:model[change['field']]=change['corrected']
    assert result['reference_main_Q']==summary['reference_main_Q']
    assert all(result['models'][mid]['main_Q']==model['main_Q']for mid,model in summary['models'].items())
    return result,changes

def main():
    lock=(O/'asset_v6_publication.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
    run=lambda args:subprocess.check_output(args,cwd=R,env={**os.environ,'GODEBUG':'http2client=0'},text=True)
    assert not run(['git','status','--porcelain']).strip();run(['git','fetch','origin','main']);run(['git','merge','--ff-only','origin/main'])
    changes=[];changed=[];checked=0
    for p in sorted(D.rglob('*summary*.json')):
        if 'original_percentage_metadata' in p.name:continue
        summary=read(p);reference=summary.get('reference_main_Q',{}).get('fixed_reference')
        if not isinstance(reference,(int,float))or reference<=0 or not summary.get('models'):continue
        checked+=sum(isinstance(m.get('main_Q'),(int,float))for m in summary['models'].values())
        corrected,issues=normalize(summary)
        if not issues:continue
        original=p.with_name(p.stem+'.original_percentage_metadata.json');assert not original.exists();shutil.copy2(p,original)
        corrected['percentage_metadata_erratum']='PERCENTAGE_METADATA_ERRATA.json at the study root; original generated summary retained alongside this corrected compact copy. Frozen paths/seed/loss/source artifacts unchanged.'
        corrected['comparison_simulation_count']=summary['paths']
        p.write_text(json.dumps(corrected,indent=2)+'\n');changed.extend([str(p.relative_to(D)),str(original.relative_to(D))]);changes.append({'path':str(p.relative_to(D)),'original_compact_sha256':sha(original),'corrected_compact_sha256':sha(p),'paths':summary['paths'],'changes':issues})
    assert len(changes)==5 and sum(len(c['changes'])for c in changes)==13,'Review unexpected scope rather than broadening this correction silently'
    receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'checked_models':checked,'affected_compact_summaries':len(changes),'affected_unique_model_stage_results':13,'cause':'Three historical driver versions multiplied1-Qmodel/Qreference by1000 rather than100. Correct reporting is100*(1-Qmodel/Qreference).','frozen_scientific_sources_losses_and_local_summaries_unchanged':True,'main_Q_reference_Q_ranks_fitting_and_promotion_unchanged':True,'current_working_incumbent_already_used_correct_percentages':True,'field_label_caveat':'Historical asset_training_blocks field name says same100path even in1000path stages; declared paths/comparison_simulation_count are authoritative. Existing key names are retained for compatibility.','changes':changes}
    f=D/'PERCENTAGE_METADATA_ERRATA.json';f.write_text(json.dumps(receipt,indent=2)+'\n');changed.append(f.name)
    shutil.copy2(Path(__file__),D/Path(__file__).name);changed.append(Path(__file__).name)
    manifest={str(p.relative_to(D)):sha(p)for p in sorted(D.rglob('*'))if p.is_file()and p.name!='PUBLIC_FILES.json'and '__pycache__'not in p.parts};(D/'PUBLIC_FILES.json').write_text(json.dumps(manifest,indent=2)+'\n')
    for line in run(['git','status','--porcelain']).splitlines():assert line[3:].startswith(REL+'/')
    run(['git','add','--',REL]);run(['git','commit','-m','Correct historical percentage display scale while preserving original summaries']);run(['git','push','origin','HEAD:main']);commit=run(['git','rev-parse','HEAD']).strip();restore=O/'percentage_metadata_recovery'/commit[:12]
    def recover(name):
        with urllib.request.urlopen(f'https://raw.githubusercontent.com/Hutch2064/Simfolio-Forecasting-Methodology/{commit}/{REL}/{name}',timeout=60)as response:data=response.read()
        assert hashlib.sha256(data).hexdigest()==manifest[name];target=restore/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(data);assert sha(target)==manifest[name]
        return {'path':name,'sha256':manifest[name],'bytes':len(data)}
    with concurrent.futures.ThreadPoolExecutor(max_workers=4)as pool:files=list(pool.map(recover,changed))
    out={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'commit':commit,'remote_main_verified':run(['git','ls-remote','origin','refs/heads/main']).split()[0]==commit,'fresh_downloaded_disk_restored_sha256_verified':len(files),'files':files,'affected_summaries':5,'corrected_unique_model_stage_results':13,'frozen_scientific_sources_or_losses_changed':False,'current_reported_development_reserved_full_panel_percentages_unchanged':True}
    (O/'percentage_metadata_correction_receipt.json').write_text(json.dumps(out,indent=2)+'\n');print(json.dumps({k:v for k,v in out.items()if k!='files'},indent=2),flush=True)
if __name__=='__main__':main()
