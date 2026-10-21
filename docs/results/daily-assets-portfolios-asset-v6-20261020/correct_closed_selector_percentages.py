"""Produce a corrected report beside immutable original evidence, never rerun it."""
from pathlib import Path
import datetime,hashlib,json,sys
from score_percentage import normalize_model_percentages
O=Path(__file__).resolve().parent;D=Path(sys.argv[1]);read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
assert read(D/'active.json')['status']=='terminal'
raw=read(D/'summary.json');audit=read(D/'verification_receipt.json');assert audit['verified']and audit['summary_sha256']==sha(D/'summary.json')
corrected,changes=normalize_model_percentages(raw);assert changes,'No correction needed; leave original report unchanged'
assert raw['reference_main_Q']==corrected['reference_main_Q']and all(raw['models'][mid]['main_Q']==m['main_Q']for mid,m in corrected['models'].items())
receipt={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'scope':'Derived percentage display metadata only; original driver, summary, forecasts and canonical score audit remain immutable','cause':'Simulation-count text replacement also replaced the display multiplier100with1000 in this one-time precision driver','original_summary_sha256':sha(D/'summary.json'),'canonical_score_verification_receipt_sha256':sha(D/'verification_receipt.json'),'corrected_fields':changes,'normalizer_source_sha256':sha(O/'score_percentage.py'),'correction_source_sha256':sha(Path(__file__)),'main_Q_reference_Q_coverage_and_ranking_unchanged':True,'no_forecast_or_fitting_reruns':True}
corrected['percentage_metadata_correction']=receipt
target=D/'summary.corrected.json'
if target.exists():assert read(target)['percentage_metadata_correction']['original_summary_sha256']==receipt['original_summary_sha256']
target.write_text(json.dumps(corrected,indent=2)+'\n');receipt['corrected_summary_sha256']=sha(target);(D/'percentage_metadata_erratum.json').write_text(json.dumps(receipt,indent=2)+'\n');assert sha(D/'summary.json')==receipt['original_summary_sha256']
print(json.dumps({'correction':receipt,'models':corrected['models'],'reference_main_Q':corrected['reference_main_Q']},indent=2),flush=True)
