"""Bind a prepared candidate to the better fully audited joint-asset baseline."""
from pathlib import Path
import sys,json,hashlib,datetime
O=Path(__file__).resolve().parent;V=O/'asset_level_v6';C=V/'candidates/asset_prior_best';sys.path.insert(0,str(V));import common,sweep_v6 as api
from candidates.asset_prior_best import candidate
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def main():
 audit=read(V/'full_pair_independent_audit.json');assert audit['complete_pair_audit']and audit['total_checkpoints_verified']==128098 and audit['native_failures']==audit['scoring_integrity_errors']==0
 assert not(C/'parent_selection.json').exists(),'Existing bound version must not be retuned'
 pre=read(C/'broad_training_preflight.json');assert not pre['failures']and pre['representative_prefixes']==710 and pre['successful_asset_fits']==1190
 for name,h in pre['source_hashes'].items():assert sha(V/name)==h
 assert read(V/'controller_active.json')['status']=='terminal'
 selection={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'both_reference_configurations_complete':True,'same_successful_tasks':64049,'better_model_id':audit['better_starting_model_id'],'worse_reference_model_id':audit['worse_reference_model_id'],'block_limits':{'minimum':1,'maximum':None},'full_pair_audit_sha256':sha(V/'full_pair_independent_audit.json'),'selection_scope':'Research architecture choice after full paired baseline comparison, not origin-level fitting on held-out outcomes'}
 api.original._atomic_json(C/'parent_selection.json',selection)
 plan=read(C/'plan.json');api.original._atomic_json(C/'plan.before_parent_binding.json',plan)
 plan.update(source_hashes={str(p.relative_to(V)):sha(p)for p in C.glob('*.py')},design_sha256=sha(C/'DESIGN.md'),parent_model_id=selection['better_model_id'],reference_model_id=selection['worse_reference_model_id'],parent_selection_sha256=sha(C/'parent_selection.json'))
 api.original._atomic_json(C/'plan.json',plan)
 api.original._atomic_json(C/'plan_receipt.json',{'utc':selection['utc'],'complete':True,'plan_sha256':sha(C/'plan.json'),'preflight_sha256':sha(C/'broad_training_preflight.json'),'parent_selection_sha256':sha(C/'parent_selection.json'),'field_forecasts_started':False})
 model=candidate.build();assert model.dependency_identity['forecast_level']=='joint_assets'
 print(json.dumps(selection),flush=True)
if __name__=='__main__':main()
