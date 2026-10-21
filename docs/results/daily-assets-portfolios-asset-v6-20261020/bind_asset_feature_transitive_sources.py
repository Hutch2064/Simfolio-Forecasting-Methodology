"""Supplement the feature grid's direct manifest with already sealed transitive fit sources."""
from pathlib import Path
import json,hashlib,datetime
O=Path(__file__).resolve().parent;V=O/'asset_level_v6';C=V/'candidates/asset_feature_batch';sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();rows={};evidence=[]
for plan,base,prefix in [(V/'candidates/asset_mean_uncertainty/AR1_mean_screen100/plan.json',O,'asset_level_v6/candidates/asset_AR1_mean/'),(V/'candidates/asset_gjr/plan.json',V,'candidates/asset_gjr/')]:
 s=json.loads(plan.read_text());selected={k:v for k,v in s['source_hashes'].items()if k.startswith(prefix)and k.endswith('.py')};assert selected
 for k,h in selected.items():
  p=base/k;assert sha(p)==h;rows[str(p.relative_to(V))]=h
 evidence.append({'previous_frozen_plan':str(plan.relative_to(O)),'plan_sha256':sha(plan),'source_sealed_before_feature_screen':s.get('utc',json.loads((V/'candidates/asset_mean_uncertainty/AR1_mean_screen100/summary.json').read_text()).get('utc'))})
r={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'complete':True,'financial_source_changes':False,'scope':'The initial feature plan bound direct batch sources and the prior fading/mean law. AR1 and GJR transitive fit modules were separately sealed in earlier experiments; this supplement verifies those exact pre-existing bytes without rewriting any original feature plan or result.','source_files_relative_to_v6':rows,'prior_sealed_evidence':evidence,'publication_recovery_receipt_sha256':sha(O/'v6_candidate_source_public_recovery.json')}
(C/'transitive_source_receipt.json').write_text(json.dumps(r,indent=2,sort_keys=True)+'\n');print('Verified additional sealed transitive source files',len(rows))
