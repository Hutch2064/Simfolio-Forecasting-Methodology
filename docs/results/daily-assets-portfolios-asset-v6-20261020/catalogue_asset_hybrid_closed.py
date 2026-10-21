"""Catalogue the complete source-bound hybrid after verified remote recovery."""
from pathlib import Path
import datetime,hashlib,json,sys
O=Path(__file__).resolve().parent;V=O/'asset_level_v6';C=V/'candidates/asset_prior_best';F=C/'full';R=O/'repository'
sys.path.insert(0,str(R/'tools/full_panel_catalogue'))
import catalogue
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    archive=read(O/'archives/asset-v6-hybrid-1000-20261021/receipt.json')
    assert archive['complete'] and not archive['scientific_originals_removed']
    result=read(F/'verification_receipt.json');official=read(V/'official_scoreboard.json')
    assert result['complete'] and result['successful_tasks']==64049 and result['native_failures']==result['scoring_integrity_errors']==0
    mid=result['model_id'];spec=read(F/'control.json')['models'][mid]
    root=F/'results/models'/hashlib.sha256(mid.encode()).hexdigest();execution=read(root/'execution.json')
    reference=result['sole_reference_model_id'];qref=result['reference_main_score']
    panel=read(V/'inputs/panel.json')
    assert catalogue.retirement_allowed({'archives':archive['archives']},archive)
    definition=('Fixed equal mixture of raw joint history-partition blocks and per-constituent variance-targeted GARCH(1,1) filtered historical simulation. Shared historical innovation rows preserve joint empirical shocks. Both halves add jointly correlated weighted-batch Gaussian mean uncertainty and smooth training-only mean shrinkage. Actual asset paths, portfolio weights and endpoint rebalancing evolve inside each scenario; no outcome-fitted asset/horizon routing. All1000paths scored; no clipping or path exclusion.')
    record={'identity':{'model_id':mid,'implementation_digest':spec['implementation_digest']},
        'display_name':'Asset Equal Mix / Smooth Gaussian Mean Uncertainty','definition':definition,
        'version':'v1_asset_joint_1000_source_bound','forecast_level':'joint_assets','retained_in_active_set':mid in official['models'],
        'score_protocol_id':official['score_protocol_id'],'execution_fingerprint':execution['fingerprint'],
        'coverage':{'panel_fingerprint':panel['panel_fingerprint'],'series':1264,'expected_tasks':64049,'successful_tasks':64049,
                    'native_failed_tasks':0,'native_fit_failures':0,'native_generation_failures':0,'missing_tasks':0,'scoring_integrity_errors':0,'status':'complete'},
        'scores':{'main':result['main_score']},
        'comparison':{'reference_model_id':reference,'matched_successful_tasks':64049,'public_scores':{'main':qref},
                      'percent_better':{'main':100*(1-result['main_score']/qref)},'preliminary':False,
                      'reference_rule':'Fixed worse of complete paired asset-level TestFolio-style configurations; exact successful coverage and1000simulations.'},
        'runtime':{'forecast_and_score_worker_seconds':result['worker_compute_seconds'],'wall_elapsed_seconds':None,
                   'scope':'Recorded worker input preparation, per-asset fitting, simulation and exact empirical CRPS, including identity-verified reused pilot forecasts; excludes checkpointIO/setup/queue. No controlled speedup claim.'},
        'reproducibility':{'actual_factory':spec['factory'],'actual_implementation_digest':spec['implementation_digest'],
                           'full_execution_manifest_sha256':sha(root/'execution.json'),'full_execution_archive_member':str((root/'execution.json').relative_to(O)),
                           'source_environment':execution['identity']['runtime'],'input_manifest_sha256':sha(V/'inputs/panel.json'),
                           'paths_per_forecast':1000,'seed_contract':'coherent_daily_seed(origin_date,max_horizon,1000)',
                           'full_independent_audit_sha256':sha(F/'verification_receipt.json'),
                           'reused_checkpoint_manifest_sha256':sha(F/'admission.json'),
                           'evidence_root':'../daily-assets-portfolios-asset-v6-20261020',
                           'preservation_receipt':'../daily-assets-portfolios-asset-v6-20261020/closed-hybrid-archive-receipt.json',
                           'frozen_input_archive_release':archive['baseline_input_archive_release'],
                           'frozen_input_archive_receipt_sha256':archive['baseline_input_archive_receipt_sha256'],
                           'source_parent_dependencies':'Restore the verified V6 baseline shared/input archive and sealed numerical parents referenced in execution identity before candidate archive.'},
        'archives':archive['archives']}
    path=R/'docs/results/daily-cross-asset-722/catalogue.json';ledger=read(path)
    catalogue.append(ledger,record)
    ledger.update(current_score_protocol_id=official['score_protocol_id'],current_panel_fingerprint=panel['panel_fingerprint'],
                  active_model_ids=list(official['models']),current_best_model_id=official['current_best_model_id'],
                  current_best_report='../daily-assets-portfolios-asset-v6-20261020/CURRENT_RESEARCH.md',
                  updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  scope='Every completed unique full-panel implementation, including losers, remains preserved. Current retained set: two joint-asset references plus verified current best. Accuracy-maximization goal remains active.')
    path.write_text(json.dumps(ledger,indent=2)+'\n');(path.parent/'CATALOGUE.md').write_text(catalogue.render(ledger))
    print(json.dumps({'unique_full_panel_models':ledger['unique_full_panel_models'],'full_panel_executions':ledger['full_panel_executions'],'goal_complete':False}),flush=True)

if __name__=='__main__':main()
