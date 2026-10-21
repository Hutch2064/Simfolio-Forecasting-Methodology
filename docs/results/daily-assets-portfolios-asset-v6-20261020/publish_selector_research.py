"""Publish scoped selector research and active goal amendment, verify remote recovery."""
from pathlib import Path
import datetime,fcntl,hashlib,json,os,shutil,subprocess
O=Path(__file__).resolve().parent;R=O/'repository';REL='docs/results/daily-assets-portfolios-asset-v6-20261020';D=R/REL;V=O/'asset_level_v6';ENV={**os.environ,'GODEBUG':'http2client=0'}
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();read=lambda p:json.loads(p.read_text())
run=lambda args:subprocess.check_output(args,cwd=R,env=ENV,text=True)
lock=(O/'asset_v6_publication.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX)
assert not run(['git','status','--porcelain']).strip(),'Preserve unrelated edits'
run(['git','fetch','origin','main']);run(['git','merge','--ff-only','origin/main'])
assert read(O/'goal_selector_amendment_receipt.json')['after']['status']=='active'
for name in ['test_asset_SAS_skew.py','screen_asset_SAS_skew.py','screen_asset_SAS_skew_precision.py','preserve_asset_SAS_skew.py','test_asset_fast_selector.py','screen_asset_fast_selector.py','run_asset_fast_selector_development.py','benchmark_asset_fast_selector.py','verify_asset_fast_selector.py','preserve_asset_fast_selector.py','test_asset_uncapped_selector.py','smoke_asset_uncapped_selector.py','preserve_asset_uncapped_selector.py','goal_selector_amendment_receipt.json','goal_selector_amendment_proposed.txt','goal_objective_updated.txt','amend_goal_selector.py','publish_selector_research.py']:
    shutil.copy2(O/name,D/name)
for folder in ['asset_SAS_skew','asset_fast_selector','asset_uncapped_selector']:
    source=V/'candidates'/folder;dest=D/'candidates'/folder;dest.mkdir(exist_ok=True)
    for p in source.iterdir():
        if p.is_file()and(p.suffix in('.py','.md')or p.name in ['source_derivation.json','qualification_receipt.json','validation_receipt.json','decision.json','preflight_diagnostic_fix.json']):shutil.copy2(p,dest/p.name)
    for phase in ['screen100','screen1000','eligible70_1000','development1000','runtime','request_smoke']:
        s=source/phase/'summary.json'
        if s.exists():
            summary=read(s);summary.pop('records',None);(dest/(phase+'_summary.json')).write_text(json.dumps(summary,indent=2)+'\n')
        p=source/phase/'execution_receipt.json'
        if p.exists():shutil.copy2(p,dest/(phase+'_execution_receipt.json'))
for tag in ['asset-v6-SAS-skew-20261021','asset-v6-fast-selector-prototype-20261021','asset-v6-uncapped-selector-reference-20261021']:
    p=O/'archives'/tag/'receipt.json';assert read(p)['complete'];shutil.copy2(p,D/(tag+'-archive-receipt.json'))
(D/'SELECTOR_RESEARCH.md').write_text('''# Adaptive selector research,21October2026

The official active goal is now the adaptive model-selection algorithm, not a fixed universal configuration. See goal_objective_updated.txt and goal_selector_amendment_receipt.json. The objective is3978characters. The prior stopping and catalogue paragraphs are verbatim unchanged. Accuracy remains primary; no arbitrary accuracy-reducing caps, noMCMC, same frozen1264series/single main score/fixed worse baseline/1000official paths/12shared workers. Portfolio-level candidates are authorized only if measured asset-level runtime is infeasible. No production deployment or web-latency claim.

The initial19option bank comprised the current trained leader, both original joint block reference methods, and16pure joint-asset options: constant/revertingEWMA/GARCH/conditionalRMS latent volatility ×Gaussian/joint empirical innovations ×sample/shrunk mean, with existing correlated mean uncertainty. This is not the complete intended component universe. Further runtime-qualified mean, volatility/GARCH, innovation, dependence, block/PW, replacement, fading and uncertainty choices remain open research. Do not claim they are implemented from this plan.

The capped1or3origin/32innerpath prototype completed355matched development cases for each policy at1000outer paths. MainQ:3origin .7155095951729757;1origin .7161946945202958; unchanged working parent .7131987298009069; fixed reference .8004022862826219. Both lose the parent. No promotion/full-panel catalogue increment.140successful pilot losses reused,150new forecasts,420bitwise-proven parent fallback reuses; no native or numerical scoring integrity errors. A70case-only reduction lacked required asset-quarter support; fail-closed canonical aggregation returnedNone. All140valid losses were preserved and completed into the original355case plan without changing scoring. Independent audit checked710candidate and710reference identities/hash vectors, exact score reduction and fresh multivariate replay. Full histories and fixed score were preserved.

The user rejected fixed origin caps and automatic parent fallback. The new complete-history reference uses all eligible origins,1000inner paths, historical leads up to the requested horizon and explicitly discloses the longest directly validated lead. It chooses the actual inner-score minimum, never an automatic previous-winner substitute. Invalid internal options are disclosed; all valid options retain identical validation origins.7qualification tests and14actual-source96row diagnostic requests across7classes passed with no errors. These are request-level diagnostics, not a canonical panel score or proof of speed at full histories. The requested21and252day horizons produced different selections for several inputs. No outer test observations enter selection.

The superseded capped prototype's36paired single-process compute measurements used2520observations,1/10/200constituents,21/252day horizons and1000final paths. Median complete selector compute: .114/.120seconds for1asset; .870/.803for10;16.996/15.463for200. Parent medians .028/.034;.213/.228;4.132/4.349seconds respectively. Repeated fitting dominates capped200asset cost. Warm numerical code/no reusable fit cache; these are local numerical timings only. Do not market this as lightning-fast or browserCTV.

All closed raw evidence, including losing prototypes, failed harness stages and runtime records, is privately archived with fresh-download, full disk restoration andSHA256verification. Public archive receipts contain exact dependencies and recovery instructions. Current working/full-panel winners and both baseline numerical definitions remain unchanged. Goal remains active: expand the compatible universe and optimize selection, fit reuse and safe pruning against complete reference computations, then perform representative and broad matched validation under the existing stopping requirements.
''')
manifest={str(p.relative_to(D)):sha(p)for p in sorted(D.rglob('*'))if p.is_file()and p.name!='PUBLIC_FILES.json'and '__pycache__'not in p.parts};(D/'PUBLIC_FILES.json').write_text(json.dumps(manifest,indent=2)+'\n')
for line in run(['git','status','--porcelain']).splitlines():assert line[3:].startswith(REL+'/')
run(['git','add','--',REL]);run(['git','commit','-m','Document adaptive selector objective and verified research prototypes']);run(['git','push','origin','HEAD:main']);commit=run(['git','rev-parse','HEAD']).strip();print(json.dumps({'pushed':True,'commit':commit,'manifest_files':len(manifest),'goal_active':True}),flush=True)
