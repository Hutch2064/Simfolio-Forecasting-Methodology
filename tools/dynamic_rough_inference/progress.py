"""Matched cumulative scores from finished portfolios, without rerunning controls."""
import argparse
import json
from pathlib import Path

import numpy as np


def snapshot(output,reference,best_reference,default_reference):
    manifest=json.loads((output/'manifest.json').read_text())
    baseline=json.loads((reference/'manifest.json').read_text())
    best=json.loads((best_reference/'manifest.json').read_text())
    assert manifest['tasks']==[rows[1] for rows in baseline['tasks']]
    assert manifest['tasks']==best['tasks']
    assert manifest['simulations']==baseline['simulations']==best['simulations']==240
    assert len(manifest['models'])==1
    tasks=manifest['tasks']
    groups={}
    for index,task in enumerate(tasks):groups.setdefault(task['portfolio_id'],[]).append(index)
    completed={int(p.stem.split('-')[1]) for p in output.glob('task-*.npz')
        if p.with_suffix('.json').exists()}
    finished=sorted(p for p,indices in groups.items() if all(i in completed for i in indices))
    progress=json.loads((output/'progress.json').read_text())
    scores={}
    if finished:
        with np.load(reference/'paired-cells.npz') as old,np.load(best_reference/'paired-cell-losses.npz') as winner:
            assert np.array_equal(old['portfolio'],winner['portfolio'])
            assert np.array_equal(old['horizon'],winner['horizon'])
            mask=np.isin(old['portfolio'],finished)
            scores['MAP predecessor']=float(old['model_0'][mask].mean())
            scores['Production frontier']=float(old['model_1'][mask].mean())
            scores['Dynamic Bayesian rough winner']=float(winner['model_0'][mask].mean())
            new=[]
            for portfolio in finished:
                selected=old['portfolio']==portfolio
                horizons=old['horizon'][selected]
                total=np.zeros(int(horizons.max()));counts=np.zeros(len(total),dtype=np.int64)
                for index in groups[portfolio]:
                    with np.load(output/f'task-{index:04d}.npz') as saved:
                        assert len(saved['losses_0'])==tasks[index]['horizon_days']
                        loss=saved['losses_0'];total[:len(loss)]+=loss;counts[:len(loss)]+=1
                assert np.array_equal(counts[horizons-1],old['origin_count'][selected])
                new.extend((total/counts)[horizons-1])
            scores['MAP predecessor + dynamic rough']=float(np.mean(new))
            if default_reference:
                default=[]
                for path in sorted(default_reference.glob('portfolio-*.npz')):
                    with np.load(path) as saved:
                        portfolio=str(saved['portfolio'])
                        if portfolio not in finished:continue
                        selected=old['portfolio']==portfolio
                        assert np.array_equal(saved['horizon'],old['horizon'][selected])
                        assert np.array_equal(saved['origin_count'],old['origin_count'][selected])
                        default.extend(saved['losses'])
                assert len(default)==int(mask.sum())
                scores['Production default naive']=float(np.mean(default))
    elapsed=progress['elapsed_seconds'];count=len(completed)
    result={'completed_origins': count,'required_origins': len(tasks),'completed_portfolios': len(finished),
        'required_portfolios': len(groups),'elapsed_seconds': elapsed,
        'estimated_remaining_seconds': elapsed*(len(tasks)-count)/count if count else None,
        'scoring': 'equal portfolio-horizon cells; completed 51-origin portfolios only',
        'comparator_models_rerun': False,'cumulative_scores': scores}
    (output/'cumulative-comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--reference',type=Path,required=True)
    parser.add_argument('--best-reference',type=Path,required=True)
    parser.add_argument('--default-reference',type=Path)
    args=parser.parse_args()
    print(json.dumps(snapshot(args.output,args.reference,args.best_reference,args.default_reference)))
