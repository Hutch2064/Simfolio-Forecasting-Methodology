"""Single uniform-shrinkage learner; compiled inner coefficients, no gate fits."""
from candidates.asset_additional_learners import builder, shrinkage
from candidates.asset_additional_learners.candidate import LearnerPool
CONFIG={'family':'uniform_shrinkage','inner_paths':2,'schedule':'half_doubling'}

def identity(config):
    if config!=CONFIG:raise ValueError('Only frozen uniform shrinkage configuration supported')
    from candidates.asset_additional_learners.candidate import identity as original_identity
    return original_identity('uniform_shrinkage')

def prepare(history,weights,horizon,training_dates,*,inner_paths=2,schedule_rule='half_doubling',seed_replication=0,width=64):
    if inner_paths!=2 or schedule_rule!='half_doubling':raise ValueError('Frozen inner2 and half_doubling required')
    bundle=builder.prepare(history,weights,horizon,training_dates,inner_paths=inner_paths,
        schedule_rule=schedule_rule,seed_replication=seed_replication,width=width)
    bundle['uniform_shrinkage_fit']=shrinkage.fit(bundle['fold_records'])
    return bundle

class ShrinkagePool:
    def __init__(self,config,training_dates,bundle):
        self.model_id=identity(config);self.bundle=bundle
        self.adapter=LearnerPool('uniform_shrinkage',training_dates,bundle,bundle['uniform_shrinkage_fit'])
    def terminal_chunks(self,history,weights,horizon,paths,context_seed,*,width=64):
        yield from self.adapter.terminal_chunks(history,weights,horizon,paths,context_seed,width=width)
        fit=self.bundle['uniform_shrinkage_fit']
        self.last_selection={**self.adapter.last_selection,'shrinkage_rho':fit['rho'],
            'shrinkage_chronological_episodes':fit['episodes'],'automatic_fallback':False}
