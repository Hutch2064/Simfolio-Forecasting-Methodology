"""Tree proposals, vertex-loss controls and causal finite-path selection."""
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor,ExtraTreesRegressor
from candidates.asset_additional_learners_v2.optimizer_progress_v3 import gate
from .objective import risk_gradient,direction_step,NumericalFailure

KINDS=('lightgbm','xgboost','cart','random_forest','extra_trees')

def features(record,context):
    z=gate.geometry(record)@context['U']/context['R']
    if 'state_basis' in context:
        from candidates.asset_additional_learners_v2.optimizer_progress_v3.state_features import transform
        z=np.column_stack((z,transform(record['state_features'],context['state_basis'])))
    if not np.isfinite(z).all():raise NumericalFailure('Nonfinite tree features')
    return z

def preparation(records,groups):
    work=gate.RequestWork(groups);context=gate.fit_context(records,work)
    return work,context,context['Z'],[work.matrix(r) for r in records]

def new_direction(kind,x,y,seed):
    # One full positive-gain tree, bounded only by the finite sample partition.
    # No row/feature subsampling: all eligible recipes/features remain available.
    if kind=='lightgbm':
        import lightgbm as lgb
        model=lgb.LGBMRegressor(n_estimators=1,learning_rate=1.,num_leaves=len(x),max_depth=-1,min_child_samples=1,min_data_in_bin=1,min_child_weight=0.,min_split_gain=0.,reg_alpha=0.,reg_lambda=0.,subsample=1.,colsample_bytree=1.,max_bin=max(3,len(x)),verbosity=-1,n_jobs=1,random_state=seed,deterministic=True,force_col_wise=True)
    elif kind=='xgboost':
        import xgboost as xgb
        model=xgb.XGBRegressor(n_estimators=1,learning_rate=1.,max_depth=0,grow_policy='lossguide',max_leaves=len(x),max_bin=max(2,len(x)),min_child_weight=0.,gamma=0.,reg_alpha=0.,reg_lambda=0.,subsample=1.,colsample_bytree=1.,tree_method='hist',n_jobs=1,random_state=seed,base_score=0.,objective='reg:squarederror')
    else:raise ValueError(kind)
    model.fit(x,y);return model

class BoostPath:
    def __init__(self,kind,records,groups,emit=lambda **kw:None):
        self.kind=kind;self.records=records;self.work,self.context,self.z,self.b=preparation(records,groups)
        self.x=np.concatenate(self.z);self.logits=[np.zeros(len(r['A'])) for r in records];self.trees=[];self.steps=[];self.endpoint=None;self.emit=emit
        value,gradient,_=risk_gradient(self.logits,records,self.b,self.context['mass'],self.context['S'])
        self.delta=64*np.finfo(float).eps*self.x.size*max(1.,abs(value));self.tolerance=np.sqrt(np.finfo(float).eps)*len(gradient)
    def extend(self,target):
        if int(target)!=target or target<0:raise ValueError('Nonnegative integer accepted-stage budget required')
        while len(self.trees)<target:
            value,gradient,_=risk_gradient(self.logits,self.records,self.b,self.context['mass'],self.context['S'])
            norm=float(np.max(np.abs(gradient)));self.emit(phase='boost_stage_start',stage=len(self.trees),objective=value,gradient_inf=norm,tolerance=self.tolerance)
            if norm<=self.tolerance:self.endpoint='logit_gradient_certificate';break
            tree=new_direction(self.kind,self.x,-gradient/norm,len(self.trees));prediction=np.asarray(tree.predict(self.x),float)
            directions=list(np.split(prediction,np.cumsum([len(z) for z in self.z])[:-1]))
            proposed,receipt=direction_step(self.logits,directions,self.records,self.b,self.context['mass'],self.context['S'],self.delta)
            if proposed is None:
                self.endpoint=receipt['endpoint'];self.emit(phase='boost_path_endpoint',stage=len(self.trees),**receipt);break
            self.logits=proposed;self.trees.append(tree);self.steps.append(receipt['effective_tree_coefficient']);self.emit(phase='boost_stage_accepted',stage=len(self.trees),**receipt)
        if len(self.trees)<target and self.endpoint!='logit_gradient_certificate':
            raise NumericalFailure(f'Cannot reproduce selected stage {target}: {self.endpoint}; actual={len(self.trees)}')
        return self
    def predict(self,record,stage):
        from scipy.special import softmax
        z=features(record,self.context);ell=np.zeros(len(z))
        for model,step in zip(self.trees[:stage],self.steps[:stage]):
            direction=np.asarray(model.predict(z));ell+=step*(direction-direction.mean())
        return softmax(ell)

def vertex_data(records,z,context):
    target=np.concatenate([(r['A']-.5*np.diag(gate.unpack_symmetric(r['B_packed'],len(r['A']))))/context['S'] for r in records])
    mass=np.concatenate([np.full(len(x),m/len(x)) for x,m in zip(z,context['mass'])])
    return np.concatenate(z),target,mass

def control_preparation(records,groups):
    _,context,z,_=preparation(records,groups);x,y,mass=vertex_data(records,z,context)
    return context,x,y,mass

def canonical_alphas(raw,y,mass):
    """Penalty-metadata repair using cancellation-aware impurity units.

    Conservative arithmetic envelope; no financial input or response clipping.
    """
    raw=np.asarray(raw,dtype=float);y=np.asarray(y,dtype=float);mass=np.asarray(mass,dtype=float)
    if not np.isfinite(raw).all():raise NumericalFailure('Nonfinite pruning path')
    negative=raw<0
    if not np.any(negative):return raw.copy(),{'raw_ccp_alphas':raw.tolist(),'canonical_ccp_alphas':raw.tolist(),'roundoff_bound':None,'canonicalized_negative_count':0,'positive_path_unchanged_without_envelope_evaluation':True}
    if not np.isfinite(y).all() or not np.isfinite(mass).all() or np.any(mass<=0):raise NumericalFailure('Nonfinite pruning target or nonpositive mass')
    p=mass/mass.sum();mean=float(p@y);second=float(p@(y*y));scale=second+mean*mean
    bound=64*np.finfo(float).eps*max(1,len(y))*scale
    if not np.isfinite(scale) or not np.isfinite(bound):raise NumericalFailure('Nonfinite pruning arithmetic envelope')
    if np.any(-raw[negative]>bound):raise NumericalFailure(f'Material negative pruning alpha; minimum={raw.min()}; roundoff_bound={bound}')
    clean=raw.copy();clean[negative]=0.
    return clean,{'raw_ccp_alphas':raw.tolist(),'canonical_ccp_alphas':clean.tolist(),'roundoff_bound':bound,'impurity_scale_M2_plus_mean_squared':scale,'training_rows':len(y),'canonicalized_negative_count':int(negative.sum()),'bound_is_conservative_arithmetic_envelope_not_library_error_proof':True}

def alpha_path(records,groups,prepared=None,diagnostics=None):
    context,x,y,mass=control_preparation(records,groups) if prepared is None else prepared
    path=DecisionTreeRegressor(random_state=0).cost_complexity_pruning_path(x,y,sample_weight=mass)
    clean,receipt=canonical_alphas(path.ccp_alphas,y,mass)
    if diagnostics is not None:diagnostics.update(receipt)
    return sorted(set([0.,*map(float,clean),float('inf')]))

def fit_control(kind,records,groups,alpha,count,prepared=None):
    context,x,y,mass=control_preparation(records,groups) if prepared is None else prepared
    if np.isinf(alpha):return {'context':context,'model':None,'null':True,'count':count}
    params={'random_state':0,'ccp_alpha':alpha}
    if kind=='cart':model=DecisionTreeRegressor(**params)
    elif kind=='random_forest':model=RandomForestRegressor(**params,n_estimators=count,n_jobs=1,max_features=1.,bootstrap=True,oob_score=False)
    elif kind=='extra_trees':model=ExtraTreesRegressor(**params,n_estimators=count,n_jobs=1,max_features=1.,bootstrap=False)
    else:raise ValueError(kind)
    model.fit(x,y,sample_weight=mass);return {'context':context,'model':model,'null':False,'count':count}

def predict_control(fit,record):
    k=len(record['A'])
    if fit['null']:return np.full(k,1./k)
    values=np.asarray(fit['model'].predict(features(record,fit['context'])),float)
    if not np.isfinite(values).all():raise NumericalFailure('Nonfinite predicted vertex CRPS')
    ties=values==np.min(values);return ties.astype(float)/sum(ties)

def matured(records,index):return [r for r in records[:index] if r['validation_stop']<=records[index]['training_rows']]

def blend_cells(accumulated,batch,v,record,b,scale=1.):
    d=v-batch;m=record['represented_origins'];accumulated[0]+=m*float(batch@record['A']-.5*batch@b@batch)
    accumulated[1]+=m*float(d@(record['A']-b@batch));accumulated[2]+=m*float(-.5*d@b@d)

def choose(cells):
    candidates=[]
    for label,(base,linear,quadratic) in cells.items():
        error=64*np.finfo(float).eps*max(1.,abs(linear),abs(quadratic))
        if quadratic < -error:raise NumericalFailure('Nonconvex blend')
        options=[0.,1.]
        if quadratic>0:options.append(float(np.clip(-linear/(2*quadratic),0.,1.)))
        tau=min(options,key=lambda t:(linear*t+quadratic*t*t,t))
        candidates.append((base+linear*tau+quadratic*tau*tau,label,tau))
    # Exact ties prefer null/stronger pruning and then fewer stages.
    return min(candidates,key=lambda item:(item[0],-item[1][0],item[1][1]))

def forest_fit(kind,records,current,groups,emit):
    """Root-admitted greedy doubling-prefix early stopping, not global search."""
    work=gate.RequestWork(groups);episodes=[(i,matured(records,i)) for i in range(len(records)) if matured(records,i)]
    if not episodes:raise ValueError('No matured chronological assessment')
    prepared={i:control_preparation(earlier,groups) for i,earlier in episodes}
    diagnostics={};alphas=alpha_path(episodes[0][1],groups,prepared[episodes[0][0]],diagnostics);count=1;previous=None;best=None;trace=[];all_labels=[]
    while True:
        cells={(a,count):[0.,0.,0.] for a in alphas}
        for index,earlier in episodes:
            record=records[index];batch=work.batch(earlier)['weights'];b=work.matrix(record)
            for label in cells:
                emit(phase='control_fit_start',assessment_index=index,alpha=label[0],tree_count=count,seed=0)
                model=fit_control(kind,earlier,groups,*label,prepared[index]);blend_cells(cells[label],batch,predict_control(model,record),record,b)
        selected=choose(cells);risk,label,tau=selected;all_labels.extend(cells)
        trace.append({'tree_count':count,'pooled_validation_risk':risk,'selected_alpha':label[0],'tau':tau})
        emit(phase='forest_prefix_assessed',**trace[-1])
        if previous is not None and risk>=previous:
            break
        best=selected;previous=risk;count*=2
    risk,label,tau=best;batch=work.batch(records)['weights'];model=None
    if tau==0.:weights=batch.copy()
    else:
        model=fit_control(kind,records,groups,*label);weights=(1-tau)*batch+tau*predict_control(model,current)
    return {'weights':weights,'model':model,'kind':kind,'groups':groups,'selected_alpha':label[0],'selected_stage_or_count':label[1],'tau':tau,'validation_risk':risk,'episodes':[{'assessment_index':i,'training_rows':[r['training_rows'] for r in earlier]} for i,earlier in episodes],'candidate_labels':all_labels,'pruning_roundoff':diagnostics,'forest_prefix_trace':trace,'forest_count_rule':'1,2,4,...; first nonimproving pooled chronological CRPS ends search; earlier best retained','seed':0,'skipped_zero_blend':tau==0.,'global_optimum_claim':False,'forest_iid_or_OOB_claim':False}

def boost_fit(kind,records,current,groups,emit):
    work=gate.RequestWork(groups);episodes=[(i,matured(records,i)) for i in range(len(records)) if matured(records,i)]
    if not episodes:raise ValueError('No matured chronological assessment')
    paths=[BoostPath(kind,earlier,groups,lambda **event:emit(assessment_index=i,**event)) for i,earlier in episodes]
    count=0;previous=None;best=None;trace=[];labels=[]
    while True:
        cells={(0.,count):[0.,0.,0.]}
        for (index,earlier),path in zip(episodes,paths):
            # Rebind each emitter: closure defaults prevent the final assessment
            # label being reported for every path.
            path.emit=lambda index=index,**event:emit(assessment_index=index,**event)
            path.extend(count);record=records[index];batch=work.batch(earlier)['weights']
            blend_cells(cells[(0.,count)],batch,path.predict(record,count),record,work.matrix(record))
        selected=choose(cells);risk,label,tau=selected;labels.append(label)
        trace.append({'accepted_stage_budget':count,'pooled_validation_risk':risk,'tau':tau,'actual_stages_by_assessment':[len(p.trees) for p in paths]})
        emit(phase='boost_prefix_assessed',**trace[-1])
        if previous is not None and risk>=previous:break
        best=selected;previous=risk;count=1 if count==0 else count*2
        # A certified stationary endpoint is absorbing. Once every path is
        # absorbing no future stage budget changes its predictions.
        if all(p.endpoint=='logit_gradient_certificate' for p in paths):break
    risk,label,tau=best;batch=work.batch(records)['weights'];model=None
    if tau==0.:weights=batch.copy()
    else:
        model=BoostPath(kind,records,groups,emit).extend(label[1]);weights=(1-tau)*batch+tau*model.predict(current,label[1])
    return {'weights':weights,'model':model,'kind':kind,'groups':groups,'selected_alpha':0.,'selected_stage_or_count':label[1],'tau':tau,'validation_risk':risk,'episodes':[{'assessment_index':i,'training_rows':[r['training_rows'] for r in earlier]} for i,earlier in episodes],'candidate_labels':labels,'boost_prefix_trace':trace,'boost_count_rule':'0(null),1,2,4,...; first nonimproving pooled chronological CRPS ends search; earlier best retained','seed':0,'skipped_zero_blend':tau==0.,'global_optimum_claim':False,'forest_iid_or_OOB_claim':False}

def fit_impl(kind,records,current,groups=(),emit=lambda **kw:None):
    if kind not in KINDS:raise ValueError(kind)
    if kind in ('random_forest','extra_trees'):
        return forest_fit(kind,records,current,groups,emit)
    if kind in ('lightgbm','xgboost'):return boost_fit(kind,records,current,groups,emit)
    labels=None;cells={};episodes=[];work=gate.RequestWork(groups)
    for index,record in enumerate(records):
        earlier=matured(records,index)
        if not earlier:continue
        emit(phase='assessment_start',assessment_index=index,training_rows=[r['training_rows'] for r in earlier])
        if labels is None:
            first_prepared=control_preparation(earlier,groups)
            diagnostics={};labels=[(a,1) for a in alpha_path(earlier,groups,first_prepared,diagnostics)]
            cells={label:[0.,0.,0.] for label in labels}
        batch=work.batch(earlier)['weights'];b=work.matrix(record)
        prepared=first_prepared if not episodes else control_preparation(earlier,groups)
        for label in labels:
            emit(phase='control_fit_start',assessment_index=index,alpha=label[0],tree_count=label[1])
            trained=fit_control(kind,earlier,groups,*label,prepared);blend_cells(cells[label],batch,predict_control(trained,record),record,b)
        episodes.append({'assessment_index':index,'training_rows':[r['training_rows'] for r in earlier]})
    if not episodes:raise ValueError('No matured chronological assessment')
    risk,label,tau=choose(cells);batch=work.batch(records)['weights'];trained=None
    if tau==0.:weights=batch.copy()
    else:
        trained=fit_control(kind,records,groups,*label);weights=(1-tau)*batch+tau*predict_control(trained,current)
    if not np.isfinite(weights).all() or weights.min()<0 or abs(weights.sum()-1)>1e-12:raise NumericalFailure('Invalid final simplex')
    return {'weights':weights,'model':trained,'kind':kind,'groups':groups,'selected_alpha':label[0],'selected_stage_or_count':label[1],'tau':tau,'validation_risk':risk,'episodes':episodes,'candidate_labels':labels,'pruning_roundoff':diagnostics,'skipped_zero_blend':tau==0.,'global_optimum_claim':False,'forest_iid_or_OOB_claim':False}

def fit(kind,records,current,groups=(),emit=lambda **kw:None):
    # Current A is deliberately not inspected by any learning/selection code.
    result=fit_impl(kind,records,current,groups,emit);weights=result['weights']
    if not np.isfinite(weights).all() or weights.min()<0 or abs(weights.sum()-1)>1e-12:raise NumericalFailure('Invalid final simplex')
    return result

def export_model(model):
    """Serializable numerical predictor; excludes callbacks and request caches."""
    if isinstance(model,BoostPath):return {'context':model.context,'trees':model.trees,'steps':model.steps,'kind':model.kind,'endpoint':model.endpoint}
    return model
