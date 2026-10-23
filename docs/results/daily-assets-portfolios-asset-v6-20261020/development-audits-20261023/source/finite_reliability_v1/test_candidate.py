from unittest.mock import patch
import numpy as np
import pytest
from . import candidate as c
from candidates.asset_selector_finite77_v1 import candidate as old77
from candidates.asset_selector_mean_reliability_v3 import candidate as oldref
from candidates.asset_selector_no_recursive_ewma_v1 import candidate as old41
from candidates.asset_selector_finite77_v1.test_candidate import Work,fixture,chunks,coefficients,solver,ELIGIBLE,FOLDS
PAIR=('finite41__current','finite41__mean_estimation_reference')
def run(module,H=21,modes=None,work=None,solve=solver):
    work=Work()if work is None else work;kwargs={}
    if module in(c,oldref):kwargs['requested_modes']=modes
    with patch.object(module,'schedule',return_value=(ELIGIBLE,FOLDS,8)),patch.object(module,'all_chunks',side_effect=chunks),patch.object(module,'coefficients',side_effect=coefficients):result=module.prepare_all(*fixture()[:2],H,fixture()[2],fit_cache=work,solver=solve,**kwargs)
    return result,work
def same(a,b):
    for key in('A','B'):assert a['coefficients']['flat'][key].tobytes()==b['coefficients']['flat'][key].tobytes()
    assert a['coefficients']['flat']['valid']==b['coefficients']['flat']['valid']
    assert a['weights_by_mode']['flat'].tobytes()==b['weights_by_mode']['flat'].tobytes()
    assert c.energy.even_allocation(a['weights_by_mode']['flat'],240).tobytes()==c.energy.even_allocation(b['weights_by_mode']['flat'],240).tobytes()
    assert [t['seed']for t in a['trace']]==[t['seed']for t in b['trace']]
def test_four_modes_exact_catalogue_and_three_qualified_control_parities():
    assert len(c.MODES)==4 and len(c.MENUS['finite41'])==41 and len(c.MENUS['finite77'])==77
    for H in(1,6041):
        result,_=run(c,H);prior,_=run(old77,H);reference,_=run(oldref,H,PAIR)
        same(result['finite41__current'],prior['finite41']);same(result['finite77__current'],prior['finite77']);same(result[PAIR[1]],reference[PAIR[1]])
        x,w,d=fixture()
        with patch.object(old41,'schedule',return_value=(ELIGIBLE,FOLDS,8)),patch.object(old41,'all_chunks',side_effect=chunks),patch.object(old41,'coefficients',side_effect=coefficients):fixed=old41.prepare(x,w,H,d,fit_cache=Work(),solver=solver,menu='finite_wealth41')
        same(result['finite41__current'],fixed)
def test_shared_bank_four_QPs_and_H1_two_exact_alias_QPs():
    for H,expected in((1,2),(21,4)):
        calls=[];draws=[]
        def tracked(A,B):calls.append(len(A));return solver(A,B)
        def draw(*args,**kwargs):draws.append(tuple(args[5]));yield from chunks(*args,**kwargs)
        work=Work()
        with patch.object(c,'schedule',return_value=(ELIGIBLE,FOLDS,8)),patch.object(c,'all_chunks',side_effect=draw),patch.object(c,'coefficients',side_effect=coefficients):result=c.prepare_all(*fixture()[:2],H,fixture()[2],fit_cache=work,solver=tracked)
        assert len(draws)==2 and len(calls)==expected and len(work.calls)==3
        assert all(set(ids)<=set(c.MENUS['finite77'])for ids in draws)
        if H==1:
            assert all(not set(ids)&set(c.DERIVATIVES)for ids in draws)
            for rule in c.RULES:same(result['finite41__'+rule],result['finite77__'+rule])
def test_requested_subset_before_all_preparation_and_QP_parity():
    allm,_=run(c)
    for modes in(PAIR,('finite77__mean_estimation_reference',),PAIR[1:]):
        result,work=run(c,modes=modes)
        assert tuple(result)==modes
        allowed={i for mode in modes for i in c.MENUS[c.split_mode(mode)[0]]}
        assert all(set(ids)<=allowed for n,ids in work.calls)
        for mode in modes:same(result[mode],allm[mode])
    for modes in((),['bad'],[PAIR[0],PAIR[0]],PAIR[0]):
        with pytest.raises(ValueError):run(c,modes=modes)
def test_reference77_formula_A_B_and_mass_once():
    b=run(c)[0]['finite77__mean_estimation_reference'];valid=b['coefficients']['flat']['valid'];K=len(valid)
    raw=[coefficients(np.broadcast_to((stop+np.asarray(valid,dtype=float))[:,None],(K,2)).copy(),0)for stop,h,m in FOLDS];factors=[1/np.sqrt(1+h/stop)for stop,h,m in FOLDS];mass=sum(m*f for(_,h,m),f in zip(FOLDS,factors))
    A=sum(m*(f*(a/np.sqrt(h)))for(_,h,m),(a,z),f in zip(FOLDS,raw,factors))/mass;B=sum(m*(f*(z/np.sqrt(h)))for(_,h,m),(a,z),f in zip(FOLDS,raw,factors))/mass
    assert b['coefficients']['flat']['A'].tobytes()==A.tobytes()and b['coefficients']['flat']['B'].tobytes()==B.tobytes()
    assert b['coefficients']['flat']['represented_mass']==mass
def test_length1_alias_before_counts_and_solver_failure_isolation():
    calls=[]
    def fit(A,B):
        calls.append(len(A))
        if len(calls)==2:raise ArithmeticError('fixture solver failure')
        v=np.zeros(len(A));v[0 if len(calls)<3 else -1]=1.;return {'weights':v}
    result,_=run(c,work=Work(length=1),solve=fit)
    assert result[PAIR[1]]['unsupported_solver']and result[PAIR[1]]['weights_by_mode']['flat']is None
    c.build_model(PAIR[0],fixture()[2],result[PAIR[0]])
    with pytest.raises(ArithmeticError):c.build_model(PAIR[1],fixture()[2],result[PAIR[1]])
    for mode in('finite77__current','finite77__mean_estimation_reference'):
        b=result[mode];alias=b['diagnostics']['final_IID_aliases'];assert len(alias)==1;i,j=next(iter(alias.items()));v=b['weights_by_mode']['flat'];assert v[i]==0 and v[j]==1
        n=c.energy.even_allocation(v,240);assert n[j]==240 and n[i]==0
        c.build_model(mode,fixture()[2],b)
def test_source_bound_identity_and_unchanged_kernel_dispatch():
    result,_=run(c);assert len({c.identity(mode)for mode in c.MODES})==4
    assert c.all_chunks is old77.all_chunks and c.Pool.terminal_chunks is old77.Pool.terminal_chunks
    with pytest.raises(ValueError):c.build_model('finite77__current',fixture()[2],result[PAIR[0]])
    with patch.object(c,'sources',return_value={}):
        with pytest.raises(ValueError):c.build_model(PAIR[0],fixture()[2],result[PAIR[0]])
def test_long_incident_geometry_parity_no_reliability_reassociation():
    n=12106;x=np.arange(n,dtype=float)[:,None]/10000;w=np.ones(1);d=np.datetime64('1980-01-01')+np.arange(n).astype('timedelta64[D]');folds=[(24,6041,1),(6065,6041,1)];results=[]
    for module in(c,oldref):
        with patch.object(module,'schedule',return_value=(ELIGIBLE,folds,2)),patch.object(module,'all_chunks',side_effect=chunks),patch.object(module,'coefficients',side_effect=coefficients):results.append(module.prepare_all(x,w,6041,d,requested_modes=PAIR,fit_cache=Work(),solver=solver))
    for mode in PAIR:same(results[0][mode],results[1][mode])
def test_prescribed_states_actual_kernel_coefficients_three_control_parities():
    # Explicit fixed finite states, no estimation or historical request execution.
    class Prescribed(Work):
        def prefix_fits(self,x,ids,w):
            self.calls.append((len(x),tuple(ids)));p=x.shape[1];z=np.zeros(p);one=np.ones(p);mean={'sample_mean':np.array([-.0002,.0004]),'center':np.array([0.,.0001]),'root':np.array([[.001,0],[.0002,.001]])};families={}
            for family in('constant','EWMA_reverting','SV_RMS'):
                f={k:z.copy()for k in('alpha','beta','positive','negative','omega','initial','decay','phi','level','sv_phi','q','m','P','V')}
                f.update(initial=np.array([1.2,.8]),decay=np.array([.9,.95]),target=np.array([.0001,.0002]),weights=np.zeros((p,3)),ring=np.ones((p,22)),pool=np.ascontiguousarray(x/.01),root=np.array([[1.,0.],[.2,.98]]),vol_root=np.eye(p),inverse_df=.1,level=np.array([-.1,-.2]),sv_phi=np.array([.8,.9]),q=np.array([.01,.02]),m=np.array([0.,.1]),P=np.array([.03,.04]))
                families[family]={'raw':{'mean':mean['sample_mean']},'mean':mean,'filter':f,'family':family,'history_sha256':c.ppw.leader.signature(x)}
            return {'families':families,'residual_PPW':{family:{'lengths':{'stationary':2.,'circular':2.}}for family in families}},{}
    ids=[i for i in c.CORE if 4<=i<166]
    for H in(1,6041):
        outputs=[]
        for module in(c,old77,oldref):
            kwargs={'requested_modes':PAIR}if module is oldref else{}
            with patch.object(module,'schedule',return_value=(ids,FOLDS,8)):
                outputs.append(module.prepare_all(*fixture()[:2],H,fixture()[2],fit_cache=Prescribed(),solver=solver,**kwargs))
        same(outputs[0]['finite41__current'],outputs[1]['finite41']);same(outputs[0]['finite77__current'],outputs[1]['finite77']);same(outputs[0][PAIR[1]],outputs[2][PAIR[1]])
def test_derivative_current_failure_isolated_from_both_170slot_controls():
    class FailDerivative(Work):
        def prefix_fits(self,x,ids,w):
            full,fail=super().prefix_fits(x,ids,w)
            return full,({i:{'type':'fixture','reason':'current'}for i in ids if i in c.DERIVATIVES}if len(x)==40 else fail)
    calls=[]
    def choose(A,B):
        calls.append(len(A));v=np.zeros(len(A));v[0 if len(calls)<3 else -1]=1.;return {'weights':v}
    result,_=run(c,work=FailDerivative(),solve=choose)
    for mode,b in result.items():
        if mode.startswith('finite41'):
            assert not b['unsupported_selected_current'];c.build_model(mode,fixture()[2],b)
        else:
            assert b['unsupported_selected_current']
            with pytest.raises(ArithmeticError):c.build_model(mode,fixture()[2],b)
