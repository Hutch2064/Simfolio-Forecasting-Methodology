from unittest.mock import patch
import numpy as np
import pytest
from . import candidate as c
from candidates.asset_selector_mean_reliability_v2 import candidate as old
from candidates.asset_selector_mean_reliability_v2.test_candidate import Work,fixture,chunks,coefficients,solver,ELIGIBLE,FOLDS
PAIR=('finite41__current','finite41__mean_estimation_reference')
def run(module,H=21,modes=None,solve=solver,work=None):
    work=Work()if work is None else work;kwargs={}if module is old else{'requested_modes':modes}
    with patch.object(module,'schedule',return_value=(ELIGIBLE,FOLDS,8)),patch.object(module,'all_chunks',side_effect=chunks),patch.object(module,'coefficients',side_effect=coefficients):result=module.prepare_all(*fixture()[:2],H,fixture()[2],fit_cache=work,solver=solve,**kwargs)
    return result,work
def assert_same(a,b):
    for key in('A','B'):assert a['coefficients']['flat'][key].tobytes()==b['coefficients']['flat'][key].tobytes()
    assert a['coefficients']['flat']['valid']==b['coefficients']['flat']['valid']
    assert a['weights_by_mode']['flat'].tobytes()==b['weights_by_mode']['flat'].tobytes()
    assert c.energy.even_allocation(a['weights_by_mode']['flat'],240).tobytes()==c.energy.even_allocation(b['weights_by_mode']['flat'],240).tobytes()
    assert [t['seed']for t in a['trace']]==[t['seed']for t in b['trace']]
def test_default_all8_byte_equal_v2():
    a,_=run(c);b,_=run(old)
    assert tuple(a)==c.MODES
    for mode in c.MODES:assert_same(a[mode],b[mode])
def test_finite_pair_and_single_reference_byte_equal_v2_H1_and_long():
    for H in(1,6041):
        reference,_=run(old,H)
        for modes in(PAIR,PAIR[1:]):
            result,work=run(c,H,modes)
            assert tuple(result)==modes
            assert all(set(ids)<=set(c.MENUS['finite41'])for n,ids in work.calls)
            for mode in modes:assert_same(result[mode],reference[mode])
def test_only_requested_union_cubes_QPs_and_selected_current_fits():
    calls=[];draws=[]
    def draw(*args,**kwargs):draws.append(tuple(args[5]));yield from chunks(*args,**kwargs)
    def fit(A,B):calls.append(len(A));return solver(A,B)
    work=Work()
    with patch.object(c,'schedule',return_value=(ELIGIBLE,FOLDS,8)),patch.object(c,'all_chunks',side_effect=draw),patch.object(c,'coefficients',side_effect=coefficients):result=c.prepare_all(*fixture()[:2],6041,fixture()[2],requested_modes=PAIR[1:],fit_cache=work,solver=fit)
    assert len(calls)==1 and len(draws)==2
    assert all(set(ids)<=set(c.MENUS['finite41'])for ids in draws)
    b=result[PAIR[1]];assert set(work.calls[-1][1])==set(np.flatnonzero(b['weights_by_mode']['flat']>0))
    assert b['requested_modes']==PAIR[1:]
def test_subset_validation_canonical_order_and_identity_generation():
    for modes in((),[],['unknown'],[PAIR[0],PAIR[0]],PAIR[0]):
        with pytest.raises(ValueError):run(c,modes=modes)
    result,_=run(c,modes=PAIR[::-1]);assert tuple(result)==PAIR
    assert c.identity(PAIR[0])!=old.identity(PAIR[0])
def test_same_failure_isolation_and_no_subset_fallback():
    calls=[]
    def fail_first(A,B):
        calls.append(len(A))
        if len(calls)==1:raise ArithmeticError('fixture')
        return solver(A,B)
    result,_=run(c,modes=PAIR,solve=fail_first)
    assert result[PAIR[0]]['unsupported_solver']and result[PAIR[0]]['weights_by_mode']['flat']is None
    assert not result[PAIR[1]]['unsupported_solver'];c.build_model(PAIR[1],fixture()[2],result[PAIR[1]])
    with pytest.raises(ArithmeticError):c.build_model(PAIR[0],fixture()[2],result[PAIR[0]])
def test_real_geometry_arithmetic_n24_H6041_reference_parity():
    n=12106;x=np.arange(n,dtype=float)[:,None]/10000;w=np.ones(1);d=np.arange(np.datetime64('1980-01-01'),np.datetime64('1980-01-01')+np.timedelta64(n,'D'));folds=[(24,6041,1),(6065,6041,1)]
    results=[]
    for module in(old,c):
        kwargs={}if module is old else{'requested_modes':PAIR}
        with patch.object(module,'schedule',return_value=(ELIGIBLE,folds,2)),patch.object(module,'all_chunks',side_effect=chunks),patch.object(module,'coefficients',side_effect=coefficients):results.append(module.prepare_all(x,w,6041,d,fit_cache=Work(),solver=solver,**kwargs))
    for mode in PAIR:assert_same(results[0][mode],results[1][mode])
    b=results[1][PAIR[1]];assert b['trace'][0]['reliability']['mean_estimation_reference']<b['trace'][1]['reliability']['mean_estimation_reference']
def test_prescribed_states_group_RNG_paths_and_real_coefficients_subset_bytes():
    # No estimation: finite manually prescribed states exercise actual unchanged path kernels.
    from candidates.asset_selector_endpoint_fit_reuse_v3 import paths
    rng=np.random.default_rng(902);x=rng.normal(0,.01,(30,2));w=np.array([.3,.7]);batch={'sample_mean':np.array([-.0002,.0004]),'center':np.array([0.,.0001]),'root':np.array([[.001,0],[.0002,.001]])}
    for family in('constant','EWMA_reverting','SV_RMS'):
        f={k:np.zeros(2)for k in('alpha','beta','positive','negative','omega','initial','decay','phi','level','sv_phi','q','m','P','V')}
        f.update(initial=np.array([1.2,.8]),decay=np.array([.9,.95]),target=np.array([.0001,.0002]),weights=np.zeros((2,3)),ring=np.ones((2,22)),pool=rng.normal(size=(30,2)),root=np.array([[1.,0.],[.2,.98]]),vol_root=np.eye(2),inverse_df=.1,level=np.array([-.1,-.2]),sv_phi=np.array([.8,.9]),q=np.array([.01,.02]),m=np.array([0.,.1]),P=np.array([.03,.04]))
        bundle={'raw':{'mean':batch['sample_mean']},'mean':batch,'filter':f,'family':family,'history_sha256':c.parent.signature(x)}
        for law in('Gaussian','empirical'):
            full=[i for i,s in enumerate(c.BANK)if s.get('family')==family and s.get('innovations')==law]
            subset=[i for i in full if i in c.MENUS['finite29']];positions=[full.index(i)for i in subset]
            for H in(1,7):
                a=np.concatenate([v for _,v in paths.group_chunks(x,w,H,2,905,full,bundle,width=3)],axis=2)
                b=np.concatenate([v for _,v in paths.group_chunks(x,w,H,2,905,subset,bundle,width=3)],axis=2)
                assert a[positions].tobytes()==b.tobytes()
                A,B=c.coefficients(np.ascontiguousarray(a[:,:,-1]),.003);C,D=c.coefficients(np.ascontiguousarray(b[:,:,-1]),.003)
                assert A[positions].tobytes()==C.tobytes()and B[np.ix_(positions,positions)].tobytes()==D.tobytes()
