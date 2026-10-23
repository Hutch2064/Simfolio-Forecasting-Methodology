import hashlib
from unittest.mock import patch
import numpy as np
import pytest
from . import candidate as c
from .geometry import map_folds
from candidates.asset_selector_no_recursive_ewma_v1 import candidate as old
from candidates.asset_selector_finite77_v1.test_candidate import fixture,chunks,coefficients,solver,ELIGIBLE,Work
FOLDS=[(24,2,3),(28,4,5)]
class MockWork(Work):
    def validate_geometry(self,H,mode,folds):self.geometry=(H,mode,tuple(folds))
def stage(H=21,mode='mapped_ratio',work=None):
    work=MockWork()if work is None else work
    with patch.object(c,'schedule',return_value=(ELIGIBLE,FOLDS,8)),patch.object(c,'all_chunks',side_effect=chunks),patch.object(c,'coefficients',side_effect=coefficients):result=c.prepare(*fixture()[:2],H,fixture()[2],mode=mode,fit_cache=work,solver=solver)
    return result,work
def same(a,b):
    for key in('A','B'):assert a['coefficients']['flat'][key].tobytes()==b['coefficients']['flat'][key].tobytes()
    assert a['weights_by_mode']['flat'].tobytes()==b['weights_by_mode']['flat'].tobytes()
    assert c.energy.even_allocation(a['weights_by_mode']['flat'],240).tobytes()==c.energy.even_allocation(b['weights_by_mode']['flat'],240).tobytes()
def test_integer_feasibility_against_bruteforce_and_projection_boundaries():
    for N in range(3,45):
        for H in range(1,55):
            m=2;full=[(m,1,1),(N-1,1,1)]if N-1>m else[(m,1,1)]
            got,meta=map_folds(N,H,m,full,len(full))
            feasible=[n for n in range(m,N)if n+(n*H+N-1)//N<=N]
            assert meta['requested_ratio_preserved']==bool(feasible)
            if feasible:assert meta['maximum_feasible_prefix']==max(feasible)
            else:assert got==[(m,N-m,len(full))]and meta['support_projection_applied']
            assert sum(t[2]for t in got)==len(full)and len(got)<=len(full)
            assert all(n>=m and h>=1 and n+h<=N for n,h,mass in got)
    with pytest.raises(ArithmeticError,match='Unsupported'):map_folds(24,1,24,[(24,1,1)],1)
def test_incident_geometry_mass_duplicates_hugeH_and_exact_H1():
    folds,meta=map_folds(12106,12106,24,[(24,6041,1),(6065,6041,1)],2);assert folds==[(24,24,1),(6053,6053,1)]and meta['requested_ratio_preserved']
    assert map_folds(6053,18159,24,[(25,3014,1),(3039,3014,1)],2)[0]==[(25,75,1),(1513,4539,1)]
    assert map_folds(100,10**20,24,[(24,38,2),(62,38,3)],5)[0]==[(24,76,5)]
    assert map_folds(100,100,24,[(24,1,2),(60,1,3),(80,1,4)],9)[0]==[(24,24,2),(50,50,7)]
    for N in(3,25,100,1000):
        eligible,oldfolds,count=c.schedule(N,1,'half_doubling');m=max(2 if c.BANK[i]['kind']=='blocks'else 24 for i in eligible)
        mapped,meta=map_folds(N,1,m,oldfolds,count);assert mapped==oldfolds and meta['requested_ratio_preserved']
def test_current_control_mock_byte_exact_frozen41_and_H1score_equivalence():
    current,_=stage(21,mode='current');x,w,d=fixture()
    with patch.object(old,'schedule',return_value=(ELIGIBLE,FOLDS,8)),patch.object(old,'all_chunks',side_effect=chunks),patch.object(old,'coefficients',side_effect=coefficients):reference=old.prepare(x,w,21,d,fit_cache=MockWork(),solver=solver,menu='finite_wealth41')
    same(current,reference)
    # Actual H1 old schedule, mocked path law/coefficients, unchanged objective and seeds.
    outputs=[]
    for mode in c.MODES:
        with patch.object(c,'all_chunks',side_effect=chunks),patch.object(c,'coefficients',side_effect=coefficients):outputs.append(c.prepare(x,w,1,d,mode=mode,fit_cache=MockWork(),solver=solver))
    same(*outputs);assert outputs[0]['fit']['flat']['objective']==outputs[1]['fit']['flat']['objective']
    assert [t['seed']for t in outputs[0]['trace']]==[t['seed']for t in outputs[1]['trace']]
def test_mapped_menu_chronology_hashes_and_no_future_access():
    result,work=stage(21);x,w,d=fixture();assert len(work.calls)-1<=len(FOLDS)
    assert all(set(ids)<=set(c.CORE)for n,ids in work.calls)
    assert work.calls[-1][0]==len(x)
    for t in result['trace']:
        n,h=t['training_rows'],t['validation_horizon'];assert n+h<=len(x)
        assert t['training_sha256']==c.parent.signature(x[:n])and t['training_last_date']==str(d[n-1])and t['validation_last_known_date']==str(d[n+h-1])
        expected=int.from_bytes(hashlib.sha256(f'horizon_pool_inner_v1:{c.parent.signature(x[:n])}:{h}:2'.encode()).digest()[:4],'little');assert t['seed']==expected
    assert result['geometry']['represented_origins']==8
def test_cache_explicit_mapped_stops_exact_prefix_weights_and_closed_guard():
    x,w,d=fixture()
    with c.GroupWork(x,w,d,horizons=(21,),skip_unused_parent_calibration=True)as work:
        for mode in c.MODES:
            folds=c.folds_for(len(x),21,mode)[1];work.validate_geometry(21,mode,folds)
        with pytest.raises(ValueError):work.validate_geometry(21,'mapped_ratio',[(2,1,1)])
        with pytest.raises(ValueError):work.prefix_fits(x[:3],list(c.CORE),w)
        # Patch only test delegate; no financial fit runs.
        with patch.object(c.parent.GroupWork,'prefix_fits',return_value=({'families':{}},{}))as call:
            n=c.folds_for(len(x),21,'mapped_ratio')[1][0][0];work.prefix_fits(x[:n],list(c.CORE),w);assert call.call_count==1
        # The inherited exact-prefix guard is exercised before any family estimation.
        bad=x[:n].copy();bad[0,0]+=1
        with pytest.raises(ValueError):work.prefix_fits(bad,list(c.CORE),w)
        with pytest.raises(ValueError):work.prefix_fits(x[:n],[148],w)
    with pytest.raises(ValueError):work.prefix_fits(x[:n],list(c.CORE),w)
def test_no_model_fallback_and_source_menu_identity_guards():
    good,_=stage();assert c.identity('current')!=c.identity('mapped_ratio')
    with pytest.raises(ValueError):c.build_model('current',fixture()[2],good)
    with patch.object(c,'sources',return_value={}):
        with pytest.raises(ValueError):c.build_model('mapped_ratio',fixture()[2],good)
    with patch.object(c,'prepare',side_effect=[ArithmeticError('fit unavailable'),good]):
        result=c.prepare_all(*fixture()[:2],21,fixture()[2],fit_cache=MockWork(),solver=solver)
    assert result['current']['unsupported']and result['mapped_ratio']is good
    with pytest.raises(ArithmeticError):c.build_model('current',fixture()[2],result['current'])
def test_prescribed_states_current_control_actual_coefficients_and_H1_paths():
    class Prescribed(MockWork):
        def prefix_fits(self,x,ids,w):
            self.calls.append((len(x),tuple(ids)));p=x.shape[1];z=np.zeros(p);mean={'sample_mean':np.array([-.0002,.0004]),'center':np.array([0.,.0001]),'root':np.array([[.001,0],[.0002,.001]])};families={}
            for family in('constant','EWMA_reverting','SV_RMS'):
                f={k:z.copy()for k in('alpha','beta','positive','negative','omega','initial','decay','phi','level','sv_phi','q','m','P','V')}
                f.update(initial=np.array([1.2,.8]),decay=np.array([.9,.95]),target=np.array([.0001,.0002]),weights=np.zeros((p,3)),ring=np.ones((p,22)),pool=np.ascontiguousarray(x/.01),root=np.array([[1.,0.],[.2,.98]]),vol_root=np.eye(p),inverse_df=.1,level=np.array([-.1,-.2]),sv_phi=np.array([.8,.9]),q=np.array([.01,.02]),m=np.array([0.,.1]),P=np.array([.03,.04]))
                families[family]={'raw':{'mean':mean['sample_mean']},'mean':mean,'filter':f,'family':family,'history_sha256':c.parent.signature(x),'pool_fit':{'raw_fraction':0.,'ewma_fit':[]}}
            parent={**families['SV_RMS'],'pool_fit':{'raw_fraction':0.,'ewma_fit':[{'initial_variance_ratio':1.,'decay':.9}for _ in range(p)]}}
            return {'families':families,'parent':parent},{}
    ids=[i for i in c.CORE if 4<=i<166]
    with patch.object(c,'schedule',return_value=(ids,FOLDS,8)):
        result=c.prepare(*fixture()[:2],21,fixture()[2],mode='current',fit_cache=Prescribed(),solver=solver)
    with patch.object(old,'schedule',return_value=(ids,FOLDS,8)):
        reference=old.prepare(*fixture()[:2],21,fixture()[2],fit_cache=Prescribed(),solver=solver,menu='finite_wealth41')
    same(result,reference)
    # Whole H1 pipeline, synthetic financial states and real path/coefficient kernels.
    outputs=[]
    for mode in c.MODES:outputs.append(c.prepare(*fixture()[:2],1,fixture()[2],mode=mode,fit_cache=Prescribed(),solver=solver))
    same(*outputs)
    paths=[]
    for mode,b in zip(c.MODES,outputs):
        model=c.build_model(mode,fixture()[2],b);paths.append(np.concatenate([v for _,v in model.terminal_chunks(*fixture()[:2],1,240,991)],axis=1))
    assert paths[0].tobytes()==paths[1].tobytes()
def test_suffix_does_not_rewrite_earlier_prefix_and_no_outside_prefix_dates():
    x,w,d=fixture();changed=x.copy();changed[38:]+=7;seen=[]
    def draw(history,*args,**kwargs):seen.append(history.copy());yield from chunks(history,*args,**kwargs)
    for history in(x,changed):
        with patch.object(c,'schedule',return_value=(ELIGIBLE,FOLDS,8)),patch.object(c,'all_chunks',side_effect=draw),patch.object(c,'coefficients',side_effect=coefficients):c.prepare(history,w,21,d,fit_cache=MockWork(),solver=solver)
    assert seen[0].tobytes()==seen[2].tobytes()==x[:24].tobytes()
    with pytest.raises(ValueError,match='dated prefix'):c.prepare(x,w,21,np.append(d,d[-1]+np.timedelta64(1,'D')),fit_cache=MockWork(),solver=solver)
