"""Bounded exact financial-state reuse within ONE dated multi-H request group.

No optimizer, forecast or persistent cache is launched by this module.
"""
from collections import Counter,OrderedDict
import numpy as np
from candidates.asset_selector_menu_components import bank as menu
from candidates.asset_selector_information_origins.validation import schedule

def bytes_owned(value,seen=None):
    seen=set() if seen is None else seen
    if id(value) in seen:return 0
    seen.add(id(value))
    if isinstance(value,np.ndarray):return value.nbytes
    if isinstance(value,dict):return sum(bytes_owned(v,seen) for v in value.values())
    if isinstance(value,(tuple,list)):return sum(bytes_owned(v,seen) for v in value)
    return 128

def same_bits(left,right):
    left=np.asarray(left,dtype=float);right=np.asarray(right,dtype=float)
    return left.shape==right.shape and np.array_equal(np.ascontiguousarray(left).view(np.uint64),np.ascontiguousarray(right).view(np.uint64))

class GroupWork:
    def __init__(self,history,weights,training_dates,*,horizons=(1,5,10,21),schedule_rule='half_doubling',byte_limit=64*1024**2):
        self.x=np.array(history,dtype=float,copy=True);self.w=np.array(weights,dtype=float,copy=True)
        self.dates=np.array(training_dates,dtype='datetime64[D]',copy=True);self.horizons=tuple(horizons);self.rule=schedule_rule
        if not isinstance(byte_limit,int) or byte_limit<0:raise ValueError('Nonnegative request-local byte bound required')
        self.limit=byte_limit;self.cache=OrderedDict();self.sizes={};self.bytes=0;self.peak_bytes=0;self.calls=Counter();self.hits=Counter();self.misses=Counter();self.closed=False
        self.expected=Counter()
        for h in self.horizons:
            _,folds,_=schedule(len(self.x),h,self.rule)
            self.expected.update(stop for stop,_,_ in folds);self.expected[len(self.x)]+=1

    def validate_request(self,history,weights,horizon,dates,rule):
        if self.closed or not same_bits(history,self.x) or not same_bits(weights,self.w) or not np.array_equal(np.asarray(dates,dtype='datetime64[D]'),self.dates) or horizon not in self.horizons or rule!=self.rule:
            raise ValueError('Work must match its exact dated multi-horizon request')

    def _get(self,key,compute):
        if key in self.cache:
            self.hits[key[1]]+=1;self.cache.move_to_end(key);return self.cache[key]
        self.misses[key[1]]+=1;value=compute()
        if self.expected[key[0]]>1:
            size=bytes_owned(value)
            if size<=self.limit:
                while self.cache and self.bytes+size>self.limit:
                    old,_=self.cache.popitem(last=False);self.bytes-=self.sizes.pop(old)
                self.cache[key]=value;self.sizes[key]=size;self.bytes+=size;self.peak_bytes=max(self.peak_bytes,self.bytes)
        return value

    def _base_prefix(self,history,eligible):
        # Same call order, exceptions, eligible-family and parent conditions as
        # asset_selector_universe.bank.prefix_fits; only pure-state reuse differs.
        base_bank=menu.base;n=len(history);prepared={'families':{}};invalid={}
        if all(base_bank.BANK[i]['kind']=='blocks' for i in eligible):return prepared,invalid
        try:base=self._get((n,'base',''),lambda:base_bank.fits.base_fit(history))
        except Exception as e:return prepared,{i:{'type':type(e).__name__,'reason':str(e)} for i in eligible if base_bank.BANK[i]['kind']!='blocks'}
        needed={base_bank.BANK[i]['family'] for i in eligible if base_bank.BANK[i]['kind']=='filtered'}|({'SV_RMS'} if 0 in eligible else set())
        for family in sorted(needed):
            try:prepared['families'][family]=self._get((n,'family',family),lambda family=family:base_bank.pure_prepare(history,family,base))
            except Exception as e:
                for i in eligible:
                    if (base_bank.BANK[i]['kind']=='filtered' and base_bank.BANK[i]['family']==family) or (i==0 and family=='SV_RMS'):invalid[i]={'type':type(e).__name__,'reason':str(e)}
        if 0 in eligible and 0 not in invalid:
            try:prepared['parent']=self._get((n,'parent',''),lambda:base_bank.parent_prepare(history,'SV_RMS','EWMA',base,prepared['families']['SV_RMS']))
            except Exception as e:invalid[0]={'type':type(e).__name__,'reason':str(e)}
        return prepared,invalid

    def prefix_fits(self,history,eligible,weights):
        n=len(history)
        if self.closed or n>len(self.x) or not same_bits(history,self.x[:n]) or not same_bits(weights,self.w):raise ValueError('Cached financial fit must use the exact same-prefix history and weights')
        # PPW is outside active flat167; preserve its original law if requested.
        if any(i>=168 for i in eligible):return menu.prefix_fits(history,eligible,weights)
        original=[i for i in eligible if i<166]
        if any(i in (166,167) for i in eligible) and 0 not in original:original.append(0)
        prepared,invalid=self._base_prefix(history,original)
        if 0 in invalid:
            for i in (166,167):
                if i in eligible:invalid[i]=dict(invalid[0])
        result=prepared,{i:error for i,error in invalid.items() if i in eligible}
        self.calls[n]+=1
        if self.calls[n]>=self.expected[n]:
            for key in [key for key in self.cache if key[0]==n]:
                self.cache.pop(key);self.bytes-=self.sizes.pop(key)
        return result

    def diagnostics(self):
        return {'cache_scope':'Exact pure financial states within one dated multi-H group; no persistent cache','cache_byte_limit':self.limit,'cache_bytes':self.bytes,'cache_peak_bytes':self.peak_bytes,'cache_hits':dict(self.hits),'cache_misses':dict(self.misses)}

    def close(self):
        self.cache.clear();self.sizes.clear();self.bytes=0;self.x=None;self.w=None;self.dates=None;self.closed=True

    def __enter__(self):return self
    def __exit__(self,*args):self.close()
