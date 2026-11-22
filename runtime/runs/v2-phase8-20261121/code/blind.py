"""Calibrate exclusively on C and seal all student rankings before scoring."""
import math
from common import *
locked_access('calibrate')
sys.path[:0]=[str(phase(5)/'code'),str(phase(3)/'pilot/vendor/cpu')]
import numpy as np
from scipy.special import expit,logit
from phase5_io import HEADS,ROLES
from phase5_calibration import calibrate,c_arrays
from phase5_ranking import rank_query
from phase5_verify import independently_fit,check_correlations,pl_gradient

def verify_cal(cal,raw,target,oof,users):
    logits=logit(raw);fold=np.arange(len(users))%3
    check=np.full_like(oof,np.nan)
    for f in range(4):
        mask=fold!=f if f<3 else np.ones(len(users),dtype=bool)
        fits=cal['folds'][f]['heads'] if f<3 else cal['full_C_heads']
        for k,h in enumerate(HEADS):
            x,y=logits[mask,:,k],target[mask,:,k];ok=np.isfinite(x)&np.isfinite(y)
            fit=fits[h]
            ref=independently_fit(x,y);temp=fit['temperature']
            assert fit['n']==ok.sum() and fit['positive']==int(y[ok].sum())
            if ref is None:assert temp is None
            else:
                assert abs(math.log(ref/temp))<2e-6,(f,h,ref,temp)
                if f<3:check[~mask,:,k]=expit(logits[~mask,:,k]/temp)
    assert np.allclose(check,oof,equal_nan=True,atol=1e-10)
    corr,deg,counts=check_correlations(check[:,:,:4],target[:,:,0])
    for i,p in enumerate(cal['pairs']):
        for s in [0,1]:
            d=p[str(s)]
            assert (d['n_users'],d['n_jobs'])==counts[i,s] and bool(d['degenerate'])==deg[i,s]
            assert abs(d['rho']-corr[i,s])<1e-10
        assert abs(p['rho']-corr[i,2])<1e-10
    w={}
    for i,r in enumerate(ROLES):
        errors=(check[:,:,i]-target[:,:,0])**2
        good=np.isfinite(errors).all(axis=1)
        assert good.sum()==cal['roles'][r]['n_users']
        if good.any():
            b=errors[good].mean(axis=1).mean()
            assert abs(b-cal['roles'][r]['brier'])<1e-10
            penalty=1+sum(max(0,p['rho']) for p in cal['pairs'] if r in (p['a'],p['b']))
            w[r]=math.exp(-4*b)/penalty
    for r in w:assert abs(cal['roles'][r]['T1']-w[r]/sum(w.values()))<1e-10

def main():
    upstream=phase(2);split=read(upstream/'data/splits.json')['standard'];c,t=split['C'],split['T']
    assert len(c)==128 and len(t)==256 and set(c).isdisjoint(t)
    lists={v['query']:v for v in records(upstream/'data/lists.jsonl') if v['query'] in set(c+t)}
    clabel=upstream/'labels/C.jsonl';assert sha(clabel)==read(phase(5)/'acceptance.json')['input_sha256'][str(clabel.relative_to(ROOT))]
    truth={v['query']:v['outcomes'] for v in records(clabel)};assert set(truth)==set(c)
    votes={(v['query'],v['role']):v for v in records(S/'reviews.jsonl')}
    assert len(votes)==1536
    original=phase(4)/'base/reviews.jsonl';assert sha(original)==read(phase(5)/'acceptance.json')['input_sha256'][str(original.relative_to(ROOT))]
    for v in records(original):
        if v['role']=='general' and v['query'] in t:votes[v['query'],'general']=v
    raw,y=c_arrays(c,truth,votes);cal,oof=calibrate(c,raw,y)
    old=read(phase(5)/'base/calibration.json')
    for a,b in zip(cal['folds'],old['folds']):
        for k in ('fold','fit_queries','held_queries'):assert a[k]==b[k]
    verify_cal(cal,raw,y,oof,c)
    cal['source_hashes']={str(p.relative_to(ROOT)):sha(p) for p in [S/'reviews.jsonl',clabel,upstream/'data/splits.json',upstream/'data/lists.jsonl',D/'adapter/adapter_model.safetensors']}
    write(S/'calibration.json',cal)
    write_rows(S/'oof.jsonl',(dict(query=q,slot=f'P{j+1:02}',job_id=lists[q]['job_ids'][j],fold=i%3,**{h:float(oof[i,j,k]) if np.isfinite(oof[i,j,k]) else None for k,h in enumerate(HEADS)}) for i,q in enumerate(c) for j in range(10)))
    results=[];grads=[]
    for q in t:
        listing=lists[q];jobs=listing['job_ids'];row={r:votes[q,r] for r in (*ROLES,'general')}
        m=rank_query(q,listing,row,cal);trusted=m['T1'];v=m['F'];v['method']='F-SFT';order=v['order_job_ids']
        assert len(set(order))==10 and set(order)==set(jobs) and v['top5']==order[:5] and v['list_sha256']==listing['sha256']
        if not trusted['fallback']:
            g=pl_gradient(trusted['scores'],{r:row[r]['order_job_ids'] for r in trusted['used_roles']},trusted['weights'])
            assert g<=1e-5;grads.append(g)
        if not v['fallback']:
            for j in jobs:
                rnk=trusted['order_job_ids'].index(j)+1
                parts=v['components'][j]
                assert parts['T1_rank']==rnk
                pred=[]
                for role,h in [('preference','pC'),('employer','pE')]:
                    p=next(z[h] for z in row[role]['predictions'] if z['job_id']==j)
                    tval=cal['full_C_heads'][role+'.'+h]['temperature'];val=expit(logit(p)/tval)
                    assert abs(parts[h]-val)<1e-12;pred.append(val)
                score=.5*(10-rnk)/9+.5*math.sqrt(pred[0]*pred[1]);assert abs(score-v['scores'][j])<1e-12
            assert order==sorted(jobs,key=lambda j:(-v['scores'][j],j))
        else:assert order==(jobs if v['fallback']['method']=='BM25' else trusted['order_job_ids'])
        results.append(v)
    write_rows(S/'rankings.jsonl',results)
    files=[S/n for n in ['reviews.jsonl','windows.jsonl','failures.jsonl','calibration.json','oof.jsonl','rankings.jsonl']]+list((D/'adapter').glob('*'))
    write(E/'blind_seal.json',dict(at=time.time(),rankings=len(results),max_gradient=max(grads,default=None),files={str(p.relative_to(RUN)):sha(p) for p in files if p.is_file()}))
    log('学生 256 份十岗排序已封存，校准仅使用 C。')
if __name__=='__main__':main()

