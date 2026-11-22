"""Independent replay from exported per-user values and paired differences."""
from experiment_paths import lock_sources
from collections import defaultdict
from math import fsum, isclose
from time import perf_counter
import numpy as np
from phase6_score import OUT, EV, P5, SEED, METRICS, METHODS, readcsv, read, records, write

def check(a,b):
    assert isclose(float(a),float(b),rel_tol=1e-11,abs_tol=1e-13),(a,b)
def replay(values, target):
    n=len(values)
    assert int(target['n'])==n
    check(fsum(values)/n,target['mean'])
    samples=np.random.default_rng(SEED).choice(n,size=(2000,n))
    means=np.mean(np.take(values,samples),axis=1)
    for k,key in [(.025,'lo'),(.975,'hi')]:
        a=np.sort(means)
        offset=k*(len(a)-1)
        i=int(offset)
        check(a[i]+(a[i+1]-a[i])*(offset-i),target[key])

def verify():
    started=perf_counter()
    s=read(OUT/'summary.json')
    den=readcsv(OUT/'denominators.csv')
    counts=s['denominators']
    assert len(den)==counts['all']==256
    totals={i['query']:(int(i['delivered']),int(i['joint'])) for i in den}
    assert len(totals)==256
    assert counts['no_delivery']==sum(d==0 for d,j in totals.values())
    assert counts['delivered_no_approval']==sum(d>0 and j==0 for d,j in totals.values())
    assert counts['no_joint']==sum(j==0 for d,j in totals.values())
    assert counts['joint']+counts['no_joint']==counts['delivered']+counts['no_delivery']==256
    raw=readcsv(OUT/'per_user.csv')
    rows={(x['query'],x['method']):x for x in raw}
    assert len(rows)==len(raw)==2560
    data={}
    for m in METHODS:
        for k in METRICS:
            v={}
            for q,(d,j) in sorted(totals.items()):
                x=rows[q,m][k]
                eligible=(d>0 if k=='delivery_ndcg' else k=='joint_hits' or j>0)
                assert bool(x)==eligible
                if eligible:
                    v[q]=float(x)
                    assert 0<=v[q]<=(5 if k=='joint_hits' else 1+1e-12)
            replay(list(v.values()),s['metrics'][k][m])
            data[m,k]=v
    ds=defaultdict(dict)
    for row in readcsv(OUT/'paired_deltas.csv'):
        group=ds[row['first'],row['second'],row['metric']]
        assert row['query'] not in group
        group[row['query']]=float(row['delta'])
    tests=readcsv(OUT/'comparisons.csv')
    assert len(tests)==len(ds)==len(s['comparisons'])==11
    pvalues=[]
    for t,summary in zip(tests,s['comparisons']):
        for key,v in summary.items():
            assert str(v)==t[key] if v is not None else t[key]==''
        a,b,k=t['first'],t['second'],t['metric']
        av,bv=data[a,k],data[b,k]
        diff={q:av[q]-bv[q] for q in av}
        assert diff==ds[a,b,k]
        v=np.array(list(diff.values()))
        replay(v,t)
        if t['primary']=='True':
            sign=np.random.default_rng(SEED).choice([-1,1],size=(10000,len(v)))
            obs=abs(fsum(v))
            null=np.abs(sign @ v)
            p=(1+sum(x>=obs or isclose(x,obs,abs_tol=1e-14,rel_tol=1e-14) for x in null))/10001
            check(p,t['p'])
            pvalues.append((p,t))
    assert len(pvalues)==4
    ordered=sorted(pvalues,key=lambda item:item[0])
    for i,(_,t) in enumerate(ordered):
        check(min(1,max((4-j)*p for j,(p,_) in enumerate(ordered[:i+1]))),t['holm'])
    ranks={(r['query'],r['method']):r for r in records(P5/'base/rankings.jsonl')}
    changed={}
    for a,b in [('T1','T0'),('F','T1')]:
        pairs=[(ranks[q,a]['order_job_ids'],ranks[q,b]['order_job_ids']) for q in totals]
        changed[a+'-'+b]=dict(whole=sum(x!=y for x,y in pairs),top5=sum(x[:5]!=y[:5] for x,y in pairs))
    fallback={m:sum(bool(rows[q,m]['fallback']) for q in totals) for m in METHODS}
    write(EV/'statistics_verified.json',dict(users=256,methods=10,absolute_intervals=50,paired_intervals=11,permutations_per_test=10000,holm_family=4,changed_orders=changed,fallback=fallback,cpu_wall_s=perf_counter()-started))
if __name__ == "__main__":
    lock_sources(6)
    verify()
