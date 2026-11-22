"""Frozen CPU evaluator. Invoke preflight and score in that order."""
from experiment_paths import ROOT, phase, lock_sources, source_hashes
import argparse
import csv
import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

RUN = phase(6)
OUT = RUN / 'eval/main'
EV = RUN / 'evidence'
P2, P4, P5 = map(phase, [2, 4, 5])
ROLES = ['skill','trajectory','preference','employer']
METHODS = ['B1','B4','T0','T1','W1','F'] + ['B4-minus-'+s for s in ROLES]
METRICS = ['joint_ndcg','joint_recall','delivery_ndcg','employer_ndcg','joint_hits']
PRIMARY = [('B4','B1'),('T1','T0'),('F','T1'),('F','B4')]
SEED = 20261121
def now():
    return datetime.now(timezone.utc).isoformat()
def read(p):
    return json.loads(p.read_text())
def write(p, data):
    p.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+'\n')
def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()
def manifest(p):
    return {name:val for val,name in (line.split(maxsplit=1) for line in p.read_text().splitlines())}
def records(p):
    return [json.loads(line) for line in p.read_text().splitlines()]
def csvout(name, rows):
    with (OUT/name).open('w', newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n')
        w.writeheader()
        w.writerows(rows)
def readcsv(p):
    with p.open() as f:
        return list(csv.DictReader(f))
def ndcg(labels):
    positives=sum(labels)
    if positives==0:
        return None
    ideal=sum(1/math.log2(i+2) for i in range(min(5,positives)))
    return sum(x/math.log2(i+2) for i,x in enumerate(labels[:5]))/ideal
def measures(delivered,joint,satisfied):
    hits=sum(joint[:5])
    sub=[s for s,d in zip(satisfied,delivered) if d]
    return [ndcg(joint),hits/sum(joint) if sum(joint) else None,ndcg(delivered),ndcg(sub),hits]
def estimate(arr):
    if not len(arr):
        return dict(n=0,mean=None,lo=None,hi=None,reason='无可评价用户')
    # Same row order and same draws for every method on a common denominator.
    rng=np.random.default_rng(SEED)
    draws=rng.integers(0,len(arr),size=(2000,len(arr)))
    lo,hi=np.quantile(arr[draws].mean(axis=1),[.025,.975])
    return dict(n=len(arr),mean=float(arr.mean()),lo=float(lo),hi=float(hi),reason='')
def permutation(diff):
    if not len(diff):
        return None
    signs=np.random.default_rng(SEED).integers(0,2,(10000,len(diff)))*2-1
    null=np.abs((signs*diff).sum(axis=1))
    observed=abs(diff.sum())
    return (1+int(np.count_nonzero((null>=observed)|np.isclose(null,observed,rtol=1e-14,atol=1e-14))))/10001
def inputs():
    split=read(P2/'data/splits.json')['standard']['T']
    assert len(split)==len(set(split))==256
    refs=[row for row in records(P2/'data/lists.jsonl') if row['query'] in split]
    lists=records(P5/'base/candidate_lists.jsonl')
    assert sorted(lists,key=lambda x:x['query'])==sorted(refs,key=lambda x:x['query'])
    byq={row['query']:row for row in lists}
    assert set(byq)==set(split) and len(lists)==256
    for row in lists:
        ids=row['job_ids']
        assert len(ids)==len(set(ids))==10
        assert hashlib.sha256(json.dumps(ids,separators=(',',':')).encode()).hexdigest()==row['sha256']
    ranks=records(P5/'base/rankings.jsonl')
    seen=set()
    for row in ranks:
        q,m=row['query'],row['method']
        order=row['order_job_ids']
        assert row['block']=='T' and (q,m) not in seen and m in METHODS
        assert len(order)==10 and set(order)==set(byq[q]['job_ids'])
        assert row['top5']==order[:5] and row['list_sha256']==byq[q]['sha256']
        seen.add((q,m))
    assert seen=={(q,m) for q in split for m in METHODS}
    return sorted(split),byq,ranks

def preflight():
    d=[int(i in (2,6,8)) for i in range(1,11)]
    j=[int(i==6) for i in range(1,11)]
    s=[a if b else None for a,b in zip(j,d)]
    vals=measures(d,j,s)
    assert vals[-1]==0 and [v for v in s if v is not None].index(1)+1==2
    assert math.isclose(vals[3],1/math.log2(3))
    assert measures([0]*10,[0]*10,[None]*10)==[None,None,None,None,0]
    assert measures([1]*10,[0]*10,[0]*10)==[None,None,1,None,0]
    assert measures([1]*10,[1]*10,[1]*10)==[1,.5,1,1,5]
    assert estimate(np.array([]))['reason'] and permutation([]) is None
    files=[*(ROOT / p for p in source_hashes(6)), EV/'source_hashes.json',EV/'experimental_setting.md',EV/'phase6_evaluation.md',EV/'acceptance_result_matrix.md']
    h2=manifest(P2/'phase2.sha256')
    for name in ['data/splits.json','data/lists.jsonl']:
        assert sha(P2/name)==h2[name]
        files.append(P2/name)
    for name,h in manifest(P5/'phase5.sha256').items():
        assert sha(P5/name)==h
    assert sha(P4/'base/reviews.jsonl')==read(P5/'acceptance.json')['input_sha256'][str((P4/'base/reviews.jsonl').relative_to(ROOT))]
    files += [P5/'phase5.sha256',P2/'phase2.sha256',P4/'base/reviews.jsonl',P5/'base/rankings.jsonl',P5/'base/candidate_lists.jsonl',P5/'base/calibration.json']
    queries,_,ranks=inputs()
    fp=hashlib.sha256()
    for row in sorted((r for r in ranks if r['method']=='F'),key=lambda r:r['query']):
        fp.update((json.dumps([row['query'],row['order_job_ids']],separators=(',',':'))+'\n').encode())
    write(EV/'preflight.json',dict(at=now(),seed=SEED,python=sys.version,numpy=np.__version__,rankings=len(ranks),users=len(queries),T_label_opens=0,synthetic_employer=vals[3],sample_joint_hits=0,sample_employer_positive_rank=2,checks=['零投递','无认可','理想排序','空分母','十种方法名单、排列与前五'],F_orders_sha256=fp.hexdigest(),expected_test_label_sha256=h2['labels/T.jsonl'],hashes={str(f.relative_to(ROOT)):sha(f) for f in files}))

def score():
    start=time.perf_counter()
    gate=read(EV/'preflight.json')
    for file,h in gate['hashes'].items():
        assert sha(ROOT/file)==h
    users,lists,ranks=inputs()
    label_file=P2/'labels/T.jsonl'
    opened=now()
    raw=label_file.read_bytes()
    assert hashlib.sha256(raw).hexdigest()==gate['expected_test_label_sha256']
    labels={}
    denom=[]
    for line in raw.splitlines():
        row=json.loads(line)
        q=row['query']
        assert q not in labels and q in lists
        ids=lists[q]['job_ids']
        slots={f'P{i:02}':job for i,job in enumerate(ids,1)}
        outcomes=row['outcomes']
        assert len(outcomes)==10 and {x['slot'] for x in outcomes}==set(slots)
        byjob={}
        for x in outcomes:
            d,s,j=x['D'],x['S'],x['J']
            assert d in (0,1) and bool(d)==x['delivered_mask']
            assert (s in (0,1) if d else s is None) and j==(s if d else 0)
            byjob[slots[x['slot']]]=(d,j,s)
        labels[q]=byjob
        denom.append(dict(query=q,delivered=sum(o[0] for o in byjob.values()),joint=sum(o[1] for o in byjob.values())))
    assert set(labels)==set(users)
    csvout('denominators.csv',sorted(denom,key=lambda x:x['query']))
    per=[]
    cube={}
    for row in ranks:
        q,m=row['query'],row['method']
        values=measures(*zip(*(labels[q][j] for j in row['order_job_ids'])))
        per.append(dict(query=q,method=m,list_sha256=row['list_sha256'],fallback=row['fallback'] or '',**dict(zip(METRICS,values))))
        cube[q,m]=values
    csvout('per_user.csv',per)
    means={}
    masks={}
    for k,metric in enumerate(METRICS):
        masks[metric]=[q for q in users if cube[q,'B1'][k] is not None]
        means[metric]={}
        for m in METHODS:
            assert masks[metric]==[q for q in users if cube[q,m][k] is not None]
            means[metric][m]=estimate(np.array([cube[q,m][k] for q in masks[metric]]))
    specs=[(a,b,0,True) for a,b in PRIMARY]+[('T1','W1',0,False)]+[(m,'B4',0,False) for m in METHODS[6:]]+[('F','T1',k,False) for k in (2,3)]
    paired=[]
    detail=[]
    for a,b,k,primary in specs:
        metric=METRICS[k]
        diff=np.array([cube[q,a][k]-cube[q,b][k] for q in masks[metric]])
        detail.extend(dict(query=q,first=a,second=b,metric=metric,delta=float(x)) for q,x in zip(masks[metric],diff))
        paired.append(dict(first=a,second=b,metric=metric,primary=primary,**estimate(diff),p=permutation(diff) if primary else None,holm=None))
    family=sorted([x for x in paired if x['p'] is not None],key=lambda x:x['p'])
    bound=0.
    for i,x in enumerate(family):
        bound=max(bound,min(1,(len(PRIMARY)-i)*x['p']))
        x['holm']=bound
    csvout('paired_deltas.csv',detail)
    csvout('comparisons.csv',paired)
    counts=dict(all=len(users),joint=sum(i['joint']>0 for i in denom),delivered=sum(i['delivered']>0 for i in denom),no_joint=sum(i['joint']==0 for i in denom),no_delivery=sum(i['delivered']==0 for i in denom),delivered_no_approval=sum(i['delivered']>0 and i['joint']==0 for i in denom))
    write(OUT/'summary.json',dict(seed=SEED,denominators=counts,methods=METHODS,metrics=means,comparisons=paired))
    write(EV/'score_receipt.json',dict(label_open_utc=opened,label_reads=1,completed=now(),sha256=gate['expected_test_label_sha256'],joined_users=len(labels),joined_pairs=sum(map(len,labels.values())),cpu_wall_s=time.perf_counter()-start,gpu_hours=0))
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("preflight", "score"))
    action = parser.parse_args().action
    lock_sources(6)
    {"preflight": preflight, "score": score}[action]()
