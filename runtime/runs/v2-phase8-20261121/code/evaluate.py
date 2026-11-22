"""Score once after hash verification on the identical test denominators."""
import math
from common import *
locked_access('score')
sys.path.insert(0,str(phase(6)/'code'))
import score as reference
import numpy as np

def verify_interval(values,summary):
    n=len(values);assert n==summary['n']
    assert math.isclose(math.fsum(values)/n,summary['mean'],abs_tol=1e-12)
    index=np.random.default_rng(SEED).choice(n,(2000,n))
    boot=np.array(values)[index].mean(axis=1)
    assert np.allclose(np.percentile(boot,[2.5,97.5]),[summary['lo'],summary['hi']],rtol=0,atol=1e-12)

def main():
    seal=read(E/'blind_seal.json')
    for path,h in seal['files'].items():assert sha(RUN/path)==h
    fref=phase(6)/'eval/main/F_per_user.csv';assert sha(fref)==read(phase(6)/'evidence/F_receipt.json')['F_per_user_sha256']
    with fref.open() as f:old={r['query']:r for r in csv.DictReader(f)}
    p2=phase(2);label=p2/'labels/T.jsonl';start=time.time()
    truth=label.read_bytes()
    expected=reference.manifest(p2/'phase2.sha256')['labels/T.jsonl']
    assert hashlib.sha256(truth).hexdigest()==expected
    labels={r['query']:r['outcomes'] for r in map(json.loads,truth.splitlines())}
    lists={r['query']:r for r in records(p2/'data/lists.jsonl') if r['query'] in old}
    ranks=records(S/'rankings.jsonl')
    assert len(labels)==len(ranks)==256 and set(labels)==set(old)=={r['query'] for r in ranks}
    values={};fallback={}; per=[]
    for r in sorted(ranks,key=lambda r:r['query']):
        q=r['query'];ids=lists[q]['job_ids']
        assert r['list_sha256']==lists[q]['sha256']==old[q]['list_sha256']
        la={x['slot']:x for x in labels[q]}
        ordered=[la[f'P{ids.index(j)+1:02}'] for j in r['order_job_ids']]
        assert len(ordered)==len(la)==10 and all(x['J']==(x['S'] if x['D'] else 0) for x in ordered)
        d,j,s=([x[k] for x in ordered] for k in ['D','J','S'])
        vals=reference.measures(d,j,s)
        def nd(z):
            total=sum(z)
            if not total:return None
            discount=lambda i:1/math.log2(i+2)
            return math.fsum(discount(i) for i,v in enumerate(z[:5]) if v)/math.fsum(discount(i) for i in range(min(total,5)))
        replay=[nd(j),sum(j[:5])/sum(j) if sum(j) else None,nd(d),nd([x['S'] for x in ordered if x['D']]),sum(j[:5])]
        for v,b in zip(vals,replay):assert (v is None and b is None) or (v is not None and math.isclose(v,b,abs_tol=1e-12))
        values[q]=dict(zip(reference.METRICS,vals));fb=r['fallback']
        if fb:fallback[q]=fb
        per.append(dict(query=q,method='F-SFT',list_sha256=r['list_sha256'],fallback=encode(fb) if fb else '',**values[q]))
    def csvfile(name,rows):
        with (OUT/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=rows[0].keys(),lineterminator='\n');w.writeheader();w.writerows(rows)
    csvfile('per_user.csv',per)
    with (OUT/'per_user.csv').open() as f:reread={r['query']:r for r in csv.DictReader(f)}
    methods={};den={}
    for k,n in zip(reference.METRICS,[228,228,250,228,256]):
        qs=sorted(q for q in values if values[q][k] is not None)
        assert qs==sorted(q for q,r in old.items() if r[k]!='') and len(qs)==n
        den[k]=qs;arr=[float(reread[q][k]) for q in qs]
        methods[k]=reference.estimate(np.array([values[q][k] for q in qs]))
        verify_interval(arr,methods[k])
    qs=den['joint_ndcg'];diff=[dict(query=q,student=values[q]['joint_ndcg'],original=float(old[q]['joint_ndcg']),delta=values[q]['joint_ndcg']-float(old[q]['joint_ndcg'])) for q in qs]
    csvfile('paired_deltas.csv',diff);effect=reference.estimate(np.array([r['delta'] for r in diff]))
    verify_interval([float(reread[q]['joint_ndcg'])-float(old[q]['joint_ndcg']) for q in qs],effect)
    write(OUT/'denominators.json',den);write(OUT/'summary.json',dict(method='F-SFT',metrics=methods,comparison='F-SFT−F',joint_ndcg_difference=effect,bootstrap=2000,seed=SEED,fallback=fallback))
    write(E/'score_receipt.json',dict(label_reads=1,opened_epoch=start,test_sha256=expected,blind_epoch=seal['at'],baseline_sha256=sha(fref),independent_intervals=6))
    assert start>=seal['at']
if __name__=='__main__':main()

