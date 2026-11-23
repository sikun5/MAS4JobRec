"""Phase 9 only. Run lock, rank, score, audit in separate processes."""
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

RUN = Path(__file__).resolve().parent
ROOT = RUN.parents[2]
sys.path.insert(0, str(ROOT / "code"))
ROLES = ("skill", "trajectory", "preference", "employer")
METRICS = ("joint_ndcg", "joint_recall", "delivery_ndcg", "employer_ndcg", "joint_hits")
DEN = (228, 228, 250, 228, 256)
SEED = 20261121
def phase(n): return RUN.parent / f"v2-phase{n}-20261121"
def sha(p):
    h = hashlib.sha256()
    with p.open("rb") as f:
        for block in iter(lambda:f.read(1<<20), b""): h.update(block)
    return h.hexdigest()
def read(p): return json.loads(p.read_text())
def save(name, obj):
    (RUN/name).write_text(json.dumps(obj, indent=2, ensure_ascii=False, allow_nan=False)+"\n")
def table(p):
    with p.open(newline="") as f: return list(csv.DictReader(f))
def csvout(name, rows):
    with (RUN/name).open("w", newline="") as f:
        w=csv.DictWriter(f, rows[0].keys(), lineterminator="\n");w.writeheader();w.writerows(rows)
def rows(p):
    with p.open() as f:
        for line in f: yield json.loads(line)
def index(values):
    result={}
    for v in values:
        q=v["query"]; assert q not in result; result[q]=v
    return result
def guard_labels(event,args):
    if event=="open" and isinstance(args[0],(str,bytes,os.PathLike)):
        if "labels" in Path(os.fsdecode(args[0])).parts: raise PermissionError("Blind stage: labels denied")
def git(*args):return subprocess.check_output(["git", "-C", str(ROOT), *args])
def code_hashes():
    return {str(p.relative_to(ROOT)):sha(p) for p in (RUN/"phase9.py",RUN/"protocol.md",RUN/"execute.py",ROOT/"code/phase5_ranking.py",ROOT/"code/phase6_score.py",ROOT/"code/experiment_paths.py")}
def lock():
    names={2:["data/splits.json","data/lists.jsonl","labels/T.jsonl"],
           4:["base/reviews.jsonl"],5:["base/rankings.jsonl"],6:["eval/main/per_user.csv"],
           7:["distill/demos.jsonl","gpu_hours.csv","distill/teacher_manifest.json"],
           8:["distill/config.json","manifest.json","evidence/blind_seal.json","evidence/reload.json","evidence/adapter.json","student/reviews.jsonl","gpu_hours.csv"]}
    vote_manifest=phase(8)/"manifest.json"
    names[8]+=["distill/adapter/"+k for k in read(vote_manifest)["adapter"]]
    head=git("rev-parse","HEAD").decode().strip()
    assert head=="a09f11b8ba0cd672b5d5286c70a5b359b134fbfe"
    inputs={};manifests={}
    for n,paths in names.items():
        folder=phase(n); f=folder/f"phase{n}.sha256"
        rel=str(f.relative_to(ROOT))
        assert git("show",head+":"+rel)==f.read_bytes()
        manifests[rel]=sha(f)
        known={str(Path(k)):h for h,k in (s.split(maxsplit=1) for s in f.read_text().splitlines())}
        for name in paths:
            p=folder/name; key=str(p.relative_to(ROOT));inputs[key]=known[name]
            if name!="labels/T.jsonl":assert sha(p)==known[name],key
    cfg=read(phase(8)/"distill/config.json"); blind=read(phase(8)/"evidence/blind_seal.json")
    demos=phase(7)/"distill/demos.jsonl"
    assert sha(demos)==cfg["demos_sha256"] and sum(1 for _ in rows(demos))==cfg["steps"]==505
    assert cfg["r"]==16 and cfg["alpha"]==32 and cfg["role_conditioned"]
    review=read(vote_manifest);reload=read(phase(8)/"evidence/reload.json")
    assert review["adapter"]==reload["adapter"]==read(phase(8)/"evidence/adapter.json")
    assert Path(review["model"]).name==Path(reload["base"]).name==cfg["revision"]
    for name,h in review["adapter"].items():assert blind["files"]["distill/adapter/"+name]==h==sha(phase(8)/"distill/adapter"/name)
    assert blind["files"]["student/reviews.jsonl"]==sha(phase(8)/"student/reviews.jsonl")
    save("input_lock.json",dict(head=head,seed=SEED,inputs=inputs,upstream_manifests=manifests,code=code_hashes(),demos=505,adapter=review["adapter"],base_revision=cfg["revision"],label_hash_source="Phase 2 committed manifest; label unopened"))
def locked(n, name):
    p=phase(n)/name
    assert sha(p)==read(RUN/"input_lock.json")["inputs"][str(p.relative_to(ROOT))],name
    return p
def aggregate(q, listing, ballots, method):
    jobs=listing["job_ids"]; used=[r for r in ROLES if r in ballots]
    weights={r:1/len(used) for r in used}
    scores=None;fallback=None;order=list(jobs)
    if len(used)<2:fallback=dict(method="BM25",reason="fewer_than_two_roles")
    else:
        scores={j:sum(9-ballots[r].index(j) for r in used)/(9*len(used)) for j in jobs}
        order=sorted(jobs,key=lambda j:(-scores[j],j))
    return dict(block="T",query=q,list_sha256=listing["sha256"],method=method,order_job_ids=order,top5=order[:5],scores=scores,used_roles=used,weights=weights,fallback=fallback)
def get_ballots(n,path,lists):
    ballots={q:{} for q in lists};seen=set();states=Counter();total=0
    for row in rows(locked(n,path)):
        total+=1
        if row["block"]!="T":assert row["query"] not in lists;continue
        r,q=row["role"],row["query"];assert q in lists
        if r=="general" and n==4:continue
        assert r in ROLES and (q,r) not in seen;seen.add((q,r))
        assert row["list_sha256"]==lists[q]["sha256"]
        state=row["status"];assert state in ("missing","valid");states[state]+=1
        if state=="valid":
            order=row["order_job_ids"];assert len(order)==10 and set(order)==set(lists[q]["job_ids"])
            ballots[q][r]=order
    assert seen=={(q,r) for q in lists for r in ROLES}
    assert total==(1792 if n==4 else 1536)
    return ballots,dict(states)
def rank():
    qs=read(locked(2,"data/splits.json"))["standard"]["T"]
    assert len(qs)==len(set(qs))==256
    lists=index(x for x in rows(locked(2,"data/lists.jsonl")) if x["query"] in qs)
    assert set(lists)==set(qs)
    for v in lists.values():
        jobs=v["job_ids"];assert len(set(jobs))==len(jobs)==10
        assert hashlib.sha256(json.dumps(jobs,separators=(',',':')).encode()).hexdigest()==v["sha256"]
    fixture={"job_ids":list(reversed(sorted(jobs))),"sha256":"fixture"}
    asc=sorted(jobs);desc=asc[::-1]
    assert aggregate("q",fixture,{"skill":asc,"employer":desc},"B4")["order_job_ids"]==asc
    for v in ({},{"skill":asc}):
        result=aggregate("q",fixture,v,"B4");assert result["fallback"]["method"]=="BM25" and result["order_job_ids"]==desc
    base,bs=get_ballots(4,"base/reviews.jsonl",lists)
    archive=index(r for r in rows(locked(5,"base/rankings.jsonl")) if r["method"]=="B4")
    assert set(archive)==set(qs)
    for q in qs:
        result=aggregate(q,lists[q],base[q],"B4")
        assert result==archive[q],("baseline replay mismatch",q)
    student,ss=get_ballots(8,"student/reviews.jsonl",lists)
    changes=Counter(full=0,top5_order=0,top5_set=0); fallback=0
    with (RUN/"rankings.jsonl").open("w") as out:
        for q in sorted(qs):
            result=aggregate(q,lists[q],student[q],"B4-SFT")
            order=result["order_job_ids"];old=archive[q]["order_job_ids"]
            assert len(order)==10 and set(order)==set(old)
            changes["full"]+=order!=old;changes["top5_order"]+=order[:5]!=old[:5];changes["top5_set"]+=set(order[:5])!=set(old[:5])
            fallback+=bool(result["fallback"]);out.write(json.dumps(result,ensure_ascii=False)+"\n")
    save("ranking_seal.json",dict(epoch=time.time(),baseline_exact=256,student_rankings=256,base_ballots=bs,student_ballots=ss,student_fallbacks=fallback,changes=changes,synthetic=["tie by job ID","zero/one role BM25"],label_access="OS Python audit hook denied label opens",rankings_sha256=sha(RUN/"rankings.jsonl"),input_lock_sha256=sha(RUN/"input_lock.json")))
def score():
    import numpy as np
    from phase6_score import measures, estimate
    seal=read(RUN/"ranking_seal.json")
    assert sha(RUN/"rankings.jsonl")==seal["rankings_sha256"] and sha(RUN/"input_lock.json")==seal["input_lock_sha256"]
    d=[int(i in (2,6,8)) for i in range(1,11)];j=[int(i==6) for i in range(1,11)]
    ex=measures(d,j,[v if k else None for v,k in zip(j,d)])
    assert ex[-1]==0 and math.isclose(ex[3],1/math.log2(3))
    assert measures([0]*10,[0]*10,[None]*10)==[None,None,None,None,0]
    assert measures([1]*10,[0]*10,[0]*10)==[None,None,1,None,0]
    b=index(v for v in table(locked(6,"eval/main/per_user.csv")) if v["method"]=="B4")
    br=index(v for v in rows(locked(5,"base/rankings.jsonl")) if v["method"]=="B4")
    sr=index(rows(RUN/"rankings.jsonl"))
    lists=index(v for v in rows(locked(2,"data/lists.jsonl")) if v["query"] in b)
    label=phase(2)/"labels/T.jsonl"; opened=time.time(); assert opened>=seal["epoch"]
    raw=label.read_bytes();assert hashlib.sha256(raw).hexdigest()==read(RUN/"input_lock.json")["inputs"][str(label.relative_to(ROOT))]
    truth=index(map(json.loads,raw.splitlines())); assert len(b)==256 and b.keys()==br.keys()==sr.keys()==truth.keys()==lists.keys()
    output=[];deltas=[];den={k:[] for k in METRICS}
    for q in sorted(b):
        jobs=lists[q]["job_ids"]; assert b[q]["list_sha256"]==sr[q]["list_sha256"]==lists[q]["sha256"]
        slot={v["slot"]:v for v in truth[q]["outcomes"]}; assert len(truth[q]["outcomes"])==len(slot)==10
        y={k:slot[f"P{i:02}"] for i,k in enumerate(jobs,1)}
        assert all(v["D"] in (0,1) and (v["S"] in (0,1) if v["D"] else v["S"] is None) and v["J"]==(v["S"] if v["D"] else 0) for v in y.values())
        results={}
        for method,ranking in (("B4",br[q]),("B4-SFT",sr[q])):
            order=ranking["order_job_ids"];assert len(order)==10 and set(order)==set(y)
            values=measures(*zip(*[(y[k]["D"],y[k]["J"],y[k]["S"]) for k in order]))
            results[method]=dict(zip(METRICS,values))
            output.append(dict(query=q,method=method,list_sha256=b[q]["list_sha256"],fallback=json.dumps(ranking["fallback"]) if ranking["fallback"] else "",**results[method]))
        for k in METRICS:
            old=b[q][k];v=results["B4"][k];new=results["B4-SFT"][k]
            assert (v is None)==(old=="")== (new is None)
            if v is not None:
                assert math.isclose(v,float(old),abs_tol=1e-14,rel_tol=0)
                den[k].append(q)
        if b[q]["joint_ndcg"]!="":
            v=float(b[q]["joint_ndcg"]);s=results["B4-SFT"]["joint_ndcg"]
            deltas.append(dict(query=q,B4=v,B4_SFT=s,delta=s-v))
    for k,n in zip(METRICS,DEN):
        assert den[k]==sorted(q for q,r in b.items() if r[k]!="") and len(den[k])==n
    csvout("per_user.csv",output);csvout("paired_deltas.csv",deltas);save("denominators.json",den)
    effects=np.array([r["delta"] for r in deltas])
    summary={m:{k:estimate(np.array([r[k] for r in output if r["method"]==m and r[k] is not None])) for k in METRICS} for m in ("B4","B4-SFT")}
    save("summary.json",dict(metrics=summary,paired=estimate(effects),changes=seal["changes"],improved=int((effects>0).sum()),tied=int((effects==0).sum()),declined=int((effects<0).sum()),label_open_epoch=opened,label_reads=1,synthetic=["no delivery","no approvals","filter employer before cutoff"]))
def audit():
    import numpy as np
    # Independent estimator built from the CSV: Python fsum, choice rather than integers, percentile endpoints.
    per=table(RUN/"per_user.csv");delta=table(RUN/"paired_deltas.csv")
    summary=read(RUN/"summary.json");den=read(RUN/"denominators.json")
    original=index(r for r in table(locked(6,"eval/main/per_user.csv")) if r["method"]=="B4")
    def interval(values,expected):
        a=np.array(values); assert len(a)==expected["n"]
        positions=np.random.default_rng(SEED).choice(len(a),size=(2000,len(a)),replace=True)
        lo,hi=np.percentile(a[positions].mean(axis=1),[2.5,97.5])
        assert np.allclose([math.fsum(values)/len(a),lo,hi],[expected["mean"],expected["lo"],expected["hi"]],atol=1e-13,rtol=0)
    seen={}
    assert len(per)==512
    for method in ("B4","B4-SFT"):
        byq=index(r for r in per if r["method"]==method);seen[method]=byq
        assert byq.keys()==original.keys()
        for k in METRICS:
            qs=sorted(q for q,r in byq.items() if r[k]!="")
            assert qs==den[k]==sorted(q for q in original if original[q][k]!="")
            interval([float(byq[q][k]) for q in qs],summary["metrics"][method][k])
    assert [r["query"] for r in delta]==den["joint_ndcg"]
    vals=[]
    for r in delta:
        q=r["query"];v=float(original[q]["joint_ndcg"]);s=float(seen["B4-SFT"][q]["joint_ndcg"])
        assert float(r["B4"])==v and float(r["B4_SFT"])==s and float(r["delta"])==s-v
        vals.append(s-v)
    interval(vals,summary["paired"])
    assert [sum(x>0 for x in vals),sum(x==0 for x in vals),sum(x<0 for x in vals)]==[summary[k] for k in ("improved","tied","declined")]
    old=index(r for r in rows(locked(5,"base/rankings.jsonl")) if r["method"]=="B4")
    new=index(rows(RUN/"rankings.jsonl")); assert new.keys()==original.keys()==old.keys()
    pairs=[(old[q]["order_job_ids"],new[q]["order_job_ids"]) for q in sorted(old)]
    assert summary["changes"]==dict(full=sum(a!=b for a,b in pairs),top5_order=sum(a[:5]!=b[:5] for a,b in pairs),top5_set=sum(set(a[:5])!=set(b[:5]) for a,b in pairs))
    assert sha(RUN/"rankings.jsonl")==read(RUN/"ranking_seal.json")["rankings_sha256"]
    costs={r["stage"]:float(r["gpu_hours"]) for r in table(locked(8,"gpu_hours.csv"))}
    ledger=table(locked(7,"gpu_hours.csv"))
    assert math.isclose(costs["teacher"],sum(float(r["gpu_hours"]) for r in ledger))
    save("audit.json",dict(status="passed",independent_intervals=11,base_replayed=256,student_rankings=256,paired_n=len(vals),costs=costs,ids_verified=True))
if __name__=="__main__":
    mode=sys.argv[1]; assert os.environ.get("CUDA_VISIBLE_DEVICES")==""
    if mode!="score":sys.addaudithook(guard_labels)
    if mode!="lock":assert read(RUN/"input_lock.json")["code"]==code_hashes()
    {"lock":lock,"rank":rank,"score":score,"audit":audit}[mode]()
