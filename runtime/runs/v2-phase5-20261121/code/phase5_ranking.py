"""Apply the frozen C parameters to a single test query without access to labels."""
import math

import numpy as np
from scipy.optimize import minimize

from phase5_io import ROLES
from phase5_calibration import normalized, probability


def score_order(scores):
    return sorted(scores, key=lambda j:(-scores[j],j))


def pl_fit(jobs, votes, weights):
    # The same suffix likelihood as the Phase 3 CPU probe, with a solver receipt.
    keys=sorted(jobs)
    ix={j:i for i,j in enumerate(keys)}
    orders=np.array([[ix[j] for j in votes[r]] for r in weights])
    weight=np.array(list(weights.values()))[:,None]
    def objective(theta):
        x=theta[orders]
        tails=np.logaddexp.accumulate(x[:,::-1],axis=1)[:,::-1]
        value=np.sum(weight*(tails[:,:-1]-x[:,:-1]))+.1*(theta@theta)
        contributions=np.exp(x)*np.cumsum(np.exp(-tails),axis=1)-1.
        grad=.2*theta.copy()
        np.add.at(grad,orders,(weight*contributions))
        return value,grad
    res=minimize(objective,np.zeros(len(keys)),jac=True,method="L-BFGS-B",
                 options={"ftol":1e-15,"gtol":1e-9,"maxiter":300,"maxls":50})
    grad=float(np.max(np.abs(res.jac)))
    ok=bool(res.success and np.isfinite(res.x).all() and grad<=1e-5)
    detail={"ok":ok,"status":int(res.status),"message":str(res.message),"iterations":int(res.nit),
            "gradient_inf":grad if math.isfinite(grad) else None, "objective":float(res.fun) if np.isfinite(res.fun) else None}
    return ({j:float(x) for j,x in zip(keys,res.x)} if ok else None),detail


def bilateral(R,pC,pE):
    return .5*R+.5*math.sqrt(pC*pE)


def rank_query(q, listing, review, calibration, solver=pl_fit):
    jobs=listing["job_ids"]
    votes={r:row["order_job_ids"] for r,row in review.items() if row and row["status"]=="valid"}
    roles=[r for r in ROLES if r in votes]
    points={r:{j:9-i for i,j in enumerate(vote)} for r,vote in votes.items()}
    out={}
    def record(method, used, weights, scores=None, order=None, fallback=None, **extras):
        ranking=(score_order(scores) if scores is not None else list(jobs)) if order is None else list(order)
        out[method]=dict(block="T",query=q,list_sha256=listing["sha256"],method=method,
                         order_job_ids=ranking,top5=ranking[:5],scores=scores,used_roles=list(used),
                         weights=weights,fallback=fallback,**extras)
    def bm(reason):
        return {"method":"BM25","reason":reason}
    if "general" in votes:
        record("B1",["general"],{"general":1.}, {j:points["general"][j]/9 for j in jobs}, votes["general"])
    else:record("B1",[],{},fallback=bm("missing_general"))
    for excluded in (None,*ROLES):
        method="B4" if excluded is None else "B4-minus-"+excluded
        used=[r for r in roles if r!=excluded]
        ws=normalized({r:1. for r in used})
        if len(used)<2:record(method,used,ws,fallback=bm("fewer_than_two_roles"))
        else:
            scores={j:sum(points[r][j] for r in used)/(9*len(used)) for j in jobs}
            record(method,used,ws,scores)
    for method in ("T0","T1"):
        stats=calibration["roles"]
        usable=[r for r in roles if stats[r]["reliability"] is not None]
        raw={r:stats[r]["reliability"]/(stats[r]["denominator"] if method=="T1" else 1.) for r in usable}
        weights=normalized(raw)
        if len(usable)<2:record(method,usable,weights,fallback=bm("fewer_than_two_weighted_roles"))
        else:
            scores,detail=solver(jobs,votes,weights)
            record(method,usable,weights,scores,fallback=bm("PL_failed") if scores is None else None,solver=detail)
    trusted=out["T1"]
    ws=trusted["weights"]
    if len(ws)<2:record("W1",list(ws),ws,fallback=bm("fewer_than_two_weighted_roles"))
    else:record("W1",list(ws),ws,{j:math.fsum(ws[r]*points[r][j] for r in ws)/9 for j in jobs})
    if trusted["fallback"]:
        record("F",list(ws),ws,order=trusted["order_job_ids"],fallback=bm("T1_"+trusted["fallback"]["reason"]),components=None)
    else:
        heads=calibration["full_C_heads"]
        sides={}
        for role,key in (("preference","pC"),("employer","pE")):
            by_job={p["job_id"]:p.get(key) for p in review[role]["predictions"]} if role in roles else {}
            temp=heads[role+"."+key]["temperature"]
            sides[key]={j:probability(by_job.get(j),temp) for j in jobs}
        comp={}
        for idx,j in enumerate(trusted["order_job_ids"]):
            R=(9-idx)/9
            pc,pe=sides["pC"][j],sides["pE"][j]
            ok=pc is not None and pe is not None
            comp[j]={"T1_rank":idx+1,"R":R,"pC":pc,"pE":pe,"bilateral_utility":math.sqrt(pc*pe) if ok else None,
                     "F":bilateral(R,pc,pe) if ok else None}
        if any(v["F"] is None for v in comp.values()):
            record("F",list(ws),ws,order=trusted["order_job_ids"],fallback={"method":"T1","reason":"missing_bilateral_head"},components=comp)
        else:record("F",list(ws),ws,{j:v["F"] for j,v in comp.items()},components=comp)
        for i,j in enumerate(out["F"]["order_job_ids"],1):comp[j]["final_rank"]=i
    return out
