"""Cross-fitted probabilities, class-conditional residual correlations and frozen weights."""
from itertools import combinations

import numpy as np
from scipy.optimize import brentq
from scipy.special import expit, logit

from phase5_io import HEADS, ROLES

PAIRS = tuple(combinations(range(len(ROLES)),2))


def c_arrays(users, truth, reviews):
    shape = (len(users),10,len(HEADS))
    raw = np.full(shape,np.nan)
    target = np.full(shape,np.nan)
    for i,q in enumerate(users):
        for j,o in enumerate(truth[q]):
            assert o["slot"]==f"P{j+1:02}" and o["D"] in (0,1) and o["J"] in (0,1)
            assert (o["S"] in (0,1) if o["D"] else o["S"] is None)
            assert o["J"]==(o["S"] if o["D"] else 0)
            target[i,j,:4]=o["J"]
            target[i,j,4]=o["D"]
            if o["D"]:target[i,j,5]=o["S"]
        for h,head in enumerate(HEADS):
            role,key=head.split(".")
            row=reviews[q,role]
            if row["status"]=="valid":
                for j,p in enumerate(row["predictions"]):
                    v=p.get(key)
                    if v is not None:
                        assert 0<v<1
                        raw[i,j,h]=v
    return raw,target


def fit_temperature(x,y):
    valid=np.isfinite(x)&np.isfinite(y)
    x,y=x[valid],y[valid]
    n=len(y)
    positive=int(y.sum())
    if not n:
        temp,status=None,"empty"
    elif positive in (0,n):
        temp,status=1.0,"single_class"
    elif not np.any(x):
        temp,status=1.0,"flat_likelihood"
    else:
        # Logistic NLL is convex in inverse temperature. Its derivative is monotone.
        def gradient(beta):
            return float(np.mean(x*(expit(beta*x)-y)))
        if gradient(.05)>=0:
            temp,status=20.0,"upper_T"
        elif gradient(20)<=0:
            temp,status=.05,"lower_T"
        else:
            temp,status=1/brentq(gradient,.05,20.,xtol=1e-13),"interior"
    return {"temperature":temp,"status":status,"n":n,"positive":positive}


def crossfit(logits,target,fold):
    pred=np.full_like(logits,np.nan)
    fits=[]
    for f in range(3):
        held=fold==f
        fitted=[]
        for h in range(logits.shape[2]):
            fit=fit_temperature(logits[~held,:,h],target[~held,:,h])
            fitted.append(fit)
            if fit["temperature"] is not None:
                pred[held,:,h]=expit(logits[held,:,h]/fit["temperature"])
        fits.append(fitted)
    return pred,fits


def stratified_pairs(probs,joint):
    residuals=probs-joint[:,:,None]
    result=[]
    for a,b in PAIRS:
        pair={"a":ROLES[a],"b":ROLES[b]}
        for y in (0,1):
            mask=(joint==y)&np.isfinite(residuals[:,:,a])&np.isfinite(residuals[:,:,b])
            x=residuals[:,:,a][mask]
            z=residuals[:,:,b][mask]
            reason=None
            if len(x)<2:reason="fewer_than_two_pairs"
            elif np.ptp(x)==0 or np.ptp(z)==0:reason="constant_residual"
            if reason:
                rho=0.
            else:
                dx=x-x.mean(); dz=z-z.mean()
                rho=float(np.clip(np.sum(dx*dz)/np.sqrt(np.sum(dx*dx)*np.sum(dz*dz)),-1,1))
            pair[str(y)]={"n_jobs":len(x),"n_users":int(mask.any(axis=1).sum()),"rho":rho,"degenerate":reason}
        pair["rho"]=(pair["0"]["rho"]+pair["1"]["rho"])/2
        result.append(pair)
    return result


def normalized(raw):
    total=sum(raw.values())
    return {k:v/total for k,v in raw.items()} if total else {}


def calibrate(c,raw,truth):
    fold=np.arange(len(c))%3
    x=logit(raw)
    oof,foldfits=crossfit(x,truth,fold)
    pairs=stratified_pairs(oof[:,:,:4],truth[:,:,0])
    role_stats={}
    for i,r in enumerate(ROLES):
        errors=(oof[:,:,i]-truth[:,:,i])**2
        keep=np.isfinite(errors).all(axis=1)
        bs=float(errors[keep].mean(axis=1).mean()) if keep.any() else None
        penalty=1+sum(max(0.,p["rho"]) for p in pairs if r in (p["a"],p["b"]))
        role_stats[r]={"n_users":int(keep.sum()),"brier":bs,"reliability":float(np.exp(-bs/.25)) if bs is not None else None,
                       "denominator":penalty}
    w0=normalized({r:s["reliability"] for r,s in role_stats.items() if s["reliability"] is not None})
    w1=normalized({r:s["reliability"]/s["denominator"] for r,s in role_stats.items() if s["reliability"] is not None})
    for r,s in role_stats.items():
        s.update(T0=w0.get(r),T1=w1.get(r),delta_pp=100*(w1[r]-w0[r]) if r in w0 else None)
    folds=[dict(fold=f,fit_queries=[q for i,q in enumerate(c) if fold[i]!=f],held_queries=[q for i,q in enumerate(c) if fold[i]==f],
                heads=dict(zip(HEADS,foldfits[f]))) for f in range(3)]
    full={h:fit_temperature(x[:,:,i],truth[:,:,i]) for i,h in enumerate(HEADS)}
    return {"calibration_queries":c,"folds":folds,"full_C_heads":full,"roles":role_stats,"pairs":pairs,
            "temperature_bounds":[.05,20.],"brier_scale":.25,"PL_L2":.1,"F_mixture":[.5,.5]},oof


def probability(p,temp):
    if p is None or temp is None:return None
    return float(expit(logit(p)/temp))
