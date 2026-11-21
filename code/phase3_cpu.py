"""Time CPU-only aggregation and evaluation work without reading formal C/T labels."""
import json
import time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import softmax, logsumexp

folder=Path("runtime/runs/v2-phase3-20261121/pilot")
rng=np.random.default_rng(20261121)


def fit_pl(perms,w):
    def objective(theta):
        costs=0.
        grad=np.zeros(10)
        for p,weight in zip(perms,w):
            x=theta[p]
            sums=np.logaddexp.accumulate(x[::-1])[::-1]
            costs+=weight*(sums[:-1]-x[:-1]).sum()
            a=np.exp(x)*np.cumsum(np.exp(-sums))
            grad[p]+=weight*(a-1)
        return costs+0.1*(theta@theta),grad+0.2*theta
    return minimize(objective,np.zeros(10),jac=True,method="L-BFGS-B").x


def sample(n_pl):
    t=time.monotonic()
    preds=rng.uniform(0.05,.95,(128,10,4));labels=rng.integers(0,2,(128,10))
    logits=np.log(preds/(1-preds))
    # Time the temperature fits (four common plus two side heads).
    for mask in [np.arange(128)%3!=i for i in range(3)]+[np.ones(128,dtype=bool)]:
        for a in [0,1,2,3,2,3]:
            x=logits[mask,:,a]; y=labels[mask]
            minimize(lambda t:np.mean(np.logaddexp(0,x/t[0])-y*x/t[0]),[1.],bounds=[(.05,20.)])
    weights=np.exp(-np.mean((preds-labels[:,:,None])**2,axis=(0,1))/.25)
    weights/=weights.sum()
    for i in range(256):
        perms=np.stack([rng.permutation(10) for _ in range(4)])
        for _ in range(n_pl):fit_pl(perms,weights)
    # Paired resampling and absolute sign-flips on dummy query contrasts.
    for _ in range(4 if n_pl==2 else 1):
        delta=rng.normal(size=256)
        np.percentile(delta[rng.integers(256,size=(2000,256))].mean(1),[2.5,97.5])
        np.abs((delta*rng.choice([-1,1],size=(10000,256))).mean(1))
    return time.monotonic()-t

if __name__=="__main__":
    data={k:[sample(n) for _ in range(3)] for k,n in [("main",2),("student",1)]}
    (folder/"cpu.json").write_text(json.dumps(data,indent=2))
