"""Generate the formal Phase 5 handoff and seal it before the supplementary stage."""
import math
import os
import platform
import time

import numpy as np
import scipy

from phase5_io import (RUN, BASE, ROOT, EVIDENCE, P2, P4, ROLES, HEADS, METHODS, FORMAL, DataAccess,
                       fingerprint, load_sources, log, read, save_hashes, sha, table, write, write_rows)
from phase5_calibration import calibrate,c_arrays
from phase5_ranking import rank_query


def nullable(x):
    return float(x) if math.isfinite(x) else None


def main():
    guard=DataAccess("formal")
    timer=time.monotonic()
    assert not (EVIDENCE/"formal_before.sha256").exists(), "formal stage already sealed"
    assert os.environ.get("CUDA_VISIBLE_DEVICES")==""
    c,t,lists,labels,reviews,input_hashes=load_sources()
    log(f"正式计算读取 C {len(c)} 人的标签、C/T 名单与 Phase 4 选票。已验证四项上游输入及 Phase 4 验收摘要的哈希。")
    raw,y=c_arrays(c,labels,reviews)
    cal,oof=calibrate(c,raw,y)
    cal["input_sha256"]=input_hashes
    write(BASE/"calibration.json",cal)
    def records():
        for i,q in enumerate(c):
            for j,jid in enumerate(lists[q]["job_ids"]):
                yield dict(query=q,block="C",job_id=jid,slot=f"P{j+1:02}",fold=i%3,list_sha256=lists[q]["sha256"],
                           J=int(y[i,j,0]),D=int(y[i,j,4]),S=nullable(y[i,j,5]),
                           raw={h:nullable(raw[i,j,k]) for k,h in enumerate(HEADS)},
                           oof={h:nullable(oof[i,j,k]) for k,h in enumerate(HEADS)})
    write_rows(BASE/"oof.jsonl",records())
    write_rows(BASE/"candidate_lists.jsonl",(lists[q] for q in t))
    ranks=[]; fallback_rows=[]
    for q in t:
        ballots={r:reviews[q,r] for r in (*ROLES,"general")}
        methods=rank_query(q,lists[q],ballots,cal)
        for name in METHODS:
            row=methods[name]; ranks.append(row)
            fallback=row["fallback"] or {}
            fallback_rows.append({"query":q,"method":name,"used_fallback":int(bool(fallback)),
                                  "fallback_target":fallback.get("method",""),"reason":fallback.get("reason","")})
    write_rows(BASE/"rankings.jsonl",ranks)
    table(BASE/"fallbacks.csv",fallback_rows)
    assert len(ranks)==2560
    guard.save()
    frozen=fingerprint()
    save_hashes(EVIDENCE/"formal_before.sha256",frozen)
    for name in FORMAL:(BASE/name).chmod(0o444)
    (EVIDENCE/"formal_before.sha256").chmod(0o444)
    write(EVIDENCE/"formal_run.json",{"input_sha256":input_hashes,"n_users_C":len(c),"n_users_T":len(t),
        "n_rankings":len(ranks),"fallback_count":sum(row["used_fallback"] for row in fallback_rows),
        "python":platform.python_version(),"numpy":np.__version__,"scipy":scipy.__version__,"gpu_hours":0,
        "elapsed_s":time.monotonic()-timer,"seal_epoch":time.time()})
    log(f"正式结果已封存，含 {len(c)*10} 份样本外人岗记录、{len(ranks)} 份完整排名及回退明细。封存基准为 evidence/formal_before.sha256。")


if __name__=="__main__":main()
