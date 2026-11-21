"""Independent scalar fits, likelihood-gradient checks and full bootstrap replay."""
import copy
import csv
import hashlib
import math
from collections import Counter
from itertools import combinations

import numpy as np
from PIL import Image
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit

from phase5_io import (ANALYSIS, BASE, EVIDENCE, FORMAL, HEADS, METHODS, OLD_PROTOCOL_HASH, P2, P4,
                       PROTOCOL, ROLES, RUN, SEED, DataAccess, encode, fingerprint, hashes,
                       load_sources, log, read, rows, sha, write)

PAIRS=list(combinations(range(4),2))


def close(a,b,tol=2e-10):
    assert abs(a-b)<=tol,(a,b,tol)


def data_arrays(c,lists,labels,votes):
    raw=np.empty((len(c),10,6)); target=np.empty_like(raw)
    for i,q in enumerate(c):
        by_id={r:{p["job_id"]:p for p in votes[q,r]["predictions"]} for r in ROLES}
        outcomes={o["slot"]:o for o in labels[q]}
        for j,jid in enumerate(lists[q]["job_ids"]):
            o=outcomes[f"P{j+1:02}"]
            target[i,j]=[o["J"]]*4+[o["D"],np.nan if o["S"] is None else o["S"]]
            for h,name in enumerate(HEADS):
                r,k=name.split(".")
                p=by_id[r][jid];assert p["slot"]==o["slot"]
                raw[i,j,h]=p[k]
    return raw,target


def independently_fit(x,y):
    mask=np.isfinite(x)&np.isfinite(y)
    a,b=x[mask],y[mask]
    if not len(b):return None
    if len(np.unique(b))==1 or np.all(a==0):return 1.
    def nll(log_t):
        z=a/np.exp(log_t)
        return np.mean(np.logaddexp(0.,z)-b*z)
    limits=(math.log(.05),math.log(20.))
    res=minimize_scalar(nll,bounds=limits,method="bounded",options={"xatol":1e-14})
    assert res.success
    return math.exp(min((*limits,res.x),key=nll))


def check_correlations(probs,y):
    vals=np.empty((6,3)); degeneracy=np.empty((6,2),bool)
    counts={}
    for k,(a,b) in enumerate(PAIRS):
        for s in (0,1):
            use=(y==s)&np.isfinite(probs[:,:,a])&np.isfinite(probs[:,:,b])
            x=probs[:,:,a][use]-s;v=probs[:,:,b][use]-s
            deg=len(x)<2 or np.ptp(x)==0 or np.ptp(v)==0
            degeneracy[k,s]=deg
            vals[k,s]=0 if deg else np.corrcoef(x,v)[0,1]
            counts[k,s]=(int(use.any(axis=1).sum()),len(x))
        vals[k,2]=vals[k,:2].mean()
    return vals,degeneracy,counts


def verify_calibration(c,t,lists,labels,votes,cal):
    assert cal["calibration_queries"]==c and set(c).isdisjoint(t)
    raw,y=data_arrays(c,lists,labels,votes);x=logit(raw)
    fold=np.arange(len(c))%3
    oof=np.empty_like(raw);max_log_delta=0.
    for f in range(4):
        held=fold==f
        train=~held
        if f==3:train[:]=True
        if f<3:
            meta=cal["folds"][f]
            assert meta["fold"]==f and meta["held_queries"]==[q for i,q in enumerate(c) if held[i]]
            assert meta["fit_queries"]==[q for i,q in enumerate(c) if train[i]]
            fits=meta["heads"]
        else:fits=cal["full_C_heads"]
        for h,key in enumerate(HEADS):
            ref=independently_fit(x[train,:,h],y[train,:,h])
            saved=fits[key]
            assert saved["n"]==int(np.isfinite(y[train,:,h]).sum())
            assert saved["positive"]==int(np.nansum(y[train,:,h]))
            diff=abs(math.log(ref/saved["temperature"]))
            max_log_delta=max(max_log_delta,diff)
            assert diff<1e-6,(f,key,ref,saved)
            if f<3:oof[held,:,h]=expit(x[held,:,h]/saved["temperature"])
    obs=list(rows(BASE/"oof.jsonl"));seen=set();ci={q:i for i,q in enumerate(c)}
    for v in obs:
        i=ci[v["query"]];j=int(v["slot"][1:])-1
        assert (i,j) not in seen;seen.add((i,j))
        assert v["block"]=="C" and v["fold"]==fold[i] and v["list_sha256"]==lists[c[i]]["sha256"]
        assert v["job_id"]==lists[c[i]]["job_ids"][j]
        assert (v["J"],v["D"],v["S"])==(y[i,j,0],y[i,j,4],None if np.isnan(y[i,j,5]) else y[i,j,5])
        for h,name in enumerate(HEADS):close(v["raw"][name],raw[i,j,h]);close(v["oof"][name],oof[i,j,h])
    assert len(seen)==1280
    coeff,deg,denoms=check_correlations(oof[:,:,:4],y[:,:,0])
    corr_sums=np.ones(4)
    for k,(a,b) in enumerate(PAIRS):
        p=cal["pairs"][k]
        assert (p["a"],p["b"])==(ROLES[a],ROLES[b])
        for s in (0,1):
            u=p[str(s)];assert (u["n_users"],u["n_jobs"])==denoms[k,s] and bool(u["degenerate"])==deg[k,s]
            close(u["rho"],coeff[k,s])
        close(p["rho"],coeff[k,2])
        corr_sums[[a,b]]+=max(0,coeff[k,2])
    brier=np.square(oof[:,:,:4]-y[:,:,:4]).mean(axis=1).mean(axis=0)
    rel=np.exp(-4*brier);w0=rel/rel.sum();adjusted=rel/corr_sums;w1=adjusted/adjusted.sum()
    for i,r in enumerate(ROLES):
        stored=cal["roles"][r]
        for k,ref in [("brier",brier[i]),("reliability",rel[i]),("denominator",corr_sums[i]),("T0",w0[i]),("T1",w1[i]),("delta_pp",100*(w1[i]-w0[i]))]:
            close(stored[k],ref)
    return x,y,fold,coeff,denoms,max_log_delta


def pl_gradient(scores,orders,weights):
    gradient={j:.2*v for j,v in scores.items()}
    for role,wt in weights.items():
        order=orders[role]
        for pos,better in enumerate(order[:-1]):
            remaining=order[pos:];peak=max(scores[j] for j in remaining)
            denom=math.fsum(math.exp(scores[j]-peak) for j in remaining)
            gradient[better]-=wt
            for j in remaining:gradient[j]+=wt*math.exp(scores[j]-peak)/denom
    return max(map(abs,gradient.values()))


def verify_rankings(t,lists,votes,cal):
    allrows=list(rows(BASE/"rankings.jsonl"))
    unique={(r["query"],r["method"]):r for r in allrows}
    assert len(allrows)==len(unique)==2560 and set(unique)=={(q,m) for q in t for m in METHODS}
    assert list(rows(BASE/"candidate_lists.jsonl"))==[lists[q] for q in t]
    with (BASE/"fallbacks.csv").open() as f:events=list(csv.DictReader(f))
    assert len(events)==2560 and {(e["query"],e["method"]) for e in events}==set(unique)
    for e in events:
        fb=unique[e["query"],e["method"]]["fallback"] or {}
        assert int(e["used_fallback"])==bool(fb) and e["reason"]==fb.get("reason","") and e["fallback_target"]==fb.get("method","")
    max_grad=0.;pfit=0
    for q in t:
        jobs=lists[q]["job_ids"]
        orders={r:votes[q,r]["order_job_ids"] for r in (*ROLES,"general")}
        pred_byid={r:{p["job_id"]:p for p in votes[q,r]["predictions"]} for r in ROLES}
        for method in METHODS:
            v=unique[q,method]
            assert v["block"]=="T" and v["list_sha256"]==lists[q]["sha256"]
            order=v["order_job_ids"];scores=v["scores"]
            assert set(order)==set(jobs) and len(order)==10 and v["top5"]==order[:5]
            assert not v["fallback"] and scores is not None
            assert set(scores)==set(jobs) and all(math.isfinite(p) for p in scores.values())
            assert order==sorted(jobs,key=lambda j:(-scores[j],j))
            if method=="B1":
                assert order==orders["general"]
            elif method=="B4" or method.startswith("B4-minus-"):
                excluded=method.removeprefix("B4-minus-")
                used=[r for r in ROLES if r!=excluded]
                assert set(v["used_roles"])==set(used)
                for j in jobs:
                    close(scores[j],sum(9-orders[r].index(j) for r in used)/(9*len(used)))
                for r in used:close(v["weights"][r],1/len(used))
            else:
                assert set(v["used_roles"])==set(ROLES)
                weights=v["weights"]
                for r in ROLES:close(weights[r],cal["roles"][r]["T0" if method=="T0" else "T1"])
                if method in ("T0","T1"):
                    assert v["solver"]["ok"]
                    grad=pl_gradient(scores,orders,weights)
                    assert grad<=1e-5
                    max_grad=max(max_grad,grad);pfit+=1
                elif method=="W1":
                    assert v["weights"]==unique[q,"T1"]["weights"] and v["used_roles"]==unique[q,"T1"]["used_roles"]
                    for j in jobs:close(scores[j],sum(weights[r]*(9-orders[r].index(j))/9 for r in ROLES))
                elif method=="F":
                    trusted=unique[q,"T1"]["order_job_ids"]
                    for j,d in v["components"].items():
                        rank=trusted.index(j)+1; R=(10-rank)/9
                        assert d["T1_rank"]==rank and d["final_rank"]==order.index(j)+1
                        close(d["R"],R)
                        for role,k in (("preference","pC"),("employer","pE")):
                            p=pred_byid[role][j][k];temp=cal["full_C_heads"][role+"."+k]["temperature"]
                            close(d[k],1/(1+((1-p)/p)**(1/temp)))
                        utility=math.sqrt(d["pC"]*d["pE"])
                        close(d["bilateral_utility"],utility);close(d["F"],(R+utility)/2);close(scores[j],d["F"])
                    assert set(v["components"])==set(jobs)
    return {"complete_rankings":len(unique),"counts":dict(Counter(v["method"] for v in allrows)),
            "real_fallbacks":sum(bool(v["fallback"]) for v in allrows),"PL_fits":pfit,
            "max_PL_score_equation_error":max_grad}


def fallback_fixtures(q,listing,votes,cal):
    from phase5_ranking import rank_query,bilateral,score_order
    from phase5_calibration import fit_temperature,stratified_pairs
    jobs=listing["job_ids"]
    fixture={r:votes[q,r] for r in (*ROLES,"general")}
    def run(items,**kwargs):return rank_query(q,listing,items,cal,**kwargs)
    sparse=run({"skill":fixture["skill"]})
    for row in sparse.values():
        assert row["fallback"]["method"]=="BM25" and row["order_job_ids"]==jobs
    for role,key in (("preference","pC"),("employer","pE")):
        missing=copy.deepcopy(fixture)
        missing[role]["predictions"][0].pop(key)
        rows_=run(missing)
        assert rows_["F"]["fallback"]=={"method":"T1","reason":"missing_bilateral_head"}
        assert rows_["F"]["order_job_ids"]==rows_["T1"]["order_job_ids"]
    broken=run(fixture,solver=lambda *_:(None,{"ok":False,"message":"simulated failure"}))
    for m in ("T0","T1","F"):
        assert broken[m]["fallback"]["method"]=="BM25" and broken[m]["order_job_ids"]==jobs
    assert broken["W1"]["fallback"] is None and broken["W1"]["weights"]==broken["T1"]["weights"]
    balanced=copy.deepcopy(fixture)
    for n,r in enumerate(ROLES):balanced[r]["order_job_ids"]=sorted(jobs,reverse=bool(n%2))
    equal=run(balanced)["B4"];assert equal["order_job_ids"]==sorted(jobs) and set(equal["scores"].values())=={.5}
    assert score_order({j:1 for j in reversed(jobs)})==sorted(jobs) and bilateral(1,1,1)==1
    assert fit_temperature(np.zeros(3),np.zeros(3))["temperature"]==1
    assert fit_temperature(np.zeros(1),np.array([np.nan]))["temperature"] is None
    ds=stratified_pairs(np.full((1,3,4),.5),np.array([[0,0,1]]))
    assert all(r["0"]["degenerate"]=="constant_residual" and r["1"]["degenerate"]=="fewer_than_two_pairs" and r["rho"]==0 for r in ds)
    return ["missing_B1","<2_votes","missing_pC","missing_pE","PL_failure","inherited_BM25","T1_side_fallback",
            "W1_shared_weights","ID_tie","F_at_unity","single_class","empty_head","degenerate_layers"]


def replay_bootstrap(x,y,fold,points,counts):
    # Check all convex optimum conditions, then rebuild held-out residuals independently.
    with np.load(ANALYSIS/"bootstrap.npz",allow_pickle=False) as archive:
        draws,temps,saved,degs=[archive[k] for k in ("draws","temperatures","correlations","degenerate")]
    assert draws.shape==(2000,128) and temps.shape==(2000,3,4) and saved.shape==(2000,6,3) and degs.shape==(2000,6,2)
    rng=np.random.default_rng(SEED);groups=[np.flatnonzero(fold==f) for f in range(3)]
    max_violation=max_coeff_error=0.
    for b,idx in enumerate(draws):
        held_probs=np.empty((128,10,4))
        for f,where in enumerate(groups):
            assert np.array_equal(idx[where],rng.choice(where,len(where),replace=True)) and np.all(fold[idx[where]]==f)
            train=idx[fold!=f]
            a=x[train,:,:4];truth=y[train,:,:4];tt=temps[b,f]
            assert ((.05<=tt)&(tt<=20)).all()
            loss_grad=np.mean(a*(expit(a/tt)-truth),axis=(0,1))
            # At max temperature the inverse-temperature derivative must be nonnegative.
            vio=np.where(tt==20,np.maximum(-loss_grad,0),np.where(tt==.05,np.maximum(loss_grad,0),abs(loss_grad)))
            max_violation=max(max_violation,float(vio.max()))
            held_probs[where]=expit(x[idx[where],:,:4]/tt)
        coeff,deg,_=check_correlations(held_probs,y[idx,:,0])
        assert np.array_equal(deg,degs[b])
        max_coeff_error=max(max_coeff_error,float(abs(coeff-saved[b]).max()))
    assert max_violation<1e-9 and max_coeff_error<1e-11
    # Compute percentiles by sorted order and linear interpolation.
    ordered=np.sort(saved,axis=0)
    def quantile(q):
        loc=(len(saved)-1)*q; lo=int(loc)
        return ordered[lo]+(loc-lo)*(ordered[lo+1]-ordered[lo])
    lo,hi=quantile(.025),quantile(.975)
    with (ANALYSIS/"correlated_error_pairs.csv").open() as f:records=list(csv.DictReader(f))
    assert len(records)==6
    for k,(r,(a,b)) in enumerate(zip(records,PAIRS)):
        assert (r["a"],r["b"])==(ROLES[a],ROLES[b])
        for s in (0,1):
            assert (int(r[f"users{s}"]),int(r[f"jobs{s}"]))==counts[k,s]
            assert int(r[f"bootstrap_degenerate{s}"])==int(degs[:,k,s].sum())
        for j,name in enumerate(("rho0","rho1","rho")):
            close(float(r[name]),points[k,j])
            close(float(r[name+"_lo"]),lo[k,j]);close(float(r[name+"_hi"]),hi[k,j])
    return dict(draws=len(draws),bootstrap_temperatures=temps.size,recomputed_statistics=saved.size,
                max_inverse_temperature_KKT_violation=max_violation,max_correlation_replay_error=max_coeff_error,
                degenerate_strata=int(degs.sum()))


def verify_protocol():
    old=(EVIDENCE/"protocol_before.md").read_bytes()
    new=PROTOCOL.read_bytes()
    assert hashlib.sha256(old).hexdigest()==OLD_PROTOCOL_HASH
    assert new.startswith(old) and new[len(old):].startswith("\n## 4. 补充实验结果｜跨 Agent 相关错误分析\n".encode())
    assert new==(EVIDENCE/"protocol.md").read_bytes()
    return {"original_bytes":len(old),"original_sha256":OLD_PROTOCOL_HASH,"current_sha256":sha(PROTOCOL),
            "prefix_unchanged":True}


def acceptance_report(result,cal):
    ranks=result["blind_ranking"];boot=result["bootstrap"]
    lines=["# Phase 5 验收｜校准、共同错误与冻结排名","",
           f'运行 `{RUN.name}` 已通过验收。服务器为 `hkust-gz-hpc3-temp2`，全部计算使用 CPU。校准集含 128 人、1,280 个人岗对。测试集 256 人的十种方法全部生成完整排名，共 {ranks["complete_rankings"]:,} 份。实际回退 {ranks["real_fallbacks"]} 次。',"",
           "## 原协议验收","",
           "| 验收项 | 实测 | 证据 |","|---|---|---|",
           "| 名单与标签隔离 | 按原散列顺序分折 43/43/42。所有拟合和验证用户逐折互斥。三个计算阶段的上游数据读取审计中，测试标签打开次数均为 0。 | [拟合与折清单](base/calibration.json) · [正式阶段读取记录](evidence/formal_data_access.json) · [验收读取记录](evidence/verify_data_access.json) |",
           "| 校准与聚合 | 独立拟合四个共同头和两个双边头的三折及全量温度，逐项复核 1,280 份样本外记录。按人复算 Brier、六对相关、惩罚分母和两套归一化权重。 | [样本外记录](base/oof.jsonl) · [复算统计](acceptance.json) |",
           f'| 盲排 | 每种方法各 256 份。核对全部十岗排列、Top-5、同分按岗位 ID、逐项双边分及 T1/W1 角色与权重一致性。{ranks["PL_fits"]} 次 PL 解均通过独立目标函数梯度检查，最大梯度绝对值 {ranks["max_PL_score_equation_error"]:.2e}。 | [十岗排名](base/rankings.jsonl) · [原始候选名单](base/candidate_lists.jsonl) |',
           "| 回退 | 构造少票、B1 缺票、双边头缺失与 PL 求解失败样例，分别核对 BM25 和整榜 T1 回退。另验证同分排序及 R、pC、pE 均为 1 时 F 为 1。 | [逐人逐方法回退记录](base/fallbacks.csv) · [验收实现](code/phase5_verify.py) |",
           "","## 温度校准","",
           "| 预测头 | 完整 C 拟合岗位数 | 正例数 | 温度 | 拟合状态 |","|---|---:|---:|---:|---|"]
    for head in HEADS:
        p=cal["full_C_heads"][head]
        lines.append(f'| {head} | {p["n"]} | {p["positive"]} | {p["temperature"]:.4f} | {p["status"]} |')
    lines+=["","Employer 的认可概率仅在已投递样本上校准。本次该头温度达到冻结上界 20，双边排序使用此值。","",
            "## 补充实验验收","",
            f'第 1–3 节的 {result["protocol"]["original_bytes"]:,} 个原始字节均与 Git 基线相符。补充分析使用已封存的 C 样本外点估计。独立核对了 2,000 次抽样的用户索引、{boot["bootstrap_temperatures"]:,} 个温度最优性条件、{boot["recomputed_statistics"]:,} 个相关统计量及全部区间端点。原始点估计与正式参数一致。五项正式产物在补充分析前后和最终验收时的 SHA-256 一致。',"",
            "[六对角色的结果、区间和权重解读](analysis/correlated_error_report.md) · [三联热图](analysis/correlation_heatmap.png) · [分析前哈希](evidence/formal_before.sha256) · [分析后哈希](evidence/formal_after.sha256)","",
            "## 第 6 期交接","",
            "完整排名按测试用户与方法定位。每行保留名单哈希、完整有序岗位 ID 和前五。F 的逐岗明细保留可信名次、双边概率、效用及最终分数。本期交接以 [运行校验清单](phase5.sha256) 为准。[工作记录](work_log.md)"]
    (RUN/"phase5_acceptance.md").write_text("\n".join(lines)+"\n")


def main():
    guard=DataAccess("verify")
    protocol=verify_protocol()
    c,t,lists,labels,votes,inputs=load_sources()
    cal=read(BASE/"calibration.json")
    assert inputs==cal["input_sha256"]
    before=hashes(EVIDENCE/"formal_before.sha256");after=hashes(EVIDENCE/"formal_after.sha256")
    assert fingerprint()==before==after and all((BASE/name).stat().st_mode&0o222==0 for name in FORMAL)
    for stage in ("formal","supplement"):
        receipt=read(RUN/"jobs"/stage/"status.json");assert receipt["state"]=="done" and receipt["exit_code"]==0
    assert read(EVIDENCE/"formal_run.json")["seal_epoch"]<=read(RUN/"jobs/supplement/status.json")["started_epoch"]
    x,y,fold,points,denoms,temp_diff=verify_calibration(c,t,lists,labels,votes,cal)
    ranks=verify_rankings(t,lists,votes,cal)
    fixtures=fallback_fixtures(t[0],lists[t[0]],votes,cal)
    boot=replay_bootstrap(x,y,fold,points,denoms)
    with (ANALYSIS/"role_weights.csv").open() as f:wrows=list(csv.DictReader(f))
    assert [r["role"] for r in wrows]==list(ROLES)
    for w in wrows:
        for k,s in cal["roles"][w["role"]].items():close(float(w[k]),s)
    with Image.open(ANALYSIS/"correlation_heatmap.png") as img:
        assert img.format=="PNG" and img.width>=1600 and img.height>=500
        fig_size=list(img.size);img.verify()
    assert fingerprint()==before
    upstream=(EVIDENCE/"phase4_after.txt").read_text().splitlines()
    assert upstream==(EVIDENCE/"phase4_before.txt").read_text().splitlines() and len(upstream)==13 and all(l.endswith(": OK") for l in upstream)
    guard.save()
    for stage in ("formal","supplement","verify"):
        data=read(EVIDENCE/(stage+"_data_access.json"))
        assert data["test_label_opens"]==0 and not data["denied"]
        if stage=="supplement":assert not data["opened_upstream_data_files"]
    result={"status":"passed","protocol":protocol,"upstream_phase4_files":len(upstream),"input_sha256":inputs,
            "formal_unchanged_sha256":before,"n_C":len(c),"n_T":len(t),"fold_sizes":[int((fold==f).sum()) for f in range(3)],
            "out_of_fold_pairs":len(c)*10,"independent_temperature_fits":24,"max_log_temperature_difference":temp_diff,
            "six_pair_points_verified":True,"blind_ranking":ranks,"synthetic_checks":fixtures,"bootstrap":boot,
            "heatmap_pixels":fig_size,"test_label_opens":0,"gpu_hours":0}
    write(RUN/"acceptance.json",result)
    acceptance_report(result,cal)
    log("独立复算通过。三折及全量温度、六对相关、两套权重、2,560 份排名、全部 bootstrap 统计量及回退样例已验收。正式封存哈希复验一致。")
    print(encode(result))


if __name__=="__main__":main()
