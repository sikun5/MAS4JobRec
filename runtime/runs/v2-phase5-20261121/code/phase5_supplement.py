"""Supplementary analysis of cross-agent residual correlations on calibration users."""
import time

import numpy as np
from scipy.special import logit

from phase5_calibration import PAIRS,crossfit,stratified_pairs
from phase5_io import (ANALYSIS,BASE,EVIDENCE,HEADS,ROLES,SEED,DataAccess,fingerprint,hashes,
                       log,read,rows,save_hashes,table,write)

DRAWS=2000
NAMES=[s.title() for s in ROLES]


def arrays(users):
    index={q:i for i,q in enumerate(users)}
    shape=(len(users),10,4)
    y=np.full(shape[:2],np.nan)
    raw=np.full(shape,np.nan)
    oof=np.full(shape,np.nan)
    for row in rows(BASE/"oof.jsonl"):
        i=index[row["query"]]; j=int(row["slot"][1:])-1
        assert row["block"]=="C" and row["fold"]==i%3 and np.isnan(y[i,j])
        y[i,j]=row["J"]
        raw[i,j]=[row["raw"][h] for h in HEADS[:4]]
        oof[i,j]=[row["oof"][h] for h in HEADS[:4]]
    assert np.isfinite(y).all()
    return logit(raw),oof,y


def bootstrap(logits,y,fold):
    rng=np.random.default_rng(SEED)
    groups=[np.flatnonzero(fold==f) for f in range(3)]
    draws=np.empty((DRAWS,len(fold)),dtype=np.int16)
    stats=np.empty((DRAWS,len(PAIRS),3))
    degenerate=np.zeros((DRAWS,len(PAIRS),2),dtype=np.uint8)
    temps=np.full((DRAWS,3,4),np.nan)
    for b,draw in enumerate(draws):
        for idx in groups:draw[idx]=rng.choice(idx,size=len(idx),replace=True)
        target=np.repeat(y[draw,:,None],4,axis=2)
        probs,fits=crossfit(logits[draw],target,fold)
        for f,fit in enumerate(fits):temps[b,f]=[x["temperature"] for x in fit]
        for k,pair in enumerate(stratified_pairs(probs,y[draw])):
            stats[b,k]=[pair["0"]["rho"],pair["1"]["rho"],pair["rho"]]
            degenerate[b,k]=[pair[str(s)]["degenerate"] is not None for s in (0,1)]
    return draws,stats,degenerate,temps


def save_heatmap(points):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.colors import Normalize
    matrices=np.full((3,4,4),np.nan)
    for row,(a,b) in zip(points,PAIRS):
        for k,stat in enumerate(("rho0","rho1","rho")):
            value=row[stat]
            if (stat=="rho0" and row["reason0"]) or (stat=="rho1" and row["reason1"]):continue
            matrices[k,a,b]=matrices[k,b,a]=value
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False})
    fig,axes=plt.subplots(1,3,figsize=(13.4,4.5),layout="constrained")
    cmap=plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("#f0f0f0")
    normal=Normalize(-1,1)
    for ax,mat,title in zip(axes,matrices,("Negative outcomes (J = 0)","Positive outcomes (J = 1)","Equal-weight mean")):
        im=ax.imshow(mat,cmap=cmap,norm=normal,aspect="equal")
        ax.set_xticks(range(4),NAMES,rotation=30,ha="right",rotation_mode="anchor")
        ax.set_yticks(range(4),NAMES)
        ax.tick_params(length=0)
        ax.set_title(title,pad=10,fontweight="bold")
        for i,j in np.ndindex(4,4):
            v=mat[i,j]
            label=("—" if i==j else "NA") if not np.isfinite(v) else f"{v:.3f}"
            ax.text(j,i,label,ha="center",va="center",color="white" if np.isfinite(v) and abs(v)>.55 else "#202020")
        for spine in ax.spines.values():spine.set_visible(False)
    fig.suptitle("Cross-agent out-of-fold residual correlations",fontweight="bold",fontsize=15)
    fig.colorbar(im,ax=axes,shrink=.78,pad=.025,label="Pearson correlation",ticks=[-1,-.5,0,.5,1])
    fig.savefig(ANALYSIS/"correlation_heatmap.png",dpi=170,facecolor="white")
    plt.close(fig)


def format_ci(row,key):
    if row[key+"_lo"] is None:return f'{row[key]:.3f} [NA]'
    return f'{row[key]:.3f} [{row[key+"_lo"]:.3f}, {row[key+"_hi"]:.3f}]'


def report(points,weights):
    text=["# 四个评审角色的共同错误","",
          "在校准集 C 的 128 名求职者中，每个角色对同一份十岗名单估计联合事件 J。下表使用按人三折得到的样本外概率。负例层中，正相关表示两位角色高估概率的程度往往同步。正例层中，正相关表示两位角色低估概率的程度往往同步。两类分别计算，再等权平均。","",
          "| 角色对 | J=0 人数 / 岗位数 | ρ₀ [95% 区间] | J=1 人数 / 岗位数 | ρ₁ [95% 区间] | 等权均值 ρ [95% 区间] |",
          "|---|---:|---:|---:|---:|---:|"]
    for p in points:
        text.append(f'| {p["a"].title()} × {p["b"].title()} | {p["users0"]} / {p["jobs0"]} | {format_ci(p,"rho0")} | {p["users1"]} / {p["jobs1"]} | {format_ci(p,"rho1")} | {format_ci(p,"rho")} |')
    high0=max(points,key=lambda p:p["rho0"])
    high1=max(points,key=lambda p:p["rho1"])
    low=min(points,key=lambda p:p["rho"])
    def names(p):return p["a"].title()+" × "+p["b"].title()
    text+=["",f'负例层中，{names(high0)} 的高估程度相关点估计最高（{high0["rho0"]:.3f}）。正例层中，{names(high1)} 的低估程度相关点估计最高（{high1["rho1"]:.3f}）。等权均值最小的是 {names(low)}（{low["rho"]:.3f}）。这些次序描述本次点估计，角色对之间的差值没有单独检验。',"",
           "![负例层、正例层及等权均值的同色标相关热图](correlation_heatmap.png)","",
           "区间为 2,000 次折内求职者 bootstrap 的第 2.5、97.5 百分位，随机种子为 20261121。每次抽取整位求职者的十岗和四角色共同头，每一折保持原人数，在另外两折重新拟合共同头温度。在原样本触发退化规则的分层标为 NA。替代值按冻结规则参与合成均值。[逐对数值及退化原因](correlated_error_pairs.csv)","",
           "## 从可靠度到相关错误降权","",
           "| 角色 | 样本外 Brier | 相关惩罚分母 | T0 权重 | T1 权重 | 变化（百分点） |",
           "|---|---:|---:|---:|---:|---:|"]
    for w in weights:
        text.append(f'| {w["role"].title()} | {w["brier"]:.5f} | {w["denominator"]:.4f} | {w["T0"]:.3%} | {w["T1"]:.3%} | {w["delta_pp"]:+.3f} |')
    top=max(weights,key=lambda w:w["delta_pp"]);bottom=min(weights,key=lambda w:w["delta_pp"])
    text+=["",f'六组均值相关中有 {sum(p["rho"]>0 for p in points)} 组为正。各角色的可靠度除以各自的惩罚分母后，再在四角色之间归一化。{bottom["role"].title()} 的相对权重下降最多（{bottom["delta_pp"]:.3f} 个百分点），{top["role"].title()} 的相对权重上升最多（{top["delta_pp"]:+.3f} 个百分点）。第 6 期使用 T1−T0 的配对比较检验这一步是否改善测试排序。[四角色权重明细](role_weights.csv)',"",
           f'原样本 12 个分层统计中退化 {sum(bool(p["reason0"])+bool(p["reason1"]) for p in points)} 个。2,000 次重采样合计触发 {sum(p["bootstrap_degenerate0"]+p["bootstrap_degenerate1"] for p in points)} 次分层退化。完整重采样序列、逐折温度和系数保存在同目录的 `bootstrap.npz`。']
    (ANALYSIS/"correlated_error_report.md").write_text("\n".join(text)+"\n")


def main():
    guard=DataAccess("supplement")
    then=time.monotonic()
    original=hashes(EVIDENCE/"formal_before.sha256")
    assert fingerprint()==original and read(BASE.parent/"jobs/formal/status.json")["state"]=="done"
    cal=read(BASE/"calibration.json")
    users=cal["calibration_queries"]
    logits,oof,y=arrays(users)
    pairs=stratified_pairs(oof,y)
    assert pairs==cal["pairs"]
    log("补充分析已确认正式封存哈希。从 C 的样本外记录复核点估计，开始 2,000 次原折内整人重采样。")
    draws,stats,deg,temps=bootstrap(logits,y,np.arange(len(users))%3)
    np.savez_compressed(ANALYSIS/"bootstrap.npz",draws=draws,correlations=stats,degenerate=deg,temperatures=temps)
    bounds=np.percentile(stats,[2.5,97.5],axis=0)
    points=[]
    for k,p in enumerate(pairs):
        row={"a":p["a"],"b":p["b"]}
        for s in (0,1):
            layer=p[str(s)]
            row.update({f"users{s}":layer["n_users"],f"jobs{s}":layer["n_jobs"],f"reason{s}":layer["degenerate"],
                        f"bootstrap_degenerate{s}":int(deg[:,k,s].sum())})
        for i,stat in enumerate(("rho0","rho1","rho")):
            undefined=bool(p[str(i)]["degenerate"]) if i<2 else all(p[str(s)]["degenerate"] for s in (0,1))
            row[stat]=p[str(i)]["rho"] if i<2 else p["rho"]
            row[stat+"_lo"]=None if undefined else float(bounds[0,k,i])
            row[stat+"_hi"]=None if undefined else float(bounds[1,k,i])
        points.append(row)
    weights=[{"role":r,**cal["roles"][r]} for r in ROLES]
    table(ANALYSIS/"correlated_error_pairs.csv",points)
    table(ANALYSIS/"role_weights.csv",weights)
    save_heatmap(points)
    report(points,weights)
    final=fingerprint()
    assert original==final
    save_hashes(EVIDENCE/"formal_after.sha256",final)
    guard.save()
    write(EVIDENCE/"supplement_run.json",{"seed":SEED,"repetitions":DRAWS,"fold_sizes":[43,43,42],
          "pair_order":[[ROLES[a],ROLES[b]] for a,b in PAIRS],"stat_order":["rho0","rho1","rho"],
          "point_estimates_equal":True,"unchanged_formal_files":len(final),"elapsed_s":time.monotonic()-then})
    log("补充分析完成。六对原始点估计与正式参数一致，五项正式产物的补充分析前后 SHA-256 一致。")


if __name__=="__main__":main()
