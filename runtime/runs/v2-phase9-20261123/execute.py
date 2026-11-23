"""Run sequential CPU processes, report evidence and seal the phase."""
import os
import sys
import time
import subprocess
import json
import traceback
from datetime import datetime, timezone
from pathlib import Path
from phase9 import RUN, ROOT, sha, save, read

def utc():return datetime.now(timezone.utc).isoformat()
def note(message):
    with (RUN/"work_log.md").open("a") as f:f.write(f"\n- {utc()}　{message}\n")
def main():
    assert os.environ["CUDA_VISIBLE_DEVICES"]==""
    assert not (RUN/"phase9.sha256").exists()
    start=time.perf_counter();times={}
    for stage in ("lock","rank","score","audit"):
        note("启动 "+stage);t=time.perf_counter()
        with (RUN/(stage+".log")).open("a") as f:
            subprocess.run([sys.executable,"-B",str(RUN/"phase9.py"),stage],cwd=ROOT,stdout=f,stderr=subprocess.STDOUT,check=True)
        times[stage]=time.perf_counter()-t;note(f"{stage} 通过，CPU 墙钟 {times[stage]:.4f} 秒。")
    s=read(RUN/"summary.json");a=read(RUN/"audit.json");r=read(RUN/"ranking_seal.json")
    wall=time.perf_counter()-start
    save("cost.json",dict(cpu_job_wall_seconds=wall,stages_seconds=times,scope="lock start through independent audit exit; excludes SSH, transfer and report/seal",phase9_incremental_gpu_hours=0,upstream_reference_gpu_hours=a["costs"],sources=["../v2-phase7-20261121/gpu_hours.csv","../v2-phase8-20261121/gpu_hours.csv"]))
    fx=s["paired"];c=s["changes"];cost=a["costs"]
    rows=["| 指标 | 人数 | B4 | B4-SFT |","|---|---:|---:|---:|"]
    for k,label in [("joint_ndcg","联合 nDCG@5"),("joint_recall","联合 Recall@5"),("delivery_ndcg","投递 nDCG@5"),("employer_ndcg","已投递子榜认可 nDCG@5"),("joint_hits","人均 Top-5 联合正例数")]:
        b=s["metrics"]["B4"][k];v=s["metrics"]["B4-SFT"][k]
        rows.append(f'| {label} | {b["n"]} | {b["mean"]:.6f} | {v["mean"]:.6f} |')
    conclusion=("区间覆盖零，当前样本未给出明确的改善证据。" if fx["lo"]<=0<=fx["hi"] else
                "区间全部高于零，在该测试池观察到改善。" if fx["lo"]>0 else "区间全部低于零，在该测试池观察到下降。")
    text=f"""# Phase 9 验收｜简单聚合下的教师蒸馏

**验收通过。** 固定四角色 B4 等权聚合后，B4-SFT − B4 的联合 nDCG@5 配对差为 {fx['mean']:+.6f}，95% 百分位区间 [{fx['lo']:+.6f}, {fx['hi']:+.6f}]，有效配对 {fx['n']} 人。{conclusion} 这是查看此前阶段结果后追加的解释性比较，适用范围为同一历史候选池内的十岗重排。

{chr(10).join(rows)}

256 人中，完整排名变化 {c['full']} 人，前五顺序变化 {c['top5_order']} 人，前五岗位集合变化 {c['top5_set']} 人。主指标提升 {s['improved']} 人、持平 {s['tied']} 人、下降 {s['declined']} 人。

原版 256 份 B4 逐字段重放一致。学生测试角色终态计数为 {r['student_ballots']}，回退 {r['student_fallbacks']} 人。排名进程禁读标签，评分进程在封存后打开一次 T 标签。五项分母逐人核对，两组均值、区间以及主效应的配对区间独立复算通过。

本期 CPU 墙钟 {wall:.3f} 秒（输入校验至独立复算结束），新增 GPU 用量 0 小时。复用教师 {cost['teacher']:.5f}、学生训练 {cost['LoRA']:.5f}、学生重评 {cost['student_reviews']:.5f} GPU 小时，均已计入前期账本。

[协议](protocol.md) · [代码](phase9.py) · [输入哈希与来源](input_lock.json) · [排名封存](ranking_seal.json) · [学生榜](rankings.jsonl) · [逐人评分](per_user.csv) · [分母](denominators.json) · [配对差](paired_deltas.csv) · [指标与区间](summary.json) · [独立复核](audit.json) · [成本](cost.json) · [工作记录](work_log.md)
"""
    (RUN/"phase9_acceptance.md").write_text(text);note("完成解释性消融验收，登记实测成本与证据。")
    seal=RUN/"phase9.sha256"
    files=sorted(p for p in RUN.iterdir() if p.is_file() and p!=seal)
    seal.write_text("".join(f"{sha(p)}  {p.name}\n" for p in files))
    subprocess.run(["sha256sum","--status","-c",seal.name],cwd=RUN,check=True)
    print(text)
if __name__=="__main__":
    try:main()
    except Exception:
        note(traceback.format_exc());raise
