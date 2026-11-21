"""Independently replay and seal a completed Phase 4 run. Run from project root."""
import csv
import hashlib
import json
import math
import os
import sqlite3
import time
from collections import Counter, defaultdict
from pathlib import Path

from phase2_data_prep import checksum, digest, encode
from phase4_agent_reviews import DATA, MODEL, RUN, SLOTS, SPECIALISTS, load_data, write_json


def lines(path):
    with path.open() as f:
        for line in f:
            yield json.loads(line)


def checked_response(text, heads):
    try:
        o=json.loads(text.strip().removeprefix("```json").removesuffix("```").strip())
        r=o["rank"]
        if not (isinstance(r,list) and all(type(v)==int for v in r) and sorted(r)==[1,2,3,4]):return None
        for k in heads:
            if not (isinstance(o[k],list) and len(o[k])==4 and all(type(v) in (float,int) and math.isfinite(v) and 0<=v<=1 for v in o[k])):return None
        return {k:o[k] for k in ["rank",*heads]}
    except (KeyError,TypeError,ValueError,AttributeError):return None


def validate():
    from transformers import AutoTokenizer
    from phase3_probe import prompt, template
    base=RUN/"base"
    split,inputs,lists=load_data()
    manifest=json.loads((RUN/"manifest.json").read_text())
    assert manifest["model"]==str(MODEL) and manifest["batch_size"]==2
    assert (manifest["dtype"],manifest["attention"],manifest["do_sample"],manifest["enable_thinking"])==("bfloat16","sdpa",False,False)
    assert manifest["starts"]==[7,5,3,1]*2 and manifest["input_cap"]==3840 and manifest["output_cap"]==256
    for f,h in manifest["source_sha256"].items():assert checksum(RUN/"code"/f)==h
    accepted=Path("runtime/runs/v2-phase1-20261121/env/model_files.sha256")
    assert checksum(accepted)==manifest["model_manifest_sha256"]
    names=[l.split(maxsplit=1)[1].strip() for l in accepted.read_text().splitlines()]
    assert (RUN/"model_check.log").read_text().splitlines()==[f+": OK" for f in names]
    tok=AutoTokenizer.from_pretrained(MODEL,local_files_only=True,padding_side="left")
    def clip(fields,n):
        keys=sorted(fields,key=lambda k:k in {"experience","job_description"})
        return tok.decode(tok.encode(" ".join(k+"="+fields[k] for k in keys),add_special_tokens=False)[:n],skip_special_tokens=True)
    docs={q:(clip(x["profile"],512),{j["slot"]:clip(j["fields"],640) for j in x["jobs"]}) for q,x in inputs.items()}
    expected={(b,q,role) for b in ("C","T") for q in split[b] for role in SPECIALISTS+(("general",) if b=="T" else ())}
    reviews={}
    for row in lines(base/"reviews.jsonl"):
        k=(row["block"],row["query"],row["role"])
        assert k in expected and k not in reviews
        reviews[k]=row
    assert set(reviews)==expected and len(reviews)==1792
    traces=defaultdict(list)
    for row in lines(base/"windows.jsonl"):
        k=(row["block"],row["query"],row["role"])
        assert k in expected
        traces[k].append(row)
    assert set(traces)==expected
    ledger=sqlite3.connect(f"file:{(base/'attempts.sqlite').resolve()}?mode=ro",uri=True)
    assert ledger.execute("pragma integrity_check").fetchone()==("ok",)
    states=Counter(); batches={}; blocks=defaultdict(Counter); timings=defaultdict(dict)
    keys=set(); expected_failures=[]; input_lengths=[]; output_lengths=[]; starts=[7,5,3,1]*2
    for k,result in reviews.items():
        block,q,role=k
        cat=block+("_B1" if role=="general" else "_four")
        c=blocks[cat];c["roles"]+=1
        slots=list(SLOTS); mapping=dict(zip(slots,lists[q]["job_ids"]))
        heads=["pJ"]+{"employer":["pE"],"preference":["pC"]}.get(role,[])
        observations=defaultdict(lambda:defaultdict(list))
        valid=True
        assert result["list_sha256"]==lists[q]["sha256"]
        for i,w in enumerate(traces[k]):
            assert valid and i<8
            pos=starts[i]-1
            inp=slots[pos:pos+4]
            assert len(set(inp))==4 and w["sweep"]==1+i//4 and w["start"]==pos+1 and w["input_slots"]==inp
            assert w["input_job_ids"]==[mapping[s] for s in inp] and w["list_sha256"]==result["list_sha256"]
            assert w["model_revision"]==MODEL.name and w["run"]==RUN.name
            parts={name:w[name] for name in ("run","model_revision","block","query","role","list_sha256","sweep","start","input_slots","input_job_ids")}
            key=digest(encode(parts))
            assert key==w["cache_key"] and key not in keys
            keys.add(key)
            profile,jobs=docs[q]
            messages=prompt(tok,profile,[jobs[s] for s in inp],role)
            prompt_hash=digest(encode(messages))
            token_count=len(tok.encode(template(tok,messages),add_special_tokens=False))
            attempts=w["attempts"];assert len(attempts) in (1,2)
            sql=ledger.execute("select n,state,payload from attempts where key=? order by n",(key,)).fetchall()
            assert len(sql)==len(attempts)
            c["first"]+=1;c["retries"]+=len(attempts)-1
            for j,(a,(n,state,payload)) in enumerate(zip(attempts,sql),1):
                assert a==json.loads(payload) and a["number"]==j==n and a["prompt_sha256"]==prompt_hash
                assert j==1 or attempts[0]["parsed"] is None
                states[state]+=1
                if state=="interrupted":
                    assert a["raw"] is None and a["error"]=="interrupted_after_dispatch"
                    continue
                assert state=="done" and a["input_tokens"]==token_count<=3840 and 0<a["output_tokens"]<=256
                input_lengths.append(token_count);output_lengths.append(a["output_tokens"])
                decoded=checked_response(a["raw"],heads)
                assert decoded==a["parsed"] and (decoded is None)==(a["error"] is not None)
                assert a["batch_seconds"]>0 and a["batch_size"] in (1,2)
                assert a["ended_epoch"]>=a["started_epoch"]
                bid=a["batch_id"];stamp=(cat,a["started_epoch"],a["batch_seconds"],a["batch_size"])
                if bid in batches:assert batches[bid]["stamp"]==stamp
                else:batches[bid]={"stamp":stamp,"count":0}
                batches[bid]["count"]+=1
                timings[cat][bid]=a["batch_seconds"]
            answer=attempts[-1]["parsed"]
            if answer is None:
                assert len(attempts)==2 and w["status"]=="missing" and w["output_slots"]==[] and w["output_job_ids"]==[]
                valid=False;c["failed_window"]+=1
            else:
                assert w["status"]=="valid"
                ranked=[inp[i-1] for i in answer["rank"]]
                assert ranked==w["output_slots"] and w["output_job_ids"]==[mapping[s] for s in ranked]
                slots[pos:pos+4]=ranked
                for h in heads:
                    for slot,p in zip(inp,answer[h]):
                        p=max(.0001,min(.9999,p))
                        observations[slot][h].append(math.log(p)-math.log1p(-p))
                c["valid_windows"]+=1
            if len(attempts)==2:
                expected_failures.append((key,k,w["sweep"],w["start"],[a["error"] for a in attempts],w["status"]))
        used=len(traces[k])
        assert result["finished_windows"]==used
        skipped=[{"sweep":1+i//4,"start":starts[i]} for i in range(used,8)]
        assert result["skipped"]==skipped
        c["skipped"]+=len(skipped)
        if valid:
            assert used==8 and result["status"]=="valid"
            assert slots==result["order_slots"] and [mapping[s] for s in slots]==result["order_job_ids"]
            assert set(slots)==set(SLOTS) and len(set(result["order_job_ids"]))==10
            preds=result["predictions"];assert [p["slot"] for p in preds]==list(SLOTS)
            for p in preds:
                slot=p["slot"];assert p["job_id"]==mapping[slot] and set(p)==set(heads)|{"slot","job_id","observations"}
                for h in heads:
                    obs=observations[slot][h];assert len(obs)==p["observations"]>0
                    expected_p=1/(1+math.exp(-sum(obs)/len(obs)))
                    assert math.isclose(p[h],expected_p,rel_tol=1e-12,abs_tol=1e-14)
            c["valid_roles"]+=1
        else:
            assert result["status"]=="missing" and not result["order_slots"] and not result["predictions"] and not result["order_job_ids"]
            c["missing_roles"]+=1
    assert sum(states.values())==ledger.execute("select count(*) from attempts").fetchone()[0]
    ledger.close()
    logged_failures=list(lines(base/"failures.jsonl"))
    expected_failures={e[0]:e for e in expected_failures}
    assert len(logged_failures)==len(expected_failures)
    for f in logged_failures:
        key,k,s,p,errors,status=expected_failures.pop(f["cache_key"])
        assert (f["block"],f["query"],f["role"])==k and (f["sweep"],f["start"],f["errors"],f["status"])==(s,p,errors,status)
        assert f["skipped"]==(reviews[k]["skipped"] if status=="missing" else [])
    assert not expected_failures
    for data in batches.values():assert data["count"]==data["stamp"][-1]
    for block,n in {"C_four":4096,"T_four":8192,"T_B1":2048}.items():
        c=blocks[block];c["planned"]=n
        assert c["first"]+c["skipped"]==n and c["valid_windows"]+c["failed_window"]==c["first"]
        assert c["valid_roles"]+c["missing_roles"]==c["roles"]
    resume=list(lines(RUN/"resume_log.jsonl"))
    assert sum(len(x["interrupted"]) for x in resume)==states["interrupted"]
    status=json.loads((RUN/"jobs/reviews/status.json").read_text())
    assert status["state"]=="done" and status["exit_code"]==0
    sessions=[json.loads(p.read_text()) for p in sorted((RUN/"jobs/reviews").glob("previous-*.json"))]+[status]
    gpu_h=sum((s["ended_epoch"]-s["started_epoch"])/3600 for s in sessions)
    with (RUN/"gpu_hours.csv").open("w") as f:
        w=csv.writer(f);w.writerow(["session_pid","started_epoch","ended_epoch","state","gpu_hours"])
        for s in sessions:w.writerow([s["pid"],s["started_epoch"],s["ended_epoch"],s["state"],(s["ended_epoch"]-s["started_epoch"])/3600])
    result=dict(run=RUN.name,users={"C":len(split["C"]),"T":len(split["T"])},roles=len(reviews),windows=len(keys),
                requests=sum(states.values()),attempt_states=dict(states),blocks={k:dict(v) for k,v in blocks.items()},
                input_tokens=[min(input_lengths),max(input_lengths)],output_tokens=[min(output_lengths),max(output_lengths)],
                batch_generation_gpu_hours={k:sum(d.values())/3600 for k,d in timings.items()},
                gpu_hours=gpu_h,resume_count=len(resume)-1,model_revision=MODEL.name)
    write_json(RUN/"acceptance.json",result)
    files=[base/n for n in ("windows.jsonl","reviews.jsonl","failures.jsonl","attempts.sqlite")]
    files += [RUN/n for n in ("manifest.json","model_check.log","gpu_hours.csv","resume_log.jsonl","acceptance.json")]
    files += sorted((RUN/"code").glob("*.py"))
    (RUN/"phase4.sha256").write_text("".join(f"{checksum(p)}  {p.relative_to(RUN)}\n" for p in files))
    for p in files:
        if p.parent==base:p.chmod(0o444)
    report=["# Phase 4 验收｜原版 8B 单评审与四角色正式推理","",
            f"运行 `{RUN.name}` 已在 `hkust-gz-hpc3-temp2` 完成。C 128 人、T 256 人全部来自 Phase 2 已验收的固定名单。独立程序从原始回复重放了 {len(keys):,} 个逻辑窗口，逐项核对 {len(reviews):,} 份角色终态及每岗预测。","","| 推理块 | 计划窗 | 实际首轮 | 格式或中断重试 | 有效窗 | 持久失败窗 | 后续跳过 | 有效选票 / 计划角色 |","|---|---:|---:|---:|---:|---:|---:|---:|"]
    for name in ("C_four","T_four","T_B1"):
        c=blocks[name];report.append(f'| {name} | {c["planned"]:,} | {c["first"]:,} | {c["retries"]} | {c["valid_windows"]:,} | {c["failed_window"]} | {c["skipped"]} | {c["valid_roles"]:,} / {c["roles"]:,} |')
    report+=["","## 验收与交接","",
        "1. 逐一验证 C/T 成员、十岗名单哈希、盲评字段白名单、窗口提示哈希与输入 token。模型从 Phase 1 校验通过的原版 Qwen3-8B 快照加载。模型推理和核验只读取 Phase 2 的划分、十岗映射与盲评文本。见 [运行配置](manifest.json) 和 [权重检查](model_check.log)。",
        "2. 各角色从相同十岗榜独立写回两遍。每次解析原始回复，检查四个编号、概率范围及额外预测头，逐窗核对输入和输出岗位。有效选票均从 8 个窗口重建，逐岗预测经截断、平均 logit、sigmoid 独立复算。",
        f'3. 计划 14,336 窗，实际模型请求 {result["requests"]:,} 次。每次重试或提前终止都与 [异常明细](base/failures.jsonl) 核对，累计恢复 {result["resume_count"]} 次。GPU 作业从启动至退出合计 {gpu_h:.4f} 单卡小时，包含模型装载、推理和落盘。[作业账本](gpu_hours.csv) · [运行工作记录](work_log.md)。',
        "", "第 5 期可读取 [完整选票与逐岗概率](base/reviews.jsonl)。[逐窗原始记录](base/windows.jsonl)、[验收统计](acceptance.json) 和 [封存校验清单](phase4.sha256) 可用于复算。",""]
    (RUN/"phase4_acceptance.md").write_text("\n".join(report))
    return result


if __name__=="__main__":
    print(json.dumps(validate(),ensure_ascii=False,indent=2))
