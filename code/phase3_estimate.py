"""Derive per-route hours from the retained H100 observations (no test labels)."""
import csv
import json
from pathlib import Path
from statistics import mean,median

OUT=Path("runtime/runs/v2-phase3-20261121/pilot")
ROLES=["skill","trajectory","preference","employer"]


def events(mode):
    p=OUT/(mode+".jsonl")
    if not p.is_file():return []
    return [json.loads(line) for line in p.read_text().splitlines() if line.strip()]


def per_window(rows,roles):
    costs=[]
    for role in roles:
        obs=[e["seconds"]/e["windows"] for e in rows if e["event"]=="inference" and e["role"]==role]
        if not obs:raise ValueError("missing "+role)
        costs.append([min(obs),median(obs),max(obs)])
    return [mean(x) for x in zip(*costs)]


def cost(rows,n,roles):
    return [n*c for c in per_window(rows,roles)]


def add(*components):
    return [sum(v) for v in zip(*components)]


def overhead(rows,keys):
    s=sum(e["seconds"] for e in rows if e["event"] in keys)
    return [s]*3


def stage_setup(stage, rows, workloads):
    """Preserve launch, tokenization, model switches, save/reload and process exit.

    The pilot-only cap stress and measured steady workloads are replaced rather
    than extrapolated. This requires a finished job receipt for each route.
    """
    receipt=json.loads((OUT.parent/"jobs"/stage/"status.json").read_text())
    assert receipt["state"]=="done"
    total=receipt["ended_epoch"]-receipt["started_epoch"]
    fixed=total-overhead(rows,[*workloads,"stress"])[0]
    assert fixed>=0
    return [fixed]*3


def main():
    obs={k:events(k) for k in ["base","teacher","full","lora"]}
    cpu=json.loads((OUT/"cpu.json").read_text())
    cpu_sec={k:[min(v),median(v),max(v)] for k,v in cpu.items()}
    details={"seconds_cpu_proxy":cpu_sec,"samples":{},"gpu_seconds":{},"data_cache":"Phase 2 accepted lists already available",
             "workload":{"calibration_users":128,"test_users":256,"main_role_windows":12288,
                         "main_general_windows":2048,"student_role_windows":12288,"epochs":1},
             "units":{"memory":"GiB, torch allocator","range":"observed speed envelope at fixed planned volume"},
             "retention_sensitivity":{}}
    routes=[]
    def register(name,gpu,cpu_time,notes):
        wall=add(gpu,cpu_time) if gpu else None
        row={"route":name,"status":notes}
        for prefix,series in [("wall",wall),("gpu",gpu)]:
            for key,val in zip(["low_h","center_h","high_h"],series or [None]*3):
                row[prefix+"_"+key]=round(val/3600,5) if val is not None else "NA"
        routes.append(row)
    for k,rs in obs.items():
        inf=[x for x in rs if x["event"]=="inference"]
        details["samples"][k]=dict(events=len(rs),complete=any(x["event"]=="completed" for x in rs),
            peak_allocated_GiB=round(max(x.get("peak_allocated",0) for x in rs)/2**30,3) if rs else None,
            peak_reserved_GiB=round(max(x.get("peak_reserved",0) for x in rs)/2**30,3) if rs else None,
            logical=sum(x["windows"] for x in inf),valid=sum(sum(x["valid"]) for x in inf),retries=sum(x["retries"] for x in inf),
            errors=[x["error"] for x in rs if x["event"]=="error"])
        stats=details["samples"][k]
        for event in ("load","warmup","inference","train_step","save","stress"):
            sub=[e for e in rs if e["event"]==event]
            if sub:
                stats[event+"_peak_allocated_GiB"]=round(max(e["peak_allocated"] for e in sub)/2**30,3)
                stats[event+"_seconds"]=round(sum(e.get("seconds",0) for e in sub),4)
        updates=[e for e in rs if e["event"]=="train_step"]
        if updates:
            stats["update_steps"]=len(updates)
            stats["trainable_parameters"]=updates[-1]["trainable"]
            stats["optimizer_state_dtype"]=updates[-1]["state_dtype"]
            stats["max_observed_parameter_delta"]=max(e["delta"] for e in updates)
            stats["step_seconds_min_med_max"]=[min(e["seconds"] for e in updates),median(e["seconds"] for e in updates),max(e["seconds"] for e in updates)]
        if inf:
            for group,roles in [("four",ROLES)]+([("general",["general"])] if k=="base" else []):
                seconds=per_window(rs,roles);details["samples"][k][group+"_windows_per_hour"]=[round(3600/v,2) for v in reversed(seconds)]
    base=obs["base"]
    assert details["samples"]["base"]["complete"]
    main_setup=stage_setup("base",base,["inference"])
    details["gpu_seconds"]["main_setup"]=main_setup
    m=add(main_setup,cost(base,12288,ROLES),cost(base,2048,["general"]))
    details["gpu_seconds"]["main"]=m
    register("main",m,cpu_sec["main"],"measured rates")
    teach=obs["teacher"]
    demos_path=OUT/"teacher_demos.json"
    for mode in ["full","lora"]:
        name="main+SFT" if mode=="full" else "main+SFT+LoRA"
        student=obs[mode]
        if not details["samples"]["teacher"]["complete"] or not details["samples"][mode]["complete"]:
            register(name,None,None,"see hardware status");continue
        demos=json.loads(demos_path.read_text())
        n=2048
        fraction=len(demos)/details["samples"]["teacher"]["logical"]
        train_n=round(n*fraction)
        details.update(teacher_windows_planned=n,pilot_teacher_demonstrations=len(demos),pilot_retention=fraction,planned_examples=train_n)
        steps=[e["seconds"] for e in student if e["event"]=="train_step"]
        # Median observed update includes optimizer work and supervised tokens; batch size one.
        train=[train_n*v for v in (min(steps),median(steps),max(steps))]
        teacher_setup=stage_setup("teacher",teach,["inference"])
        details["gpu_seconds"]["teacher_setup"]=teacher_setup
        teacher=add(teacher_setup,cost(teach,n,ROLES))
        rating=cost(student,12288,ROLES)
        setup=stage_setup(mode+"-matched",student,["inference","train_step"])
        details["gpu_seconds"].update({mode+"_training":train,mode+"_student_recheck":rating,mode+"_setup":setup,"teacher":teacher})
        common=add(m,teacher,rating,setup)
        cpu_total=add(cpu_sec["main"],cpu_sec["student"])
        register(name,add(common,train),cpu_total,"measured rates; one shared pass")
        details["retention_sensitivity"][mode]=[
            {"retention":rate,"training_examples":round(n*rate),
             "wall_hours_at_median_speed":round((common[1]+cpu_total[1]+round(n*rate)*median(steps))/3600,5)}
            for rate in sorted({0.25,0.5,1.0,fraction})]
    (OUT/"hardware.json").write_text(json.dumps(details,indent=2)+"\n")
    with (OUT/"runtime_estimates.csv").open("w") as f:
        w=csv.DictWriter(f,fieldnames=routes[0].keys());w.writeheader();w.writerows(routes)
    print(json.dumps(routes,ensure_ascii=False))

if __name__=="__main__":main()
