"""Non-overlapping GPU accounting; CPU durations are recorded separately."""
from experiment_paths import ROOT, phase, lock_sources
from datetime import datetime
from phase6_score import phase, P4, P5, EV, OUT, readcsv, read, write, csvout

def build():
    formal=read(P4/'acceptance.json')
    session=readcsv(P4/'gpu_hours.csv')
    total=sum(float(r['gpu_hours']) for r in session)
    assert abs(total-formal['gpu_hours'])<1e-10
    pilot=readcsv(phase(3)/'gpu_hours.csv')
    spans=sorted((datetime.fromisoformat(x['start_utc']),datetime.fromisoformat(x['end_utc'])) for x in pilot)
    assert all(a[1]<=b[0] for a,b in zip(spans,spans[1:]))
    preparation=float(readcsv(phase(1)/'gpu_hours.csv')[0]['gpu_hours'])
    gpu_pilot=sum(float(r['gpu_hours']) for r in pilot)
    rates=readcsv(phase(3)/'pilot/runtime_estimates.csv')[0]
    planned=float(rates['gpu_center_h'])
    rows=[]
    def add(stage,scope,windows,retries,gpu,cpu,basis,evidence):
        rows.append(dict(stage=stage,scope=scope,first_windows=windows,retries=retries,gpu_hours=gpu,cpu_wall_seconds=cpu,basis=basis,source=str(evidence.relative_to(ROOT))))
    add('Phase 1','准备估计','','',preparation,'','开发实例占用上界估记，平台计费未核实',phase(1)/'gpu_hours.csv')
    add('Phase 3','pilot',184,0,gpu_pilot,'','独立探测作业（含中断），不计入主线',phase(3)/'gpu_hours.csv')
    for block,h in formal['batch_generation_gpu_hours'].items():
        b=formal['blocks'][block]
        add(block,'主线',b['first'],b['retries'],h,'','批次生成耗时',P4/'acceptance.json')
    rest=total-sum(formal['batch_generation_gpu_hours'].values())
    assert rest>=0
    add('Phase 4 其他占卡','主线','','',rest,'','作业占卡总量减批次生成；含装载、准备、写盘及退出',P4/'gpu_hours.csv')
    phase5_seconds=sum(read(P5/'evidence/run_receipt.json')['job_elapsed_s'].values())
    add('Phase 5','主线',0,0,0,phase5_seconds,'正式、补充和验收三项 CPU 作业墙钟之和',P5/'evidence/run_receipt.json')
    step6=read(EV/'score_receipt.json')['cpu_wall_s']
    verify=read(EV/'statistics_verified.json')['cpu_wall_s']
    add('Phase 6','主线',0,0,0,step6+verify,'评分和独立统计复算；文档整理与 SSH 等待另计',EV/'statistics_verified.json')
    csvout('cost_ledger.csv',rows)
    result=dict(main_gpu_hours=total,estimate_gpu_hours=planned,estimate_low=float(rates['gpu_low_h']),estimate_high=float(rates['gpu_high_h']),delta_actual_minus_estimate=total-planned,percent_delta=100*(total/planned-1),phase4_other=rest,pilot=gpu_pilot,prep_estimate=preparation,measured_total=total+gpu_pilot,including_preparation_estimate=total+gpu_pilot+preparation,phase5_cpu_s=phase5_seconds,phase6_score_s=step6,phase6_verify_s=verify)
    write(OUT/'cost_summary.json',result)
if __name__ == "__main__":
    lock_sources(6)
    build()
