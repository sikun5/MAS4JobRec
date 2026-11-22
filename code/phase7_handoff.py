"""CPU-only handoff after the teacher exits. Rebuild results directly from its journal."""
from experiment_paths import ROOT, lock_sources
import csv,datetime,os,subprocess,time
from phase7_prepare_windows import (D, EV, RUN, read, save, sha)
from phase7_filter_demos import filter_demos
from phase7_verify_demos import verify
def now():return datetime.datetime.now().astimezone().isoformat(timespec='seconds')
def log(msg):
    with (RUN/'work_log.md').open('a') as f:f.write(f'\n- {now()}　{msg}\n')
def main():
    status_file=RUN/'jobs/teacher/status.json'
    while read(status_file)['state']=='running':time.sleep(30)
    teacher=read(status_file)
    assert teacher['state']=='done' and teacher['exit_code']==0,teacher
    log('教师进程已退出，开始 CPU 筛选并重放训练窗口、角色提示与原始答复。')
    filter_demos();verify()
    v=read(EV/'verified.json')
    sessions = [read(p) for p in sorted(status_file.parent.glob("previous-*.json"))] + [teacher]
    hours = [(s["ended_epoch"] - s["started_epoch"]) / 3600 for s in sessions]
    gh = sum(hours)
    with (RUN/'gpu_hours.csv').open('w') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['stage','start_epoch','end_epoch','gpu_hours','basis'])
        for session, duration in zip(sessions, hours):
            w.writerow(['teacher', session['started_epoch'], session['ended_epoch'], duration,
                        'complete GPU session including import/load/generation/save/exit'])
        w.writerow(['prepare/filter/verify','','',0,'CPU only'])
    p6=RUN.parent/'v2-phase6-20261121'
    with (EV/'phase6_after.txt').open('w') as f:subprocess.run(['sha256sum','-c','phase6.sha256'],cwd=p6,stdout=f,check=True)
    assert (EV/'phase6_after.txt').read_bytes()==(EV/'phase6_before.txt').read_bytes()
    head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
    assert head==(EV/'git_head.txt').read_text().strip()
    assert sha(ROOT/'global_guide/v2_frozen/acceptance_result_matrix.md')==read(p6/'acceptance.json')['matrix_sha256']
    role=v['role_statistics']
    passed=v['usable_demos']>0
    state='Phase 7 通过，可进入 Phase 8' if passed else '中断于教师阶段（无可训练示范）'
    save(RUN/'acceptance.json',dict(status=state,accepted=passed,finished=now(),eligible_users=214,skipped_users=42,teacher_gpu_hours=gh,revision=read(D/'teacher_manifest.json')['revision'],**v))
    body=['# Phase 7 验收｜32B 教师逐角色示范',state+'。',f"预留 256 名训练用户。214 人各抽两个窗口；42 人无合格四岗组合。428 个窗口均为一正三负。四角色共 1,712 次逻辑判断，实际尝试 {v['raw_attempts']} 次，共保留 {v['usable_demos']} 条示范，覆盖 {v['retained_unique_users']} 名不同求职者。", '| 角色 | 逻辑判断 | 重试 | 格式有效 | 最终无效 | 首位错误 | 保留 | 保留率 | 保留示范覆盖人数 |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|\n'+'\n'.join('| '+ ' | '.join([r['role'],r['logical'],r['retries'],r['valid'],r['invalid'],r['wrong_first'],r['retained'],f"{float(r['retention_rate']):.2%}",r['retained_users']])+' |' for r in role),f"GPU 总量 {gh:.5f} 单卡小时，含加载、推理及退出阶段。筛选和复核使用 CPU。按原版 8B 学生 tokenizer 将每条真实角色提示与目标回答编码并附加 EOS，总长度范围为 {v['student_token_range']} token。Phase 3 已验证可重新加载 LoRA 适配器。", '独立验收按训练划分及简历组复核互斥关系，固定种子重放组合和乱序。按正式提示模板及字段白名单重建 1,712 条消息，重解析每次原始答复并与训练标签对齐，逐条核对筛选理由、保留集合和四角色汇总。训练标签用于选窗及生成后的筛选。Phase 6 交付文件复核通过。','[配置与版本](distill/teacher_manifest.json) · [训练窗口](distill/train_windows.jsonl) · [盲评请求](distill/blind_requests.jsonl) · [原始输出](distill/teacher_outputs.jsonl) · [筛选明细](distill/filter_decisions.jsonl) · [训练示范](distill/demos.jsonl) · [角色统计](distill/role_results.csv) · [独立复核](evidence/verified.json) · [成本](gpu_hours.csv) · [工作记录](work_log.md)']
    (RUN/'phase7_acceptance.md').write_text('\n\n'.join(body)+'\n')
    log(state+'。已复核全部原始答复、学生 token 长度、角色统计、GPU 账本及 Phase 6 封存。')
if __name__ == "__main__":
    lock_sources(7)
    main()
