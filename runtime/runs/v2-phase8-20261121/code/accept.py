"""Verify the training handoff and publish only the Optional entries of the matrix."""
import collections,math,random,re,subprocess,tempfile
from common import *
def audit_train():
    source=records(phase(7)/'distill/demos.jsonl');hist=records(D/'training.jsonl');idx=list(range(505));random.Random(SEED).shuffle(idx)
    assert len(hist)==len(idx)==len(source) and [v['source_index'] for v in hist]==idx
    for i,row in enumerate(hist,1):
        demo=source[row['source_index']]
        assert row['step']==i and all(demo[k]==row[k] for k in ('window','role','prompt_sha256'))
        assert math.isfinite(row['loss']) and math.isfinite(row['grad_norm']) and row['grad_norm']>0 and row['answer_tokens']>1
    from transformers import AutoTokenizer
    tok=AutoTokenizer.from_pretrained(BASE,local_files_only=True)
    for h in hist:
        d=source[h['source_index']]
        msg=tok.apply_chat_template(d['messages'],add_generation_prompt=True,enable_thinking=False,tokenize=False)
        assert len(tok.encode(msg,add_special_tokens=False))==h['prompt_tokens']
        assert len(tok.encode(d['answer'],add_special_tokens=False))+1==h['answer_tokens']
    conf=read(D/'adapter/adapter_config.json')
    assert conf['r']==16 and conf['lora_alpha']==32 and conf['lora_dropout']==0
    assert read(E/'adapter.json')==read(E/'reload.json')['adapter'] and read(E/'reload.json')['base'].endswith(BASE.name)
    for n,h in read(E/'adapter.json').items():assert sha(D/'adapter'/n)==h
    return dict(steps=len(hist),coverage=dict(collections.Counter(r['role'] for r in hist)),trainable=read(D/'config.json')['trainable'],gradient_min=min(x['grad_norm'] for x in hist),max_memory=max(x['peak_allocated'] for x in hist))
def session_hours(stage):
    job=RUN/'jobs'/stage
    sessions=[read(p) for p in sorted(job.glob('previous-*.json'))]+[read(job/'status.json')]
    assert sessions[-1]['exit_code']==0
    return sum(x['ended_epoch']-x['started_epoch'] for x in sessions)/3600
def interval(s,signed=False):
    f='+.5f' if signed else '.5f'
    return f"{s['mean']:{f}} [{s['lo']:{f}}, {s['hi']:{f}}]"
def main():
    training=audit_train()
    rv=read(E/'review_verified.json');assert rv['roles']==1536 and rv['users']=={'C':128,'T':256}
    for b,n in [('C_four',4096),('T_four',8192)]:
        counts=rv['blocks'][b];assert counts['planned']==n==counts['first']+counts['skipped']
    before=E/'matrix_before.md';guide=ROOT/'global_guide/v2_frozen/acceptance_result_matrix.md'
    assert sha(before)==read(phase(6)/'acceptance.json')['matrix_sha256']
    for i in [6,7]:
        with (E/f'phase{i}_after.txt').open('w') as f:
            subprocess.run(['sha256sum','-c',f'phase{i}.sha256'],cwd=phase(i),stdout=f,check=True)
        assert (E/f'phase{i}_before.txt').read_bytes()==(E/f'phase{i}_after.txt').read_bytes()
    summary=read(OUT/'summary.json');effect=summary['joint_ndcg_difference'];assert effect['n']==228
    upstream=read(phase(7)/'acceptance.json');teacher=upstream['teacher_gpu_hours']
    train=session_hours('train');reviews=session_hours('reviews');main_h=read(phase(6)/'eval/main/cost_summary.json')['main_gpu_hours']
    extra=teacher+train+reviews;total=main_h+extra
    cost=dict(teacher=teacher,train=train,reviews=reviews,added_gpu_hours=extra,main_gpu_hours=main_h,cumulative_main_plus_distill=total,pilot_and_preparation_excluded=True)
    write(OUT/'cost.json',cost)
    with (RUN/'gpu_hours.csv').open('w') as f:
        w=csv.writer(f,lineterminator='\n');w.writerow(['stage','scope','gpu_hours','basis'])
        for name,h,p in [('main',main_h,phase(6)/'eval/main/cost_summary.json'),('teacher',teacher,phase(7)/'gpu_hours.csv'),('LoRA',train,RUN/'jobs/train/status.json'),('student_reviews',reviews,RUN/'evidence/review_gpu_hours.csv')]:w.writerow([name,'主线' if name=='main' else '扩展',h,p.relative_to(ROOT)])
    result=before.read_text()
    def lk(label,p):return f'[{label}]({os.path.relpath(p,guide.parent)})'
    stats=lk('学生逐人指标',OUT/'per_user.csv');delta=lk('配对差值',OUT/'paired_deltas.csv');ledger=lk('工时账本',RUN/'gpu_hours.csv')
    def replace_line(prefix,line):
        nonlocal result
        targets=[x for x in result.splitlines() if x.startswith(prefix)]
        assert len(targets)==1,prefix
        result=result.replace(targets[0],line)
    replace_line('| Optional 逐角色蒸馏 |',f"| Optional 逐角色蒸馏 | 32B 教师示范；8B 共享 LoRA | 256 人预留，214 人可选窗；505 条保留示范覆盖 124 人 | 已完成。F-SFT−F 联合 nDCG@5 差 {interval(effect,True)}。{delta} |")
    replace_line('| Optional 蒸馏 |',f"| Optional 蒸馏 | 完成 505 步、学生 C/T 同榜盲评、独立校准及配对评分 | {lk('Phase 8 验收',RUN/'phase8_acceptance.md')} |")
    cells=[interval(summary['metrics'][k]) for k in ['joint_ndcg','joint_recall','delivery_ndcg','employer_ndcg','joint_hits']]
    replace_line('| Optional F-SFT |','| Optional F-SFT | '+' | '.join(cells+[stats])+' |')
    replace_line('| Optional F-SFT−F |',f"| Optional F-SFT−F | 联合 nDCG@5 | 228 | {interval(effect,True)} | {delta} |")
    role_table=['| 教师角色 | 尝试 | 格式有效 | 保留 | 保留率 |','|---|---:|---:|---:|---:|']
    for r in upstream['role_statistics']:role_table.append(f"| {r['role']} | {r['logical']} | {r['valid']} | {r['retained']} | {float(r['retention_rate']):.2%} |")
    fallback=collections.Counter(v['reason'] for v in summary['fallback'].values())
    retries=sum(x['retries'] for x in rv['blocks'].values());skips=sum(x['skipped'] for x in rv['blocks'].values())
    replace_line('**Optional 蒸馏状态',f"**Optional 蒸馏已完成。** 32B 教师 {teacher:.5f}、LoRA 训练 {train:.5f}、学生重评 {reviews:.5f} 单卡小时。扩展新增 {extra:.5f}，主线加扩展累计 {total:.5f} GPU 小时。主线复用一次，pilot 与准备时段单列。{ledger}\n\n"+'\n'.join(role_table)+f"\n\n学生计划 12,288 窗，实际调用 {rv['requests']:,} 次，重试 {retries} 次，跳过 {skips} 窗。F-SFT 回退 {len(summary['fallback'])} 人，原因计数 {dict(fallback)}。五项指标与原版使用相同用户，人数依次为 228、228、250、228、256。教师示范与学生微调的效果按同榜配对区间解释。{stats} · {lk('分角色示范统计',phase(7)/'distill/role_results.csv')}")
    write(RUN/'acceptance.json',dict(status='passed',training=training,windows=rv['blocks'],effect=effect,cost=cost,metrics=summary['metrics'],fallback=dict(fallback),matrix_sha256=hashlib.sha256(result.encode()).hexdigest()))
    report=f"""# Phase 8 验收｜学生同榜重评

**验收通过，Optional 完成。** 唯一扩展比较为 F-SFT−F。联合 nDCG@5 配对人数 228，差值 {interval(effect,True)}，区间按用户配对重采样 2,000 次得到。

| 协议要求 | 实测与证据 |
|---|---|
| 共享适配器训练与重载 | 505 条示范各用一次，505 次 AdamW 更新，只监督回答及 EOS。原版 BF16 Qwen3-8B，LoRA rank 16、alpha 32。逐条复核训练顺序、token 与梯度；独立评审进程重载同一份适配器。[训练记录](distill/training.jsonl) · [重载回执](evidence/reload.json) |
| 同池四角色评审 | 原 C 128 人、T 256 人，计划 12,288 窗，实际调用 {rv['requests']:,} 次。逐窗重放，检查名单、提示、排列、概率、重试与跳过。[重评复核](evidence/review_verified.json) |
| 学生可信聚合与双边排序 | C 内按原折拟合温度、Brier、六对相关与角色权重，独立复核温度、相关及 PL 解。学生 256 个完整榜或回退榜先封存再评分。[学生参数](student/calibration.json) · [冻结排名](student/rankings.jsonl) · [封存记录](evidence/blind_seal.json) |
| 评测与记账 | 每项指标逐人 ID 与原版相同。独立复算五项指标及配对区间。教师、训练、重评分别为 {teacher:.5f}、{train:.5f}、{reviews:.5f} GPU 小时，新增 {extra:.5f} 小时。[五项结果](eval/student/summary.json) · [逐人差值](eval/student/paired_deltas.csv) · [GPU 账本](gpu_hours.csv) |

[结果矩阵快照](acceptance_result_matrix.md) · [工作记录](work_log.md)
"""
    (RUN/'phase8_acceptance.md').write_text(report)
    def redirect(m):
        p=(guide.parent/m[1]).resolve()
        assert p.is_file(),p
        return ']('+os.path.relpath(p,RUN)+')'
    snap=re.sub(r'\]\(([^)]+)\)',redirect,result)
    (RUN/'acceptance_result_matrix.md').write_text(snap)
    assert guide.read_bytes()==before.read_bytes() or guide.read_text()==result
    with tempfile.NamedTemporaryFile('w',dir=guide.parent,delete=False) as f:
        f.write(result);tmp=Path(f.name)
    tmp.chmod(guide.stat().st_mode&0o777);tmp.replace(guide)
    assert sha(guide)==read(RUN/'acceptance.json')['matrix_sha256']
    log('独立验收通过，服务器结果矩阵 Optional 栏已更新。')
if __name__=='__main__':main()

