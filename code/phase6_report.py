"""Render the authoritative matrix and a run-local snapshot with working relative links."""
from experiment_paths import ROOT, phase, lock_sources
import os
from phase6_score import (P2, P4, P5, EV, METHODS, METRICS, OUT, RUN, read, readcsv)

GUIDE=ROOT/'global_guide/v2_frozen'
MATRIX=GUIDE/'acceptance_result_matrix.md'
SNAP=RUN/'acceptance_result_matrix.md'
def display(x, signed=False, digits=5):
    return format(x,('+' if signed else '')+f'.{digits}f')
def interval(x,signed=False):
    if x['n']==0:
        return f"NA（{x['reason']}）"
    return f"{display(x['mean'],signed)} [{display(x['lo'],signed)}, {display(x['hi'],signed)}]"
def render(location):
    def link(text,path):
        return f'[{text}]({os.path.relpath(path,location.parent)})'
    def table(header,rows):
        def line(cells):
            return '| '+' | '.join(map(str,cells))+' |'
        return '\n'.join([line(header),line(['---']*len(header)),*(line(row) for row in rows)])
    template=(EV/'acceptance_result_matrix.md').read_text()
    # Preserve the frozen source-data and pilot interpretation sections, rebasing their links.
    import re
    def section(start,end):
        body=template.split(start,1)[1].split(end,1)[0]
        return re.sub(r'\[([^\]]+)\]\(([^)]+)\)',lambda m:link(m[1],GUIDE/m[2]),body).strip()
    summary=read(OUT/'summary.json')
    stats=summary['metrics']
    paired=summary['comparisons']
    bykey={(r['first'],r['second'],r['metric']):r for r in paired}
    count=summary['denominators']
    costs=read(OUT/'cost_summary.json')
    weights=readcsv(P5/'analysis/role_weights.csv')
    pairs=readcsv(P5/'analysis/correlated_error_pairs.csv')
    heads=read(P5/'base/calibration.json')['full_C_heads']
    changed=read(EV/'statistics_verified.json')['changed_orders']
    lp=link('逐人指标',OUT/'per_user.csv')
    ld=link('逐人差值',OUT/'paired_deltas.csv')
    lt=link('配对检验',OUT/'comparisons.csv')
    lc=link('成本账本',OUT/'cost_ledger.csv')
    ls=link('分母明细',OUT/'denominators.csv')
    corr=link('六对相关与区间',P5/'analysis/correlated_error_pairs.csv')
    lw=link('权重明细',P5/'analysis/role_weights.csv')
    result=[]
    add=result.append
    add('# v2 验收与结果矩阵｜面向求职者的多 Agent 岗位重排')
    add(f"完整方法 F 相对指定基线 B4 的联合 nDCG@5 差值为 {interval(paired[3],True)}，Holm p = {paired[3]['holm']:.4f}。主线实耗 {costs['main_gpu_hours']:.5f} 单卡 GPU 小时。本次未观察到完整方法相对基线的显著提升。结论限于每人已浏览或投递的十岗池。{lt} · {lc}")
    add('## 1. 研究问题、设置、数据和结果')
    add(table(['研究问题','实验设置','数据使用','预期检验及实测结论'],[
        ['RQ1 增加专业评审','B4 对 B1；每人 32 窗对 8 窗','同一测试名单与原版 8B','估计角色与调用共同增量。联合 nDCG@5 差 '+interval(paired[0],True)+'，区间跨零。'+ld],
        ['RQ2 共同错误降权','T1 对 T0；T1 对 W1 为聚合器对照','C 分析误差；T 检验排名','校准集显示六对正相关；降权后测试前五顺序改变 0 人。'+corr+' · '+ld],
        ['RQ3 双边效用','F 对 T1；F 对 B4','T 的联合、投递及已投递认可子榜','F 与 T1 的全部十岗顺序一致，双边指标增量均为零。'+ld],
        ['Optional 逐角色蒸馏','32B 教师与 8B LoRA 学生','独立训练用户预留 256，正式使用 0','未运行（本次执行至 Phase 6）；F-SFT 及配对效应均为 NA。']]))
    add('## 2. 运行登记与协议验收')
    add('运行 v2-phase6-20261121，主机 hkust-gz-hpc3-temp2。标准档 Pilot / C / T 为 16 / 128 / 256。每位 T 用户固定十岗，输出前五。D 为投递，S 为已投递条件下的认可，J 为投递且获认可。'+link('冻结设置',GUIDE/'experimental_setting.md')+' · '+link('评分前封存',EV/'preflight.json'))
    add('### 已核对的原始数据')
    add(section('### 已核对的原始数据','### 执行验收'))
    add('### 执行验收')
    add(table(['环节','验收实测','证据'],[
        ['划分与固定名单','656 人，用户及规范化简历组跨区块重叠均为 0；T 256 人的名单与各方法的初始哈希逐一一致',link('数据验收',P2/'phase2_acceptance.md')+' · '+link('名单核验',EV/'preflight.json')],
        ['评审推理','原版 Qwen3-8B，14,336 次首轮判断，1,792 份有效选票，重试与缺失均为 0',link('运行配置',P4/'manifest.json')+' · '+link('评审验收',P4/'phase4_acceptance.md')],
        ['校准与配对对照','C 内三折 43/43/42。T0/T1/W1/F 共用选票，T1/W1 有效角色和权重一致；pE 仅在已投递样本上校准。估参阶段读取 T 标签 0 次',link('可信排序验收',P5/'phase5_acceptance.md')],
        ['回退与评分','每方法 256 份完整排名，全部计入分母核对。各方法实际回退 0。开封前合成样例雇主 nDCG 为 0.63093',link('回退清单',P5/'base/fallbacks.csv')+' · '+link('评分回执',EV/'score_receipt.json')],
        ['硬件与复现','H100 单卡评审，Phase 5、6 使用 CPU。固定统计种子 20261121',link('探测验收',phase(3)/'phase3_acceptance.md')+' · '+link('独立复算',EV/'statistics_verified.json')],
        ['Optional 蒸馏','未运行（本次执行至 Phase 6）','正式教师示范、学生训练和重评均未启动']]))
    add('## 3. 排序结果与配对比较')
    add('### 共同评价分母')
    add(table(['指标','可评价人数','排除或保留情况','逐人依据'],[
        ['联合 nDCG@5、Recall@5',count['joint'],f"无联合正例 {count['no_joint']} 人",ls],
        ['投递 nDCG@5',count['delivered'],f"无投递 {count['no_delivery']} 人",ls],
        ['已投递子榜认可 nDCG@5',count['joint'],f"无投递 {count['no_delivery']} 人，有投递但无认可 {count['delivered_no_approval']} 人",ls],
        ['Top-5 联合正例数/人',count['all'],f"包含 {count['no_joint']} 名候选池无联合正例用户",ls]]))
    add('### 各方法实测结果')
    add('用户等权均值及 95% 百分位区间。每个指标在所有方法间共用同一组可评价用户和 2,000 次用户重采样。雇主侧先从完整十岗筛选已投递岗位，再截前五。'+link('冻结排名',P5/'base/rankings.jsonl'))
    add(table(['方法','联合 nDCG@5','联合 Recall@5','投递 nDCG@5','已投递认可 nDCG@5','Top-5 联合正例数/人','依据'],[[m,*[interval(stats[k][m]) for k in METRICS],lp] for m in METHODS]+[['Optional F-SFT',*['NA']*5,'未运行']]))
    add('### 四个预登记比较')
    add('指标为联合 nDCG@5。差值按前者减后者。双侧配对符号置换 10,000 次，p 值加一修正，四项一起做 Holm 校正，显著性水平 0.05。')
    add(table(['比较','配对用户','均值差 [95% CI]','原始 p','Holm p','依据'],[[r['first']+'−'+r['second'],r['n'],interval(r,True),f"{r['p']:.5f}",f"{r['holm']:.5f}",ld+' · '+lt] for r in paired[:4]]))
    add('### 解释性比较')
    names=dict(joint_ndcg='联合 nDCG@5',delivery_ndcg='投递 nDCG@5',employer_ndcg='已投递认可 nDCG@5')
    add(table(['比较','指标','配对用户','均值差 [95% CI]','依据'],[[r['first']+'−'+r['second'],names[r['metric']],r['n'],interval(r,True),ld] for r in paired[4:]]+[['Optional F-SFT−F','联合 nDCG@5','NA','NA','未运行']]))
    add('## 4. 角色贡献与跨 Agent 相关错误')
    add('温度来自完整 C；Brier 和残差相关来自 C 的样本外预测。移除效应按测试集上的“移除该角色后 B4 − 完整 B4”计算。')
    role_rows=[]
    for w in weights:
        role=w['role']
        effect=bykey['B4-minus-'+role,'B4','joint_ndcg']
        role_rows.append([role.title(),f"{heads[role+'.pJ']['temperature']:.5f}",w['n_users'],f"{float(w['brier']):.5f}",f"{100*float(w['T0']):.3f}%",f"{100*float(w['T1']):.3f}%",f"{float(w['delta_pp']):+.3f}",interval(effect,True),lw+' · '+ld])
    add(table(['角色','共同头温度','C 人数','Brier','T0 权重','T1 权重','变化（百分点）','移除效应 [95% CI]','依据'],role_rows))
    add('各对按同人同岗对齐，残差为校准共同概率减 J。负例层反映高估程度的同步性，正例层反映低估程度的同步性。区间沿用 Phase 5 在原折内按人 bootstrap 并重拟合温度 2,000 次的结果。')
    pair_rows=[]
    for row in pairs:
        vals=[f"{float(row[k]):.4f} [{float(row[k+'_lo']):.4f}, {float(row[k+'_hi']):.4f}]" for k in ['rho0','rho1','rho']]
        pair_rows.append([row['a'].title()+' × '+row['b'].title(),row['users0']+'/'+row['jobs0'],vals[0],row['users1']+'/'+row['jobs1'],vals[1],vals[2],(row['reason0'] or '无')+' / '+(row['reason1'] or '无'),row['bootstrap_degenerate0']+' / '+row['bootstrap_degenerate1'],corr])
    add(table(['角色对','J=0 人/岗','ρ₀ [95% CI]','J=1 人/岗','ρ₁ [95% CI]','两类等权 ρ [95% CI]','原样本退化原因（0/1 层）','重采样退化次数（0/1 层）','依据'],pair_rows))
    add('![负例、正例与等权均值的残差相关热图]('+os.path.relpath(P5/'analysis/correlation_heatmap.png',location.parent)+')')
    add('Skill 与 Trajectory 的等权相关最高（0.794）。Preference 权重降幅最大（−0.653 个百分点），Employer 升幅最大（+0.743 个百分点）。T1 对 T0 仅 1 人的后五名顺序有变化，全部 256 人的前五顺序一致。这支持校准集存在冗余，降权在本次测试上未带来联合 nDCG@5 增量。'+link('逐榜核对',EV/'statistics_verified.json')+' · '+link('Phase 5 分析',P5/'analysis/correlated_error_report.md'))
    add(f"投递概率温度为 {heads['preference.pC']['temperature']:.5f}（C 上 {heads['preference.pC']['n']} 岗）。条件认可概率温度为 {heads['employer.pE']['temperature']:.5f}（{heads['employer.pE']['n']} 个已投递人岗对），达到冻结上界。F 与 T1 在所有测试用户上的完整排序一致，投递侧和雇主侧的变化均为零。"+link('正式校准参数',P5/'base/calibration.json')+' · '+ld)
    add('## 5. 硬件、成本、失败记录与 Optional 状态')
    add(section('## 5. 硬件探测、耗时、失败记录与 Optional 状态','| 阶段 | 计划工作量'))
    ledger=readcsv(OUT/'cost_ledger.csv')
    add(table(['阶段','首轮窗','重试','单卡小时','CPU 墙钟秒','计时口径','依据'],[[r['stage'],r['first_windows'] or '不适用',r['retries'] or '不适用',f"{float(r['gpu_hours']):.5f}",f"{float(r['cpu_wall_seconds']):.3f}" if r['cpu_wall_seconds'] else '—',r['basis'],lc+' · '+link('来源',(ROOT / r['source']))] for r in ledger]))
    add(f"主线共 14,336 次判断、0 次重试。在 H100 上实耗 {costs['main_gpu_hours']:.5f} 小时，较 pilot 外推 {costs['estimate_gpu_hours']:.5f} 小时减少 {abs(costs['delta_actual_minus_estimate']):.5f} 小时（{abs(costs['percent_delta']):.2f}%）。批次生成合计 {costs['main_gpu_hours']-costs['phase4_other']:.5f} 小时，其余 {costs['phase4_other']:.5f} 小时包括装载、准备与落盘，记录支持合并计量。CPU 评分耗时 {costs['phase6_score_s']:.3f} 秒，独立统计复算 {costs['phase6_verify_s']:.3f} 秒，均为零 GPU 小时。"+lc)
    add(f"主线加 pilot 的实测占卡共 {costs['measured_total']:.5f} 小时。另计 Phase 1 开发实例准备时段的保守估计 {costs['prep_estimate']:.5f} 小时后为 {costs['including_preparation_estimate']:.5f} 小时，平台账单未核实。"+link('成本汇总',OUT/'cost_summary.json'))
    add('评审首轮失败、重试后持久失败、缺失角色均为 0。六种主方法与四个角色移除对照逐方法回退均为 0，F 沿用 BM25 与缺双边头退回 T1 分别为 0。'+link('评审统计',P4/'acceptance.json')+' · '+link('逐方法复核',EV/'statistics_verified.json'))
    add('**Optional 蒸馏状态：未运行（本次执行至 Phase 6）。** 正式教师生成、学生 LoRA 训练和重评的新增 GPU 小时均为 0。四个教师角色尝试及保留示范均为 0；保留率、学生效果和学生回退为 NA。Phase 3 的教师探测样本计入 pilot。')
    return '\n\n'.join(result)+'\n'
if __name__ == "__main__":
    lock_sources(6)
    SNAP.write_text(render(SNAP))
    (EV/'matrix_for_publication.md').write_text(render(MATRIX))
