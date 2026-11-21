# Phase 6 验收｜主实验评分、跨 Agent 相关错误与成本

**通过，可以交付主线。** 运行 v2-phase6-20261121 使用 CPU。主方法相对基线的联合 nDCG@5 差为 −0.00368，95% CI [−0.01084, +0.00304]，四项检验的 Holm p 均为 1。主实验在已浏览或已投递的十岗池中未显示 F 优于 B4。[服务器结果矩阵](../../../global_guide/v2_frozen/acceptance_result_matrix.md) · [运行快照](acceptance_result_matrix.md)

| 验收项 | 实测 | 证据 |
|---|---|---|
| 开启 T 标签的前置条件 | 先按 37 项清单复核 Phase 5，评分代码及输入哈希冻结。256 人各有十种方法，同一有序候选池、全排列与 Top-5 均通过。评分程序随后一次读取 T 文件，校验其冻结哈希并关联 2,560 个人岗对，按原始槽位映射岗位 ID。 | [预检](evidence/preflight.json) · [评分记录](evidence/score_receipt.json) |
| 指标、分母与特殊样例 | 联合与雇主认可分母 228，投递分母 250，联合正例个数覆盖全体 256。28 人无联合正例，其中 6 人无投递、22 人有投递无认可。协议样例原榜前五联合正例 0，雇主子榜正例第 2 名，nDCG 为 0.63093。每个方法包含全部回退用户，实际回退 0。 | [分母](eval/main/denominators.csv) · [逐人记录](eval/main/per_user.csv) |
| 均值、配对区间与置换检验 | 从逐人结果重算十方法五指标、11 项配对差值和 95% 区间，以及四个原始 p 与 Holm p。用每项共同用户重采样 2,000 次，确认性比较置换 10,000 次。 | [汇总](eval/main/summary.json) · [差值](eval/main/paired_deltas.csv) · [检验](eval/main/comparisons.csv) · [独立复算](evidence/statistics_verified.json) |
| 共同错误与排序影响 | 已核对六对三项相关与区间、分层人数与岗位数、退化项、四角色权重迁移与移除效应。T1 对 T0 仅 1 人的后五名变化；F 对 T1 无十岗顺序变化。 | [跨角色报告](../v2-phase5-20261121/analysis/correlated_error_report.md) · [热图](../v2-phase5-20261121/analysis/correlation_heatmap.png) · [矩阵第 4 节](acceptance_result_matrix.md) |
| 成本与交接 | 主实验 14,336 个首轮窗口，0 次重试，3.39318 GPU 小时。P5、P6 的 CPU 作业另记秒数。准备时段的计费不确定性单列。F 逐人结果及上游冻结排名指纹已登记。 | [逐阶段账本](eval/main/cost_ledger.csv) · [成本汇总](eval/main/cost_summary.json) · [F 交接](evidence/F_receipt.json) |
| 证据与 Optional | 已验证两份矩阵的数字、证据路径和一致性，快照采用相对于运行目录的链接。Optional 蒸馏未运行（本次执行至 Phase 6）。 | [机器验收](acceptance.json) · [工作记录](work_log.md) · [封存清单](phase6.sha256) |

完整评分程序、统计复算程序和矩阵发布核验分别见 [score.py](code/score.py)、[verify.py](code/verify.py)、[accept.py](code/accept.py)。已填矩阵跨角色部分沿用 Phase 5 的 C 样本外统计；其排序效应由 T 上预登记的 T1−T0 给出。
