# Phase 5 验收｜校准、共同错误与冻结排名

运行 `v2-phase5-20261121` 已通过验收。服务器为 `hkust-gz-hpc3-temp2`，全部计算使用 CPU。校准集含 128 人、1,280 个人岗对。测试集 256 人的十种方法全部生成完整排名，共 2,560 份。实际回退 0 次。

## 原协议验收

| 验收项 | 实测 | 证据 |
|---|---|---|
| 名单与标签隔离 | 按原散列顺序分折 43/43/42。所有拟合和验证用户逐折互斥。三个计算阶段的上游数据读取审计中，测试标签打开次数均为 0。 | [拟合与折清单](base/calibration.json) · [正式阶段读取记录](evidence/formal_data_access.json) · [验收读取记录](evidence/verify_data_access.json) |
| 校准与聚合 | 独立拟合四个共同头和两个双边头的三折及全量温度，逐项复核 1,280 份样本外记录。按人复算 Brier、六对相关、惩罚分母和两套归一化权重。 | [样本外记录](base/oof.jsonl) · [复算统计](acceptance.json) |
| 盲排 | 每种方法各 256 份。核对全部十岗排列、Top-5、同分按岗位 ID、逐项双边分及 T1/W1 角色与权重一致性。512 次 PL 解均通过独立目标函数梯度检查，最大梯度绝对值 8.03e-08。 | [十岗排名](base/rankings.jsonl) · [原始候选名单](base/candidate_lists.jsonl) |
| 回退 | 构造少票、B1 缺票、双边头缺失与 PL 求解失败样例，分别核对 BM25 和整榜 T1 回退。另验证同分排序及 R、pC、pE 均为 1 时 F 为 1。 | [逐人逐方法回退记录](base/fallbacks.csv) · [验收实现](code/phase5_verify.py) |

## 温度校准

| 预测头 | 完整 C 拟合岗位数 | 正例数 | 温度 | 拟合状态 |
|---|---:|---:|---:|---|
| skill.pJ | 1280 | 294 | 0.7300 | interior |
| trajectory.pJ | 1280 | 294 | 0.8670 | interior |
| preference.pJ | 1280 | 294 | 0.9158 | interior |
| employer.pJ | 1280 | 294 | 1.1759 | interior |
| preference.pC | 1280 | 657 | 11.0639 | interior |
| employer.pE | 657 | 294 | 20.0000 | upper_T |

Employer 的认可概率仅在已投递样本上校准。本次该头温度达到冻结上界 20，双边排序使用此值。

## 补充实验验收

第 1–3 节的 2,330 个原始字节均与 Git 基线相符。补充分析使用已封存的 C 样本外点估计。独立核对了 2,000 次抽样的用户索引、24,000 个温度最优性条件、36,000 个相关统计量及全部区间端点。原始点估计与正式参数一致。五项正式产物在补充分析前后和最终验收时的 SHA-256 一致。

[六对角色的结果、区间和权重解读](analysis/correlated_error_report.md) · [三联热图](analysis/correlation_heatmap.png) · [分析前哈希](evidence/formal_before.sha256) · [分析后哈希](evidence/formal_after.sha256)

## 第 6 期交接

完整排名按测试用户与方法定位。每行保留名单哈希、完整有序岗位 ID 和前五。F 的逐岗明细保留可信名次、双边概率、效用及最终分数。本期交接以 [运行校验清单](phase5.sha256) 为准。[工作记录](work_log.md)
