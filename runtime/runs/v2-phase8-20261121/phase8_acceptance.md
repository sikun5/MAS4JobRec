# Phase 8 验收｜学生同榜重评

**验收通过，Optional 完成。** 唯一扩展比较为 F-SFT−F。联合 nDCG@5 配对人数 228，差值 +0.03051 [+0.00105, +0.05957]，区间按用户配对重采样 2,000 次得到。

| 协议要求 | 实测与证据 |
|---|---|
| 共享适配器训练与重载 | 505 条示范各用一次，505 次 AdamW 更新，只监督回答及 EOS。原版 BF16 Qwen3-8B，LoRA rank 16、alpha 32。逐条复核训练顺序、token 与梯度；独立评审进程重载同一份适配器。[训练记录](distill/training.jsonl) · [重载回执](evidence/reload.json) |
| 同池四角色评审 | 原 C 128 人、T 256 人，计划 12,288 窗，实际调用 12,288 次。逐窗重放，检查名单、提示、排列、概率、重试与跳过。[重评复核](evidence/review_verified.json) |
| 学生可信聚合与双边排序 | C 内按原折拟合温度、Brier、六对相关与角色权重，独立复核温度、相关及 PL 解。学生 256 个完整榜或回退榜先封存再评分。[学生参数](student/calibration.json) · [冻结排名](student/rankings.jsonl) · [封存记录](evidence/blind_seal.json) |
| 评测与记账 | 每项指标逐人 ID 与原版相同。独立复算五项指标及配对区间。教师、训练、重评分别为 1.05584、0.09041、5.29288 GPU 小时，新增 6.43913 小时。[五项结果](eval/student/summary.json) · [逐人差值](eval/student/paired_deltas.csv) · [GPU 账本](gpu_hours.csv) |

[结果矩阵快照](acceptance_result_matrix.md) · [工作记录](work_log.md)
