# Phase 7 验收｜32B 教师逐角色示范

Phase 7 通过，可进入 Phase 8。

预留 256 名训练用户。214 人各抽两个窗口；42 人无合格四岗组合。428 个窗口均为一正三负。四角色共 1,712 次逻辑判断，实际尝试 1712 次，共保留 505 条示范，覆盖 124 名不同求职者。

| 角色 | 逻辑判断 | 重试 | 格式有效 | 最终无效 | 首位错误 | 保留 | 保留率 | 保留示范覆盖人数 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| skill | 428 | 0 | 428 | 0 | 304 | 124 | 28.97% | 101 |
| trajectory | 428 | 0 | 428 | 0 | 297 | 131 | 30.61% | 105 |
| preference | 428 | 0 | 428 | 0 | 301 | 127 | 29.67% | 97 |
| employer | 428 | 0 | 428 | 0 | 305 | 123 | 28.74% | 103 |

GPU 总量 1.05584 单卡小时，含加载、推理及退出阶段。筛选和复核使用 CPU。按原版 8B 学生 tokenizer 将每条真实角色提示与目标回答编码并附加 EOS，总长度范围为 [1037, 2375] token。Phase 3 已验证可重新加载 LoRA 适配器。

独立验收按训练划分及简历组复核互斥关系，固定种子重放组合和乱序。按正式提示模板及字段白名单重建 1,712 条消息，重解析每次原始答复并与训练标签对齐，逐条核对筛选理由、保留集合和四角色汇总。训练标签用于选窗及生成后的筛选。Phase 6 交付文件复核通过。

[配置与版本](distill/teacher_manifest.json) · [训练窗口](distill/train_windows.jsonl) · [盲评请求](distill/blind_requests.jsonl) · [原始输出](distill/teacher_outputs.jsonl) · [筛选明细](distill/filter_decisions.jsonl) · [训练示范](distill/demos.jsonl) · [角色统计](distill/role_results.csv) · [独立复核](evidence/verified.json) · [成本](gpu_hours.csv) · [工作记录](work_log.md)
