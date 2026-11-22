# v2 实验代码入口

主线从原版 8B 角色评审开始，经校准、相关错误修正、PL 聚合与双边效用生成排名，再开启测试标签评分。Optional 从训练用户的 32B 教师示范开始，经 8B LoRA 训练、学生盲排，最终只比较同榜的 `F-SFT−F`。协议见[冻结实验设置](../global_guide/v2_frozen/experimental_setting.md)。

| 阶段 | 可执行入口（依执行顺序） | 主要交接产物 |
|---|---|---|
| 4 · 原版选票 | [角色评审](phase4_agent_reviews.py)、[独立重放](phase4_verify_reviews.py) | `base/reviews.jsonl`、权重检查、选票与封存清单 |
| 5 · 可信排序 | [正式校准和排序](phase5_formal.py)、[跨角色相关错误分析](phase5_error_analysis.py)、[独立复算](phase5_verify.py) | `base/calibration.json`、`base/rankings.jsonl`、`analysis/` |
| 6 · 主实验评分 | [预检与评分](phase6_score.py)、[统计复算](phase6_verify_statistics.py)、[成本](phase6_cost.py)、[结果矩阵](phase6_report.py)、[验收发布](phase6_accept.py) | `eval/main/`、已发布矩阵和阶段快照 |
| 7 · 教师示范 | [选窗](phase7_prepare_windows.py)、[教师生成](phase7_generate_demos.py)、[筛选与验收交接](phase7_handoff.py)、[封存](phase7_seal.py) | `distill/teacher_outputs.jsonl`、`distill/demos.jsonl`、分角色统计 |
| 8 · 同榜重评 | [学生调度](phase8_pipeline.py)依次启动[训练](phase8_train_student.py)、[评审](phase8_student_reviews.py)、[重放](phase8_verify_reviews.py)、[盲排](phase8_rank_blind.py)、[评分](phase8_score_student.py)、[更新 Optional](phase8_accept.py)。最后[封存](phase8_seal.py)。 | `distill/adapter/`、`student/rankings.jsonl`、`eval/student/`、Optional 矩阵 |

[共享选票实现](review_windows.py)负责解析、两遍滑窗、缓存键和逐岗概率。原版在 T 上增加综合评审 B1，学生只重评四个专业角色。两者分别使用本期的作业目录和数据库。学生盲排直接复用 [Phase 5 校准](phase5_calibration.py)和[排序](phase5_ranking.py)，同榜评分直接复用 Phase 6 指标。Phase 7 交接会调用[筛选](phase7_filter_demos.py)和[逐条复核](phase7_verify_demos.py)。同时提交异步 handoff 作业时可使用[等待交接后封存](phase7_wait_and_seal.py)。

## 复现工作区和输入

在**独立的复现工作区**执行下列命令。工作区须包含当前版本的整个 `code/`、冻结协议、已验收的 Phase 1–3 输入和模型。若从后续阶段开始，准备此前阶段的完整封存交付。只给本次要执行的阶段使用新的空运行目录。Git 仅归档精选证据，完整盲评文本、训练示范、选票数据库、逐用户结果、模型和适配器需从实验存储取得。历史工作区中的原始脚本用于按既有清单核验。

源码以自身位置定位工作区，运行产物写入 `runtime/runs/v2-phaseN-20261121/`。每期首次启动将维护源码及其跨阶段依赖的 SHA-256 写入 `evidence/source_hashes.json`，后续阶段和恢复作业复核同一清单。已有阶段封存文件时，计算入口会终止。Phase 6 和 8 会发布到复现工作区的 `global_guide/v2_frozen/acceptance_result_matrix.md`。

以下命令使用 **Bash、冻结 Linux 环境的 Python 3.10、GNU sha256sum**。Phase 4、7、8 的推理或训练步骤使用单张 H100。先在复现工作区根目录定义路径和函数，之后可从任何工作目录调用。命令返回非零时先检查本期的 `jobs/<阶段>/stage.log`。

```bash
set -euo pipefail
PROJECT=$(pwd -P)
CODE="$PROJECT/code"
RUNS="$PROJECT/runtime/runs"
PY="$PROJECT/runtime/envs/v2/bin/python"
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=2
export PYTHONPATH="$RUNS/v2-phase3-20261121/pilot/vendor/cpu${PYTHONPATH:+:$PYTHONPATH}"

stage() {
    local number="$1" name="$2" device="$3" script="$4"
    shift 4
    CUDA_VISIBLE_DEVICES="$device" "$PY" "$CODE/launch_stage.py" --wait \
        "$RUNS/v2-phase${number}-20261121" "$name" maintained-v2 -- \
        "$PY" -u "$CODE/$script" "$@"
}
handoff() { CUDA_VISIBLE_DEVICES="" "$PY" "$CODE/experiment_handoff.py" "$@"; }
```

### 主线

Phase 4 要求 Phase 1 的模型文件清单，以及 Phase 2 的 splits、候选池和盲评输入。Phase 5 读取 C 标签；Phase 6 的预检执行后才由评分步骤打开 T 标签。以下命令包含作业收据、准备文件和上下游校验。

```bash
handoff prepare 4
stage 4 reviews 0 phase4_agent_reviews.py
stage 4 acceptance "" phase4_verify_reviews.py

handoff prepare 5
stage 5 formal "" phase5_formal.py
stage 5 supplement "" phase5_error_analysis.py
handoff verify-upstream 5
stage 5 verify "" phase5_verify.py
handoff close 5
```

在复现工作区中，Phase 6 的矩阵输入应为 Phase 5 完成时的模板。历史模板可从提交 `7cb9652` 取得。以下重定向会写入复现工作区的矩阵。

```bash
git -C "$PROJECT" show 7cb9652:global_guide/v2_frozen/acceptance_result_matrix.md \
    > "$PROJECT/global_guide/v2_frozen/acceptance_result_matrix.md"
handoff prepare 6
stage 6 preflight "" phase6_score.py preflight
stage 6 score "" phase6_score.py score
stage 6 statistics "" phase6_verify_statistics.py
stage 6 cost "" phase6_cost.py
stage 6 report "" phase6_report.py
stage 6 accept "" phase6_accept.py
handoff close 6
```

### Optional 教师蒸馏

Phase 7 另需 Phase 2 的 `labels/train.jsonl`、已验证的 32B 教师快照、Phase 3 的 AWQ kernel 和验收记录。Phase 8 使用冻结的 505 条示范、原版 F 的逐人结果和**已通过 Phase 6 的矩阵**。若直接复用历史 Phase 6 交付，发布版矩阵可从提交 `0e00198` 取得；运行目录内的矩阵快照使用另一套相对链接。

```bash
handoff prepare 7
stage 7 prepare "" phase7_prepare_windows.py
stage 7 teacher 0 phase7_generate_demos.py
stage 7 handoff "" phase7_handoff.py
"$PY" "$CODE/phase7_seal.py"

handoff prepare 8
stage 8 pipeline "" phase8_pipeline.py
"$PY" "$CODE/phase8_seal.py"
```

学生调度按顺序等待每个子作业成功。重启时，启动器按阶段名、版本和命令核对已完成作业；失败阶段由原训练检查点或选票数据库恢复。训练和评审进程隔离测试标签，学生排名及其哈希先封存，再做配对评分。
