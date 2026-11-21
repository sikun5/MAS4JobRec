# 第 1 期｜服务器、环境与原版 8B 模型准备

[主入口与服务器实测配置](./experimental_setting.md) · 本期可与[第 2 期数据准备](./phase2_data_prep.md)并行，两期共同交接给[第 3 期 pilot](./phase3_pilot_budget.md)。

## 1. 实验目标

在项目实验服务器上建立一个能重复调用的 Qwen3-8B 环境，确定代码、权重、数据与结果的位置。主线以原版 8B 能离线加载为准备目标。为第 3 期准备 32B-AWQ 快照及推理依赖，登记版本、磁盘需求和本地可用性。

## 2. 实验内容

1. 使用 `ssh hkust-gz-hpc3-temp2` 登录，进入 `/data/user/swu073/iota6910l/project`，核对主入口约定的五个一级目录。在 `runtime/runs/<run_id>/` 建立本次运行目录 `RUN_ROOT`。用小文件验证实际读写，记录可用空间和个人配额的可查询情况。同步实验所需的代码、配置、数据或模型文件时按清单选择，本地认证文件留在客户端。
2. 优先在 `$RUNTIME_ROOT/envs/` 使用独立 Conda 前缀，默认选择 Python 3.10。准备能加载 Qwen3 的 Transformers、PyTorch、tokenizer 及选用的批推理后端。远端若无法获取依赖，从可联网机器准备匹配 Linux x86_64、Python ABI 和所选 CUDA runtime 的包或环境归档。登记实际环境路径、包版本、后端和驱动。
3. 优先查可复用的 `Qwen/Qwen3-8B` 快照。需要传输时，在可联网机器获取固定 revision 的配置、tokenizer、聊天模板、权重索引及全部分片，生成 SHA-256 清单后传入 `$MODEL_ROOT/`。核对索引中的每个分片，验证配置和 tokenizer 可离线打开。通用评审与四角色使用同一快照。
4. 用项目解释器实测 PyTorch 版本、CUDA runtime、可见卡数、设备名称和显存。将模型首次完整加载及窗口生成并入第 3 期 pilot。登记第 1 期实际占卡时间。准备长任务入口，使每次启动保存 PID、命令、配置版本、阶段日志以及进程退出码，恢复时从已完成记录续跑。

**交接产物**为 `env/manifest.json`（路径、软件版本、设备和模型 revision）、`env/model_files.sha256`、环境重建清单及 `gpu_hours.csv`。后续用登记的解释器绝对路径启动。

例如，若环境建在约定的 `runtime/envs/v2`，可在远端用下列命令复核框架与设备，并把输出写入运行清单。

```sh
/data/user/swu073/iota6910l/project/runtime/envs/v2/bin/python - <<'PY'
import torch

print(torch.__version__, torch.version.cuda, torch.cuda.device_count())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0), torch.cuda.get_device_properties(0).total_memory)
PY
```

## 3. 验收且可推进标准

2026-11-21 已在 hkust-gz-hpc3 完成逐项验收。[本次验收记录](../../runtime/runs/v2-phase1-20261121/phase1_acceptance.md)

- [x] 代码目录、数据目录和本次运行目录实测可用，项目环境能导入推理依赖，`torch.cuda` 在该环境中识别目标 H100。
- [x] 8B 的模型标识、revision、tokenizer、分片数及校验结果可复核。配置和 tokenizer 的离线加载成功。
- [x] 运行清单能定位同一环境与模型，账本能区分准备时间和 pilot 时间。原验收时 32B 仅登记为 Optional。新增的 32B 推理、8B 全参数与 LoRA 训练能力在第 3 期独立核验。

满足以上条件且第 2 期的盲评输入已就绪，进入第 3 期。依赖缺失或权重校验失败时，先补齐对应文件，其他 CPU 数据准备可继续。
