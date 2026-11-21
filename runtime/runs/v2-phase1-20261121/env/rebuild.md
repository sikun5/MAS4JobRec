# 第 1 期环境重建与模型验收

项目根目录为 `/data/user/swu073/iota6910l/project`。本次源环境为账号已有的 `/hpc2hdd/home/swu073/envs/dsaa`，只读复用。本项目的推理后端为 Transformers，支持在同一批次中处理多个提示。以下命令从项目根目录执行。

1. 使用 `/opt/conda/bin/conda create --offline --yes --prefix "$PWD/runtime/envs/v2" --clone /hpc2hdd/home/swu073/envs/dsaa` 复制解释器、Conda 包及源环境中由 pip 安装的依赖。源 Conda 包版本、官方地址与 SHA-256 见 `conda-packages.json`。离线包归档存放在 `runtime/envs/phase1-conda-packages.tar`。最终环境由验收时导出的 `conda-explicit.txt` 和 `pip-freeze.txt` 标识。
2. 将 `source.json` 中固定 revision 的全部文件放入 `model/Qwen3-8B/<revision>/`。大文件使用上游 LFS SHA-256，小文件先用上游 Git blob ID 核对，再登记 SHA-256。
3. 长任务入口为 `python3 code/launch_stage.py RUN_DIR STAGE CONFIG_REVISION -- COMMAND ARG...`。阶段目录下有 `status.json` 和 `stage.log`。同配置重入时检查活动锁或已完成状态。失败阶段可重试，配置改变须使用新阶段名或新运行 ID。
4. 使用 `runtime/envs/v2/bin/python code/verify_phase1.py` 进行本次运行的验收。验收仅打开 safetensors 头部、离线配置、tokenizer 和聊天模板，并执行小张量 CUDA 测试。8B 完整模型加载属于第 3 期。


## 本次已验证的离线恢复

源环境有 30 个 Conda 基础包。克隆先复制了 pip 安装的推理包，在获取基础包时因离线模式退出。本次从源环境元数据列出的官方地址下载这 30 个精确版本，并按 SHA-256 校验。恢复脚本 `finish_preparation.py` 把它们解包到项目私有的 `runtime/envs/phase1-archives/`，生成本地 `@EXPLICIT` 清单，再用项目内的包缓存补齐已有前缀。复建同一前缀时可沿用此路径，空前缀需要先完成源环境中 pip 依赖的克隆。

验收作业位于 `jobs/phase1-acceptance/`。项目解释器为 Python 3.10.13，已实测可导入 PyTorch 2.14.0+cu130、Transformers 4.57.6 及 Qwen3ForCausalLM，并识别 NVIDIA H100 80GB HBM3。长任务入口的重复启动及完成复用记录位于 `jobs/launcher-check/`。32B 教师按 Optional 分支规划，未下载。

本机 Conda 4.9 的 noarch 路径处理未将 pip 暴露给 Python 3.10。将已校验的 pip-26.2.1 私有缓存中 site-packages/pip 和对应 dist-info 复制到项目 lib/python3.10/site-packages 后，以项目 Python -m pip 验证。
