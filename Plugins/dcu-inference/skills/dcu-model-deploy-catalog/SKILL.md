---
name: dcu-model-deploy-catalog
description: "DCU 模型部署目录：100 个已验证模型（LLM/VLM/Embedding/CV/图像视频生成/时序/科学计算）的部署配方速查——按模型查镜像/框架/启动命令/共性坑，命令摘自各模型《海光DCU模型部署手册》原文。触发场景：要部署某个具体模型、先看别人用哪个镜像和启动参数、查跨模型共性坑。不适用：部署前可行性与选型评估走 dcu-model-portability；部署后逐项验收走 dcu-model-deploy-verify；vLLM 专项部署调优走 dcu-vllm-deploy。"
version: 1.0.2
author: DCU Engineer
license: MIT
---

# DCU 模型部署目录（dcu-model-deploy-catalog）

> ## 🔗 官方 cookbook 上游引用（P0 更新源）
> 本 skill 是**内部快照**（100 篇《海光 DCU 模型部署手册》，BW1100 为主全类别）。
> **HYGON-AI/inference-cookbook-das** 是官方**持续更新**的模型×框架×卡型（K100_AI/BW1000/BW1100）**验证矩阵**（85 行、59 行已验证，含最新 Qwen3.8/GLM-5.2/DeepSeek-V4/Kimi-K3/MiniMax-M3 等），
> 两者互补：本 skill 有具体命令/坑，官方有**三卡型交叉验证状态** + 新模型。**查新模型/最新 SOTA 时先用官方矩阵**——见 `references/hygon-cookbook-upstream.md`。
> 更新机制（每次用前刷最新）：`python3 <agent-home>/skills/dcu/scripts/hygon-upstream-sync.py --check`（快检）/ `--full`（刷新矩阵）。
> 量化工具补充：官方 `one-click-quant`（BF16→W8A8 FP8/INT8、Block FP8→BF16、DeepseekV3/Qwen3.5 MoE 定制、`model_free_ptq`、slimquant w4a8、MoE-Quant 单机大权重 GPTQ）。

本 skill 是**按模型查部署配方**的速查目录，收录 100 个已在海光 DCU（BW1100 已验证）上跑通的
模型的部署手册要点：**用哪个镜像、哪个框架、怎么起容器、怎么装依赖、怎么启动/推理、有哪些坑**。
命令均摘自各模型《海光 DCU 模型部署手册》原文，照抄即跑（路径/卡号/容器名按实际调整）。

> **定位**：本 skill 回答「**某个具体模型**在 DCU 上怎么部署」。
> 若任务是「**评估一个全新模型**能否即插即用 + 从零产出部署手册」→ 转 `dcu-model-portability`（方法学）。
> 若任务是「**部署前真机实测验证**纪律」→ 转 `dcu-model-deploy-verify`。
> 若任务是「**vLLM 服务调优/已知坑**」→ 转 `dcu-vllm-deploy`。
> 本 skill 是「查现成配方」，上述是「做适配/调优」，互补不重叠。

## 一、按类别查（路由表）

| 模型类别 | 数量 | 配方文件 | 典型模型 |
|---|---|---|---|
| LLM（语言模型） | 29 | `references/llm.md` | Qwen3/DeepSeek/GLM/Llama/Baichuan/Kimi/ChatGLM/CodeLLaMA/telechat |
| VLM（视觉语言） | 14 | `references/vlm.md` | Qwen-VL/GLM-V/InternVL/LLaVA/ERNIE-VL/PaddleOCR-VL |
| Embedding / Reranker | 11 | `references/embedding.md` | bge/M3E/BGE-M3/Conan/Qwen3-Embedding/BGE-Reranker |
| CV（检测/分割/分类） | 21 | `references/cv.md` | yolo/ResNet/Swin/Mobilenet/convnext/DINO/DETR/sam3/CycleGAN/RT-DETRv2/BiRefNet/DA3/segformer |
| 图像生成 | 8 | `references/image-gen.md` | Stable Diffusion/FLUX.1/Kolors/GLM-Image/HunyuanImage/Qwen-Image |
| 视频生成 | 3 | `references/video.md` | Wan2.1/Wan2.2/HunyuanVideo |
| 音频/多模态 | 2 | `references/audio.md` | Qwen2-Audio/Qwen2.5-Omni |
| 时序预测 | 5 | `references/timeseries.md` | Timer/Timer-XL/Chronos/UNITS/Time-MoE |
| 科学计算 | 7 | `references/science.md` | DeePMD/FNO/PINN/ESM/alphafold/Openfold/powerflownet |

**机器可读总索引**（模型 → 镜像 tag → 框架 → 原文 内部文档 链接）：`references/index.md`。
查「某模型用哪个镜像/框架」先翻 `index.md`，要完整命令再进对应类别文件。

## 二、共性部署范式（100 篇高度一致，先看这里）

几乎所有模型手册都遵循同一套容器/环境范式，新模型部署可直接套用：

1. **容器启动（硬性，雷打不动）**：
   - 必挂 `-v /opt/hyhal:/opt/hyhal:ro`（99/100 篇），否则 DCU 不可见（DEVCOUNT=0）。
   - 必带 `--device=/dev/kfd --device=/dev/dri`（部分加 `--device=/dev/mkfd`）。
   - 大共享内存 `--shm-size`（100/100 篇，LLM 常 16G~256G，CV/科学计算 32~64G）。
   - 常用 `--privileged --group-add video --cap-add=SYS_PTRACE --security-opt seccomp=unconfined --network=host`。
   - 指定卡用 `HIP_VISIBLE_DEVICES=0`（不靠 `-g N`）。
2. **镜像**：绝大多数走 `harbor.sourcefind.cn:5443/dcu/admin/base/...`（DTK 26.04 为主，95/100 篇）；
   少数科学计算/OCR 走 `image.sourcefind.cn:5000/...` 的 custom 镜像。镜像 tag 见 `index.md`。
3. **权重下载**：72/100 篇走**魔搭 modelscope**（`https://www.modelscope.cn/models/...`）；
   走 HuggingFace 的必须 `export HF_ENDPOINT=https://hf-mirror.com`（国内镜像，9 篇显式）。
4. **依赖安装**：29/100 篇显式设 pip 镜像源（清华 tuna / 阿里 aliyun）；
   装 transformers 系前常 `pip uninstall -y cupy-cuda12x`（6 篇，避免 CUDA 版 cupy 与 numpy 冲突）；
   镜像预装 DCU 版 torch 时勿 `pip install torch` 覆盖，numpy 锁 `<2`（防 kornia 等拉到 2.x）。
5. **框架分布**：vLLM 32 · PyTorch/transformers 30 · torchrun 9 · XInference 6 · SGLang 4 · TensorRT 1 · MIGraphX 1。
   - LLM 服务化多用 `vllm serve` / `sglang serve`；embedding/reranker 多用 `xinference launch`；
     大模型原生推理用 `torchrun generate.py`；CV/科学计算多为纯 `python` 脚本或 `bash run.sh`。
6. **Day0 适配流程（新模型发布当天跟进）**：前沿模型（如 Qwen3.8-Flash-Next / GLM-5.3-Flash 系列）发布后，
   在目标平台（BW1000 等）用当前主流框架版本（vLLM / SGLang 各自最新稳定版）做 Day0 适配——
   跑通部署 + 基础精度/性能验证，**优先在 MAAS 平台验证**，再同步到各行业项目。
   新模型适配先查本目录是否已有同类模型手册可套用（§二范式），没有再按范式新建。

## 三、共性坑 / 排障（跨文档提炼）

| 现象 | 根因 | 处理 |
|---|---|---|
| DCU 不可见 / DEVCOUNT=0 | 未挂 `/opt/hyhal` | `docker run` 加 `-v /opt/hyhal:/opt/hyhal:ro` 重启 |
| `Failed to initialize NumPy: _ARRAY_API not found` | 残留 CUDA 版 `cupy-cuda12x` 与 numpy 2.x 冲突 | `pip uninstall -y cupy-cuda12x`（必要时 `cupy`），锁 numpy<2 |
| HF 权重下载 401/超时 | 内网不可达 / Xet 分块对镜像站 401 | `export HF_ENDPOINT=https://hf-mirror.com`；必要时 `HF_HUB_DISABLE_XET=1` |
| 权重走 modelscope 更稳 | 国内网络 | 优先魔搭 `modelscope.cn/models/...` 下载 |
| pip 装包慢/失败 | 默认源不可达 | `pip config set global.index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple`（或 aliyun） |
| 显存 OOM（视频/大模型） | batch/序列过长 | 减小 batch；`--offload_model True --t5_cpu`（Wan 系）；多卡拆分；`hy-smi -a` 监控 |
| 多卡 NCCL 异常 | 镜像预置 NCCL 变量 | 进容器先 `unset NCCL_ALGO` 等再测 |
| 性能掉 15~20% 无报错 | 启动命令未走登录 shell，DTK 加速库未加载 | `bash -lc 'source /opt/dtk/env.sh && <cmd>'`（详见 `dcu-vllm-deploy` §4.3） |
| 科学计算需关加速 | BladeDISC 等加速默认开 | 按手册注释 `disc.enable()` 等（见 `references/science.md`） |

> 完整 vLLM/SGLang 调优坑（MTP tool call 漂移、fp8 KV、mamba 限流等）见 `dcu-vllm-deploy`；
> 通用容器/镜像/多机见 `dcu-docker`；包版本绑定矩阵见 `dcu-env-packages`。

## 四、使用纪律

- **命令照抄即跑，但路径/卡号/容器名/镜像 tag 按实际环境调整**；本目录是「配方索引」，
  不是「保证可复现的脚本」——正式交付部署手册前须按 `dcu-model-deploy-verify` 真机实测。
- **不存性能数字/基线/逐行日志**（会变）；只存「怎么做、注意什么」。
- 某模型手册原文（含完整 stdout/截图）以 `index.md` 里的 内部文档 链接为准，本目录只摘命令要点。
- 新增模型：按类别追加到对应 `references/<cat>.md` + `index.md` 一行，保持「镜像/框架/原文」三要素齐全。

## 五、来源与溯源

- 根文档：《<某能源集团客户>2026年网络设备项目测试项汇总》
  `<internal-doc-link>`（file_id `<internal-file-id>`），
  「模型适配支持度」sheet（worksheet_id 6）「部署文档交付链接（标黄代表BW1100已验证）」列。
- 各模型 内部文档 链接见 `references/index.md` 每行「原文」列（96 篇来自上表 sheet，另 4 篇 RT-DETRv2/DA3/BiRefNet/segformer 为本项目自验证新增）。
- 抓取方式：`内部文档-cli`（见 `内部文档-doc-extraction` skill）；抓取原文仅作临时中间产物、处理完后清理，不入仓库。
