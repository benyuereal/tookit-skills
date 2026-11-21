---
name: dcu-vllm-deploy
description: DCU 上 vLLM 部署与调优经验速查：容器启动硬约束、关键启动参数（TP/显存/graph）、性能调优闭环入口、已知问题与部署选型。触发场景：在 DCU 上起 vLLM 服务、调吞吐或首 token 延迟、启动报错或性能不达预期。不适用：输出错乱/数值偏差等精度问题走 dcu-vllm-precision；非 vLLM 框架的部署配方走 dcu-model-deploy-catalog。
version: 1.0.1
author: DCU Engineer
license: MIT
kind: tool
---

# DCU 上 vLLM 部署调优经验速查（dcu-vllm-deploy）

本 skill 是**经验速查**，沉淀 DCU 上部署 vLLM 的约束与已知坑。**不存性能数字/基线**（会变），只存"怎么做、注意什么"。具体 patch 与默认行为以海光 fork 实时核验。

## 0. 操作范围门（scope gate，执行前必读）

本 skill 涉及的一切执行动作（起容器、占卡、占端口、重启服务）**默认只允许作用于"本会话自己启动"的资源**：

- **容器**：只动自己 `docker run` 起的；他人容器只读（`ps`/`logs`/`stats` 可，`rm -f`/`kill`/`restart` 不可）。
- **端口**：起服务前 `ss -lntp` 查占用，只选空闲端口，不顶掉他人进程。
- **卡**：`hy-smi` 选空卡起服务；卡被占时做有界轮询或先问用户，**绝不杀他人 vLLM/SGLang 服务腾显存**。
- 例外（须动他人资源才能完成任务）：逐次向用户报确切命令获批准后执行；长期成立的例外登记到下方白名单，其余情形仍按默认规则。

**例外白名单（经用户批准后逐条登记）**：暂无

## 1. 容器启动约束（硬性，雷打不动）
- 必挂运行时：`docker run ... -v /opt/hyhal:/opt/hyhal:ro`，否则 DCU 不可见（DEVCOUNT=0）。
- 指定部分卡：`HIP_VISIBLE_DEVICES=0,1,2,3`（编号从 0 起），不靠 `-g N` 挑卡。
- 进容器跑测试前先查 `NCCL_NCHANNELS_PER_PEER` 是否已设（镜像常预置=2），有则 `unset` 再测。
- gfx / 卡型号用**物理机** `rocminfo` 查（带 `ROCM_PATH`/`LD_LIBRARY_PATH`），容器内不可靠。
- 编译/适配在**指定镜像起的专用容器**内做；多卡多机参考 `dcu-docker`。

## 2. 关键启动参数（调优入口）
| 参数 | 作用 | 备注 |
|---|---|---|
| ~~`--device rocm`~~ | ~~指定 ROCm/DCU 后端~~ | ⚠ **已废弃**：vLLM 0.15.1+/0.18.1 实测无此参数（platform plugin 自动激活 hcu，不再需要显式传）。海光 fork 走 `vllm_hcu:hcu_platform_plugin`。无需再设 |
| `--enforce-eager` | 关 CUDA/ROCm Graph | 排查期定位 kernel 问题 |
| `--tensor-parallel-size` | TP 并行度 | 按卡数与模型定 |
| `--dtype` | 精度 BF16/FP16/FP8 | FP8 需海光支持 [待核验] |
| `--kv-cache-dtype` | KV 量化 | 省显存常用 fp8；`fp8_e5m2` 需 patch 放行（见 §4.4） |
| `--enable-prefix-caching` | 前缀缓存 | 多轮对话提速 |
| `VLLM_TARGET_DEVICE=rocm` | 构建/运行目标 | 编译期环境变量 |

> 完整参数见 vLLM 官方 `--help`；DCU 专属默认项以海光 fork 为准 [待海光fork核验]。

## 3. 性能调优闭环入口
1. 起服务 → 用 `vllm/benchmarks/benchmark_serving.py` 压测（TTFT/TPOT/吞吐）。
2. 发现慢 → `dcu-perf-profiling` 定位瓶颈 kernel。
3. 要改 kernel → `dcu-vllm-source` 定位源码 + `dcu-op-optimization` 走闭环。

## 4. 已知问题 / 排查入口
- 设备不可见 → 查 `/opt/hyhal` 挂载、`hy-smi` 是否正常（见 `dcu-hy-smi`）。
- 编译失败 → 查 DTK 版本与 `ROCM_HOME` 路径（见 `dcu-dtk`）。
- 运行报错关键词 → `dcu-ontrack-experience`（含 vLLM/编译/驱动报错反查）。
- 多卡通信差 → `dcu-rccl-test` 先验 RCCL 带宽。

### 4.1 MTP 投机解码导致 tool call 参数结尾漂移（高频踩坑）
开 MTP（`num_speculative_tokens>1`）后，客户端（OpenAI SDK / Claude Code）约 10% 的
tool call 失败报 `Invalid tool parameters` / `json.loads` 失败。根因：MTP 在生成
tool call 参数 JSON 结尾处偶发多吐 `{}` 顶掉收尾的 `"` `}`，JSON 未闭合。**关闭 MTP 即正常**，
但加速不可放弃。漂移形态规律（永远结尾 `"}` 被 `{}` 取代及衍生），可被确定性修复。

- 完整根因 / 畸形形态 / 旁路修复代理方案：`references/mtp_toolcall_drift.md`
- 即用 OpenAI 版修复代理（FastAPI 旁路，已端到端验证）：`templates/openai_repair_proxy.py`
- 思路：客户端 <-> vLLM 间插转发代理，拦截 SSE 流把畸形 arguments 原地修复。零改动
  vLLM / 客户端，MTP 照开。治标方案；根治应改 vLLM 侧 MTP 解码或 tool call 序列化出口兜底。

### 4.2 多模态 SGLang 并发被静默限流（max-mamba-cache-size 缺省陷阱）
使用 SGLang 跑**含 mamba 结构的多模态模型**（如 Qwen3.5/3.6 系列）压测时，即使显式设了
`--max-running-requests 32` 与 bench `--max-concurrency 32`，实际只能跑 ~20 并发、KV Cache 未跑满。
根因：`max-mamba-cache-size` 默认自动分配过小，在压测阶段悄悄限制并发（非算力/带宽问题）。
解法：显式设 `max-mamba-cache-size`（或确认自动分配足够）；完整实战启动脚本（Aiter/Marlin/fp8/mamba/EAGLE 投机一堆 env）见 `dcu-perf-cases` 案例 7。另 SGLang 源码适配点见 `dcu-sglang-source`。

### 4.3 启动命令未走登录 shell 导致 rocblas/hipblaslt 未加载（性能掉 15–20%）
`docker run ... <image> vllm serve ...` 直接把命令当 PID 1 执行，若镜像把 DTK 加速库
路径（rocblas/hipblaslt 的 LIBPATH 与 tuning）放在 `/etc/profile.d/` 或 `.bashrc`
而非镜像 ENV 里，则该非登录非交互 shell **跳过加载**，推理性能掉 15–20% 且无任何报错。
实测对比见 dcu-docker「测评执行方式」：非登录 shell 52→43 Req/s。

- 判别：容器内 `bash -lc 'echo $LD_LIBRARY_PATH' | grep dtk` 无 `/opt/dtk` 即中招；
  或同镜像同参数只改启动方式跑 benchmark，tok/s 差 15%+ 即实锤。
- 修法：用登录 shell 包裹启动命令，双保险显式 source：
  `docker run ... <image> bash -lc 'source /opt/dtk/env.sh && vllm serve /model_weight ...'`
- 同样适用于 `docker exec`：性能相关命令一律 `docker exec -d bash -ilc '...'`，不要
  `docker exec bash -c`。

### 4.4 fp8_e5m2 KV cache + flash 路由（ROCM_AITER_UNIFIED_ATTN 路径速查）
BW10 上把 KV cache 存成 fp8_e5m2 并经 aiter unified_attention 路由到 flash 的部署组合
（gemma-4-31B AWQ 实战，TPOT 2.59x，详见 `dcu-perf-cases` 案例 13）：

- **`--kv-cache-dtype fp8_e5m2` 需 patch 放行**：上游 vllm 对 compressed-tensors 模型
  无条件报错 "fp8_e5m2 kv-cache is not supported with fp8 checkpoints"，但 AWQ-4bit
  checkpoint 里并没有 fp8 KV scale（kv_cache_scheme=None），属误伤。需改 vllm 3 文件
  （attention.py 放行 + 读侧 view 用 e5m2 + 写侧 triton reshape_and_cache_flash 按
  字符串选 e5m2 dtype，C++ op 不支持 e5m2）。
- **e5m2 优于 e4m3**：reshape 性能基本相同，但精度 e5m2 有明显优势。
- **flash 路由环境变量**（aiter unified_attention.py 侧）：
  | 变量 | 作用 | 注意 |
  |---|---|---|
  | `ATTN_FLASH_PREFILL=1` | prefill 走 flash | **必须=1**：设 0 回退 aiter triton 2D 会在 fp8 KV prefill 崩（`'constexpr_type' object has no attribute 'is_ptr'`），无退路 |
  | `ATTN_FLASH_HEAD512=1` | draft head_dim=512 的 decode 走 flash | MTP draft 层 head_dim 与主模型不同时必开 |
- **配套参数**：`--attention-backend ROCM_AITER_UNIFIED_ATTN` + `--dtype float16`
  （对齐 aiter kernel fp16 输出）；fp8 KV 时 flash prefix decode 要求 Q/K 同 dtype，
  aiter 侧需 Q cast 成 KV dtype + unit descale。
- **已知回退**：该路径 TTFT 会涨（Q-cast/路由开销，实测明显上升），无 env 可绕，
  生产以 TPOT 为主指标时可接受。

### 4.5 hybrid 线性注意力模型（qwen3_5/qwen3_next 系）启动三坑

来源：VeriLoop-E2（Qwen3_5ForConditionalGeneration，64 层=48 GDN 线性+16 全注意力+MTP，55.7G bf16）在 vLLM 0.28.x DCU 镜像 TP4 部署实测（2026-10，BW100/DTK 26.04）。适用：所有 hybrid（mamba/GDN）模型 × DCU vLLM 0.28。

1. **retention 整除性校验启动即死（DCU 0.28 真 bug，最恶毒）**：`VLLM_PREFIX_CACHE_RETENTION_INTERVAL` 默认 2048，hybrid 模型 `scheduler_block_size=784`，校验方向是 `retention_interval % scheduler_block_size == 0`（2048%784=480≠0 → 判死），容器**无声退出**，报错埋日志深处。修复：设 `0`（0 满足校验且关闭 retention 调度）；或设为 block_size 整数倍。
2. **VLM 壳崩溃 → `--language-model-only`**：仓库带 `video_preprocessor_config.json` 时，vLLM 初始化 processor 撞 transformers 5.x `video_processing_auto` NoneType 缺陷崩死。纯文本推理必加。仅特定场景：仓库带 VLM 配置才触发。
3. **libcudart shim**：worker 报 libcudart 未加载时设 `VLLM_CUDART_SO_PATH=<镜像内dtk>/cuda/cuda-12/targets/x86_64-linux/lib/libcudart.so.12`（按镜像实际 dtk 目录核对）。按镜像而定：0.28.0 内部镜像必设，0.28.1rc1 公开镜像不设也通。

TP 合法性通式（hybrid）：全 attention heads 集合 % TP==0 且 KV heads % TP==0（或 TP%KV==0），GDN 额外要求 linear_num_key_heads / linear_num_value_heads % TP==0。VeriLoop-E2（16/48/24/4/5120）→ TP∈{1,2,4,8}。

### 4.6 serving 镜像跑 HF 基线缺 accelerate + safetensors 精确校验

精简 serving 镜像无 accelerate，HF `device_map` 报 `requires accelerate`。修复：`pip install --target /挂载卷/pylibs accelerate` + `PYTHONPATH` 注入，不污染镜像。配套：safetensors 分片校验必须逐片与 manifest **字节精确相等**——断点续传（`curl -C -`）可能写出**超长**残缺文件，"大小≥预期"判据检不出，加载期才报 `SafetensorError: incomplete metadata`；CDN 限速时串行单连接 + `--speed-limit 20000 --speed-time 60` 看门狗比并行快。

## 5. 部署方案与框架选型经验

> 沉淀"某模型/某卡型上的推荐部署配置与框架选择"，不存性能数字。具体镜像/版本以实时核验为准。

| 场景 | 推荐配置/手段 | 备注 |
|---|---|---|
| Qwen3.6-35B（SGLang） | 额外加 `--max-running-requests 64 --max-mamba-cache-size 320` | 含 mamba 结构，缺省 mamba cache 过小会静默限流并发（见 §4.2） |
| 短上下文 + 小并发 | TP8 方案更优 | 大上下文 + 大并发则 DP8 更优；按上下文长度与并发数选型 |
| 大模型多机（BW1000 类） | `tp8pp2 + mtp + fp8` 组合 | 短文本与超长文本均显著优于单参数方案 |
| 多实例负载均衡 | Nginx 挂 vLLM 多实例 | 整体吞吐/最大并发优于单实例双机 TP16，但难以随硬件线性提升 |
| Qwen3.6-27B（K100-AI） | vllm-0.21 版本 4 卡推理 | 该版本在 K100-AI 上可用 |
| 小参数量模型 / FP8 精度（BW1100） | SGLang 框架 | BW1100 上 SGLang 部分场景优于 vLLM，尤其 FP8 精度、小参数量模型 |
| 请求接口选择 | 注意 `/v1/completions` 与 `/v1/chat/completions` 性能差异显著 | 同模型两接口吞吐可差较多，压测/对比时固定同一接口 |
| SGLang 文生图（图像生成）推理 | 加 `--enable-torch-compile` 开启 torch compile | 单卡与多卡（tp=1）均有效，多卡可叠加 |
| 超节点稳定 decode 场景 | 调整 `--cuda-graph-bs` 覆盖 D 节点不同并发数 | 稳定 decode 场景下固化 CUDA Graph 推理效率，提高解码效率 |
| MoE 模型显存不足（专家层放不下） | 用 ktransformer 把专家层全部 offload 到 CPU（显存换吞吐） | Qwen3.5-35B 实战，吞吐可大幅提升 |
| ASR/TTS 模型高并发（Qwen3-ASR/TTS 类） | 单卡/多卡多实例负载均衡 + 开启多个 api-server 进程 | 瓶颈常在 CPU（非 DCU）；相同硬件资源下满足 SLO 支持更多并发 |

## 6. 性能症状速查 + 最小实现 POC（vendored reference）

> 遇到 vLLM 推理性能/精度问题，先按症状定位案例，再进原理。两份 reference 已随本 skill 落库（换环境 clone 即带，无外部路径依赖）：

- **症状速查**：`references/perf_symptom_lookup.md` —— 按「症状 / 场景 / 指标 / 模型硬件」定位到 `dcu-perf-cases` 案例（17-37）+ 高频优化手段速查表（APC / KV FP8 / W8A8 / MTP / aiter GEMM / PD 分离 / 锁频 等开关与效果）。
- **最小实现 POC**：`references/poc/` —— 6 个纯 Python 零依赖脚本（调度器 / PagedAttention / KV cache+APC / InputBatch 固定 buffer / 采样管线 / TP 并行），CPU 直接跑、看 trace，把 vLLM 机制"跑起来看见"。`python3 references/poc/<name>/mini_*.py`。
- **原理层**：vLLM 0.26.x 上游源码深潜（16 篇）在 `dcu-vllm-source/references/upstream-vllm-026/`（按需加载）。
- **实测案例**：`dcu-perf-cases` 案例 17-37（现象→根因→方案→量化，已脱敏）。

## 使用须知（必读）
- **经验仅供参考，不保证修复。** 动任何配置/代码前先核对：① 设备型号（gfx）② 驱动/DTK 版本 ③ 报错原文。
- **重启/重装驱动/刷 vbios/改 BIOS/拔电源** 等高危操作需谨慎：确认环境（产线/客户）与带外恢复通道，优先在测试机验证。
- 本 skill **不存性能数据/基线/逐行日志**；数值类结论用 `dcu-op-optimization` 模板在各项目留痕。
- 海光 fork 的具体行为与本文差异，以 sourcefind `codes/OpenDAS/vllm` 实时核验。
