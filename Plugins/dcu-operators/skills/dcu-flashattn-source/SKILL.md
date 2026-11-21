---
name: dcu-flashattn-source
description: flash-attn 海光 DCU 适配源码索引：三层适配结构 + FA3 不可移植结论 + 版本谱系
version: 1.0.1
author: DCU Engineer
license: MIT
---

# flash-attn 海光 DCU 适配源码索引（dcu-flashattn-source）

## When to Use

- 问「海光 flash-attn 怎么适配 DCU 的 / 改了哪些 / 怎么做的」→ 看 §1 三层适配结构。
- 问「flash-attn 开源最新版能否适配 DCU / 升级能否提速」→ 看 §3 核心结论（FA3 不可移植、已在最新 FA2）。
- 查「DCU 上 flash-attn 有哪些 build / 版本串怎么读 / 哪个最新」→ 看 §2 版本谱系。
- 要「验证/复现 flash-attn 适配点」→ 看 §4 验证方法。
- 与 `dcu-aiter-source`（AITER 算子库）、`dcu-vllm-source`（vLLM 框架）区分：本 skill 专讲 **flash-attn 这个独立 attention kernel 库**的 DCU 适配。

> 基于 2026-09-01 实测：DCU 源（pypi.sourcefind.cn）最新 release wheel `2.8.3+das185.dtk2604.torch2100.2608132155.g9febea` 解包，与上游 flash-attn 2.8.3 sdist 做 Python 层 diff + csrc kernel 清单对照 + hopper/FA3 架构守卫核验。
> 定位：回答「海光 flash-attn 怎么适配的 / 开源最新版能否适配 DCU / 能否带来性能提升」三类问题。与 `dcu-aiter-source`（AITER 算子库）、`dcu-vllm-source`（vLLM 框架）互补——flash-attn 是**独立的 attention kernel 库**，vLLM/SGLang 通过它或 AITER 走 attention 路径。

## §0 定位：flash-attn 在 DCU 栈里的位置

- flash-attn = **独立的 attention kernel 库**（CUDA/HIP kernel + Python 绑定），不是推理框架。
- 上游 = Dao-AILab/flash-attention（FA2 主线 + FA3 hopper 子目录）。
- 海光版 = **上游 2.8.3 + 三层 DCU 适配**（见 §1），以 wheel 交付（编译产物 `.so`，**csrc 源码不在包内**）。
- 与 AITER 的关系：vLLM ROCm 路径的 attention 可走 AITER（`rocm_aiter_unified_attn`）或 flash-attn；两者是**并行的 attention 后端**，不是包含关系。SGLang 侧 flash-attn 是 fa2 后端之一。

## §1 三层适配结构（实测 diff 结论）

### 层 1：Kernel 层（csrc → HIP，性能核心）
- 上游 kernel 全是 `*_sm80.cu`（NVIDIA Ampere 架构），`csrc/flash_attn/src/` 下 60+ 个文件：`flash_fwd_hdim{32,64,96,128,192,256}_{bf16,fp16}_{causal,}_sm80.cu` + `flash_bwd_*` + `flash_fwd_split_*` + 头文件（`flash_fwd_kernel.h`/`kernel_traits.h`/`softmax.h`/`utils.h` 等）。
- 海光把这些 CUDA kernel **hipify 成 HIP、重定向 gfx936 编译**，产物：
  - `flash_attn/lib/libflash_attention.so`
  - `flash_attn_2_cuda.cpython-310-x86_64-linux-gnu.so`
- **这是"怎么做的"核心：CUDA→HIP 移植 + gfx936 重编译，不是改算法。** 移植方法学见 `dcu-cuda-hip-port`。

### 层 2：Python 接口层（大幅扩展）
- `flash_attn_interface.py`：上游 **1606 行 → DCU 3988 行**（diff 2862 行），几乎翻倍。
- 海光**新增** DCU 专属入口函数（上游没有）：
  | 函数 | 用途 |
  |---|---|
  | `hg_flash_attn_varlen_func` | 海光 varlen 主入口（`hg_` 前缀 = Hygon） |
  | `flash_attn_func_blasst` | BLASST 注意力（带 `blasst_threshold_scale_factor`） |
  | `sparse_attn_func` / `sparse_attn_with_sla` | 稀疏注意力（SLA = Sparse Linear Attention） |
  | `flash_mla_with_kvcache` / `get_mla_metadata` | MLA（DeepSeek 类模型） |
  | `varlen_fwd_unified` | 统一注意力入口 |
  | `vllm_flash_attn_varlen_func` / `vllm_flash_attn_with_kvcache` | vLLM 集成入口 |
- 新增文件：`attn_torch_function.py`、`bwd_preprocess.py`、`bwd_split_kernel.py`、`dropout_rng.py`、`fwd_kernel.py`、`fused_softmax.py`、`flash.py`、`flash_attn_triton_mqa_gqa.py`、`flash_attn_triton_interface.py`、`utils/sparse_utils.py` 等。
- `__init__.py` 导出上述全部新入口 + `version.py` 带 `dtk`/`hcu_version`/`git_hash`/`abi`/`torch_version` 字段（DCU 专属版本元信息）。

### 层 3：gfx 运行时分支（按卡型 dispatch）
运行时读 `torch.cuda.get_device_properties("cuda").gcnArchName` 判断走哪条：
- **`is_gfx936` 分支**（BW100）：**mixed paged attention（192/128 head dim + FP8 E5M2）**，decode 场景专用（`max_seqlen_q==1` 或 `<16`、`uniform_decode`、`bshd`、`q192/k192/v128`），带 descale 校验（E5M2 Q 要求 singleton unit descale）。
- **`is_938` 分支**（gfx938/gfx92a）：长序列（`max_seqlen_q>2048`）走 `pagedkv_to_contiguouskv` 连续化路径；非 938 的 E5M2 KV 会 cast 到 q dtype。
- Triton 路径（`flash_attn_triton.py`/`flash_attn_triton_mqa_gqa.py`）也有 `torch.version.hip` 分支。

### 删掉的部分：FA3（hopper/cute）
- 上游的 `hopper/`（FA3）和 `flash_attn/cute/` 在 DCU 版里**被移除**——因为 FA3 是 Hopper 专属（见 §3），gfx936 跑不了。

## §2 版本谱系（DCU 源实测）

- **DCU 源上 flash-attn 的上游版本全部是 2.8.3**（主仓 42 个 wheel 全是 2.8.3）。"新 build"只是海光 fork 的**本地 commit** 在变，不是上游算法升级。
- 版本串格式：`2.8.3+das.opt1.dtk2604.torch<标记>.<build日期>.<git短hash>[.aicc]`
  - `torch<标记>` = 编译时 torch 大版本（torch2100=torch 2.10.0，须与环境 torch 一致，否则 ABI 不匹配，见 `dcu-das-hub-api` §5）
  - `<git短hash>` = 海光 fork 本地 commit
- torch2100（匹配容器 torch 2.10.0）这条线时间线：
  | 本地 commit | build 日期 | 状态 |
  |---|---|---|
  | gca46e7 | 06-19 | vLLM 0.18.1 容器在用 |
  | g5e8a9e | 08-04 | SGLang 0.5.12 容器在用 |
  | **g9febea** | 08-13 | **最新 release**（das1.8.5） |
  | gfc2740 | 09-01 | 最新 nightly（rc4） |
- 查版本/下 wheel 方法见 `dcu-das-hub-api`（DAS Hub JSON API + 内网→公网 URL 改写 + sha256 校验）。

## §3 开源最新版 + 能否适配 DCU + 能否提升性能（核心结论）

**开源最新版 = 2.8.3.post1**（公共 PyPI），DCU 基于 2.8.3，**基本就是最新 FA2**。

| 版本 | 能否适配 DCU | 性能提升 |
|---|---|---|
| **2.8.3.post1**（最新 FA2） | 能，但 DCU 已是 2.8.3，post1 只是 patch，无算法变化 | **≈ 无**（已在最新） |
| **FA3**（hopper 目录，"更快"的那版） | **不能** | 不适用 |

**FA3 为什么不能适配 DCU**（重点）：
- FA3 的 `hopper/setup.py` 明确 `arch=compute_sm90a, code=sm_90a`，**只支持 Hopper（H100/H800）**。
- 它靠 **WGMMA / TMA**（Hopper 专属 tensor core 指令）拿性能，README 写明 "Requirements: H100/H800, CUDA>=12.3"。
- gfx936 是海光 CDNA 架构，**根本没有 WGMMA/TMA 这些指令**，FA3 kernel 无法移植。
- 所以海光把 `hopper/`+`cute/` 从 DCU 版删掉——**不是没做，是做了也跑不了**。

**结论：升开源最新版带不来性能提升**（已在最新可移植版 FA2 2.8.3 上）。

**那 DCU 上 flash-attn 的性能提升从哪来？** 只能靠**海光本地 build 的 gfx936 kernel 调优**（§2 那条 gca46e7→g9febea→gfc2740 线），是针对 gfx936 的 tile 大小 / 量化路径 / paged attention 调优，跟"开源版本"无关。要确认某个新 build 是否更快，**必须 A/B 实测**（同模型同 workload，对比 prefill/decode 吞吐 + 尾延迟，输出完整原始 stdout、不算均值、异常原样保留）。

## §4 验证方法（可复现）

1. **下 DCU wheel**：`dcu-das-hub-api` 查 `das_release/dtk2604` 仓库 flash-attn 最新 release，内网 URL 改公网域名下载 + sha256 校验。
2. **解包**：`zipfile` 解 wheel，看 `flash_attn/*.py`（Python 层）+ `*.so`（编译产物）。
3. **下上游 sdist**：DCU 源 `open/pypi` 仓库的 `flash_attn-2.8.3.tar.gz`（比 pythonhosted 快）。
4. **diff**：`diff -rq 上游/flash_attn DCU/flash_attn`（注意 locale 中文输出）+ `flash_attn_interface.py` 行级 diff 计数。
5. **FA3 架构守卫**：`grep sm90 hopper/setup.py` 确认 Hopper 专属。

## §5.5 paged decode 路径实测事实（2026-10-09，<device-2>gfx936）

- `flash_attn_with_kvcache`（paged 路径）**要求 page size 是 256 的倍数**（vLLM DCU 用 page=16 走不了 FA2 路径，是平台选 Triton 后端的因素之一）；`out=` 参数在 DCU 构建有形状校验 bug，用普通调用
- paged decode 路径**自带 split-K**，比无 paged 的 `flash_attn_func` 连续版快 ~3.2x（32k 上下文 0.44ms vs 1.42ms，batch=1 Hq=32/Hkv=8/d=128）——**对比 FA2 必须用 paged 版当基线**，否则收益虚高
- `flash_attn_func` 默认布局是 **BSHD**（`[batch, seqlen, heads, dim]`）不是 BHSD：传 `(1, Hq, 1, d)` 形状不传 `bhsd=True` 会报 "Number of heads in key/value must divide number of heads in query"（seqlen 维被当成 head 数）

## §5 使用须知

- 本 skill 是**源码适配索引**，**不存性能数据/基线**；性能数字用 `dcu-op-optimization` 模板留痕。
- **wheel 内无 csrc 源码**——kernel 层改动只能从 Python 层 + 编译产物推断，要看 csrc 需拉海光 fork 源码（sourcefind 私有仓，需 PAT）。
- 海光 fork 本地 commit 改了什么**版本串看不出来**（私有仓 changelog 不公开），"新 build 更快"是假设不是事实，须 A/B 实测。
- 换 torch 大版本必须换对应 `torch<标记>` 的 build，否则 ABI 不匹配（`undefined symbol: _ZN3c10...`）。
- 经验仅供参考，先核对设备型号（gfx936/938）+ DTK 版本 + 报错原文。
