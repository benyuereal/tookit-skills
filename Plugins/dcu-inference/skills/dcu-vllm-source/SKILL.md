---
name: dcu-vllm-source
description: vLLM 各版本 DCU 源码结构与适配点索引（合并 0.15/0.18/0.21.0/0.23.0/0.26 五版）
version: 1.0.1
author: DCU Engineer
license: MIT
---

# vLLM 各版本源码结构与 DCU 适配点索引（dcu-vllm-source）

> ## 🔗 开源上游锚点（vllm-plugin-das）
> 官方开源版 **vllm-plugin-das**（`github.com/HYGON-AI/vllm-plugin-das`，20⭐）对齐上游 **vLLM v0.25.1**（commit `752a3a5`），
> 采用**运行时插件**架构：3 个 vLLM 插件入口 `hcu / hcu_model / hcu_ops`，`vllm_config.additional_config["hcu"]` 存 HCU 特性开关，
> `apply_platform_patches()` + `apply_worker_patches()` 两阶段打补丁，`patch_report()` 报角色/符号/状态，`vllm-hcu-doctor` 诊断，
> 兼容门仅收 `0.25.x`；`VLLM_PLUGINS=hcu,hcu_model,hcu_ops`（留空则全载）。目录 `patch/`（import_coordinator + module_exchange + platform/worker）、
> `model_executor/layers/`（linear + fused_moe 配 DeepEP/DeepGEMM + quantization compressed-tensor/SlimQuant）、`models/`（DeepSeek/HY/GLM）、`platforms/hcu.py`。
> 刷新：`python3 scripts/hygon-upstream-sync.py --check`（快照 `references/hygon-upstream/upstream-snapshot.json`，索引 `repos-index.md`）。

> 定位：**合并 vLLM 0.15/0.18/0.21.0/0.23.0/0.26 五版源码适配索引**。原五版 split 为独立 skill，内容高度重复，此处合并为单 skill 覆盖多版本，供 agent 按目标版本查结构/适配点。
> 适用方式：查「某版本 vLLM 的 DCU 适配落点」时，在本 skill 内按版本段取路径/适配结论，再与 `dcu-vllm-op-replace` / `dcu-vllm-custom-op` 组合走算子开发闭环。
> 版本拆分原则的原文纪律：每个大版本曾独立 skill，但重复度太高 → 合并收敛为单 skill（沿用【四象限方法论】的收敛判断）。

## 一、版本索引总表

| 版本 | 身份/基线 | 结构探查方式 | 关键差异（DCU 适配点） |
|---|---|---|---|
| **0.26.x** | 发布说明级（`/tmp/vllm_v0.26.0_article.md`） | 占位（0.26 源码包未获取，待核验） | fp32 head_dtype torch.mm、DeepSeek-V4 两阶段压缩、MXFP8 GEMM、AITER 稀疏分页注意力+投机解码、MiniMax-M2 融合 QK-norm+all-reduce、Transformers 5.13.0/FlashInfer 0.6.14/NIXL 1.3.1、移除 TeleChat/Persimmon/Fuyu |
| **0.23.0** | DAS 发布版 | 源码/容器内实测 | `_custom_ops` 3997 行、`_aiter_ops` 2767 行；编译产物多了 `_C_stable_libtorch.abi3.so` / `spinloop*.so`；结构高度一致（platforms/attention 列表同 0.21） |
| **0.21.0** | `0.21.0+das.8c979d2.dtk2604`（DTK 2604 体系） | 容器 `<container_vllm0210>`（<device-B> 设备），wheel 内无 csrc 源码 | 平台后端复用上游 `RocmPlatform`；注意力主推 `rocm_aiter_unified_attn`（AITER 集成层）；`_custom_ops` 3825 行、`_aiter_ops` 2399 行；编译产物 `_C.abi3.so`/`_moe_C.abi3.so`/`_rocm_C.abi3.so`/`cumem_allocator.abi3.so` |
| **0.18.x** | 0.18.1 海光 fork | `dcu-vllm-source` 实测（0.18 容器） | csrc 无新增（内容级改造）；`_custom_ops` 已核验大小；注册机制未改（0.15.1 实测同）；rocm.py 重定向、VLLM_USE_HYGON 开关（0.15.x 需要，0.18 已移除） |
| **0.15.x** | 0.15.1 海光 fork | `dcu-vllm-source` 实测 | 深度 kernel 新增（区别于 0.18 轻量改造）；csrc 有 moe/quantization 专属 kernel；VLLM_USE_HYGON 开关 |

> 迁移方向：0.15（深度 kernel 新增）→ 0.18（轻量/内容级）→ 0.21/0.23（wheel AITER 集成加深）→ 0.26（发布说明占位待核验）。同一框架演进，适配点从「手写 kernel」转向「AITER 集成层启用」。

## 二、五版共通结构骨架（高重叠部分）

> 从 0.18 / 0.21.0 / 0.23.0 实测归纳，五版**结构高度一致**（差异仅在行数与编译产物）。通用骨架如下：

| 适配层 | 文件 | 0.18 路径 | 0.21/0.23 路径 |
|---|---|---|---|
| 平台后端 | `vllm/platforms/` | `rocm.py`（复用上游 RocmPlatform） | 同（0.21 无 hygon.py，全局 grep hygon 零命中） |
| 注意力后端 | `vllm/v1/attention/backends/` | `rocm_attn.py`（Triton prefix prefill + AITER rope） | `rocm_aiter_unified_attn.py`（AITER 统一封装，DCU 主推） |
| 算子注册 | `vllm/_custom_ops.py` | 0.15.1 实测机制未改、torch.library + @register_fake | 0.21：3825 行；0.23：3997 行 |
| AITER 集成 | `vllm/_aiter_ops.py` | 未核验（0.18 期） | 0.21：2399 行；0.23：2767 行（融合注意力/MoE/量化在此注册） |
| MoE 层 | `vllm/model_executor/layers/fused_moe/` | `fused_humming_moe.py`（hummingbird）+ `fused_batched_moe.py` | 0.21：同 + `triton_deep_gemm_moe.py`（DeepGEMM，对应 `dcu-deepgemm-source`） |
| 量化层 | `vllm/model_executor/layers/quantization/` | fp8/awq/gptq/experts_int8 等 | 同 + `humming.py`（hummingbird 量化，名字巧合非海光专属） |
| 构建系统 | `setup.py` | VLLM_TARGET_DEVICE=rocm | 同（wheel 内无 csrc 源码，构建期 gfx 裁剪） |
| 编译产物 | `vllm/*.so` | `_C.abi3.so` 等 | 0.21：3 个 + `cumem_allocator.abi3.so`；0.23：多 `_C_stable_libtorch`/`spinloop` |

## 三、各版本特有适配要点

### 3.1 0.15.x（深度 kernel 新增）
- csrc 有 moe/quantization 专属 kernel（区别于后续轻量改造）。
- `VLLM_USE_HYGON=1` 是 0.15.x 需要、0.18 已移除的开关。
- 0.15.1 注册机制实测未改（torch.library + `@register_fake`），与 0.18 一致。

### 3.2 0.18.x（轻量/内容级改造）
- 平台后端与 0.15 同复用 `RocmPlatform`；结构无新增文件（csrc 内容级改同名）。
- `VLLM_USE_HYGON` 开关已移除。
- 注册中心 `_custom_ops.py` 未改机制；与 0.15.1 实测一致。
- **decode attention 后端实测（2026-10-09，<device-B> gfx936，容器 `dcu/admin/base/vllm:0.18.1`）**：实际走 `TritonAttentionImpl` → `unified_attention`（`vllm/v1/attention/backends/triton_attn.py`，3D 并行 softmax 分段 `NUM_PAR_SOFTMAX_SEGMENTS=16`，自带 split-K），**不是 flash_attn 也不是 rocm_attn**（模型层 probe：`type(layer.impl).__name__ == 'TritonAttentionImpl'`，36 层全命中）。unified kernel 无 GQA query-head 打包，长上下文小 batch decode 是优化空白点（swap-AB 布局 kernel 实测 E2E graph 模式 +14.8% TPOT@32k，方法论详见 `dcu-perf/dcu-external-kernel-validation`）。

### 3.3 0.21.0 / 0.23.0（wheel 内无 csrc，AITER 集成加深）
- 包以 wheel 交付（编译产物 `.so`），**csrc 源码不在包内**（与 0.26 占位不同——本版明确 wheel 内无 csrc，差异核验需另行拉 DAS 仓库源码）。
- 注意力主推 `rocm_aiter_unified_attn`（AITER 集成层 `vllm/_aiter_ops.py`），DCU 主推后端；调优优先看 `rocm_aiter_ops.is_enabled()`。
- 平台后端完全复用上游 `RocmPlatform`，海光差异在 DTK 运行时报告 gfx 标识 + 构建期 gfx 裁剪（不在 Python 层加分支）。
- `_custom_ops` / `_aiter_ops` 行数增长（0.21：3825/2399；0.23：3997/2767）——算子扩充。

### 3.4 0.26.x（发布说明占位，待源码核验）
- 变更集中在：fp32 head_dtype torch.mm、DeepSeek-V4 两阶段压缩、MXFP8 GEMM、AITER 稀疏分页注意力+投机解码、MiniMax-M2 融合 QK-norm+all-reduce、HybridW4A16 线性内核。
- 依赖门槛：Transformers 5.13.0、FlashInfer 0.6.14、NIXL 1.3.1——升级前先确认海光 DTK/torch 版本。
- 已移除模型：TeleChat、Persimmon、Fuyu（适配时勿再为这些做后端）。
- 源码级路径/适配点：以 0.18 同款探查方法、拿到 0.26 源码包后逐项填真实路径（替换所有「待核验」标记）。

### 3.5 vllm-hcu 插件 patch 层（0.21+ 时代，独立包，★ 详见 dcu-vllm-hcu-source）
- **vllm-hcu 是独立 pip 包**（非 vLLM fork），装到 vLLM 旁，经 `vllm.platform_plugins` / `vllm.general_plugins` entry point 钩进 vLLM 插件系统。
- 适配方式：**运行时字符串 patch**（105 个 `*.patch.py`，每个 target 一个 vLLM 模块）+ **模块交换**（`MODULE_EXCHANGE_MAP` 7 条，`sys.modules` 替换）+ **HCUPlatform 平台插件**（`platforms/hcu.py:216`）。
- 唯一改 vLLM 安装目录处：post_install 软链 `modular_kernel.py`（`post_install.py:5`）。
- 实测版本：`vllm_hcu 0.21.0+das.dtk2604.torch2100`（对应 vLLM 0.21，绑 torch2100）。
- **演进定位**：0.15（深度 kernel fork）→ 0.18（轻量 fork）→ 0.21+（**vllm-hcu 插件 patch**，vLLM 本体尽量不动）。查 0.21+ 的 DCU 适配落点，优先看 `dcu-vllm-hcu-source`（105 patch 分类 + HCUPlatform 落点 + 模块交换映射）。

## 四、与兄弟 skill 关系

- `dcu-vllm-hcu-source`：★ vllm-hcu 插件 patch 层（0.21+ 时代独立包，105 patch + HCUPlatform + 模块交换）→ 查 0.21+ DCU 适配落点优先看它。
- `dcu-vllm-custom-op`：从零造算子（开发流程）→ 对应 vLLM 注册机制，0.18/0.15.1 实测一致。
- `dcu-vllm-op-replace`：替换/重定向已有算子到 vLLM 6 步流程 → 0.15.1 实测（VLLM_USE_HYGON 开关、rocm.py 重定向）。
- `dcu-aiter-source` / `dcu-deepgemm-source`：AITER/DeepGEMM 算子级展开 → 本 skill §二 MoE 层、§三.3 AITER 集成层对应。
- `dcu-op-optimization` / `dcu-perf-profiling`：定位热点后走闭环。
- `dcu-vllm-deploy` / `dcu-inference-infra`：部署/总入口。

## 五、vLLM 0.26.x 上游源码深潜（vendored reference）

> 本 skill 主表是**海光 fork 的 DCU 适配落点**（0.15/0.18/0.21/0.23/0.26 五版）。
> 若要查 **vLLM 上游机制本身**（调度器 / KV Cache / ModelRunner / MoE / Attention 后端 / 量化 / 并行 / 编译 / 投机解码 / Engine / KV 传输 / V2 / 稀疏注意力），见 `references/upstream-vllm-026/`（16 篇，vLLM 0.26.1 upstream 基线，已脱敏、按需加载，主题索引见其 `README.md`）。
> 配套：实测案例在 `dcu-perf-cases` 案例 17-37；症状速查在 `dcu-vllm-deploy/references/perf_symptom_lookup.md`；最小实现 POC 在 `dcu-vllm-deploy/references/poc/`。

## 六、使用须知

- 本 skill 合并五版索引，**不存性能数据/基线**；数字用 `dcu-op-optimization` 模板留痕。
- 海光 fork 具体行为以 sourcefind `codes/OpenDAS/vllm` 各版本分支实时核验。
- 高版本（0.21/0.23）wheel 内无 csrc——需拉 DAS 仓库源码才能看 csrc 层；0.26 是发布说明占位，动代码前必须先拿源码包完成核验。
- 高危操作谨慎，优先测试机验证。