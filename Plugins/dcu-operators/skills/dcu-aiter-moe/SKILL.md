---
name: dcu-aiter-moe
description: AITER MoE融合算子四后端调度与适配索引
version: 1.0.0
author: DCU Engineer
license: MIT
kind: knowledge
---

# AITER MoE 算子深度索引（dcu-aiter-moe）

> 从 `dcu-aiter-source` 析出的 MoE 专题。总览/源码地图/构建见 `dcu-aiter-source`；本文聚焦 MoE 的实现、调度、踩坑与调优。
> 事实来源：海光自写 `dcu-aiter-source/references/aiter算子库说明.md`（§2 MoE，785 行文档中占比最大）+ 实测源码 `aiter-0.1.3-das184.dtk2604/`。

## §0 定位

MoE 层 = 门控路由（Gating/Router）+ 专家网络（Experts），核心靠 **Token 路由重排 + 两阶段融合 GEMM（Gate→Activation→Down）** 实现稀疏计算。
AITER 的 MoE 不是单一 kernel，而是一套**四后端调度框架**：同一份 `aiter_moe` 接口，按量化类型和 dtype 自动选 moe_c / asm / triton / ck 四种后端之一执行。这是 AITER MoE 区别于裸 GEMM 库的核心抽象。

## §1 调度框架：get_aiter_moe_config（后端选择器）

`get_aiter_moe_config(M, E, N1, N2, K, top_k, block_size, dtype, quant_type, activation, gated, spec_sol_type, use_shuffle)` 是总入口，返回 `AiterMoeConfig(status, solution_type, config, need_shuffle)`。

**后端优先级**（由 `quant_type × dtype` 决定，部分）：
- W4A16 + BF16 → 先 ASM，次 Triton，最后 moe_c
- W8A8 + channel-wise（block_size=0）→ 先 moe_c，次 Triton
- W8A8 + block-wise → 先 ASM，次 Triton
- 指定 `spec_sol_type` 时只考虑该后端

**关键参数语义**：
- `N1` = GEMM1 输出维度，gated 场景（`silu`/`gelu`）= `2*intermediate_size`，非 gated = `intermediate_size`
- `block_size=0` 或空 → channel-wise；非 0 → block-wise（`[0, block_size]`）
- `quant_type` 支持：w16a16 / w4a16 / int8_w8a8 / fp8_w8a8 / w4a8 / int8_w8a16
- `need_shuffle`：调用 `aiter_moe` 前是否需先 `aiter_moe_shfl_weight` 重排权重（ASM/CK 路径常需）

**降级保护**：所有后端均无匹配配置时 `status=False`，调用方应跳过或用默认。

## §2 aiter_moe 调用树（四后端路径）

```
aiter_moe(hidden_states, w1, w2, topk_weights, topk_ids, moe_config, ...)
└── 按 moe_config.solution_type 分发：
    ├── [A] MOE_C   → moe_c_fused_experts
    │     ├── inplace/outplace 分支
    │     └── fused_experts_impl_marlin / fused_experts_impl
    │           ├── moe_align_block_size / moe_sorting_ck → aiter.moe_sorting_fwd
    │           ├── invoke_fused_moe_kernel_marlin → aiter.moe_c_moe_gemm_marlin_w4a16 (GEMM1)
    │           ├── moe_c_silu_and_mul
    │           ├── invoke_fused_moe_kernel_marlin (GEMM2)
    │           └── triton_moe_sum / moe_c_moe_sum
    ├── [B] ASM     → fused_experts_asm_impl
    │     ├── moe_sorting_ck → aiter.moe_sorting_fwd
    │     ├── aiter.asm_fmoe_stage1          # ASM GEMM1
    │     ├── triton_silu_and_mul            # Triton 激活
    │     ├── aiter.asm_fmoe_stage2          # ASM GEMM2
    │     └── triton_moe_sum                 # Triton reduce
    ├── [C] TRITON  → fused_experts_impl (ops.triton.fused_moe)
    │     ├── moe_sorting_ck → aiter.moe_sorting_fwd
    │     ├── invoke_fused_moe_kernel → fused_moe_kernel (GEMM1, JIT)
    │     ├── _apply_activation (silu_and_mul)
    │     ├── invoke_fused_moe_kernel (GEMM2)
    │     └── triton_moe_sum
    └── [CK]       → run_fused_experts_ck_impl (Composable Kernel)
```

**标准调用顺序**：
1. `status, moe_cfg = get_aiter_moe_config(...)` —— 必须先调，检查 status
2. `if moe_cfg.need_shuffle: w1, w2 = aiter_moe_shfl_weight(w1, w2, moe_cfg)`
3. `out = aiter_moe(hidden_states, w1, w2, topk_weights, topk_ids, moe_cfg, ...)`
   - 若 `moe_config.solution_type` 或 `quant_type` 为 None → 直接抛 `ValueError`

## §3 踩坑（海光文档重点强调）

- **shuffle 必须做**：moe_c 后端调用前必须 `moe_layout_shuffle_gemm1/2`（或经 `aiter_moe_shfl_weight`）。**不做 shuffle 直接传 `moe_c_fused_experts`，底层 Marlin/CUDA kernel 会按错误内存偏移读权重 → 结果错误（无声崩溃，不报错）**。这是最高频踩坑。
- **status 检查**：`get_aiter_moe_config` 返回 `status=False` 时必须跳过/降级，不能直接传空 config 给 `aiter_moe`（会 ValueError）。
- **gated 推断**：`activation=silu/gelu` 自动判 gated（`N1=2*intermediate`），`relu2` 为非 gated；显式传 `gated=` 可覆盖。
- **W8A8/W8A16/FP8 channel-wise**：Triton 路径在 `block_shape is None` 时自动 `per_channel_quant=True`，不要重复设。
- **EP 场景**：`global_num_experts` / `expert_map` 用于 Expert Parallel；`expert_map` 把全局 expert 映射到本地。

## §4 调优表与实现落点（海光实测）

| 项 | 位置 | 说明 |
|---|---|---|
| MoE_C 分卡调优 | `aiter/configs/moe_c_configs/{gfx928,gfx92a,gfx936,gfx938}/` + `silu_configs/` | 海光按 gfx 沉淀的 Marlin/CK 调优（改自上游 MI300X/MI350X 配置） |
| ASM 方案 manifest | `csrc/py_itfs_asm/optKernelManifest.json` + `asm_gemm_kernel_config.*` | ASM GEMM SOL_ID 配置 |
| Triton 调优 | `aiter/ops/triton/configs/gemm/*.json` | GPU 标识已改为海光 gfx |
| ASM kernel | `asm_kernels/*.s`（W4A16/Dequant/Marlin，按 MT×BN×BK 分块命名） | 海光补充 gfx936/938 |
| C++/HIP kernel | `csrc/kernels/moe_fused_gate.cu` | 融合门控（四后端共用入口 kernel） |
| pybind 注册（MoE_C 后端） | `csrc/py_itfs_moe_c/moe_c.cu` | MoE_C 的 pybind + kernel 绑定 |
| pybind 注册（ASM 后端） | `csrc/py_itfs_asm/asm_fmoe_2stage.cpp`、`asm_fmoe_a8.cpp`、`asm_fmoe_solutions.cpp` | ASM 两阶段 GEMM + 方案选择 |
| Python 调度接口 | `aiter/moe.py`：`get_aiter_moe_config`(L307)、`aiter_moe`(L449)、`aiter_moe_shfl_weight`(L651) | 后端选择器 + 主入口 + shuffle |
| Python 后端实现 | `aiter/fused_moe_c.py`：`moe_c_fused_experts`(L3238)、`fused_experts_impl`(L3791)；`aiter/fused_moe_asm_wna16.py`：`fused_experts_asm_impl`(L178) | moe_c / triton / asm 三后端实现 |
| MoE 测试（精确） | `op_tests/ci_tests/test_aiter_moe_with_config_w16a16.py`、`_w16a16_nogate.py`、`_w4a16.py`、`_w4a8_perchannel.py`、`_w8a16.py`、`_w8a8_blockwise.py`、`_w8a8_channelwise.py`、`_w8a8_tensorwise.py`、`test_moe_sorting.py`、`test_moeTopkSoftmax.py` | 按量化类型分脚本；运行 `python op_tests/ci_tests/<名>` 或 `python -m op_tests.ci_tests.<名>` |

## §5 和 vLLM MoE 的衔接

- vLLM 0.15.1（海光深度改造版）里有 `csrc/moe/moe_fused_gate.cu`，和 AITER 的 `moe_fused_gate.cu` 是同类算子的两套实现——**对照两者可快速定位海光 MoE kernel 的异同**（dcu-vllm-source §csrc 已记）。
- 把 AITER MoE kernel 适配进 vLLM 走 `dcu-vllm-op-replace` 6 步（HIP kernel → torch_bindings → _custom_ops.py → rocm.py 重定向 → 重编译 → 验证）。
- kernel 开发规范（HIP 语法/编译调试）见 `dcu-kernel-dev`；性能验证闭环见 `dcu-op-optimization` + `dcu-perf-profiling`。

## §6 验证入口（实测命令）

```bash
# 开发模式跑 MoE 测试（需先 setup.py develop + GPU_ARCHS=gfx936;gfx938）
PYTHONPATH=. python op_tests/ci_tests/test_aiter_moe_with_config_w8a8_channelwise.py
# 或
python -m op_tests.ci_tests.test_aiter_moe_with_config_w8a8_channelwise
# 打印算子参数（调试 shuffle/后端选择）
AITER_LOG_OP_PARAM=1 python op_tests/ci_tests/test_aiter_moe_with_config_w8a8_channelwise.py
```

## §7 gfx936 MXFP4 软件回退（绕开 AITER 原生 DotScaleOp 阻塞）

> 来源：开源仓库 `lutianshu824/mxfp4-gfx936`（MIT）。它不是新算子库，而是 **AITER MoE 在 gfx936 上的第 5 条回退后端 + vLLM 适配 + Qwen3 混合精度示例**。核心结论：gfx936 上 AITER 原生 MXFP4 MoE（`tt.dot_scaled` + E2M1）编译器在 `convert-triton-amdgpu-to-llvm` 报 `Unsupported DotScaleOp`，**加 gfx936 白名单没用**（缺编译器 lowering，非架构开关）。本仓库绕开它，走「packed MXFP4 权重在设备上显式反量化成 BF16 → 平台 BF16 GEMM」。

**两后端**（env 控制，`MXFP4_GFX936_BACKEND` / vLLM 侧 `MXFP4_IMPLEMENTATION`）：
- `safe`（默认，唯一精度过关）：权重保持 uint8 packed，设备上查表/`torch.exp2` 反量化成 BF16 后调 BF16 matmul。入口函数：`src/aiter/ops/triton/moe_op_mxfp4_soft.py:dequant_mxfp4`（表查找反量化，E8M0 scale 逐 32 值套一个，code==255→NaN）、`linear_mxfp4_soft`（dense W4A16）、`fused_moe_mxfp4_soft`（routed MoE）。
- `triton`（实验性，显式开启）：在 Triton tile 内用位运算无表解码 E2M1（低 nibble `((packed<<12)>>6)&mask`、高 nibble `(packed<<2)&mask`，配合 `fp4_to_bf16_scale=0x7E800000` 位转 bf16=128），单次 dot + FP32 累加（刻意不做两次独立 dot，避免 MoE routing 放大舍入差）。编译器不可用时**直接报错**不静默回退。实测 PPL 57.7941 vs 参考 56.9245（+1.5276%），未过 0.1% 实现阈值 → 仅暴露为 experimental，不用于通过示例。

**反量化原理（OCP MXFP4 = E2M1 值 + E8M0 scale，block_size=32）**：
- 16 个 E2M1 码值：`0/±0.5/±1/±1.5/±2/±3/±4/±6`（见 `_MXFP4_E2M1_VALUES`）。
- E8M0 scale：`scale = 2^(exponent-127)`，每 32 个 MXFP4 值共享一个 uint8 scale。
- BF16 解码：`value_bits × 128 × block_scale`（128 即 E2M1→BF16 的位缩放常数）。

**适配流程（怎么接进 AITER / vLLM）**：
1. **AITER 集成（拷贝 + dispatch 注入）**：基于 OpenDAS AITER 提交 `e99c6755`。把仓库两文件拷进 AITER 源码树：
   - `src/aiter/ops/triton/moe_op_mxfp4_soft.py` → `<aiter>/aiter/ops/triton/moe_op_mxfp4_soft.py`
   - `integration/aiter/moe_op_mxfp4.py` → `<aiter>/aiter/ops/triton/moe_op_mxfp4.py`
   - 注入点：`integration/aiter/moe_op_mxfp4.py:fused_moe_mxfp4` 内 `if triton.runtime.driver.active.get_current_target().arch == "gfx936": from aiter.ops.triton.moe_op_mxfp4_soft import fused_moe_mxfp4_soft; return fused_moe_mxfp4_soft(...)`——非 gfx936 走原生 `_fused_moe_kernel_mxfp4`（`tt.dot_scaled`）。
   - 测试入口：`tests/test_moe_mxfp4_soft.py` → `<aiter>/op_tests/triton_tests/test_moe_mxfp4_soft.py`，`pytest -q` 发布套件 26/26 通过。
2. **vLLM 集成（runtime monkey-patch adapter，不动 vLLM 源码）**：`integration/vllm/vllm_mxfp4_gfx936.py` 在 `import vllm` 后 import 本模块即生效，覆盖 compressed-tensors 量化层方法：
   - `CompressedTensorsW4A16Mxfp4.process_weights_after_loading` / `.apply_weights`（dense W4A16）
   - `CompressedTensorsW4A4Mxfp4MoEMethod.process_weights_after_loading` / `.apply`（W4A4 MoE）
   - 环境变量：`MXFP4_IMPLEMENTATION=safe|triton|reference`（默认 safe；`reference`=纯 BF16 反量化对照）、`MXFP4_SOFT_MODULE` 指向 soft 模块路径。要求 vLLM >= 0.15。
   - 用法：`export MXFP4_IMPLEMENTATION=safe; export MXFP4_SOFT_MODULE=$PWD/src/aiter/ops/triton/moe_op_mxfp4_soft.py; python -c 'import integration.vllm.vllm_mxfp4_gfx936; import your_server'`。

**Qwen3-30B-A3B 混合精度结论（精度来源定位）**：
- 全 MXFP4 未过质量门，尽管软件实现与独立反量化参考一致。
- 消融定位：routed expert 的 gate/up/down 是最大精度损失源。
- 最终方案：从同源 BF16 checkpoint 还原全部 routed expert 权重（gate/up/down），非 expert 权重保持 MXFP4。通过全部 11 个同 TP 精度门（TP=2；PPL 混合 56.9245 vs 官方 BF16 58.3569）。
- 代价：模型 17GiB→56GiB（显存翻倍），且仅为 380 token 探针，非大规模 benchmark。
- 构建器：`examples/qwen3_30b_a3b/build_all_experts_bf16_checkpoint.py`（fail-loud：不覆盖输出、校验 18432 个 expert 张量精确还原、删除所有 expert packed/scale 张量）。

**范围边界（不要误用）**：
- 是软件 fallback，**不声称 gfx936 硬件原生 MXFP4 吞吐**；只移除了 DotScaleOp 编译阻塞。
- 不支持 swizzled MX scales（safe/triton 后端显式 `NotImplementedError` 或 fail-loud）。
- 无专用调优表：safe 复用平台 BF16 GEMM 默认；triton 实验性 kernel 固定 `BLOCK_M=16/64, BLOCK_N=64, BLOCK_K=128, num_warps=8, num_stages=3`。
- 实测栈：Hygon BW100(gfx936) / DTK 26.04 / PyTorch 2.9 / Triton 3.3 / vLLM 0.15.1。

## 使用须知
- 经验仅供参考，不保证修复。**先核对**：gfx 型号（936/938）、DTK 版本、报错原文。适配点随 DAS 版本变化（das184 基线可能非严格 v0.1.3 tag，见 dcu-aiter-source §3）。
- 改 MoE kernel 后必须重编（`rebuild_aiter.sh` 或 `setup.py develop`）；`PREBUILD_KERNELS=1` 预编译 kernel 与 `GPU_ARCHS` 强相关，**换 gfx 必须重编**。
- 高危操作（重装驱动/刷 vbios）需谨慎，确认带外恢复通道、优先测试机验证。
