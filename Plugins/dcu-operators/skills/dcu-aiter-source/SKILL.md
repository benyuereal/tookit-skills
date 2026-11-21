---
name: dcu-aiter-source
description: AITER算子库DCU源码结构与适配点索引
version: 1.1.2
author: DCU Engineer
license: MIT
---

# AITER 源码适配索引（dcu-aiter-source）

> ## 🔗 开源上游锚点（aiter-das）
> 官方开源版 **aiter-das**（`github.com/HYGON-AI/aiter-das`）上游参考 commit `552f4b8`（ROCm/aiter `main` 分支，**非 v0.1.3 tag**——可据此修正本 skill §3「das184 可能不是严格对齐 v0.1.3」的待核验项）。AITER 依赖的 hipc MoE 算子库另见 **Moe** 仓（`github.com/HYGON-AI/Moe`）。刷新：`python3 ../_dcu-toolkit-shared/scripts/hygon-upstream-sync.py --check`（快照 `../_dcu-toolkit-shared/references/hygon-upstream/upstream-snapshot.json`）。

> 基于 `aiter-0.1.3-das184.dtk2604.zip`（海光 DAS 适配版，dtk2604）实测。上游对照基线：`ROCm/aiter` v0.1.3 tag（curl 拉取，实测 diff 见 §3）。
> 完整中文算子说明（海光自写）见 `references/aiter-operator-reference.md`（785 行，含 MoE/Norm/Activation/Quant/MQA 全部参数表）。本 skill 提炼结构、机制与适配点，不照抄参数表。

## §0 定位：AITER 是什么、和 vLLM 的差异

AITER = AI Tensor Engine for ROCm，**底层算子加速库**（C++/HIP kernel + PyTorch/Python 绑定），不是推理框架。
- 整合四种后端：**moe_c（Marlin/CK）/ asm（汇编 kernel）/ triton（JIT）/ ck（Composable Kernel）**，统一通过 Python API 暴露。
- 覆盖算子族：MoE（混合专家）、Norm（RMSNorm 等）、Activation（SwiGLU 等）、Quant（per-token/group FP8/INT8 等）、Attention（MHA/MQA/MLA/paged）、RoPE。
- **和 vLLM 的边界**：vLLM 是推理框架、在上面；AITER 是它下面的算子实现库。vLLM 的 ROCm 路径里很多算子会调用 AITER（尤其 MoE/GEMM）。所以"用 AITER 写适配和优化算子"的链路是：
  `AITER 取/改 kernel` → `dcu-kernel-dev`（HIP 开发规范）→ `dcu-vllm-op-replace`（把算子接入 vLLM 原生路径）。

## §1 源码地图（实测目录结构）

```
aiter-0.1.3-das184.dtk2604/
├── aiter/                    # Python 包（PyTorch 绑定 + 调度逻辑）
│   ├── __init__.py
│   ├── moe.py                # MoE 总入口 + get_aiter_moe_config（后端选择器）
│   ├── fused_moe.py / fused_moe_c.py / fused_moe_asm.py / fused_moe_ck.py  # 四后端实现
│   ├── ops/                  # 算子族 Python 接口
│   │   ├── moe_op.py moe_sorting.py rmsnorm.py rope.py quant.py norm.py
│   │   ├── attention.py mha.py pos_encoding.py activation.py topk.py shuffle.py
│   │   ├── triton/           # Triton JIT 后端（gemm_*/attention/*/fused_moe.py）
│   │   └── tilelang/         # TileLang 后端（sparse_mla_fwd 等）
│   ├── configs/              # 调优表（asm_tune/ck_tune/fmoe csv/json）
│   │   └── moe_c_configs/    # 分 gfx 的 MoE_C 调优：gfx928 gfx92a gfx936 gfx938 silu_configs
│   ├── dist/                 # 分布式通信（通信原语、custom_all_reduce、EP 映射）
│   └── jit/                  # JIT 编译工具链（cpp_extension / hipify / chip_info）
├── csrc/                     # HIP/C++ kernel 与 pybind 绑定
│   ├── kernels/              # 算子 kernel（.cu）：moe_fused_gate / rmsnorm_* / quant_* / mqa_logits_* / rope_* / topk_* / attention_*
│   ├── py_itfs_asm/ py_itfs_ck/ py_itfs_cu/ py_itfs_moe_c/  # 各后端 pybind 入口 + 调优 manifest
│   ├── ck_batched_gemm_* ck_gemm_*   # CK 后端 GEMM 实例
│   └── rocm_ops.cpp          # 总注册
├── asm_kernels/              # 手写汇编 kernel（.s，W4A16/Dequant/Marlin 等，按 MT/BH 分块命名）
├── 3rdparty/                 # 子模块：CK、moe_c（首次须 git submodule update --init）
├── op_tests/                 # 算子测试（ci_tests/ 下有 per-算子测试脚本）
├── das-build.sh rebuild_aiter.sh gen_co.sh   # 构建脚本（海光版，见 §2）
├── aiter-operator-reference.md        # 海光自写中文算子文档（references/ 已归档）
└── setup.py pyproject.toml requirements.txt
```

## §2 构建与安装（海光实测）

- **依赖子模块**：首次必须 `git submodule update --init`（CK、moe_c）。否则 3rdparty 空、编译失败。
- **清理旧产物**（rebuild_aiter.sh）：删 `aiter/jit/{aiter_.so,build,*.so,jit/build}`、`build`、`aiter.egg-info`、`aiter_meta`，`pip uninstall aiter -y`。
- **develop 安装**（按 gfx 指定架构）：
  ```bash
  GPU_ARCHS="gfx936;gfx938" python setup.py develop   # gfx936=BMZ, gfx938=NMZ
  ```
- **打 whl**（预编译 kernel，慢但部署快）：
  ```bash
  PYTHONUNBUFFERED=1 GPU_ARCHS="gfx936;gfx938" PREBUILD_KERNELS=1 \
    AITER_PREBUILD_LOG_PROGRESS=1 python setup.py bdist_wheel
  ```
- **诊断环境变量**（卡住时用）：`AITER_PREBUILD_LOG_PROGRESS=1`（每模块 START/DONE）、`AITER_PREBUILD_VERBOSE=1`（hipcc/ninja -v）、`AITER_LOG_MORE=2`（JIT 流程）、`MAX_JOBS=1`（串行）、`AITER_REBUILD=1`（强制全重编）。
- **运行测试**：`PYTHONPATH=. python op_tests/ci_tests/test_aiter_moe_with_config_w8a8_channelwise.py`（或 `python -m op_tests.ci_tests.<mod>`）。
- **关键日志开关**：`AITER_LOG_MORE`、`AITER_LOG_OP_PARAM=1`（打印 aiter_moe 等算子参数）。

## §3 海光适配点（diff 实测，对照 ROCm/aiter v0.1.3）

> 方法同 `dcu-vllm-source/../../../dcu-inference/skills/dcu-vllm-source/scripts/diff-hygon-release.sh`：`diff -rq` + 行级计数。上游 tarball 已缓存于 `<local_src_cache>/aiter-v0.1.3-upstream.tar.gz`，解压于 `<local_src_cache>/aiter-0.1.3`。

实测结论（aiter-0.1.3-das184 vs 上游 v0.1.3）：
- **Python 层大面积改动**：`aiter/` 几乎全改——`dist/`（通信重写）、`jit/`（工具链适配）、`configs/`（GPU 调优表标识）、`ops/` 各算子接口、`__init__.py`（加 torch._dynamo.ignore_logger_methods）。
- **csrc/ 层无新增文件**：`diff -rq csrc` 的 `Only in` 为空——海光是**同名 .cu/.cpp 改内容**（如 moe_fused_gate.cu / rmsnorm_kernels.cu 内做 gfx 分支），不是新增专属文件。这和 vLLM 0.18.1 的"csrc 无新增"模式一致。
- **configs GPU 标识改写**：Triton `configs/gemm/*.json` 的 `MI300X/MI350X` 命名被改为海光 gfx 标识；`moe_c_configs/` 下分 `gfx928/gfx92a/gfx936/gfx938` 沉淀各卡调优。
- **asm_kernels/**：海光补充 gfx936/938 的汇编 kernel（W4A16/Dequant/Marlin 等，按 MT×BN×BK 分块命名）。
- **[待核验] das184 基线偏移**：`__init__.py` 版权年 `2024-2025`→`2024-2026`，且加了上游 0.1.3 未必有的 dynamo 改动——**das184 可能不是严格对齐 v0.1.3 tag，而是基于更新的提交**。skill 内引用上游 tag 仅为近似基线，精确适配点以海光发布版为准。
- **适配策略定性**："ROCm 上游 + 海光 gfx/调优表适配 + 工具链适配"，改动量大于 vLLM 0.18.1（轻量）但小于 vLLM 0.15.1（深度 kernel 新增）。属于**中等改造**：复用 AMD 四后端框架，差异集中在 gfx 标识、调优表、通信/工具链。

## §4 四后端调度机制（AITER 核心抽象）

所有 MoE/融合算子都走"配置选择器 → 分发到后端"模式，理解这个才能做适配/优化：

- **`get_aiter_moe_config(M, E, N1, N2, K, top_k, block_size, dtype, quant_type, ...)`** 是 MoE 总调度器：
  - 按 `quant_type × dtype` 的组合确定后端优先级（例：W4A16+BF16 → 先 ASM 后 Triton 后 moe_c；W8A8 channel-wise → 先 moe_c 后 Triton；W8A8 block-wise → 先 ASM 后 Triton）。
  - 返回 `AiterMoeConfig(solution_type, config, need_shuffle, quant_type)`；`status=False` 时无可用后端。
  - 指定 `spec_sol_type` 可强制只用某后端。
- **`aiter_moe(...)`** 按 `moe_config.solution_type` 分发：
  - `MOE_C` → `moe_c_fused_experts`（Marlin/CK，W4A16/W8A8 等）
  - `ASM` → `fused_experts_asm_impl`（汇编 kernel，stage1/stage2 GEMM + Triton 激活/reduce）
  - `TRITON` → `fused_experts_impl`（Triton JIT）
  - `CK` → `run_fused_experts_ck_impl`（Composable Kernel）
- **关键注意（海光文档强调）**：moe_c 后端调用前必须做 `moe_layout_shuffle_gemm1/2`（或经 `aiter_moe_shfl_weight`）；**不做 shuffle 直接传 `moe_c_fused_experts` 会导致底层 Marlin kernel 按错误偏移读权重、结果错误**。这是踩坑高发点。
- **[gfx936 专用坑] MXFP4 原生路径不可用**：gfx936 上 AITER 原生 MXFP4 MoE（Triton `tt.dot_scaled` + E2M1）编译器在 `convert-triton-amdgpu-to-llvm` 报 `Unsupported DotScaleOp`，加 gfx936 白名单无效（缺编译器 lowering，非架构开关）。需走软件回退方案（packed MXFP4 反量化成 BF16 再 GEMM），适配流程详见 `dcu-aiter-moe` §7。
- 调用链树（详）：见 `references/aiter-operator-reference.md` 的 §2.1（路径 A/B/C 三棵调用树）。

## §5 算子族索引（实现位置速查）

| 算子族 | Python 入口 | csrc kernel（精确文件） | 调优表/后端 | 测试入口 | 说明 |
|---|---|---|---|---|---|
| MoE（融合专家） | `aiter/moe.py`、`fused_moe*.py` | `csrc/kernels/moe_fused_gate.cu` + `csrc/py_itfs_moe_c/moe_c.cu` + `csrc/py_itfs_asm/asm_fmoe_*.cpp` + `asm_kernels/*.s` | `configs/moe_c_configs/{gfx936,gfx938}` + asm/ck manifest | `op_tests/ci_tests/test_aiter_moe_with_config_*.py`、`test_moe_sorting.py`、`test_moeTopkSoftmax.py` | **深度专题见 `dcu-aiter-moe`**（四后端调度/shuffle 踩坑/精确落点） |
| RMSNorm / AddRMSNorm | `aiter/ops/rmsnorm.py`、`norm.py` | `csrc/kernels/rmsnorm_kernels.cu`、`rmsnorm_autograd.cu`、`fused_rmsnorm_rope.cu` | — | `op_tests/ci_tests/test_rmsnorm2d.py` | head_rms_norm / fused_add_rms_norm_cu |
| SwiGLU / Activation | `aiter/ops/activation.py` | `csrc/kernels/swiglu_variant.cu`、`activation_kernels.cu` | — | `op_tests/ci_tests/test_aiter_sigmoid.py` | moe_swiglu_dynamic_quant_wrapper |
| Quant（FP8/INT8 动态量化） | `aiter/ops/quant.py` | `csrc/kernels/quant_kernels.cu`、`per_token_group_quant_fp8_kernels.cu` | — | `op_tests/ci_tests/test_quant.py` | per_token_group_quant_fp8 |
| Attention（MHA/MQA/MLA/paged） | `aiter/ops/mha.py`、`attention.py` + `ops/triton/attention/*` | `csrc/kernels/{attention_v1.cu,attention_ragged.cu,mha_common.cu,mqa_logits_kernels.cu,mhc_kernels.cu}` + `csrc/py_itfs_ck/{mha_fwd_kernels.cu,mha_bwd_kernels.cu,mha_varlen_fwd_kernels.cu,mha_varlen_bwd_kernels.cu,attention_kernels.cu,grouped_gemm_kernels.cu,moe_c_api_kernels.cu}` | ck manifest（`py_itfs_ck/import_ck_grouped_gemm.md`） | — | mqa_logits |
| RoPE / 位置编码 | `aiter/ops/rope.py`、`pos_encoding.py` | `csrc/kernels/rope_general_fwd_kernels.cu`、`rope_general_bwd_kernels.cu`、`rope_pos_fwd_kernels.cu`、`fused_rmsnorm_rope.cu` + `csrc/kernels/rope/`（子目录） | — | — | — |
| TopK / 路由 | `aiter/ops/topk.py` | `csrc/kernels/{topk_plain_kernels.cu,topk_softmax_kernels.cu,topk_softmax_kernels_group.cu,topk_transform.cu,topk_per_row_kernels.cu}` | — | `op_tests/ci_tests/test_topk_plain.py`、`test_topk_transform.py`、`test_moeTopkSoftmax.py` | — |
| 通信（AllReduce/EP） | `aiter/dist/*`、`aiter/ops/custom_all_reduce*` | `csrc/kernels/custom_all_reduce.cu`、`custom_kernels.cu` | — | `op_tests/ci_tests/test_custom_allreduce.py`、`test_fused_ar_rms.py` | 分布式通信原语 |

## §6 和 vLLM 的衔接（写适配/优化算子的入口）

你"用 AITER 写适配和优化算子"的实操链路（详见关联 skill）：
1. **取/改 kernel**：从本 skill 的 §5 定位算子 → 改 `csrc/kernels/*.cu` 或 `asm_kernels/*.s`（仿 `moe_fused_gate.cu` 的 `#ifdef USE_ROCM` 写法）。开发规范见 `dcu-kernel-dev`（HIP 语法/编译调试）。
2. **绑定**：在 `csrc/py_itfs_*` 加 pybind，在 `aiter/ops/*.py` 暴露 Python 接口。
3. **接 vLLM**：经 `dcu-vllm-op-replace` 的 6 步流程（HIP kernel → torch_bindings → _custom_ops.py 登记 → rocm.py 重定向 → 重编译 → 验证），把 AITER 算子替换进 vLLM 原生路径。
4. **验证闭环**：`dcu-op-optimization`（profile→定位→改→验证）+ `dcu-perf-profiling`（rocprof/rocTracer）。
5. **构建前置**：海光 DTK 见 `dcu-dtk`；容器内必挂 `-v /opt/hyhal:/opt/hyhal:ro`，否则 DCU 不可见。

## 何时选它 / 何时别选

- **选它**：标准 MoE / Norm / Activation / Quant / Attention 接入，要一层统一 Python API，还要按 gfx 自动挑后端；已在 vLLM/SGLang 的 ROCm-HCU 路径上，改的是「被框架调到的那条实现」。
- **选它**：`get_aiter_moe_config` 返回的 `solution_type` 是 `ASM` / `CK` / `TRITON` 之一，问题确实落在该后端 kernel 里。
- **选它**：需要四后端回退——某量化档在 asm 上没解但 triton 有，交给调度器比自己做分支便宜。
- **别选**：MoE 慢但 `solution_type` 是 `MOE_C`——那是独立仓库 `Moe` 的实现（见 `dcu-aiter-moe`），改 AITER 的 Python 层调不到它；先确认后端再动手。
- **别选**：需要 GEMM 的 layout 级控制（masked/contiguous/low-latency、HIPC/W6 权重布局）——走 `dcu-deepgemm-source`，AITER 的 GEMM 是 MoE 流程的一部分。
- **别选**：memory-bound 的融合链（norm+rope、激活+量化）——LightOp 的融合 kernel 才是为这条链设计的。
- **结构性吃亏**：查表不是自动调优。同一 shape 换一档量化可能整体降级到更慢的后端而**无告警**（`status=False` 才报），性能回归的根因常在表项缺失而不是 kernel 退化——这种情况继续调 kernel 是白花时间。

## 使用须知
> **别混淆**：z-file Cat4 DAS 目录里的 `aitemplate`（DAS1.8.3/1.8.4 新增独立包）是 Meta AITemplate 框架线，**不是**本 skill 的 AITER 算子库（aiter）——找 aiter 的包/源码时别错拿 aitemplate（2026-10-14 下载站实测备注）。

- 经验仅供参考，不保证修复。**先核对**：设备型号（gfx936/938 等）、DTK 版本、报错原文；适配点随 DAS 版本变化。
- 改 AITER kernel 后必须重编（`rebuild_aiter.sh` 或 `setup.py develop`），且 `PREBUILD_KERNELS=1` 预编译的 kernel 与 `GPU_ARCHS` 强相关——**换 gfx 必须重编**。
- 重启/重装驱动/刷 vbios 等高危操作需谨慎，确认环境与带外恢复通道、优先测试机验证（见 dcu-ontrack-experience 使用须知）。

## 支持文件
- `references/aiter-operator-reference.md`：海光自写中文算子文档（785 行，MoE/Norm/Activation/Quant/MQA 完整参数表与调用树）。从发布包 `aiter-operator-reference.md` 归档。
- 上游基线（已缓存）：`<local_src_cache>/aiter-v0.1.3-upstream.tar.gz` → 解压 `<local_src_cache>/aiter-0.1.3`。
- 海光发布版（已解压）：`<local_src_cache>/aiter-0.1.3-das184.dtk2604/`。
- diff 方法：复用 `dcu-vllm-source/../../../dcu-inference/skills/dcu-vllm-source/scripts/diff-hygon-release.sh`（本 skill §3 结论用其方法得出）。
