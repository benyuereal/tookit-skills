# HYGON-AI 高性能算子库注册表

> 上游组织：`https://github.com/HYGON-AI`（61 个公开仓库，快照见 `upstream-snapshot.json`）
> 本文只记**元数据与入口**：默认分支、关键路径、算子面、构建方式。源码内容不复制。
>
> 刷新方式：`python3 scripts/hygon-upstream-sync.py --check`（快照）/ 按本文路径直接读上游。

## 一、算子库（直接影响 `op-stack-design` 的选型）

| 库 | 默认分支 | 上游 | 许可 | 一句话 |
|---|---|---|---|---|
| **hytlass** | `release` | CUTLASS 3.5.0 `7d49e6c` | BSD-3 | HCU 版 CUTLASS：**C++ 模板库，自己搭 GEMM** |
| **lightop** | `master` | 自研（无上游） | Apache-2.0 | 融合算子库，覆盖 GEMM/MoE/Attention/norm/quant/采样 |
| **deepgemm-das** | `main` | deepseek-ai/DeepGEMM `559d79f` | MIT | 三路径 × 8 种量化，纯 GEMM 专精 |
| **Moe**（moe_c） | `main` | 自研，兼容 AITER 调用路径 | MIT | MoE GEMM 后端，8 种精度模式 |
| **aiter-das** | `main` | ROCm/aiter `552f4b8` | MIT | 四后端调度框架 |
| **BoltOPs** | `release/0.1.0` | 自研（AI Compiler 团队） | — | Triton/TileLang DSL 低延迟算子 |
| **TileKernels-das** | `staging` | DeepSeek TileKernels | — | tilelang 算子，**含 mHC 全套** |
| **FlashMLA-das** | — | DeepSeek FlashMLA | — | MLA，BF16/FP8 + 稀疏 |
| **flash-attention-das** | — | Dao-AILab/flash-attention | — | attention，含 qsa/dsa 变体 |
| **composable_kernel-das** | — | ROCm CK | — | CK 的 HCU 移植 |
| **one-click-quant** | — | llmcompressor / MoE-Quant | — | 一键量化工具链 |
| **ao-das** | — | torchao v0.18.0 `5f2baf9` | BSD-3 | torchao 的 HCU 适配 |

## 二、四个新库的入口（现有 skill 未覆盖）

### hytlass —— 补上「自己搭 GEMM」这条中间路径

```
include/hute/algorithm/    gemm.hpp copy.hpp prefetch.hpp axpby.hpp — HuTe 层原语
examples/                  19 个（见下）
```

示例清单（比 README 的「十余个」更全，按用途分组）：

| 类别 | 示例 |
|---|---|
| 基础 GEMM | `00_hytlass_basic_gemm` |
| Split-K | `01_hytlass_serial_splitk_gemm`、`02_hytlass_parallel_splitk_gemm` |
| Stream-K | `03_hytlass_streamk_gemm`、`06_hute_streamk`（HuTe 模型） |
| Batch / Group | `04_hytlass_batch_gemm`、`05_hytlass_group_gemm`、`07_hute_batch_gemm`、`15_hute_group_gemm` |
| 融合 | `08_hytlass_fused_gemm`、`16_hytlass_gemm_softmax`、`18_gemm_with_abs_max` |
| 卷积 | `09_hytlass_tensorop_conv2d`、`10_hytlass_tensorop_wgrad_split_k`、`11_hytlass_tensorop_group_conv`、`12_depthwise_simt_conv2dfprop`、`13_hytlass_tensorop_fused_conv2d_fprop` |
| 稀疏 / 特殊 | `14_gather_scatter_fusion`、`17_ell_block_sparse_gemm` |

**何时选它**：想自己搭一个 GEMM（自定义 tile / epilogue / 调度），又不想从零写 HIP。它给了 Split-K / Stream-K / Group / 融合 epilogue 的现成骨架。
**何时别选**：只要一个开箱即用的 GEMM（用 deepgemm / hipBLASLt）；或需要的 dtype/布局不在 TF32/FP16/BF16/I8/U8 之内。
**边界**：README 明说示例「不代表 best practice」，是骨架不是调优结果。需 DTK 25.10+、CMake 3.19+、C++17。

### Moe（moe_c）—— AITER 的 MoE 后端本体

```
csrc/                    per-precision 入口
csrc_for_aiter/936_938/  gfx936/938 头文件、configs、intrinsics
moe_c_configs/           按 gfx × 量化类型组织（运行时配置选择）
op_test/                 从 AITER MOE CI 迁来的正确性测试
tune/                    moe_c 调优脚本
```

精度矩阵：W4A8(channelwise) / W8A8 INT8(channelwise·blockwise·tensorwise) / W8A8 FP8(同上三种) / W4A16(groupwise) / W8A16(channelwise·blockwise) / WFP4A8(channelwise·groupwise32) / WFP4A16(groupwise) / W16A16(channelwise)。

**何时选它**：AITER 的 MoE 路径慢或没有解 —— **先查 `moe_c_configs/` 里这个 (gfx, 量化类型, E/N/K) 组合是否有条目**，没有就补配置，而不是改 kernel。
**何时别选**：只要 attention/GEMM（用 lightop / deepgemm）。
**构建**：`./build.sh`；`MOE_C_TARGET=moe_w4a8` 选单模块；`MOE_C_HIP_ARCHITECTURES=gfx938`、`MOE_C_BUILD_DIR`、`MOE_C_BUILD_JOBS`。

### BoltOPs —— DSL 路线的低延迟算子

```
boltops/fa/          decode_attention / extend_attention / triangle_attention / unified_attention
                     （全 Triton；unified 含 int4_per_token_head）
boltops/fla/         GDN 等线性注意力：cumsum / fused_sigmoid_gating_recurrent / gdn_verify
boltops/gemm/        matmul_persistent / gemm_a16w16 / w8a8_triton_block_scaled_mm /
                     triton_scaled_mm / awq_gemm_triton
boltops/fused_moe/   fused_experts / moe_activation
boltops/generic/     attn_res / causal_conv1d_fn / causal_conv1d_update /
                     fused_mamba_state_scatter_with_mask
```

**何时选它**：形状不定或数据相关的算子（gating、routing、causal conv1d、状态 scatter），以及需要快速试一个 Triton 实现时。
**何时别选**：把 `tl.dot` 当 GEMM 引擎 —— DCU 上它在 M 64–256 只有参考实现的 0.29–0.41×。
**自带调优**：`boltops-fused-moe-tune-cli <device_name> <test_type> [--benchmark]`。

### TileKernels-das —— mHC 的官方实现

```
tile_kernels/mhc/    expand / hc_split_sinkhorn / head_compute_mix / multilayer_recompute /
                     norm_fn / post / pre_apply_mix / pre_big_fuse
tests/mhc/           含 bench_mhc_pre_big_fuse.py
```

**何时选它**：要 mHC / Engram / gating / MoE routing 的 tilelang 实现，或要一个可训练层封装（`torch.autograd.Function`）。
**何时别选**：需要手写 HIP 级别的控制，或 dtype/形状不在其覆盖内。
**要求**：Python 3.10+、PyTorch 2.10+、TileLang 1.12.0+、BW1000/BW1100/BW150/K100_AI。

## 三、lightop 自带的 agent skill（重要：不要重复造）

`lightop` 仓库根下 `.agents/skills/` 有两个 skill：

| 路径 | 内容 |
|---|---|
| `lightop-kernel-agent-loop/scripts/measure-device-bandwidth.sh` | **带宽测量**（read/copy/write，扫 buffer 尺寸，含 docker 模式） |
| `lightop-kernel-agent-loop/references/hip-optimization-rules.md` | **按证据选方法**的 HIP 优化规则表 |
| `lightop-kernel-agent-loop/assets/hcu-profiler-report/scripts/hcu_profile_digest.py` | profiler 摘要 |
| `lightop-kernel-agent-loop/assets/lightop-kernel-knowledge/` | 知识库（`query.py` + `index.jsonl` + `fetch_evidence.py`） |
| `build-hcu-wheel/SKILL.md` | 多架构 wheel 构建 |

**纪律**：仓内 `scripts/probe_stream.py` 与本仓的带宽脚本是同一口径的两套实现。引用带宽数字时必须写明是哪一套、什么 buffer 尺寸。**同一个数两处算必然漂**（本仓已有一次 87%/55% 的事故记录）。

## 四、仓库能力分布（选型时的第二问）

| 需要的算子 | 首选 | 备选 |
|---|---|---|
| 稠密 GEMM（大 M） | deepgemm-das / hipBLASLt | hytlass（自己搭） |
| 稠密 GEMM（小 M 解码） | lightop / 手写 ladder | BoltOPs gemm |
| Grouped / MoE GEMM | deepgemm-das 三路径 | Moe(moe_c) / lightop MoE |
| Attention（dense / paged） | flash-attention-das | BoltOPs fa / lightop |
| MLA（含稀疏） | FlashMLA-das | lightop attention |
| 线性注意力 / GDN | BoltOPs fla | — |
| norm / 激活 / 量化融合 | lightop | BoltOPs generic |
| mHC / Engram | TileKernels-das | 手写（本仓 mhc 记录） |
| 稀疏 GEMM（ELL block） | hytlass `17_ell_block_sparse_gemm` | — |
| 量化（离线） | one-click-quant | ao-das |

## 五、更新方式

```bash
python3 scripts/hygon-upstream-sync.py --check   # 对账仓内快照 vs 上游实际
```

新增库时：先跑上面的命令确认元数据，再回 `op-stack-design/references/libraries.md` 加画像（结构定律 / 可调面 / 已知遗留税 / 何时选它）。
