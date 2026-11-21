---
name: dcu-lightop-fusion
description: Use when writing or understanding LightOp fusion kernels.
version: 1.0.0
author: DCU Engineer
license: MIT
---

# LightOp 算子融合方法论（dcu-lightop-fusion）

> 本 skill 是**融合方法论**专题，与 `dcu-lightop-source`（源码地图）、`dcu-lightop-moe`（MoE 专题）互补。
> 源码基准：`<local_src_cache>/lightop-master/csrc/` 30+ 融合 kernel，其中 8 个代表 kernel 完整读过（dropout_add / bias_swiglu / silu_mul_quant / moe_swiglu_dynamic_quant / layernorm_rope / groupgemm_permute_quant / DSV4 专用 kernel / export.cpp 注册面）。
> 写新融合 kernel 或理解既有融合实现时，先加载本 skill。

## §0 定位：融合是什么、不是什么

LightOp 的融合**只打 memory-bound 的算子链**（norm / elementwise / 激活 / 量化 / rope），不碰 compute-bound 的 GEMM 主体。
收益来源单一且明确：**消除 HBM 往返**。torch native 每个算子一次读一次写；融合后中间态留在寄存器/smem，HBM 只读一次写一次。README 实测 norm 类 2-5× 就是这么来的。

**判断标准（写融合 kernel 第一步）**：目标算子链是不是 memory-bound？不是（如大 GEMM、attention matmul）就别融合——融合无益甚至有害。

## §1 六大融合模式族（源码实证）

| 模式族 | 代表 kernel | 融合内容 | 方向 |
|---|---|---|---|
| elementwise 链 | `bias_swiglu.cu`、`fuse_bias_gelu.cu`、`fuse_linear_bias.cu` | bias+gelu / relu+dropout / linear+bias | 推理 |
| norm+residual+dropout | `fuse_ln_add_dropout_bias`、`fuse_rn_add_dropout`（一整套） | residual add → norm → dropout 一个 kernel | 训练（带 autograd） |
| norm+RoPE | `fuse_layernorm_rope.cu`、`fuse_rms_roped.cu`、`fuse_rms_mrope_cuda.cu` | norm + rope 共用 smem head 数据 | 推理 |
| 激活+动态量化 | `fuse_silu_mul_quant.cu`、`moe_swiglu_dynamic_quant.cu`、`fuse_silu_mul_fp8_quant.cu` | silu*mul 结果直接量化 int8/fp8，中间 bf16 从不落 HBM | 推理（MoE 核心路径） |
| GEMM epilogue | `gemm_bias.h`、`fused_linear_bias`、`fuse_groupgemm_permute_quant.cu` | GEMM+bias / group GEMM+permute+quant 一个 kernel | 推理 |
| attention 特化大融合 | `fused_deepseek_v4_qnorm_rope_kv_insert_kernel.cu` | per-head RMSNorm + RoPE + UE8M0 FP8 量化 + paged KV cache 写入，一个 kernel | 推理（DSV4 专用） |

**最激进例子（DSV4）**：Q 和 KV 用 **block-specialized grid** 区分（`block_id < total_q_blocks` 走 Q 逻辑，否则走 KV 逻辑），一次 launch 处理两类异构工作，省 launch 开销。

## §2 八大核心手法（代码里反复出现的套路）

1. **向量化访存**：`aligned_vector<scalar_t, VEC>`（VEC=4/8）+ reinterpret_cast 成 LoadT，128/256-bit 对齐读写，所有 kernel 无例外。
2. **单 kernel 内两遍扫描（动态量化的灵魂）**：第一遍算 row_max（BlockReduceMax：warp shuffle + smem 两级归约），第二遍用 inv_s 量化。**关键：第一遍的输入值留在寄存器数组里（r_x1/r_x2/r_y），第二遍直接用，HBM 只读一次。** 这是"激活+量化"融合能成立的前提。
3. **编译期分发代替运行时分支**：act_first（silu(x)*y 还是 x*silu(y)）、HAS_RESIDUAL、IS_NEOX 全是 template bool + DISPATCH_BOOL 宏，constexpr if 把死分支消掉。
4. **block 映射按语义切**：attention 类 kernel 一个 block = 一个 token 的 HEADS_PER_BLOCK 个 head（2D block：threadIdx.x 管 head 内维度，threadIdx.y 管 head 编号），且 HEADS_PER_BLOCK 自适应（token>=1024 时 4→8）。
5. **smem 做流水线中转**：cos/sin cache 每 block 只从 HBM 搬一次；head 数据 norm 完留在 smem，rope 直接读 smem，最后才写回。
6. **DCU 平台适配宏（跨 CUDA/HIP 三件套）**：
   - `VLLM_LDG`：CUDA 用 `__ldg`，ROCm 退化成普通读
   - exp 用 `__builtin_amdgcn_exp2f(-x*LOG2E)` 代替 expf（DCU 快速路径）
   - `float_to_int8_rn`：HIP 用 nearbyint + 手动 saturate（绕开 hip-clang 的 std::clamp 编译 bug，注释引 llvm issue 95183），CUDA 用 `cvt.rni.sat.s8.f32` PTX
7. **grid-stride + 提前退出**：`for(token_idx; token_idx < num_local_tokens; token_idx += gridDim.x)`，配合 num_local_tokens_tensor 动态截断，处理 MoE 里 token 数不定的情况。
8. **没有原生 bf16 atomic 就手写**：`atomic_add_bf16` 用 atomicCAS 循环 + 拆两个 uint16 转 float 加完再转回。

## §3 工程约定

- 命名空间用 `at::native`（伪装成 torch 原生算子，方便框架侧直接替换）
- 所有 pybind 入口集中在 `csrc/export.cpp`，Python 侧 `python/lightop/` 一个算子一个 .py
- 训练向算子必须配 `torch::autograd::Function`（forward 存中间态如 mask，backward 独立 kernel）
- 推理向算子（MoE 量化、DSV4）不包 autograd，裸函数
- 测试在 `test/` 下按算子分目录，`run_moe_tests.py` 批量跑

## §4 可迁移 checklist（写自己融合 kernel 时）

1. 先判断目标算子链是不是 memory-bound，不是就别融合
2. 中间态必须留在寄存器/smem，落 HBM 一次融合就白做
3. 需要"先归约再计算"（动态量化、norm）就用单 kernel 两遍扫描 + 寄存器缓存
4. 异构工作（Q/KV、不同专家）优先 block-specialized grid 合一次 launch，省 launch 开销
5. 所有可选行为（residual、neox、act_first）做成 template bool，不做运行时 if
6. 跨平台差异收敛到少数宏（VLLM_LDG / 量化转换 / exp），kernel 主体不写 #ifdef
7. 向量化宽度 VEC 是模板参数，按 head_dim 整除性选 4 或 8
8. 训练场景记得 backward 也要融合（bias_grad 用单独的 reduce kernel 收尾即可）

## §5 源码落点（去哪看）

- 源码根：`<local_src_cache>/lightop-master/`
- 融合 kernel 目录：`csrc/fuse_*.cu`、`csrc/fused_*.cu`、`csrc/bias_swiglu.cu`、`csrc/moe_swiglu_dynamic_quant.cu`
- pybind 注册面：`csrc/export.cpp`（所有融合算子入口）
- Python 接口：`python/lightop/fuse_*.py`（一个算子一个文件）
- 源码地图 / 构建 / DTK 矩阵：见 `dcu-lightop-source`
- MoE 多量化路径：见 `dcu-lightop-moe`

## §6 何时选它 / 何时别选

- **选它**：目标算子链是 memory-bound（norm / 激活 / 量化 / rope / 采样 / 视觉预处理 / loss），且中间态当前要落 HBM 多次——融合的收益只有「省往返」这一项，能省才做。
- **选它**：融合后 kernel **个数减少而并行度不变**（各段独立、可 block-specialized grid 合成一次 launch，如 Q/KV 异构工作）。
- **选它**：要写训练向算子（residual+norm+dropout 带 backward），库里已有 autograd.Function 的成套样板。
- **别选（结构性吃亏）**：主体是 compute-bound 的 GEMM / attention matmul——融合不省时间还多占 smem 与寄存器，净负。
- **别选（结构性吃亏）**：把**独立、可并行**的 kernel 串成一条依赖链的「大融合」。链条变长后延迟由最长段决定、smem 峰值累加——本项目实测一例这种融合比非融合组合慢 1.7×（同形态实现在 `TileKernels-das` 的 mHC 族 `pre_big_fuse`，见 `op-stack-design` 的库画像）。
- **别选**：只想拿单个算子的现成实现、没有可融的邻居——没有往返可省，融合纯增复杂度。
- **先量的东西**：融合前后的 HBM 字节数（读+写）与 launch 次数。两者都没降就不是融合问题，回去查是不是 memory-bound。

## 使用须知
- 本 skill 是方法论索引，具体 kernel 实现细节以源码为准。**先核对**：gfx 型号、DTK 版本、pytorch 版本。
- 改 kernel 后必须重编（`setup.py install`）；`PYTORCH_ROCM_ARCH` 与 gfx 强相关，**换 gfx 必须重编**。
- 开发规范 / 验证闭环：`dcu-kernel-dev`（HIP 语法/编译）、`dcu-op-optimization` + `dcu-perf-profiling`（profile→定位→改→验证）。
