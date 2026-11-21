---
name: dcu-training-frameworks-source
description: DCU 训练框架适配合集——deepspeed（wheel 带 14 个 .hip 源码）+ megatron-core + dcu-megatron（patch 适配层）+ apex + transformer-engine
version: 1.0.0
author: DCU Engineer
license: MIT
---

# DCU 训练框架适配合集索引（dcu-training-frameworks-source）

> ## 🔗 开源上游锚点（Megatron-LM-das）
> 官方开源版 **Megatron-LM-das**（`github.com/HYGON-AI/Megatron-LM-das`，即 **hcu-megatron**）基于上游 **core_r0.18.0 / core_v0.18.2**（commit `571370c`）；
> 要求 `dtk >= 25.04`、`transformer-engine >= 2.4.0`、`torch >= 2.6.0`；能力覆盖 LLM 预训练 / LLM SFT / VLM SFT（文档 `docs/getting-started.md`）。
> 本 skill §3 记的是内网 `dcu_megatron 0.15.0+das.opt1.dtk2604`（海光自研 patch 层、无上游）——**迁移/对齐前先核对两者 commit 关系**；RL 强化学习训练（verl/slime/AReaL 等上游 HCU 套件）见 `training/dcu-train-cookbook`。
> 相关开源：apex-das（PyTorch Apex 扩展，混合精度/分布式/HCU HIP kernel 融合）；**pytorch-das**（上游 PyTorch 二次开发：HCU 编译/测试/CI 全量单测，官方 PyTorch 下游主线锚点）；**TransformerEngine-das**（上游 TE 下游适配，与 Megatron-LM-das 的 `transformer-engine >= 2.4.0` 依赖对齐）；flash-train（训练算子库）；TileKernels-das（DeepSeek-AI/TileKernels 下游，用 `tilelang-hygon` 构建，见 `operators/dcu-tilelang-source`）。
> **TE 部署铁三角（2026-10-14 实测 `TE_GPU_PASS`）**：TE **2.13.0**（`dtk2604.torch2100`）+ torch **2.10.0** + flash-attn **2.8.4**（torch2100 构建）组合跑通；TE 2.14.1 要更新 DTK-26.10 的 `GemmEpilogue` V1 构造器符号，常规容器跑不了。torch 升降级后 TE/flash-attn 必须同步换 torch 标记构建。详见 `env/dcu-software-adaptation-guide` §8。
> 刷新：`python3 scripts/hygon-upstream-sync.py --check`（快照 `references/hygon-upstream/upstream-snapshot.json`，索引 `repos-index.md`）。

> 定位：**五个训练侧框架包的 DCU 适配合集**——deepspeed / megatron-core / dcu-megatron / apex / transformer-engine。本 skill 讲各包在 DCU 上的结构、适配形态、patch 机制。
> 与 `dcu-train-precision` 的关系：那个讲**训练精度排查**（NaN/逐层比对）；本 skill 讲**训练框架在 DCU 上的包结构与适配落点**。
> 实测来源：五个 wheel（release/dtk2604 源）解包 + ldd 实测（2026-09-01）。

## §0 五包总览

| 包 | 版本（release/dtk2604 最新） | 上游 | DCU 适配形态 | 源码在 wheel 内？ |
|---|---|---|---|---|
| **deepspeed** | 0.18.2+das.opt1.dtk2604.torch290 | microsoft/DeepSpeed | HIP kernel 移植（inference/v2）+ op_builder | ✅ **14 个 .hip + 86 .cpp + 145 .h** |
| **megatron-core** | 0.15.4+das.opt1.dtk2604.torch290 | NVIDIA/Megatron-LM core | 纯 Python（内容级适配，hip 提及仅 1 处） | ✅ 344 .py（无 kernel） |
| **dcu-megatron** | 0.15.0+das.opt1.dtk2604.torch290 | 海光自研（无上游） | **patch 适配层**（adaptor/patch_utils.py）+ op_builder | ✅ 134 .py + 1 .cpp + 1 .so |
| **apex** | 1.7.0+das.opt1.dtk2604.torch290 | NVIDIA/apex | 27 个 .so（fused kernel HIP 移植） | ❌ 仅 .so（152 .py 是 Python 侧） |
| **transformer-engine** | 2.10.0+das.opt1.dtk2604.torch271 | NVIDIA/TransformerEngine | 2 个 .so（TE 核心 + torch 绑定）+ 44 .h/28 .cpp | 部分（.h/.cpp 头文件在，.so 编译产物） |

> 五包 .so 均链接 **`libgalaxyhip.so.5`**（DTK）+ torch HIP。TE 额外链 `librccl.so.1` + `libhipblaslt.so.0` + `libhipblas.so.2` + `librocblas.so.4`。

## §1 deepspeed（wheel 带完整 HIP 源码）

**结构**（1034 文件，wheel 内带 C++/HIP 源码——区别于 apex/TE 的纯 .so）：
```
deepspeed/
├── ops/                          # 训练算子（adam/lamb/lion/adagrad/quantizer/sparse_attention/spatial/transformer/aio/gds）
│   ├── op_builder/               # ★ 算子构建器（builder.py/async_io.py/gds.py，HIP 编译路径）
│   ├── dc_op.so / utils_op.so / random_ltd_op.so   # 编译产物
│   └── csrc/                     # C++ 源码
├── inference/v2/                 # ★ 推理 v2（MoE 推理引擎）
│   └── kernels/
│       ├── ragged_ops/           # ★ 14 个 .hip（top_k_gating/moe_scatter/moe_gather/logits_gather/blocked_kv_rotary/embed）
│       ├── cutlass_ops/          # moe_gemm.hip / mixed_gemm.hip
│       └── core_ops/             # hip_rms_norm/hip_linear/hip_layer_norm/gated_activations/bias_activations（.hip）
├── runtime/zero/                 # ZeRO 优化器（stage_1_and_2/offload_config/muon）
└── ...
```

**DCU 适配点**：
- **inference/v2 kernels 14 个 .hip**：MoE 推理的 ragged ops（top_k_gating/moe_scatter/moe_gather）+ cutlass GEMM（moe_gemm/mixed_gemm）+ core ops（rms_norm/linear/layer_norm/activations）——CUDA 同名文件移植 HIP（`*_cuda.hip` 命名保留）。
- **op_builder**：`ops/op_builder/builder.py` 的 HIP 编译路径（`hip|dtk|gfx9` 提及）——训练算子（fused adam 等）在 DCU 上经 op_builder 编译。
- **ZeRO**：`runtime/zero/stage_1_and_2.py` / `offload_config.py` 有 hip 分支（offload 路径 DCU 适配）。

> deepspeed 是五包里**源码最全**的（.hip/.cpp/.h 都在 wheel 内）——改 MoE 推理 kernel 可直接看 `inference/v2/kernels/`。

## §2 megatron-core（纯 Python，内容级适配）

**结构**（365 文件，`megatron/core/` 359 文件）：
- 纯 Python 包（344 .py），**无 kernel 源码**（kernel 在 apex/TE 里）。
- DCU 适配是**内容级**（hip 提及仅 1 处：`core/optimizer/distrib_optimizer.py:122`，且是注释误命中"reducing"）——即 megatron-core 本体几乎无 DCU 专属代码，适配靠**配套的 dcu-megatron patch 层**（§3）。

> megatron-core 本身是上游 NVIDIA Megatron-LM core 的 DCU 构建（torch HIP 版），DCU 特化在 dcu-megatron。

## §3 dcu-megatron（海光 patch 适配层，★ 重点）

**dcu-megatron 是海光自研的 Megatron DCU 适配层**（无上游），形态同 vllm-hcu（patch 机制）：

```
dcu_megatron/
├── adaptor/
│   ├── megatron_adaptor.py       # ★ 适配入口（add_args/parser_unknown_args/get_adaptor_args）
│   └── patch_utils.py            # ★★ Patch 类 + MegatronPatchesManager（patch 引擎）
├── training/                     # 训练入口（arguments/checkpoint/initialize/training/utils/edgc_utils）
├── optimizer/distrib_optimizer.py
├── op_builder/                   # ★ 算子构建（algorithm_builder.py/builder.py）
├── ops/
├── core/                         # parallel_state/quantization_utils/utils
├── miscellaneous/gpt_builders.py
└── legacy/
```

**patch 机制**（`adaptor/patch_utils.py`）：
| 类/函数 | 作用 |
|---|---|
| `class Patch` | ★ 单条 patch（`orig_func_or_cls_name` → `new_func_or_cls`，支持 `create_dummy`/`apply_wrapper`/`remove_origin_wrappers`） |
| `MegatronPatchesManager` | patch 管理器（批量应用/回滚） |
| `dummy_function_wrapper` | 原函数不存在时生成 dummy（raise RuntimeError） |
| `get_func_name` | 函数全名（`module.qualname`） |

**适配入口**（`adaptor/megatron_adaptor.py`）：
- `add_args(args, key, value)`：给 megatron args 加 DCU 参数。
- `parser_unknown_args`：解析未知参数（DCU 专属 CLI 参数）。
- `get_adaptor_args()`：取 adaptor 参数（`process_adaptor_args`）。
- `is_te_min_version`（import 自 megatron.core.utils）：TE 版本检查。

> **dcu-megatron 与 vllm-hcu 同范式**：都是海光「不改上游源码、用 patch 层适配」的产物。区别：vllm-hcu 是**字符串 patch + 模块交换**（运行时），dcu-megatron 是**函数/类 patch**（`Patch` 类替换函数引用）。

## §4 apex（27 个 .so，fused kernel HIP 移植）

**结构**（185 文件）：
```
apex/
├── amp/ / fp16_utils/ / normalization/ / optimizers/ / parallel/ / transformer/ / contrib/
├── fused_dense/ / mlp/ / multi_tensor_apply/ / RNN/
└── version.py
# 顶层 27 个 .so（编译产物，无 C++ 源码在 wheel 内）：
# amp_C / apex_C / bnp / distributed_adam_cuda / distributed_lamb_cuda / fast_multihead_attn /
# focal_loss_cuda / fused_adam_cuda / fused_bias_swiglu / fused_dense_cuda / fused_index_mul_2d /
# fused_lamb_cuda / fused_layer_norm_cuda / ...
```

- apex 的 fused kernel（fused_adam/fused_layer_norm/fused_dense/fast_multihead_attn 等）全部 HIP 移植，**27 个 .so 链接 `libgalaxyhip.so.5`** + torch HIP。
- **C++ 源码不在 wheel 内**（仅 .so）——看 kernel 实现需拉 apex 上游 + DTK 构建。
- Python 侧 152 .py（amp/fp16_utils/normalization 等）是上游代码。

## §5 transformer-engine（TE 核心 + torch 绑定）

**结构**（246 文件）：
```
transformer_engine/
├── pytorch/                      # ★ PyTorch 集成（135 文件：module/attention/quantization/cpp_extensions/optimizers）
├── jax/                          # Jax 集成（34 文件）
├── common/                       # 公共（51 文件，含 44 .h/28 .cpp 头文件）
├── debug/
├── libtransformer_engine.so      # ★ TE 核心（链接 libgalaxyhip + librccl + libhipblaslt + libhipblas + librocblas）
└── transformer_engine_torch.cpython-312-x86_64-linux-gnu.so   # ★ torch 绑定
```

- TE 的 GEMM/attention/quantization（FP8）在 DCU 上走 HIP 实现（`libtransformer_engine.so` 链接 DTK 全套：galaxyhip + rccl + hipblaslt + hipblas + rocblas）。
- **pytorch/cpp_extensions/gemm.py** 等 17 个 py 文件有 hip/dtk 分支（DCU 适配点）。
- 头文件（44 .h/28 .cpp）在 wheel 内（common/），但 .so 是编译产物。

## §6 DTK 链接（ldd 实测汇总）

| 包 | .so | DTK 库链接 |
|---|---|---|
| deepspeed | `ops/dc_op.so` 等 | `libgalaxyhip.so.5` + torch HIP |
| apex | 27 个 .so | `libgalaxyhip.so.5` + torch HIP |
| TE | `libtransformer_engine.so` | **`libgalaxyhip.so.5` + `librccl.so.1` + `libhipblaslt.so.0` + `libhipblas.so.2` + `librocblas.so.4`** |
| TE | `transformer_engine_torch.so` | `libgalaxyhip.so.5` + torch HIP |
| dcu-megatron | 1 .so | `libgalaxyhip.so.5` |

> TE 是五包里**链接 DTK 库最全**的（rccl + hipblaslt + hipblas + rocblas 全链）——FP8 GEMM 走 hipblaslt，集合通信走 rccl。

## §7 与兄弟 skill 关系

- `dcu-train-precision`：训练精度排查（NaN/逐层比对/NV 对齐）→ 本 skill 是训练框架的包结构/适配落点。
- `dcu-kernel-dev` / `dcu-cuda-hip-port`：kernel 开发/移植 → deepspeed 的 14 个 .hip 是移植样板。
- `dcu-vllm-hcu-source`：vllm-hcu patch 层 → dcu-megatron 是训练侧同范式（patch 适配）。
- `dcu-dtk` / `dcu-env-packages`：DTK 安装/包清单 → 五包 .so 的 DTK 库依赖。

## §8 验证入口

```bash
# 安装（release/dtk2604，torch 标记对齐）
pip install deepspeed megatron-core dcu-megatron apex -i https://pypi.sourcefind.cn/release/dtk2604/
# TE 注意 torch 标记（2.10.0 绑 torch271，2.7.0 绑 torch251）
pip install transformer-engine -i https://pypi.sourcefind.cn/release/dtk2604/

# 确认 DTK 链接（DCU 容器内）
ldd $(python3 -c "import apex,os;print(os.path.dirname(apex.__file__))")/../fused_adam_cuda*.so | grep galaxyhip

# deepspeed HIP 源码位置
python3 -c "import deepspeed,os;print(os.path.join(os.path.dirname(deepspeed.__file__),'inference/v2/kernels'))"

# dcu-megatron patch 机制
python3 -c "from dcu_megatron.adaptor.patch_utils import Patch, MegatronPatchesManager; print('dcu-megatron OK')"
```

## 使用须知

- **deepspeed 源码最全**（14 .hip + 86 .cpp 在 wheel 内）——改 MoE 推理 kernel 看 `inference/v2/kernels/`。
- **dcu-megatron 是海光 patch 适配层**（同 vllm-hcu 范式）——Megatron DCU 特化在 dcu-megatron，不在 megatron-core 本体。
- **apex/TE 无 C++ 源码在 wheel 内**（仅 .so）——看 kernel 实现需拉上游 + DTK 构建。
- **TE 链接 DTK 库最全**（rccl + hipblaslt + hipblas + rocblas）——FP8 GEMM 走 hipblaslt。
- **torch 标记**：deepspeed/megatron-core/dcu-megatron/apex 绑 torch290，TE 2.10.0 绑 torch271（2.7.0 绑 torch251）——装前对齐。
- 五包 .so 均链 `libgalaxyhip.so.5`（DTK 主 HIP 库）。
- 不存性能数据/基线；数字用 `dcu-op-optimization` 模板留痕。
- 高危操作谨慎，优先测试机验证。
