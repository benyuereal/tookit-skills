---
name: dcu-mhc-kernels-source
description: MHC/线性注意力 kernel 包 DCU 适配——causal-conv1d + fast-hadamard-transform + mhc-fused-kernel（Megatron MHC 融合 HIP kernel）
version: 1.0.0
author: DCU Engineer
license: MIT
---

# MHC/线性注意力 kernel 包 DCU 适配索引（dcu-mhc-kernels-source）

> 定位：**三个小 kernel 包的 DCU 适配合集**——causal-conv1d（Mamba 因果卷积）、fast-hadamard-transform（Hadamard 变换）、mhc-fused-kernel（Megatron MHC 融合 kernel）。三者都是**单 .so + 薄 Python 接口**形态，DCU 适配 = CUDA kernel 移植 HIP + 链接 DTK。
> 与 `dcu-kernel-dev` / `dcu-cuda-hip-port` 的关系：那两个讲**怎么移植/开发**；本 skill 讲**这三个包在 DCU 上的结构与接口落点**。
> 实测来源：三个 wheel 均出自 **`customer/dtk2604-rc4` 仓库**（release/dtk2604 主索引**没有**这三个包，rc4 独有），解包 + ldd 实测（2026-09-01）。

## §0 三包总览

| 包 | 版本（rc4 最新） | 上游 | 用途 | 形态 |
|---|---|---|---|---|
| **causal-conv1d** | 1.5.4+das.opt1.dtk2604.torch2100.2608041357.gda8b16 | Dao-AILab/causal-conv1d（CUDA） | Mamba/SSM 因果卷积（forward/update） | 1 .so + 4 py |
| **fast-hadamard-transform-cuda** | 0.1.0+das.opt1.dtk2604.torch290.2605291057.g71fa17 | 海光自研（无公开上游） | Hadamard 变换（12N/20N/28N/40N 多尺寸） | 1 .so + 2 py |
| **mhc-fused-kernel** | 0.1.0+das.opt1.dtk2604.torch290.2608191027.g36a382 | Megatron-LM fused_mhc_kernels（CUDA） | MHC 融合 kernel（sinkhorn/h_aggregate/h_post_bda/proj_rms） | 1 .so + 1 py（顶层） |

> 三包 .so 均链接 **`libgalaxyhip.so.5`**（DTK）+ torch HIP（`libtorch_hip.so`/`libc10_hip.so`）。torch 标记：causal-conv1d 有 torch2100/torch271 多版本，fast-hadamard/mhc 绑 torch290。

## §1 causal-conv1d（Mamba 因果卷积）

```
causal_conv1d/
├── __init__.py
├── causal_conv1d_interface.py    # ★ Python 接口（autograd.Function + 函数）
├── causal_conv1d_varlen.py       # varlen 接口
└── cpp_functions.py
causal_conv1d_cuda.cpython-310-x86_64-linux-gnu.so   # ★ HIP kernel（链接 libgalaxyhip）
```

**接口落点**（`causal_conv1d_interface.py`）：
| 函数 | 行号 | 作用 |
|---|---|---|
| `CausalConv1dFn.forward` | `:12` | autograd forward |
| `CausalConv1dFn.backward` | `:68` | autograd backward |
| `causal_conv1d_fn` | `:100` | ★ 因果卷积 forward（Mamba 用） |
| **`causal_conv1d_fn_hcu`** | `:132` | ★★ **HCU 专属接口**（海光新增，DCU 路径） |
| `causal_conv1d_ref` | `:152` | 参考实现（纯 PyTorch，对拍用） |
| `causal_conv1d_update` | `:194` | 增量 update（decode 用，conv_state 更新） |
| `causal_conv1d_update_ref` | `:225` | update 参考实现 |

> **`causal_conv1d_fn_hcu` 是海光新增的 DCU 专属入口**（上游 causal-conv1d 无此函数）——DCU 上 Mamba 模型（Jamba/Mamba2 类）的因果卷积走它。

## §2 fast-hadamard-transform-cuda（Hadamard 变换）

```
fast_hadamard_transform/
├── __init__.py
└── fast_hadamard_transform_interface.py   # ★ Python 接口
fast_hadamard_transform_cuda.cpython-312-x86_64-linux-gnu.so   # ★ HIP kernel
```

**接口落点**（`fast_hadamard_transform_interface.py`）：
| 函数 | 行号 | 作用 |
|---|---|---|
| `HadamardTransformFn.forward/backward` | `:19/:24` | autograd |
| `hadamard_transform` | `:30` | ★ 通用 Hadamard 变换 |
| `hadamard_transform_12N` | `:59` | 12N 尺寸特化 |
| `hadamard_transform_20N` | `:88` | 20N 尺寸特化 |
| `hadamard_transform_28N` | `:116` | 28N 尺寸特化 |
| `hadamard_transform_40N` | `:143` | 40N 尺寸特化 |
| `hadamard_transform_ref` | `:156` | 参考实现（对拍） |

> 多尺寸特化（12N/20N/28N/40N）对应不同 hidden dim——Hadamard 变换用于某些 attention/MLA 的 K 旋转（如 DeepSeek 系）。海光自研，无公开上游。

## §3 mhc-fused-kernel（Megatron MHC 融合 kernel）

```
fused_mhc.py                                  # ★ Python API（顶层，非包）
fused_mhc_ops.cpython-312-x86_64-linux-gnu.so # ★ HIP kernel（链接 libgalaxyhip）
```

**接口落点**（`fused_mhc.py`，匹配 Megatron-LM `fused_mhc_kernels` 公开 API）：
| 函数 | 作用 |
|---|---|
| `fused_sinkhorn` | ★ Sinkhorn-Knopp 投影（`fused_sinkhorn(logits, num_iterations=5, eps=1e-6)`） |
| `fused_h_aggregate` | ★ 加权 n-stream 聚合（`fused_h_aggregate(x_streams, h_pre)`） |
| `fused_h_post_bda` | ★ `H_res.T @ residual + H_post * (x + bias)` |
| `fused_proj_rms` | ★ 投影 + RMS norm（`fused_proj_rms(x, weight, eps=1e-6)`） |

> MHC = Megatron 的 **Multi-Head Cross/聚合** 类融合 kernel（sinkhorn 投影 + 多流聚合 + 投影 RMSNorm）。`fused_mhc.py` 自带 `LD_LIBRARY_PATH` 设置（把 torch/lib 加进去，解决 .so 加载）+ 懒加载 `fused_mhc_ops`。

## §4 与框架的衔接

- **Mamba/线性注意力模型**（Jamba/Mamba2/Qwen3.5 混合架构）：causal-conv1d 是 Mamba 块的核心算子。SGLang/vLLM 的 mamba 路径（`dcu-sglang-source` 的 mamba_mixer2 / `dcu-vllm-hcu-source` 的 `vllm__model_executor__layers__mamba__*.patch.py`）会调 causal-conv1d。
- **MLA/DeepSeek 系**：fast-hadamard-transform 用于 K 旋转；mhc-fused-kernel 用于 Megatron 训练侧 MHC 融合。
- **vllm-hcu 模块交换**：`vllm.model_executor.layers.mhc` → `vllm_hcu.model_executor.layers.mhc`（见 `dcu-vllm-hcu-source` §4）——mhc 模块被 vllm-hcu 接管，底层可能调 mhc-fused-kernel。

## §5 与兄弟 skill 关系

- `dcu-kernel-dev` / `dcu-cuda-hip-port`：kernel 开发/移植方法学 → 本 skill 是三个具体包的落点。
- `dcu-sglang-source` / `dcu-vllm-source` / `dcu-vllm-hcu-source`：推理框架 → 这三个 kernel 包是框架 mamba/MLA 路径的底层算子。
- `dcu-op-optimization`：优化闭环。

## §6 验证入口

```bash
# 安装（rc4 仓库，release 主索引没有这三个包）
pip install causal-conv1d fast-hadamard-transform-cuda mhc-fused-kernel \
  -i https://pypi.sourcefind.cn/customer/dtk2604-rc4/

# 确认 DTK 链接（DCU 容器内）
ldd $(python3 -c "import causal_conv1d,os;print(os.path.dirname(causal_conv1d.__file__))")/../*.so | grep galaxyhip

# causal-conv1d HCU 接口
python3 -c "from causal_conv1d import causal_conv1d_fn_hcu; print('HCU conv1d OK')"

# mhc fused kernel
python3 -c "from fused_mhc import fused_sinkhorn, fused_h_aggregate; print('MHC OK')"
```

## 使用须知

- **三包均 rc4 独有**：`customer/dtk2604-rc4` 仓库，release/dtk2604 主索引没有——装包用 rc4 索引（见 `dcu-das-hub-api` 跨仓库枚举）。
- **causal-conv1d 有 HCU 专属接口** `causal_conv1d_fn_hcu`（海光新增，上游无）——DCU 上 Mamba 因果卷积走它。
- **torch 标记**：causal-conv1d 有 torch2100/torch271 多版本，fast-hadamard/mhc 绑 torch290——装前对齐 torch 大版本。
- 三包都是**单 .so + 薄 Python 接口**，DCU 适配 = CUDA→HIP 移植 + 链 libgalaxyhip，无复杂源码结构。
- 不存性能数据/基线；数字用 `dcu-op-optimization` 模板留痕。
- 高危操作谨慎，优先测试机验证。
