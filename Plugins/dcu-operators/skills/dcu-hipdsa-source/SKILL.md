---
name: dcu-hipdsa-source
description: hip-dsa / hip-dsa-kernels / hipdnn DCU 适配——cudnn.DSA（DeepSeek 稀疏注意力）HIP 替代（14 个 .hip kernel 源码在 wheel 内）+ hipdnn PReLU
version: 1.0.0
author: DCU Engineer
license: MIT
kind: knowledge
---

# hip-dsa / hipdnn DCU 适配源码索引（dcu-hipdsa-source）

> 定位：**hip-dsa 是 cudnn.DSA（DeepSeek Sparse Attention）的 HIP 替代**——提供与 `cudnn.deepseek_sparse_attention` 相同的公开 API，让现有 `dsa_kernels.py` 集成在 DCU 上不改代码。本 skill 讲 hip-dsa 的 kernel 源码结构、wrapper API、hipdnn 小包。
> 与 `dcu-flashattn-source` 的关系：flash-attn 是**稠密** attention 的 DCU 适配；hip-dsa 是**稀疏** attention（DeepSeek-V3.2/V4 的 DSA indexer + 稀疏 KV 选择）的 DCU 适配。
> 实测来源：`hip_dsa` / `hip_dsa_kernels`（customer/dtk2604-rc4 源，**rc4 独有**，release 主索引无）+ `hipdnn`（release/dtk2604 源）wheel 解包 + ldd 实测（2026-09-01）。

## §0 定位：hip-dsa 是什么

cudnn.DSA = NVIDIA cuDNN 的 **DeepSeek Sparse Attention** 算子集（indexer 前向/反向 + 稀疏注意力前向/反向 + topk 选择 + score 重算）。海光把这套 CUDA kernel 移植到 HIP，包名 `hip_dsa`，**API 与 cudnn.DSA 完全对齐**（`from hip_dsa import DSA` 后 `DSA.sparse_attention_backward_wrapper(...)` 等调用不变）。

**关键设计**（`wrappers.py` 注释）：
- 每个算子由**单个融合 HIP kernel** 服务（`hip_dsa_kernels` 扩展），**无纯 PyTorch fallback**——扩展缺失或 shape 不支持直接 raise，不静默降级。
- `sparse_attn_forward` 是 **FlashMLA forward 的 HIP 替代**。
- `q_causal_offsets` 非零时**显式报错**（`_check_q_causal_offsets`）：cuDNN kernel 用 per-segment causal limit `(q_causal_offset + q + 1) // ratio`，HIP kernel 只从 packed sequence position 推导（仅 offset 全零时精确）——DSv4 context-parallel ratio-4 indexer 的非零 offset 会静默产生错误 causal mask，故 fail loudly。

## §1 源码地图（wheel 实测目录）

```
hip_dsa/
├── __init__.py              # DSA = DSANamespace()；导出 sparse_attn_forward / sparse_attn_fwd_indexer
├── wrappers.py              # ★ DSANamespace + 各算子 wrapper（cudnn.DSA 同 API）
├── reference.py             # 参考实现（对拍用）
└── kernels/                 # ★★ 14 个 .hip/.h 源码（wheel 内带，可看可改）
    ├── sparse_attn_fused.hip / sparse_attn_fused.h / sparse_attn_fused_hip.hip   # 稀疏注意力融合
    ├── sparse_attn_fwd_indexer.hip / sparse_attn_fwd_indexer_hip.hip             # 前向 indexer
    ├── indexer_fwd.hip / indexer_bwd.hip / indexer_topk.hip                      # indexer 前/反向/topk
    ├── csa_compressor.hip / csa_compressor.h                                     # CSA 压缩
    ├── dense_mmac.hip / dense_mmac.h / dense_mmac_hip.hip                        # 稠密 MMAC
    ├── sparse_mmac.hip / sparse_mmac.h / sparse_mmac_hip.hip                     # 稀疏 MMAC
    ├── compactify.hip                                                           # 紧凑化
    ├── score_recompute.hip                                                      # score 重算
    ├── hip_dsa_kernels.h / thd_utils.h                                          # 头文件
hip_dsa_kernels.cpython-312-x86_64-linux-gnu.so   # ★ 编译产物（链接 libgalaxyhip.so.5）
```

> **14 个 .hip 源码在 wheel 内**（区别于 vLLM 0.21+ wheel 无 csrc）——可直接看 kernel 实现、改后重编。

## §2 wrapper API（wrappers.py，cudnn.DSA 同 API）

| 入口 | 作用 |
|---|---|
| `DSA = DSANamespace()`（`__init__.py`） | ★ 命名空间对象（对齐 `cudnn.DSA`） |
| `DSA.sparse_attention_backward_wrapper(q, kv, ...)` | 稀疏注意力反向 |
| `DSA.indexer_forward_wrapper(q, k, w, ratio=4)` | indexer 前向（ratio=4 对应 DSv4） |
| `sparse_attn_forward`（模块级导出） | ★ FlashMLA forward 的 HIP 替代 |
| `sparse_attn_fwd_indexer`（模块级导出） | 前向 indexer |
| `_check_q_causal_offsets` | q_causal_offsets 非零校验（fail loudly） |

> 集成方式：现有 `dsa_kernels.py`（cuDNN 版）把 `import cudnn; cudnn.DSA` 换成 `from hip_dsa import DSA` 即可，调用不变。

## §3 hip-dsa-kernels（独立小包）

`hip_dsa_kernels` 是**纯 .so 包**（无 Python 源码，只有 `hip_dsa_kernels.cpython-312-x86_64-linux-gnu.so`）——`hip_dsa` 的 kernel 扩展可独立分发/升级。`hip_dsa/wrappers.py` 的 `_load` 逻辑会找 `hip_dsa_kernels` 扩展（glob 加载）。

> 两个包装法：`hip_dsa`（Python API + kernel 源码）+ `hip_dsa_kernels`（纯 .so）。装 `hip_dsa` 时通常连带 `hip_dsa_kernels`。

## §4 hipdnn（小包，release 主索引有）

```
hipdnn/
├── __init__.py              # import torch_wrappper.torch_prelude + hipdnn_native
└── torch_wrappper/
    └── torch_prelu.py       # ★ TorchPReLU（PyTorch PReLU 封装）
hipdnn_native.cpython-311-x86_64-linux-gnu.so   # 编译产物（链接 libhipdnn_backend.so + libgalaxyhip.so.5）
```

- hipdnn = 海光自研小库，当前只暴露 **TorchPReLU**（PyTorch PReLU 的 DCU 加速版）+ `hipdnn_native` 扩展。
- `hipdnn_native.so` 链接 **`libhipdnn_backend.so`**（DTK 的 hipdnn 后端库）+ `libgalaxyhip.so.5`。
- 版本 `1.0.1+dtk2604`（cp311，注意 Python 版本约束）。

## §5 DTK 链接（ldd 实测）

| .so | DTK 库链接 |
|---|---|
| `hip_dsa_kernels.so`（hip_dsa 包内） | **`libgalaxyhip.so.5`** + torch HIP（`libtorch_hip.so`/`libc10_hip.so`） |
| `hip_dsa_kernels.so`（hip_dsa_kernels 包） | 同上 |
| `hipdnn_native.so` | **`libhipdnn_backend.so`** + `libgalaxyhip.so.5` |

## §6 与框架的衔接

- **DeepSeek-V3.2/V4（DSA 模型）**：hip-dsa 是 DSA 稀疏注意力的 DCU 实现。SGLang 海光 fork 0.5.15 的 `dsa/` 目录（见 `dcu-sglang-source` 版本演进表）+ vllm-hcu 的 `vllm__v1__attention__backends__mla__{flashmla_sparse,indexer}.patch.py`（见 `dcu-vllm-hcu-source` §7）是 DSA 接入推理框架的落点。
- **FlashMLA 替代**：`sparse_attn_forward` 是 FlashMLA forward 的 HIP 替代——DCU 上 DSA 模型的稀疏 KV 选择走它。

## §7 与兄弟 skill 关系

- `dcu-flashattn-source`：稠密 attention DCU 适配 → hip-dsa 是稀疏 attention（DSA）适配，互补。
- `dcu-kernel-dev` / `dcu-cuda-hip-port`：kernel 开发/移植方法学 → hip-dsa 的 14 个 .hip 是移植样板。
- `dcu-sglang-source` / `dcu-vllm-source` / `dcu-vllm-hcu-source`：推理框架 → hip-dsa 是 DSA 模型的底层算子。
- `dcu-op-optimization`：优化闭环。

## §8 验证入口

```bash
# 安装（rc4 仓库，release 主索引没有 hip-dsa）
pip install hip-dsa hip-dsa-kernels -i https://pypi.sourcefind.cn/customer/dtk2604-rc4/
# hipdnn 在 release 主索引
pip install hipdnn -i https://pypi.sourcefind.cn/release/dtk2604/

# 确认 DTK 链接（DCU 容器内）
ldd $(python3 -c "import hip_dsa,os;print(os.path.dirname(hip_dsa.__file__))")/hip_dsa_kernels*.so | grep galaxyhip

# DSA API（cudnn.DSA 同 API）
python3 -c "from hip_dsa import DSA; print('DSA OK:', hasattr(DSA, 'indexer_forward_wrapper'))"

# hipdnn PReLU
python3 -c "from hipdnn.torch_wrappper.torch_prelu import TorchPReLU; print('hipdnn OK')"
```

## 使用须知

- **hip-dsa 是 cudnn.DSA 的 HIP 替代**，API 对齐——集成时把 `cudnn.DSA` 换成 `from hip_dsa import DSA` 即可。
- **无纯 PyTorch fallback**：扩展缺失/shape 不支持直接 raise（不静默降级）——排查"算子不可用"先看扩展是否加载。
- **q_causal_offsets 非零会报错**：DSv4 context-parallel ratio-4 的非零 offset 在 HIP kernel 上不支持（会静默错），fail loudly 是设计行为。
- **14 个 .hip 源码在 wheel 内**：可直接看/改 kernel（改后重编 .so）。
- **hip-dsa 是 rc4 独有**（customer/dtk2604-rc4），hipdnn 在 release 主索引——装包注意仓库。
- **torch 标记**：hip-dsa 绑 torch290，hipdnn 是 cp311（Python 3.11）。
- 不存性能数据/基线；数字用 `dcu-op-optimization` 模板留痕。
- 高危操作谨慎，优先测试机验证。
