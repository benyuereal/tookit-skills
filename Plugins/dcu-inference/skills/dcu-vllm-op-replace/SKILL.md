---
name: dcu-vllm-op-replace
description: vLLM DCU版自定义算子替换原生算子流程
version: 1.0.0
author: DCU Engineer
license: MIT
kind: knowledge
---

# 自定义算子替换 vLLM 原生算子（dcu-vllm-op-replace）

本 skill 定义**如何把一个自定义/海光优化算子替换进 vLLM、重定向原生算子调用**的标准流程。是基于 **vLLM 官方 custom_op 机制**（`torch.library` + `@register_fake`）与 **海光 0.15.1 发布版真实范例**（`csrc/opt/`、`csrc/moe/moe_fused_gate.cu`、`vllm/platforms/rocm.py` 的 `compilation_config.custom_ops` 重定向）提炼的可复用方法。

> 机制事实（实测，0.15.1 海光版 `_custom_ops.py` 与上游 v0.15.1 官方完全一致）：vLLM 的自定义算子注册机制本身**未被海光改动**，所以标准流程对海光版与上游版通用。海光 0.15.1 的"替换/重定向"主要通过 `vllm/platforms/rocm.py` 里的 `compilation_config.custom_ops` 列表（`+opname`/`-opname`）实现——这是官方支持的入口。

## 标准 6 步流程

### 第 1 步 · 写 HIP kernel（csrc/ 层）
- 海光范例：`csrc/opt/activation_kernels_opt.cu`、`csrc/opt/transpose_kernels.cu`、`csrc/moe/moe_fused_gate.cu`。
- 写法参照海光 `moe_fused_gate.cu` 头：`#ifdef USE_ROCM` 分支用 `hip/hip_bf16.h`，`typedef __hip_bfloat16 __nv_bfloat16;` 兼容 CUDA 命名；非 ROCm 走 NVIDIA 路径。
- 落点选择：已有对应算子的改 `csrc/rocm/` 下文件；海光新增优化算子放 `csrc/opt/`（海光既成目录，避免污染上游）。

### 第 2 步 · torch 绑定（torch_bindings.cpp）
- 在对应 `csrc/.../torch_bindings.cpp` 里 `m.def("op_name", ...)` + `m.impl("op_name", ...)` 绑定到你的 kernel。
- 海光范例：`csrc/moe/torch_bindings.cpp`（被海光改）、`csrc/quantization/gptq/torch_bindings.cpp`（海光新增）。

### 第 3 步 · 在 vllm/_custom_ops.py 登记（`@custom_op` + `@register_fake`）
- 海光/上游通用的注册模式（源码实测）：
  ```python
  # vllm/_custom_ops.py
  from torch.library import register_fake
  # 实际算子在 csrc 通过 torch_bindings 注册到 torch.library
  # register_fake 提供抽象实现（给 torch.compile 用，不参与实际计算）
  @register_fake("_C::your_op")
  def _your_op_fake(...): ...
  ```
- 每个自定义算子都需要 `register_fake`（否则 `torch.compile` 图捕获会失败）。这是官方强制要求，海光版未改。

### 第 4 步 · 重定向：让 vLLM 调用你的算子（关键）
两条官方支持的路径，海光版均可用：

**路径 A —— `compilation_config.custom_ops` 开关（海光 0.15.1 rocm.py 主用）**
在 `vllm/platforms/rocm.py` 的 platform 初始化里，通过 `+opname` 启用自定义实现、`-opname` 禁用：
```python
compilation_config.custom_ops.append("+rms_norm")      # 启用自定义 rms_norm
compilation_config.custom_ops.append("+quant_fp8")    # 启用自定义 fp8 量化
compilation_config.custom_ops.remove("-grouped_topk") # 移除对某算子的禁用
# 海光 0.15.1 实例：+sparse_attn_indexer（重定向到 rocm 稀疏注意力实现）
```
含义：`+X` 表示"X 走自定义/torch 实现"，`-X` 表示"X 走原生"。这是**重定向原生算子到自定义实现的标准开关**。

**路径 B —— 直接替换 Dispatch 注册**
在 `vllm/platforms/rocm.py` 的 `Dispatch` 映射里，把某算子的 dispatch key 指向你的 HIP 实现（海光版 `dispatch_key` 默认 `"CUDA"`，对 ROCm 复用 CUDA dispatch 键）。适用于需要替换底层算子实现的场景。

### 第 5 步 · 重编译（容器内，必挂 hyhal）
```bash
# 指定镜像起的专用容器内，必挂 -v /opt/hyhal:/opt/hyhal:ro
export VLLM_TARGET_DEVICE=rocm
export VLLM_USE_HYGON=1            # 海光 0.15.x 需要（0.18 不需要）
export ROCM_HOME=/opt/dtk/...     # ls /opt/dtk 为准
pip install -e .                  # 重新编译 csrc/*，使新 kernel/绑定生效
```

### 第 6 步 · 验证闭环（强制）
走 `dcu-op-optimization` 的闭环：
1. **正确性**：数值对齐（与未替换前 / CPU 参考，tolerance 按精度定）。
2. **性能**：同条件 rocprof 对比替换前后（见 `dcu-perf-profiling`），给量化加速比。
3. 端到端：起服务 + `benchmark_serving.py` 确认无回归。

## 海光 0.15.1 真实替换范例（可直接参照）
| 海光新增/改造 | 作用 | 对应步骤 |
|---|---|---|
| `csrc/opt/activation_kernels_opt.cu` | 海光优化版 activation 算子 | 1 |
| `csrc/opt/transpose_kernels.cu` | 海光优化版 transpose | 1 |
| `csrc/moe/moe_fused_gate.cu` | 海光融合 MoE gate（ROCm 分支） | 1+2 |
| `csrc/quantization/gptq/{setup.py,torch_bindings.cpp}` | GPTQ 量化海光路径 | 2 |
| `vllm/platforms/rocm.py` 中 `+rms_norm/+quant_fp8/+grouped_topk/+sparse_attn_indexer` | 重定向开关 | 4-A |
| `setup.py` 中 `VLLM_USE_HYGON` → 读 `hygon.txt` | 海光专属依赖 | 5 |

> 注：0.18.1 海光版 Python 层零差异（见 `dcu-vllm-source`），上述重定向机制在 0.18 同样存在（rocm.py 属上游），但 0.15 的 `VLLM_USE_HYGON` 开关在 0.18 已移除。跨版本请先读对应版本 skill。

## TileLang 替代路径（DCU 算子，免 csrc 重编译）

常规流程（第 1–6 步）是改 csrc/ 重编译 vLLM。另有 **TileLang 路径**：用类 Triton 的 Python DSL 写算子、编译进 vLLM，**绕开 csrc 重编译**，开发周期约为手写 HIP 的 1/5。完整规范/安装/已知问题见 `dcu-kernel-dev`（路径 B）。

端到端示例（GemmaRMSNorm 替换，源自 内部文档 主文档 file_id=<internal-file-id>）：
1. 用 TileLang 写 fused 算子（参考 `dcu-kernel-dev/../../../dcu-operators/skills/dcu-kernel-dev/scripts/tl_gemma_rmsnorm_naive.py`），`tl.compile(..., target="hip")` **显式指定 hip 后端**（DCU 必选，否则出 CUDA 代码）。
2. 环境变量开关切原生 / tilelang：`os.environ["USE_TILELANG"] = "1"`，在 `vllm/model_executor/models/qwen3_5.py` 内按开关选实现。
3. 注册为 torch 自定义 op（同官方 custom_op 机制）：`torch.ops.vllm.qwen3_5_rms_norm(...)`，配 `@register_fake`（漏写 torch.compile 图捕获失败）。
4. 精度验证：单独起精度测试脚本（f32 参考 vs tilelang 实现，max diff < 0.02 视为安全），通过后再起服务对话。
5. baseline 性能：hipprof 跟踪（`hipprof --hip-trace --output-type 0 bash tilelang.sh`，脚本起服务+压测），对比替换前后 kernel 耗时。
6. 优化方向（tilelang 算子）：向量化访存（VEC=8 打满 128-bit load/store）、Block Size 调优（256→1024）、Warp Shuffle（**须显式 `warp_size=64`**）。

⚠ 该路径与 csrc 路径二选一；DCU 专属坑（backend=hip / warp_size=64 / alloc_local 用寄存器 vs alloc_shared 竞争写）见 `dcu-kernel-dev` §3，勿漏。

## 引用与组合
- 算子优化闭环/留痕 → `dcu-op-optimization`；度量 → `dcu-perf-profiling`。
- 源码落点定位 → `dcu-vllm-source`（各版本源码结构与适配点索引）。
- 部署验证 → `dcu-vllm-deploy`。

## 使用须知
- 经验/方法仅供参考，不保证修复。先核对设备型号(gfx)+驱动/DTK版本+报错原文。
- 改动 csrc/构建：在指定镜像容器内含 `-v /opt/hyhal:/opt/hyhal:ro`；gfx 用物理机 `rocminfo` 查。
- 不存性能数据/基线；数字用 `dcu-op-optimization` 模板留痕。
- 高危操作谨慎，优先测试机验证。
- `register_fake` 是 `torch.compile` 必需，漏写会导致图捕获失败——务必为每个新算子补 fake 实现。
