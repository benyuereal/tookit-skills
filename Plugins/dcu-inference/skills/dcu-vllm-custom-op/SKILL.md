---
name: dcu-vllm-custom-op
description: vLLM DCU版从零开发自定义算子流程
version: 1.0.0
author: DCU Engineer
license: MIT
kind: tool
---

# 海光 DCU 版 vLLM 自定义算子开发流程（dcu-vllm-custom-op）

本 skill 定义**在 vLLM 中从零开发一个自定义算子**的标准流程，并标注海光 DCU / ROCm 下的关键改造点。
与 [`dcu-vllm-op-replace`](../dcu-vllm-op-replace/) **互补不重叠**：

| skill | 解决的问题 |
|---|---|
| `dcu-vllm-op-replace` | 已有一个自定义/海光优化算子，**替换/重定向**进 vLLM 原生调用（6 步：HIP kernel→torch_bindings→`_custom_ops.py` 登记→`rocm.py` 重定向→重编译→验证） |
| **`dcu-vllm-custom-op`（本 skill）** | **从零造一个算子**：参考实现→Triton→CUDA/HIP 优化→dispatch→binding→fallback→单测→性能/精度评估；并给出 ROCm 适配差异 |

> 机制事实：vLLM 的自定义算子注册机制（`torch.library` + `@register_fake`）**海光版未改动**，对上游与海光版通用（实测 0.15.1/0.18 一致）。本 skill 的改造点只在 **dispatch 判断** 与 **kernel 后端（CUDA→HIP）** 两处。

## 一、通用开发流程（11 步，源自 vLLM 官方最佳实践与 PR #16173）

以 `merge_attn_states` 算子（把 prefix KV cache 与 suffix 新 token 的 attention 输出按 LSE 重缩放合并）为范例，这是社区公认的「足够简单、能讲清流程」的入门算子。

1. **算子数学/语义定义**：先写清公式与张量 shape（如 `merge_attn_states` 输入 `[NUM_TOKENS, NUM_HEADS, HEAD_SIZE]` + LSE `[NUM_HEADS, NUM_TOKENS]`，输出按 LSE 加权合并）。
2. **PyTorch 参考实现**：用纯 `torch` 写正确但慢的参考（用于正确性比对与 `register_fake` 思路）。
3. **Triton 基础算子**：用 `@triton.jit` 写第一版 GPU 实现（开发快，作为后续 CUDA 的 baseline 与 fallback）。
4. **Triton 算子分析**：定位瓶颈（CPU 下发开销、访存、占用率），决定要不要上 CUDA。
5. **CUDA/HIP kernel 优化**：手写为求极致性能（PR #16173 用 CUDA 替代 Triton，**最高 3x~4.4x 加速**，原因是减少 CPU 下发开销）。⚠️ DCU 上这步写 **HIP**（见 §四）。
6. **Dispatch 逻辑**（关键）：按平台/dtype/headdim 选择走 CUDA 还是 Triton（见 §三骨架）。
7. **PyTorch binding**：在 `vllm/_custom_ops.py` 注册 Python 入口，转发到 `torch.ops._C.xxx`（见 §三）。
8. **custom ops 封装**：把 C++/kernel 通过 `torch_bindings.cpp` 的 `m.def`/`m.impl` 挂到 `torch.ops._C`。
9. **fallback 逻辑**：对不支持的分支（如 FP8 输入、非对齐 headdim）自动回退 Triton/原生，避免硬崩。
10. **单元测试**：`tests/kernels/test_xxx.py` 跑正确性 + 自动生成性能对比 markdown 表。
11. **性能评估 + 精度评估**：rocprof/hipprof 给加速比；用 evalscope 跑 CEval 等端到端精度回归。

> PR #16173 实测（L20, MLA TP=8, headsize=128）：float16/bf16 各配置 **3.2x~4.4x** 加速（torch 0.15ms→CUDA 0.015ms）；已 merged 进 vLLM main，端到端 R1(PP=3,TP=8) 无退化。

## 二、DCU / 海光适配要点（本 skill 的核心增量）

在 ROCm/海光上重走上述流程时，有 **3 个官方示例没讲、但必须改** 的点：

### ②-1 Dispatch 判断：`is_cuda()` 在 ROCm 下为 False —— 必改
官方 dispatch 层（如 `merge_attn_states.py`）用：
```python
if current_platform.is_cuda() and supported_dtypes(...) and supported_headdim(...):
    from vllm._custom_ops import merge_attn_states   # CUDA 路径
else:
    from vllm.v1.attention.ops.triton_merge_attn_states import merge_attn_states  # fallback
```
**在海光 ROCm 上 `is_cuda()` 返回 False**，会永远走 Triton fallback，你的 HIP kernel 永远不会被调用。海光版应改为：
```python
# 海光适配：CUDA 或 ROCm 都走优化 kernel（前提是 HIP 版 kernel 已注册）
if (current_platform.is_cuda() or current_platform.is_rocm()) and supported_dtypes(...) and supported_headdim(...):
    from vllm._custom_ops import merge_attn_states
else:
    from vllm.v1.attention.ops.triton_merge_attn_states import merge_attn_states
```
或直接 `current_platform.is_rocm()`。这是 DCU 上「写了 HIP kernel 却不生效」的头号原因。

### ②-2 Kernel 后端：CUDA → HIP
- 第 5 步的 CUDA kernel，海光上用 **HIP** 编写（语法近 CUDA，详见 `dcu-kernel-dev`）。csrc 里用 `#ifdef USE_ROCM` 分支；`typedef __hip_bfloat16 __nv_bfloat16;` 兼容 CUDA 命名（海光 0.15.1 范例 `moe_fused_gate.cu` 同法）。
- Triton 路径对 DCU 通常可直接用（Triton 后端选 ROCm），但 DCU 上 Triton 性能/稳定性需实测；HIP 路径对极致性能更可控。

### ②-3 重编译纪律（强制）
改 csrc/ 或 `_custom_ops.py` 后必须重编译，且**必须在指定镜像容器内、必挂 `/opt/hyhal:ro`**：
```bash
# 指定镜像起的专用容器内
export VLLM_TARGET_DEVICE=rocm
export VLLM_USE_HYGON=1          # 海光 0.15.x 需要（0.18 已移除该开关）
export ROCM_HOME=/opt/dtk         # ls /opt/dtk 为准
pip install -e .                  # 重新编译 csrc/*，使新 kernel/绑定生效
```
- gfx 用物理机 `rocminfo` 查（不写死，见 `dcu-kernel-dev` §12.5）。
- 指定部分卡：`HIP_VISIBLE_DEVICES=0,1,2,3`。

## 三、核心代码骨架（基于 vLLM 官方真实源码）

### 3.1 Dispatch 层（`vllm/v1/attention/ops/merge_attn_states.py`，111 行精华）
完整逻辑就是「平台/dtype/headdim 判断 → 走优化 kernel 或 fallback」：
```python
def merge_attn_states(output, prefix_output, prefix_lse, suffix_output, suffix_lse,
                      output_lse=None, prefill_tokens_with_context=None, output_scale=None):
    # 约束：prefix/suffix 必须同 head stride
    assert prefix_output.stride(1) == suffix_output.stride(1)
    # fallback 约束：非 f32/f16/bf16 必须带 output_scale(FP8)
    if output.dtype not in (torch.float32, torch.half, torch.bfloat16):
        assert output_scale is not None
    def supported_dtypes(t): return t.dtype in [torch.float32, torch.half, torch.bfloat16]
    def supported_headdim(t):
        hd = t.shape[2]
        return hd % 4 == 0 if t.dtype == torch.float32 else hd % 8 == 0  # 128b 对齐
    # —— 海光适配点：is_cuda() → is_cuda() or is_rocm() ——
    if current_platform.is_cuda() and supported_dtypes(prefix_output) and supported_headdim(prefix_output):
        from vllm._custom_ops import merge_attn_states
        return merge_attn_states(output, prefix_output, prefix_lse, suffix_output,
                                suffix_lse, output_lse, prefill_tokens_with_context, output_scale)
    else:
        from vllm.v1.attention.ops.triton_merge_attn_states import merge_attn_states
        return merge_attn_states(output, prefix_output, prefix_lse, suffix_output,
                                suffix_lse, output_lse, prefill_tokens_with_context, output_scale)
```

### 3.2 注册层（`vllm/_custom_ops.py` 片段）
```python
def merge_attn_states(output, prefix_output, prefix_lse, suffix_output, suffix_lse,
                      output_lse=None, prefill_tokens_with_context=None, output_scale=None) -> None:
    # 转发到 C++/kernel 注册进 torch.library 的 _C 命名空间
    torch.ops._C.merge_attn_states(output, output_lse, prefix_output, prefix_lse,
                                    suffix_output, suffix_lse, prefill_tokens_with_context, output_scale)
```
- 每个自定义算子都需在 `torch_bindings.cpp` 用 `m.def`/`m.impl` 绑定，并在 Python 侧补 `@register_fake`（给 `torch.compile` 图捕获用，**漏写会图捕获失败**）。

### 3.3 本 skill 附带骨架脚本
`scripts/vllm_custom_op_dispatch_template.py`：可直接复制的 dispatch 层模板（含 `is_cuda() or is_rocm()` 海光适配分支 + fallback + dtype/headdim 校验注释），抄它改算子名即可。

## 四、验证闭环（强制，呼应 dcu-op-optimization）

1. **正确性**：数值对齐（你的实现 vs PyTorch 参考 / CPU 参考；tolerance 按精度定，bf16 通常 <1e-2）。
2. **性能**：同条件 `hipprof` 对比替换前后（见 `dcu-perf-profiling`），给量化加速比（参照 PR #16173 的 markdown 表）。
3. **端到端**：起服务 + `benchmark_serving.py` 确认无回归；可用 evalscope 跑 CEval 精度回归。

## 五、一手源与溯源

> **原文《vLLM算子开发流程保姆级详细记录》（知乎 zhuanlan.zhihu.com/p/1892966682634473987）在本环境无法读取**（知乎 403 反爬 + archive.org 出口超时 + 百度验证，三条自动路径均不可达）。以下材料为公开一手溯源，比读二手转载更权威：

- **vLLM 官方 PR #16173** `[Kernel] support merge_attn_states CUDA kernel, 3x speedup by DefTruth`（已 merged）：本 skill 第 1–11 步与 §三骨架的权威依据，含完整性能表/正确性流程。
- **同源知乎问答**（zhihu.com/question/1894449141901395185「如何进行vLLM算子开发」）：确认该文章即此问答的「保姆级扩写版」，大纲一致。
- **vLLM 官方 CustomOp 文档**：Developer Guide → CustomOp（`https://docs.vllm.ai/en/latest/dev/custom_ops.html`，侧栏 `CustomOp` 条目）。
- **海光既有机制事实**来自 `dcu-vllm-op-replace`（0.15.1 实测：注册机制未改、rocm.py 重定向、VLLM_USE_HYGON 开关、csrc 落点）。

## 六、组合引用

- 算子内核怎么写（HIP/TileLang）→ `dcu-kernel-dev`（含运行时设备探测、不写死 gfx）。
- 替换/重定向已有算子 → `dcu-vllm-op-replace`。
- 优化闭环/性能度量 → `dcu-op-optimization`、`dcu-perf-profiling`。
- 源码落点定位 → `dcu-vllm-source`（上游复用版）。
- 部署验证 → `dcu-vllm-deploy`。

## 使用须知

- 经验/方法仅供参考，不保证修复。先核对设备型号(gfx)+驱动/DTK版本+vLLM版本+报错原文。
- 改动 csrc/构建：在指定镜像容器内含 `-v /opt/hyhal:/opt/hyhal:ro`；gfx 用物理机 `rocminfo` 查（不写死）。
- 不存性能数据/基线；数字用 `dcu-op-optimization` 模板留痕。
- `register_fake` 是 `torch.compile` 必需，漏写导致图捕获失败——务必为每个新算子补 fake 实现。
- `is_cuda()` 在 ROCm 为 False：DCU 上 dispatch 必须显式加 `is_rocm()` 分支（★头号坑）。
