#!/usr/bin/env python3
# vllm_custom_op_dispatch_template.py
# ─────────────────────────────────────────────────────────────────────────────
# vLLM 自定义算子「dispatch 层」模板（DCU/海光适配版）。
# 直接复制本文件，把 YourOp / your_op / 张量参数改成你的算子即可。
#
# 来源：基于 vLLM 官方 vllm/v1/attention/ops/merge_attn_states.py 的 dispatch
#       逻辑提炼；★标记处为海光 ROCm 必须加的适配分支（官方原版只有 is_cuda）。
# 配合：vllm/_custom_ops.py 的 Python 入口 + csrc/.../torch_bindings.cpp 的 C++ 绑定。
# ─────────────────────────────────────────────────────────────────────────────
import torch

from vllm.platforms import current_platform


def your_op(
    output: torch.Tensor,
    # TODO: 按你的算子补充参数（参考实现应在此列出全部张量 shape）
    output_scale: torch.Tensor | None = None,
) -> None:
    """你的自定义算子。

    语义/公式在此写清（见 skill §一 第1步）。
    """

    # ── 约束校验（dispatch 前必须保证 kernel 能正确处理）──
    # TODO: 按算子补 dtype / shape / stride / 对齐 校验
    if output.dtype not in (torch.float32, torch.half, torch.bfloat16):
        # 例：FP8 输出必须带 output_scale（merge_attn_states 的实际约束）
        assert output_scale is not None, (
            f"output_scale is required when output is {output.dtype}"
        )

    def supported_dtypes(t: torch.Tensor) -> bool:
        return t.dtype in [torch.float32, torch.half, torch.bfloat16]

    def supported_headdim(t: torch.Tensor) -> bool:
        # 例：HIP/CUDA kernel 通常按 128bit(16B) 对齐 load/store
        hd = t.shape[-1]
        if t.dtype == torch.float32:
            return hd % 4 == 0
        return hd % 8 == 0

    # ── ★ 海光适配点：官方的 is_cuda() 在 ROCm 下为 False，会永远走 fallback ──
    # 必须显式加 is_rocm()（或 is_cuda() or is_rocm()），否则你的 HIP kernel 永不被调用。
    if (
        (current_platform.is_cuda() or current_platform.is_rocm())   # ← 海光关键改造
        and supported_dtypes(output)
        and supported_headdim(output)
    ):
        # 走优化 kernel（CUDA 或 HIP，取决于平台编译目标）
        from vllm._custom_ops import your_op  # 对应 csrc 注册进 torch.ops._C.your_op
        return your_op(output, output_scale)
    else:
        # fallback：Triton / 原生 / PyTorch 参考实现
        from vllm.v1.ops.triton_your_op import your_op  # TODO: 你的 Triton 路径
        return your_op(output, output_scale)


# ─────────────────────────────────────────────────────────────────────────────
# 配套清单（开发时逐项核对，见 skill §一 11 步）：
#   [ ] 1. 算子数学/语义 + 张量 shape 定义
#   [ ] 2. PyTorch 参考实现（正确性比对 + register_fake 思路）
#   [ ] 3. Triton 基础算子（@triton.jit，作为 baseline + fallback）
#   [ ] 4. Triton 算子分析（瓶颈定位）
#   [ ] 5. CUDA/HIP kernel 优化（DCU 写 HIP，见 dcu-kernel-dev）
#   [ ] 6. 本文件的 dispatch 逻辑（含 ★is_rocm 分支）
#   [ ] 7. vllm/_custom_ops.py 增加 Python 入口（转发 torch.ops._C.your_op）
#   [ ] 8. csrc/.../torch_bindings.cpp 增加 m.def / m.impl 绑定
#   [ ] 9. fallback 逻辑（不支持分支自动回退，避免硬崩）
#   [ ] 10. tests/kernels/test_your_op.py（正确性 + 自动性能表）
#   [ ] 11. 性能(hipprof) + 精度(evalscope CEval) 评估
#   [ ] ★ register_fake：torch.compile 必需，漏写图捕获失败
#   [ ] ★ 重编译：容器内 -v /opt/hyhal:/opt/hyhal:ro + VLLM_TARGET_DEVICE=rocm
# ─────────────────────────────────────────────────────────────────────────────
