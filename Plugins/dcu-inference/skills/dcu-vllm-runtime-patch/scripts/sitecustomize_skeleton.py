"""
DCU vLLM 运行时 monkey-patch 骨架（通用模板）。

用法：
1. 把本文件放到 PYTHONPATH 第一个目录，命名为 sitecustomize.py
   （或在启动脚本里 `export PYTHONPATH=/path/to/this/dir:$PYTHONPATH`）。
2. 通过环境变量开关启用各专用路径；默认全部关闭、走上游，保留安全网。
3. 只对 decode 形状（M <= MAX_M）注入专用 kernel，其余 fallback 上游 heuristic。

这是模板，需按目标模型/算子把 `TODO` 处替换为真实 kernel 调用与形状表。
本文件仅做语法与结构示范：torch/vllm 的导入在 _apply_* 内惰性进行，
因此本骨架可不依赖运行环境直接 `python -m py_compile` 通过。
"""
from __future__ import annotations

import os

# ---- 1. 环境变量总开关（默认关，按形状精确开）--------------------------------
# 约定：每个专用路径一个 *_ENABLED 与 *_MAX_M，只对 M<=MAX_M 的形状替换。
_LINEAR_ENABLED = os.getenv("DCU_PATCH_LINEAR_A8W8", "0") == "1"
_LINEAR_MAX_M = int(os.getenv("DCU_PATCH_LINEAR_A8W8_MAX_M", "4"))


def _shape_specific_config(m: int, k: int, n: int) -> dict | None:
    """返回 (M,K,N) 的专用 Triton/AITER config；未命中返回 None -> fallback。

    TODO: 用真实 (M,K,N) 形状（含 M=1~MAX_M 各档）离线搜出的 config 填表。
    表内必须保证与该形状的「上游输出 bit-identical」（relative_l2==0）。
    """
    table: dict[tuple[int, int, int], dict] = {
        # (1, 2048, 12288): {"bm": 32, "bn": 32, "bk": 128, "warps": 8, "waves": 2},
        # (1, 2048, 9216):  {"bm": 32, "bn": 32, "bk": 128, "warps": 8, "waves": 2},
    }
    return table.get((m, k, n))


def _apply_linear_patch() -> None:
    """hook triton_scaled_mm：对命中形状的 decode 走专用 GEMM，其余走上游。

    TODO: 用 `import importlib` 惰性导入 vllm 的 triton_scaled_mm 与你的 GEMM，
    在 torch.library.custom_op 内：命中 shape -> 专用 kernel；否则 _orig_*(...)。
    所有 custom_op 必须带 register_fake 以支持 torch.compile / inductor。

    示例结构（伪代码，仅展示范式，非可运行实现）：
        import importlib, torch
        ct = importlib.import_module(
            "vllm.model_executor.layers.quantization.compressed_tensors.triton_scaled_mm")
        lin = importlib.import_module(
            "vllm.model_executor.kernels.linear.scaled_mm.triton")
        orig = ct.triton_scaled_mm

        @torch.library.custom_op("dcu::patched_linear", mutates_args=(), device_types="cuda")
        def _patched(x, w, sa, sb):
            m, k, n = int(x.shape[0]), int(x.shape[1]), int(w.shape[1])
            cfg = _shape_specific_config(m, k, n)
            if m <= _LINEAR_MAX_M and cfg is not None:
                return _your_gemm(x, w, sa, sb, cfg)   # 专用 kernel
            return orig(x, w, sa, sb, torch.bfloat16)  # 上游 fallback

        @_patched.register_fake
        def _patched_fake(x, w, sa, sb):
            return x.new_empty((x.shape[0], w.shape[1]), dtype=torch.bfloat16)

        ct.triton_scaled_mm = _patched
        lin.triton_scaled_mm = _patched
    """
    raise NotImplementedError(
        "实现示例见 dcu-perf-cases Case 10（K100 AI Qwen3.6 W8A8）："
        "真实 sitecustomize.py 含 INT8 Linear / lm_head / native MoE / QKVZ+BA 融合四类 hook"
    )


def install() -> None:
    if _LINEAR_ENABLED:
        _apply_linear_patch()


install()
