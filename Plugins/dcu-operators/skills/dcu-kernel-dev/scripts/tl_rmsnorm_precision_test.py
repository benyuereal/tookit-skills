# TileLang GemmaRMSNorm 精度验证 demo（源自 内部文档 主文档 §2.4）
# 用途: 算子替换后，先单独跑本脚本验证精度，通过后再做对话/端到端测试
# 平台: DCU 海光; 须在已注册 torch.ops.vllm.qwen3_5_rms_norm 的环境下运行
#        （即 vllm 仓库里已按 USE_TILELANG 开关接入 tl_gemma_rmsnorm_v11）
# 溯源: 内部文档 主文档《Tilelang算子开发实践》file_id=<internal-file-id> §2.4
#
# 关键认知（见 SKILL.md §八.2 精度陷阱）:
#   本实现全程以 float32 执行，不会发生尾数截断误差；
#   与 PyTorch mean() 的差异仅源于「规约加法顺序」不同（IEEE754 舍入顺序敏感），
#   强转 bf16 写回时可能撞上舍入悬崖。block_size 越大该差异越明显，
#   block=256 时可 0 误差，block=1024 时 max diff ≤ 0.03125 (1 ULP) 视为安全。
#
# 运行:
#   export USE_TILELANG=1
#   python tl_rmsnorm_precision_test.py
#
# 依赖 torch / vllm（DCU 容器内已装）；离线 LSP 报 import 未解析属正常误报，非运行错误。

import os
import torch  # pyright: ignore[reportMissingImports]  (依赖 DCU 容器内 torch)

# 文档原始写法依赖 vllm 注册算子；若仅验证 tilelang kernel 本身，
# 可改为直接 import get_cached_tilelang_kernel 调用（见 tl_gemma_rmsnorm_naive.py）。
from vllm.model_executor.my_tl_op.tl_gemma_rmsnorm_v11 import (  # noqa: F401
    TileLangAOTQwen3_5RMSNorm,
    _TILELANG_KERNEL_CACHE,
)

os.environ.setdefault("USE_TILELANG", "1")
os.environ.setdefault("PROFILE_RMSNORM", "0")


def test_rms_norm():
    device = "cuda" if torch.cuda.is_available() else "hip"
    M, hidden_size = 4, 4096
    eps = 1e-6

    x = torch.randn(M, hidden_size, dtype=torch.bfloat16, device=device)
    weight = torch.randn(hidden_size, dtype=torch.bfloat16, device=device)

    # ---- f32 参考（精确参考） ----
    x_f32 = x.float()
    w_f32 = weight.float()
    var_f32 = x_f32.pow(2).mean(-1, keepdim=True)
    ref_f32 = (x_f32 * torch.rsqrt(var_f32 + eps) * (1.0 + w_f32)).bfloat16()

    # ---- TileLang 实现（经 vllm 注册的 torch.ops 入口） ----
    my_out = torch.ops.vllm.qwen3_5_rms_norm(x, weight, eps)

    # ---- 对比 ----
    print("=== vs f32 reference (high precision) ===")
    print("  Max diff:", (ref_f32 - my_out).abs().max().item())
    print("  Mean diff:", (ref_f32 - my_out).abs().mean().item())

    print("=== Sanity checks ===")
    print("  Has NaN in my_out?", torch.isnan(my_out).any().item())
    # 阈值判据：bf16 在 [4,8) 区间 1 ULP = 0.03125；< 0.03 视为安全
    print("  Are max diffs < 0.02 (f32) ?",
          (ref_f32 - my_out).abs().max().item() < 0.02)


if __name__ == "__main__":
    _TILELANG_KERNEL_CACHE.clear()
    torch.manual_seed(42)
    test_rms_norm()
