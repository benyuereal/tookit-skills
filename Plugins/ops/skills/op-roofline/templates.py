#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""访问模式模板：从符号形状算 FLOPs 与 Bytes。

为什么要有这个文件
------------------
`derive.py` 现在只支持 `sparse-mla` 与 `mqa-logits` **两个特例**——
每个都是一个独立模型文件、一套专用参数。要覆盖 53 张卡，得写 53 个。

而实际上大部分算子的 FLOPs/Bytes 只取决于**访问模式**。10 个模板就够：

    elementwise-unary   rope / quant / gelu            ~N        2·N·d
    elementwise-binary  silu×mul / add / combine       ~N        3·N·d
    reduce-lastdim      rmsnorm / layernorm / qk-norm  ~3N       (2~3)·N·d
    matmul              gemm-* / qkv / o_proj          2MNK      (MK+KN+MN)·d
    batched-matmul      bmm / grouped-gemm             B·2MNK    …
    attention-prefill   flash-attn / mha               4S²hd     KV 全读
    attention-decode    paged / mla-*                  4Shd      KV 全读
    gather-scatter      moe-permute / embedding        ~0        moved·d
    topk-reduce         topk-gating / sampling / mqa   ~N        N·d + k·d
    matvec              lm-head                        2NK       (N+1)·K·d

**纪律**：模板只回答「这个形态下搬多少字节、算多少次」。
它不回答「该用什么语言」、也不回答「能优化多少」——那是 plan.py 的事。

special-case 必须声明理由：模板表达不了的，用 `model_override` + `why_not_generic`。
`validate.py` 会查（缺 why_not_generic 直接 error）。

用法
----
    from templates import evaluate, by_name
    evaluate("reduce-lastdim", {"M": 8192, "H": 4096, "dtype_bytes": 2})
    # → {"flops": 0, "bytes": 268435456, "ai": 0.0, "note": "..."}

    python3 -I templates.py --selftest
    python3 -I templates.py --list
"""
import sys

DTYPE_BYTES = {"fp32": 4, "bf16": 2, "fp16": 2, "fp8": 1, "int8": 1}


def _b(p):
    """参数取字节宽度，默认 bf16。"""
    return float(p.get("dtype_bytes") or DTYPE_BYTES.get(str(p.get("dtype") or "bf16"), 2))


# 每个模板：name -> (fn(params) -> dict, 覆盖的算子示例, 说明)
def _elementwise_unary(p):
    n = p["N"]
    d = _b(p)
    # 读 N 个元素，写 N 个
    return {"flops": 0, "bytes": 2 * n * d, "note": "1读1写"}


def _elementwise_binary(p):
    n = p["N"]
    d = _b(p)
    return {"flops": 0, "bytes": 3 * n * d, "note": "2读1写"}


def _reduce_lastdim(p):
    M, H = p["M"], p["H"]
    d = _b(p)
    # 读 x（H）+ 写 y（H）；归约本身 FLOPs 约 3N 但相对带宽可忽略
    return {"flops": 0, "bytes": 2 * M * H * d, "note": "1读1写（含行内归约）"}


def _reduce_lastdim_2out(p):
    """融合残差版：2读2写（含 residual_out）。rmsnorm-fused 属这类。"""
    M, H = p["M"], p["H"]
    d = _b(p)
    return {"flops": 0, "bytes": 4 * M * H * d, "note": "2读2写（含 residual_out）"}


def _matmul(p):
    M, N, K = p["M"], p["N"], p["K"]
    d = _b(p)
    return {"flops": 2 * M * N * K,
            "bytes": (M * K + K * N + M * N) * d, "note": "标准 GEMM"}


def _batched_matmul(p):
    B, M, N, K = p["B"], p["M"], p["N"], p["K"]
    d = _b(p)
    return {"flops": B * 2 * M * N * K,
            "bytes": B * (M * K + K * N + M * N) * d, "note": "B 个独立 GEMM"}


def _attention_prefill(p):
    B, S, h, dim = p["B"], p["S"], p["h"], p["D"]
    d = _b(p)
    # causal 实际 FLOPs 约满阵一半；这里给满阵（保守，且与卡片 ratio 同口径）
    flops = 4 * B * h * S * S * dim
    byts = 3 * B * S * h * dim * d          # q/k/v 全读 + out 写
    return {"flops": flops, "bytes": byts, "note": "满阵口径（causal 约一半）"}


def _attention_decode(p):
    B, S, h, dim = p["B"], p["S"], p["h"], p["D"]
    d = _b(p)
    # decode：1 个 query 读全部 KV
    flops = 4 * B * h * S * dim
    byts = 2 * B * S * h * dim * d          # KV 读 + out 写
    return {"flops": flops, "bytes": byts, "note": "1 query 读全 KV"}


def _gather_scatter(p):
    moved = p["moved"]
    d = _b(p)
    return {"flops": 0, "bytes": 2 * moved * d, "note": "gather 读 + scatter 写"}


def _topk_reduce(p):
    N, k = p["N"], p.get("k", 1)
    d = _b(p)
    return {"flops": 0, "bytes": (N + k) * d, "note": "读 N 选 k"}


def _matvec(p):
    N, K = p["N"], p["K"]
    d = _b(p)
    return {"flops": 2 * N * K, "bytes": (N + 1) * K * d, "note": "1 个向量乘矩阵"}


TEMPLATES = {
    "elementwise-unary": (_elementwise_unary, ["rope", "quant", "gelu"]),
    "elementwise-binary": (_elementwise_binary, ["silu×mul", "add", "combine"]),
    "reduce-lastdim": (_reduce_lastdim,
                       ["rmsnorm", "layernorm", "qk-norm", "fused-add-rmsnorm"]),
    "reduce-lastdim-2out": (_reduce_lastdim_2out, ["rmsnorm-fused"]),
    "matmul": (_matmul, ["gemm-bf16", "gemm-fp8", "gemm-int8", "qkv", "o_proj"]),
    "batched-matmul": (_batched_matmul, ["bmm", "grouped-gemm"]),
    "attention-prefill": (_attention_prefill, ["flash-attention-fwd", "mha-prefill"]),
    "attention-decode": (_attention_decode, ["paged-decode", "mla-full", "mla-absorb"]),
    "gather-scatter": (_gather_scatter, ["moe-permute", "embedding-lookup"]),
    "topk-reduce": (_topk_reduce, ["topk-gating", "sampling", "mqa-logits"]),
    "matvec": (_matvec, ["lm-head"]),
}


def by_name(name):
    return TEMPLATES.get(str(name))


def evaluate(name, params):
    """算一个模板的 flops / bytes / ai。**缺参数就报错，不猜。**"""
    t = by_name(name)
    if t is None:
        return {"error": "未知模板 %r（可选 %s）" % (name, sorted(TEMPLATES))}
    fn = t[0]
    try:
        r = fn(params)
    except KeyError as e:
        return {"error": "模板 %s 缺参数 %s（需要 %s）"
                         % (name, e, _required(name))}
    if not r.get("bytes"):
        return {"error": "模板 %s 算出 bytes=0，无法判 AI" % name}
    r["ai"] = (r["flops"] / r["bytes"]) if r["flops"] else 0.0
    r["template"] = name
    return r


def _required(name):
    """每个模板必需的参数，用于报错时提示。"""
    return {
        "elementwise-unary": ["N"], "elementwise-binary": ["N"],
        "reduce-lastdim": ["M", "H"], "reduce-lastdim-2out": ["M", "H"],
        "matmul": ["M", "N", "K"], "batched-matmul": ["B", "M", "N", "K"],
        "attention-prefill": ["B", "S", "h", "D"],
        "attention-decode": ["B", "S", "h", "D"],
        "gather-scatter": ["moved"], "topk-reduce": ["N"], "matvec": ["N", "K"],
    }.get(name, [])


def selftest():
    # 1) reduce-lastdim：rmsnorm prefill-8k 应算出 2读2写 = 268.4MB
    r = evaluate("reduce-lastdim-2out", {"M": 8192, "H": 4096, "dtype_bytes": 2})
    assert abs(r["bytes"] - 268435456) < 1, r
    print("[ok] reduce-lastdim-2out：M=8192 H=4096 bf16 → 268.4MB（与卡片一致）")

    # 2) matmul 8192³，两种精度 AI 不同——这本身就是个判据：
    #    bf16: AI = 2·N³ / (3·N²·2) = N/3 = 2730
    #    fp8 : AI = 2·N³ / (3·N²·1) = 2N/3 = 5461
    #    两者都 ≫ ridge(161)，都判 compute。但**绝对值不同**，
    #    引用 AI 时必须说明精度，否则「AI=4096」这种数是哪个口径都说不清。
    r = evaluate("matmul", {"M": 8192, "N": 8192, "K": 8192, "dtype_bytes": 2})
    assert r["flops"] == 2 * 8192 ** 3, r
    assert abs(r["ai"] - 8192 / 3) < 1, r["ai"]
    r8 = evaluate("matmul", {"M": 8192, "N": 8192, "K": 8192, "dtype_bytes": 1})
    assert abs(r8["ai"] - 2 * 8192 / 3) < 1, r8["ai"]
    print("[ok] matmul 8192³：bf16 AI=%.0f / fp8 AI=%.0f（都判 compute，但绝对值差 2×）"
          % (r["ai"], r8["ai"]))

    # 3) 缺参数 → 报错，不猜
    r = evaluate("matmul", {"M": 4})
    assert "error" in r and "缺参数" in r["error"], r
    print("[ok] 缺参数 → 明确报错：%s" % r["error"][:40])

    # 4) 未知模板
    r = evaluate("not-a-template", {})
    assert "error" in r
    print("[ok] 未知模板 → 报错")

    # 5) 全部 10 个模板齐全
    assert len(TEMPLATES) >= 10, len(TEMPLATES)
    print("[ok] 模板数 %d（目标 ≥10）" % len(TEMPLATES))

    # 6) 每个模板都至少覆盖 1 个算子
    bad = [k for k, v in TEMPLATES.items() if not v[1]]
    assert not bad, bad
    print("[ok] 每个模板都有覆盖算子")

    print("\nselftest PASS（6 项）")
    return 0


def main(argv=None):
    argv = argv or sys.argv[1:]
    if "--selftest" in argv:
        return selftest()
    if "--list" in argv:
        for k, (_, ops) in sorted(TEMPLATES.items()):
            print("%-24s %s" % (k, " / ".join(ops)))
        return 0
    print("用法：--list 或 --selftest")
    return 0


if __name__ == "__main__":
    sys.exit(main())
