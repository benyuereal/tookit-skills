#!/usr/bin/env python3
"""算子 `mha` 测试脚本 (含 gqa / mqa / flash_attention / paged_attention)。

用法:
    python3 test_mha.py
"""

# 需先安装 op_analyzer wheel:  pip install operator_analyzer-*.whl
import operator_analyzer as opa


def make_op_info(op_name, in_shapes, out_shapes, in_dtypes, out_dtypes, mode):
    """构造 OpInfo"""
    op = opa.OpInfo(op_name)
    op.set_input_shape(in_shapes)
    op.set_output_shape(out_shapes)
    op.set_input_data_type(in_dtypes)
    op.set_output_data_type(out_dtypes)
    op.set_compute_mode(mode)
    return op


# ---- 输入参数 (按需修改) ----
OP_TYPE = "mha"
IN_SHAPES = [[1]]
OUT_SHAPES = [[1]]
IN_DTYPES = [opa.HIPDATA_BF16]
OUT_DTYPES = [opa.HIPDATA_BF16]
ATTRS = {'batch': 1, 'num_q_heads': 8, 'num_kv_heads': 1, 'seq_q': 1192, 'seq_kv': 1192, 'head_dim': 128, 'v_head_dim': 128}

DEVICE = "nmz"
UTIL_TENSOR_CORE = 0.8
UTIL_VECTOR = 0.6
UTIL_SFU = 0.6
UTIL_BANDWIDTH = 0.8


def _build(mode):
    op = make_op_info(OP_TYPE, IN_SHAPES, OUT_SHAPES, IN_DTYPES, OUT_DTYPES, mode)
    if ATTRS:
        for k, v in ATTRS.items():
            op.set_attribute(k, str(v))
    return op


def test_compute_mode():
    """compute 模式: 输入实测耗时, 输出算力 / 带宽"""
    print(f"\n{'=' * 60}")
    print("compute 模式")
    print(f"{'=' * 60}")
    op = _build("compute")
    compute_time_ns = 0_372_319.0  # 1 ms
    m = opa.OperatorAnalyzer().analyze_operator(OP_TYPE, op, compute_time_ns)
    print(f"  输入 compute_time_ns = {compute_time_ns}")
    print(f"  total_flops          = {m['total_flops']}")
    print(f"  tflops               = {m['tflops']:.4f}")
    print(f"  bandwidth            = {m['bandwidth']:.2f} GB/s")
    print(f"  compute_time         = {m['compute_time_ns']} ns")


def test_eval_mode():
    """eval 模式: 估算耗时"""
    print(f"\n{'=' * 60}")
    print("eval 模式")
    print(f"{'=' * 60}")
    op = _build("eval")
    m = opa.OperatorAnalyzer().analyze_operator(OP_TYPE, op, 0.0)
    print(f"  total_flops = {m['total_flops']}")
    print(f"  eval_time   = {m['eval_time']} ns")


def test_utilization_mode():
    """指定利用率估算耗时"""
    print(f"\n{'=' * 60}")
    print("指定利用率模式")
    print(f"{'=' * 60}")
    op = _build("eval")
    m = opa.OperatorAnalyzer().analyze_operator(
        OP_TYPE, op, 0.0,
        device=DEVICE,
        tensor_core=UTIL_TENSOR_CORE,
        vector=UTIL_VECTOR,
        sfu=UTIL_SFU,
        bandwidth=UTIL_BANDWIDTH)
    print(f"  device                = {DEVICE}")
    print(f"  util(tc/vec/sfu/bw)   = {UTIL_TENSOR_CORE}/{UTIL_VECTOR}/{UTIL_SFU}/{UTIL_BANDWIDTH}")
    print(f"  total_flops           = {m['total_flops']}")
    print(f"  eval_time             = {m['eval_time']:.1f} ns")


def test_causal_folding():
    """因果注意力 (causal) 折半: prefill score 矩阵只需算下三角。

    causal_scores(sq, skv) = sq*skv - sq*(sq-1)/2 ;  非 causal = sq*skv。
    tensor_core_flops 全部由 QK^T / attn·V 两个 matmul 构成, 与 score 元素数严格成正比,
    故 causal/非causal 的 tensor_core_flops 比值应精确等于 score 比值。
    seq_q==seq_kv (单段 prefill) 时 full/causal ≈ 2 (长序列下趋近 2x)。
    """
    print(f"\n{'=' * 60}")
    print("causal 折半")
    print(f"{'=' * 60}")
    sq = int(ATTRS["seq_q"])
    skv = int(ATTRS["seq_kv"])
    full_scores = sq * skv
    causal_scores = sq * skv - sq * (sq - 1) // 2

    op_c = _build("eval")
    op_c.set_attribute("causal", "true")
    op_f = _build("eval")
    op_f.set_attribute("causal", "false")
    tc_c = opa.OperatorAnalyzer().analyze_operator(OP_TYPE, op_c, 0.0)['tensor_core_flops']
    tc_f = opa.OperatorAnalyzer().analyze_operator(OP_TYPE, op_f, 0.0)['tensor_core_flops']
    print(f"  scores causal/full          = {causal_scores}/{full_scores}")
    print(f"  tensor_core_flops causal/full = {tc_c}/{tc_f}")
    # tensor_core_flops 与 score 数严格成正比 → 整数交叉积相等
    assert tc_c < tc_f, f"causal flops 应 < 非causal: {tc_c} vs {tc_f}"
    assert tc_c * full_scores == tc_f * causal_scores, \
        f"flops 比值应 == score 比值: {tc_c}/{tc_f} vs {causal_scores}/{full_scores}"
    if sq == skv and sq > 1:
        ratio = tc_f / tc_c
        print(f"  full/causal ratio           = {ratio:.6f} (seq_q==seq_kv → ≈2)")
        assert abs(ratio - 2.0) < 2.0 / sq + 1e-9, f"ratio={ratio}"
    print("  [OK] causal 折半口径正确")


def main():
    print("=" * 60)
    print("  mha 算子测试")
    print("=" * 60)
    print(f"  op_type   = {OP_TYPE}")
    print(f"  in_shape  = {IN_SHAPES}")
    print(f"  out_shape = {OUT_SHAPES}")
    print(f"  in_dtype  = BF16")
    print(f"  out_dtype = BF16")
    print(f"  attrs     = {ATTRS}")

    for t in (test_compute_mode, test_eval_mode, test_utilization_mode, test_causal_folding):
        try:
            t()
        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
