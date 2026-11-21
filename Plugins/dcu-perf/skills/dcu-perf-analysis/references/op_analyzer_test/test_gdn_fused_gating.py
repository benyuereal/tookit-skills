#!/usr/bin/env python3
"""算子 `gdn_fused_gating` 测试脚本。

用法:
    python3 test_gdn_fused_gating.py
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
OP_TYPE = "gdn_fused_gating"
IN_SHAPES = [[4096, 32]]
OUT_SHAPES = [[4096, 32]]
IN_DTYPES = [opa.HIPDATA_FP32]
OUT_DTYPES = [opa.HIPDATA_FP32]
ATTRS = None

DEVICE = "nmz"
UTIL_TENSOR_CORE = 0.6
UTIL_VECTOR = 0.3
UTIL_SFU = 0.3
UTIL_BANDWIDTH = 0.7


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
    compute_time_ns = 1_000_000.0  # 1 ms
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


def main():
    print("=" * 60)
    print("  gdn_fused_gating 算子测试")
    print("=" * 60)
    print(f"  op_type   = {OP_TYPE}")
    print(f"  in_shape  = {IN_SHAPES}")
    print(f"  out_shape = {OUT_SHAPES}")
    print(f"  in_dtype  = FP32")
    print(f"  out_dtype = FP32")
    print(f"  attrs     = {ATTRS}")

    for t in (test_compute_mode, test_eval_mode, test_utilization_mode):
        try:
            t()
        except Exception as e:
            print(f"  [ERROR] {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
