

import os
from pathlib import Path

_DATA_DIR = Path(__file__).resolve().parent / 'data'
if _DATA_DIR.is_dir():
    os.environ.setdefault('OP_ANALYZER_DEVICE_CONFIG_ROOT',
                          str(_DATA_DIR / 'configs' / 'devices'))
    os.environ.setdefault('OP_ANALYZER_PERF_DATA_ROOT',
                          str(_DATA_DIR / 'perf_data'))

try:
    from .op_analyzer_py import (
        OpInfo,
        OperatorAnalyzer,
        create_op_info,
        HIPDATA_FP32,
        HIPDATA_FP16,
        HIPDATA_BF16,
        HIPDATA_I32,
        HIPDATA_I64,
        HIPDATA_I8,
        HIPDATA_FP8E4M3,
        HIPDATA_FP4E2M1,
        HIPDATA_FP8E8M0,
    )

    def _bmm(analyzer, name, in1, in2, out, dtype,
             device, tensor_core, vector, sfu, bandwidth):
        """构建一个 BMM (标准布局 [b,M,K]x[b,K,N]) 并走 Roofline 估算。"""
        op = OpInfo(name)
        op.set_input_shape([in1, in2])
        op.set_output_shape([out])
        op.set_input_data_type([dtype, dtype])
        op.set_output_data_type([dtype])
        op.set_compute_mode("eval")
        return analyzer.analyze_operator(
            "bmm", op,
            device=device,
            tensor_core=tensor_core, vector=vector, sfu=sfu, bandwidth=bandwidth,
            force_roofline=True)

    def _softmax(analyzer, name, shape, dtype,
                 device, tensor_core, vector, sfu, bandwidth):
        op = OpInfo(name)
        op.set_input_shape([shape])
        op.set_output_shape([shape])
        op.set_input_data_type([dtype])
        op.set_output_data_type([dtype])
        op.set_compute_mode("eval")
        return analyzer.analyze_operator(
            "softmax", op,
            device=device,
            tensor_core=tensor_core, vector=vector, sfu=sfu, bandwidth=bandwidth,
            force_roofline=True)

    def analyze_attention(seq,
                          qk_head_dim=192,
                          v_head_dim=128,
                          q_heads=16,
                          kv_heads=1,
                          kv_seq=None,
                          dtype=HIPDATA_BF16,
                          tensor_core=-1.0,
                          vector=-1.0,
                          sfu=-1.0,
                          bandwidth=-1.0,
                          device="",
                          include_softmax=True,
                          analyzer=None):
        """计算注意力的两个 batched matmul (QK^T + P·V), 可选 softmax, 在指定
        TC/vector/SFU/带宽利用率下用 Roofline 估算耗时。

        语义 (GQA/MQA broadcast: q_heads 个 query head 共享 kv_heads 个 KV head):
            QK^T : Q[q_heads, seq, qk_head_dim] x K[kv_heads, kv_seq, qk_head_dim]
                       -> P[q_heads, seq, kv_seq]
            P·V  : P[q_heads, seq, kv_seq]      x V[kv_heads, kv_seq, v_head_dim]
                       -> O[q_heads, seq, v_head_dim]

        例 (用户示例): analyze_attention(seq, qk_head_dim=192, v_head_dim=128,
                                        q_heads=16, kv_heads=1)

        FLOPs 按 q_heads 个 head 各做一次完整 matmul 计 (KV broadcast 到每个 head),
        与标准 attention FLOP 口径一致。底层 BMM 要求标准布局 [b,M,K]x[b,K,N], 本
        函数已把 K 的 192 维放到收缩位 (即传入 K^T = [q_heads, qk_head_dim, kv_seq])。

        利用率参数 <0 表示该项不覆盖, 用 device JSON 默认值; 详见
        doc/device_config_refactor.md。

        返回 dict:
            {'qk': <metrics>, 'softmax': <metrics 或 None>, 'pv': <metrics>,
             'total_time_ns': float, 'total_flops': int}
        每个 <metrics> 是底层算子的完整 PerformanceMetrics dict (含 eval_time(ns) /
        tensor_core_flops / tflops / bandwidth 等)。
        """
        if kv_seq is None:
            kv_seq = seq
        own = analyzer is None
        if own:
            analyzer = OperatorAnalyzer()

        # QK^T: [q_heads, seq, qk_head_dim] x [q_heads, qk_head_dim, kv_seq]
        qk = _bmm(analyzer, "qk_matmul",
                  [q_heads, seq, qk_head_dim], [q_heads, qk_head_dim, kv_seq],
                  [q_heads, seq, kv_seq], dtype,
                  device, tensor_core, vector, sfu, bandwidth)

        sm = None
        if include_softmax:
            sm = _softmax(analyzer, "attn_softmax",
                          [q_heads, seq, kv_seq], dtype,
                          device, tensor_core, vector, sfu, bandwidth)

        # P·V: [q_heads, seq, kv_seq] x [q_heads, kv_seq, v_head_dim]
        pv = _bmm(analyzer, "pv_matmul",
                  [q_heads, seq, kv_seq], [q_heads, kv_seq, v_head_dim],
                  [q_heads, seq, v_head_dim], dtype,
                  device, tensor_core, vector, sfu, bandwidth)

        parts = [qk, pv] + ([sm] if sm is not None else [])
        total_time_ns = sum(p["eval_time"] for p in parts)
        total_flops = sum(int(p["total_flops"]) for p in parts)
        return {
            'qk': qk,
            'softmax': sm,
            'pv': pv,
            'total_time_ns': total_time_ns,
            'total_flops': total_flops,
        }

    __all__ = [
        'OpInfo',
        'OperatorAnalyzer',
        'create_op_info',
        'analyze_attention',
        'HIPDATA_FP32',
        'HIPDATA_FP16',
        'HIPDATA_BF16',
        'HIPDATA_I32',
        'HIPDATA_I64',
        'HIPDATA_I8',
        'HIPDATA_FP8E4M3',
        'HIPDATA_FP4E2M1',
        'HIPDATA_FP8E8M0',
    ]
except ImportError:
    # Module not compiled yet
    __all__ = []

__version__ = '1.0.0'
