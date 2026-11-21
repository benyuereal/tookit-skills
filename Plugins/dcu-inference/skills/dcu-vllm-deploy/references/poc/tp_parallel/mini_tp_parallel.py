#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vLLM Tensor Parallel (TP) 最小化 POC —— Column/Row Parallel Linear + all-reduce

只依赖 Python 标准库，CPU 直接跑：
    python3 mini_tp_parallel.py

对应真实 vLLM 源码：
  - vllm/model_executor/layers/linear.py
      ColumnParallelLinear.forward  按输出维切 W，各 rank 算局部结果，不通信
      RowParallelLinear.forward     按输入维切 W，输出是"部分和"，
                                    最后 tensor_model_parallel_all_reduce
  - vllm/distributed/communication_op.py
      tensor_model_parallel_all_reduce  (get_tp_group().all_reduce)

这个 POC 把 TP 的核心浓缩成 ~200 行，让你"看见"：

  1. 权重怎么切：Column 并行按输出维切、Row 并行按输入维切
  2. 每个 rank 只算出"部分结果"
  3. Row 并行的部分和 → all-reduce 求和 → 完整结果
  4. 哪一步需要通信（Column 不需要、Row 需要 all-reduce）
  5. 单卡完整计算 == 多卡 TP 计算（assert 逐元素相等）
"""

import math
import random

HIDDEN = 8        # hidden size
INTER = 16        # MLP intermediate size
TP = 2            # tensor parallel 度
NUM_LAYERS = 2    # transformer 层数
SEED = 42
TOL = 1e-5


# ---------------------------------------------------------------------------
# 纯 Python 二维列表矩阵运算（零依赖）
# ---------------------------------------------------------------------------

def matmul(A, B):
    """(m,k) x (k,n) -> (m,n)"""
    k, p = len(A[0]), len(B[0])
    return [[sum(A[i][t] * B[t][j] for t in range(k)) for j in range(p)]
            for i in range(len(A))]


def shape(M):
    return f"({len(M)}x{len(M[0])})"


def add(A, B):
    """逐元素相加（residual / all-reduce 用）"""
    return [[a + b for a, b in zip(ra, rb)] for ra, rb in zip(A, B)]


def all_reduce(parts):
    """模拟 tensor_model_parallel_all_reduce：所有 rank 的张量逐元素求和"""
    out = [row[:] for row in parts[0]]
    for p in parts[1:]:
        out = add(out, p)
    return out


def silu(x):
    return x / (1.0 + math.exp(-x))


def silu_mul(gu):
    """gu shard = [gate | up]，各 (1, INTER/TP) → silu(gate) * up"""
    half = len(gu[0]) // 2
    gate, up = gu[0][:half], gu[0][half:]
    return [[silu(g) * u for g, u in zip(gate, up)]]


def fmt(row, n=3):
    return "[" + " ".join(f"{v:+.{n}f}" for v in row) + "]"


def rand_mat(rows, cols, rng, scale=0.5):
    return [[rng.uniform(-scale, scale) for _ in range(cols)] for _ in range(rows)]


# ---------------------------------------------------------------------------
# 并行 Linear（对齐 vllm/model_executor/layers/linear.py）
# ---------------------------------------------------------------------------

class ColumnParallelLinear:
    """Y = X @ W，W (in, out) 沿输出维（最后一维）切。
    rank r 拿 W[:, r*out/TP : (r+1)*out/TP]，输出 (1, out/TP) 是局部结果。
    对应 ColumnParallelLinear：output_size_per_partition = divide(output_size, tp_size)
    forward 里 gather_output=False 时不通信。"""

    def __init__(self, W, tp_size):
        self.W = W
        self.tp_size = tp_size
        self.out_per = len(W[0]) // tp_size

    def weight(self, rank):
        s = rank * self.out_per
        return [row[s:s + self.out_per] for row in self.W]

    def forward(self, X, rank):
        return matmul(X, self.weight(rank))   # 局部结果，不通信


class RowParallelLinear:
    """Y = X @ W，W (in, out) 沿输入维（第一维）切。
    rank r 拿 W[r*in/TP : (r+1)*in/TP, :]，输入必须是切好的 shard，
    输出 (1, out) 是"部分和"，需要 all-reduce 求和才是完整结果。
    对应 RowParallelLinear：input_size_per_partition = divide(input_size, tp_size)
    forward 里 reduce_results=True 时调 tensor_model_parallel_all_reduce。"""

    def __init__(self, W, tp_size):
        self.W = W
        self.tp_size = tp_size
        self.in_per = len(W) // tp_size

    def weight(self, rank):
        s = rank * self.in_per
        return self.W[s:s + self.in_per]

    def forward(self, X_shard, rank):
        return matmul(X_shard, self.weight(rank))   # 部分和，通信在外部做


class MiniTransformerLayer:
    """一个 transformer 层：attention(QKV+O) + MLP(gate/up+down)

    gate/up 对齐真实 vLLM 的 MergedColumnParallelLinear：完整权重是
    [W_gate | W_up]，每个 rank 从 gate 和 up 里各拿 inter/TP 列，
    即 rank r 的 shard = [W_gate[:, r] | W_up[:, r]]（交错排列）。"""

    def __init__(self, rng):
        self.qkv = ColumnParallelLinear(rand_mat(HIDDEN, 3 * HIDDEN, rng), TP)
        self.o = RowParallelLinear(rand_mat(3 * HIDDEN, HIDDEN, rng), TP)
        self.W_gate = rand_mat(HIDDEN, INTER, rng)
        self.W_up = rand_mat(HIDDEN, INTER, rng)
        # 交错拼成 [G0 G1 | U0 U1]，让 ColumnParallelLinear 按列连续切分
        # 时，rank r 恰好拿到 [G_r | U_r]
        per = INTER // TP
        self.W_gate_up = [
            [v for r in range(TP)
             for v in (self.W_gate[i][r * per:(r + 1) * per]
                       + self.W_up[i][r * per:(r + 1) * per])]
            for i in range(HIDDEN)
        ]
        self.gate_up = ColumnParallelLinear(self.W_gate_up, TP)
        self.down = RowParallelLinear(rand_mat(INTER, HIDDEN, rng), TP)


# ---------------------------------------------------------------------------
# TP 前向（带 trace）
# ---------------------------------------------------------------------------

def forward_tp(layers, X, verbose=True):
    """TP 前向。X (1, H) 是 residual stream，在所有 rank 上复制（不切）。"""
    comm = 0
    for li, layer in enumerate(layers):
        if verbose:
            print(f"\n===== Layer {li} =====")

        # ---- Attention ----
        # 1) QKV：Column 并行 —— 按输出维切，各 rank 算自己的 shard
        qkv_shards = [layer.qkv.forward(X, r) for r in range(TP)]
        if verbose:
            print(f"  [QKV Column] W{shape(layer.qkv.W)} 按输出维切 -> "
                  + "  ".join(f"W{r}{shape(layer.qkv.weight(r))}" for r in range(TP))
                  + "   (不通信)")
            for r in range(TP):
                print(f"    rank{r} qkv_shard {shape(qkv_shards[r])}: {fmt(qkv_shards[r][0])}")
        # 2) attention：各 rank 用自己那部分 head 本地算，不通信
        #    （简化：直接透传 shard；真实 attention 在这里做 per-head 计算）
        attn_shards = qkv_shards
        if verbose:
            print(f"  [Attention ] 各 rank 本地算自己的 head   (不通信)")
        # 3) O：Row 并行 —— 按输入维切，各 rank 得到"部分和"
        o_partials = [layer.o.forward(attn_shards[r], r) for r in range(TP)]
        if verbose:
            print(f"  [O Row     ] W{shape(layer.o.W)} 按输入维切 -> "
                  + "  ".join(f"W{r}{shape(layer.o.weight(r))}" for r in range(TP)))
            for r in range(TP):
                print(f"    rank{r} o_partial {shape(o_partials[r])}: {fmt(o_partials[r][0])}")
        # 4) all-reduce：部分和求和 -> 完整结果
        o_out = all_reduce(o_partials)
        comm += 1
        if verbose:
            print(f"  [all-reduce #{comm}] 部分和求和 -> {shape(o_out)}: {fmt(o_out[0])}")
        X = add(X, o_out)   # residual：所有 rank 复制，不通信

        # ---- MLP ----
        # 5) gate/up：Column 并行 —— 按输出维切
        gu_shards = [layer.gate_up.forward(X, r) for r in range(TP)]
        if verbose:
            print(f"  [gate/up Column] W{shape(layer.gate_up.W)} 按输出维切 -> "
                  + "  ".join(f"W{r}{shape(layer.gate_up.weight(r))}" for r in range(TP))
                  + "   (不通信)")
            for r in range(TP):
                print(f"    rank{r} gu_shard {shape(gu_shards[r])}: {fmt(gu_shards[r][0])}")
        # 6) silu(gate) * up：逐元素，各 rank 本地算自己的 shard，不通信
        h_shards = [silu_mul(gu_shards[r]) for r in range(TP)]
        if verbose:
            for r in range(TP):
                print(f"    rank{r} h_shard  {shape(h_shards[r])}: {fmt(h_shards[r][0])}")
        # 7) down：Row 并行 —— 按输入维切，各 rank 得到"部分和"
        d_partials = [layer.down.forward(h_shards[r], r) for r in range(TP)]
        if verbose:
            print(f"  [down Row    ] W{shape(layer.down.W)} 按输入维切 -> "
                  + "  ".join(f"W{r}{shape(layer.down.weight(r))}" for r in range(TP)))
            for r in range(TP):
                print(f"    rank{r} d_partial {shape(d_partials[r])}: {fmt(d_partials[r][0])}")
        # 8) all-reduce：部分和求和 -> 完整结果
        d_out = all_reduce(d_partials)
        comm += 1
        if verbose:
            print(f"  [all-reduce #{comm}] 部分和求和 -> {shape(d_out)}: {fmt(d_out[0])}")
        X = add(X, d_out)   # residual：所有 rank 复制，不通信

    return X, comm


def forward_single(layers, X):
    """单卡完整计算（同一套权重，不切分）—— 作为正确性参照

    注意：gate/up 这里用"不交错"的 [W_gate | W_up] 布局（gate 全量在前、
    up 全量在后），这样 silu_mul 按前后各一半切分才正确。它和 TP 用的
    交错布局 [G0 U0 G1 U1] 数学上等价：
        单卡 h = [silu(X@G0)*(X@U0), silu(X@G1)*(X@U1)] = [h0, h1]
        h @ D  = h0 @ D0 + h1 @ D1  （D 按行堆叠 [D0; D1]）
    正好等于 TP 各 rank 算 h_r 再 all-reduce 求和。"""
    for layer in layers:
        qkv = matmul(X, layer.qkv.W)          # (1, 3H)
        o = matmul(qkv, layer.o.W)            # (1, H)
        X = add(X, o)
        g = matmul(X, layer.W_gate)[0]        # (1, INTER)
        u = matmul(X, layer.W_up)[0]          # (1, INTER)
        gu = [g + u]                          # 拼成 [gate | up]
        h = silu_mul(gu)
        d = matmul(h, layer.down.W)           # (1, H)
        X = add(X, d)
    return X


# ---------------------------------------------------------------------------
# 演示
# ---------------------------------------------------------------------------

def main():
    rng = random.Random(SEED)
    layers = [MiniTransformerLayer(rng) for _ in range(NUM_LAYERS)]
    X0 = [[rng.uniform(-1, 1) for _ in range(HIDDEN)]]

    print("=" * 72)
    print(f"场景：{NUM_LAYERS} 层 mini-transformer，TP={TP}，hidden={HIDDEN}, "
          f"intermediate={INTER}")
    print("  每层：QKV(Column) -> Attention(本地) -> O(Row) -> gate/up(Column)")
    print("        -> silu*up(本地) -> down(Row)")
    print("=" * 72)

    # 权重切分一览（以 Layer 0 为例）
    l0 = layers[0]
    print("\n===== 权重切分（Layer 0）=====")
    for name, lin, how in [
        ("qkv", l0.qkv, "输出维"), ("o", l0.o, "输入维"),
        ("gate_up", l0.gate_up, "输出维"), ("down", l0.down, "输入维"),
    ]:
        shards = "  ".join(f"W{r}{shape(lin.weight(r))}" for r in range(TP))
        print(f"  {name:8s} {shape(lin.W)} 按{how}切 -> {shards}")

    print(f"\n输入 X {shape(X0)}: {fmt(X0[0])}  (residual stream，所有 rank 复制)")

    X_tp, comm = forward_tp(layers, X0)
    X_ref = forward_single(layers, X0)

    # ---- 正确性：单卡完整计算 == 多卡 TP 计算 ----
    max_diff = max(abs(a - b) for ra, rb in zip(X_tp, X_ref)
                   for a, b in zip(ra, rb))
    print("\n===== 正确性校验 =====")
    print(f"  单卡完整计算: {fmt(X_ref[0])}")
    print(f"  TP={TP} 计算结果: {fmt(X_tp[0])}")
    print(f"  最大逐元素差: {max_diff:.2e} (容差 {TOL:.0e})")
    assert max_diff < TOL, "TP 结果与单卡结果不一致！"
    print("  ✓ assert 通过：单卡完整计算 == 多卡 TP 计算")

    # ---- 通信统计 ----
    print("\n===== 通信统计 =====")
    print(f"  一个 transformer 层 (TP={TP}) 需要 {comm // NUM_LAYERS} 次 all-reduce："
          f"attention 的 O 投影 + MLP 的 down 投影")
    print(f"  {NUM_LAYERS} 层共 {comm} 次 all-reduce")
    print("  trade-off：TP 越大，每层 all-reduce 次数不变（仍是 2 次），")
    print("    但每次 all-reduce 参与的 rank 更多、数据量更大、延迟更高；")
    print("    好处是每卡权重/激活显存按 TP 线性下降。")
    print("    所以 TP 一般限制在单机内（如 8 卡），跨机扩展用 PP/EP。")


if __name__ == "__main__":
    main()
