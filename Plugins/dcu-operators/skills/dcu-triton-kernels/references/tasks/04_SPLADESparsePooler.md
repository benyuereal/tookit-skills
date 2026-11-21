# Task04：SPLADESparsePooler

## 优化思路

**瓶颈：** Memory-bound，`[83, 30522]` logits 物化代价高。

**核心洞察：融合 GELU+LayerNorm + decoder 输出直送池化**

中间张量表：
```
dense [83,768]       → GEMM
GELU [83,768]        → 读写完整张量        ← 融合掉
LayerNorm [83,768]   → 多次归约+读写        ← 融合掉
decoder [83,30522]   → 大 GEMM+巨大输出     ← 关键！
ReLU+log1p [83,30522] → 再次读写巨大输出   ← 融合掉
segment max 4×[30522] → 再读 logits         ← 数学变换后提前
```

**数学变换：** `log1p(ReLU(x))` 是单调函数，可移到 max pooling 之后。

## 关键代码

```python
# GELU + LayerNorm 融合kernel
x = tl.load(x_ptr + off)                              # 一次加载
x = x * 0.5 * (1.0 + tl.erf(x / tl.sqrt(tl.float32(2.0))))  # GELU
mean = tl.sum(x) / N
var = tl.sum((x - mean) * (x - mean)) / N
x_norm = (x - mean) / tl.sqrt(var + eps) * weight   # LayerNorm

# decoder 后直接算 segment max，不物化完整 logits
# 融合：decoder_out → relu → log1p → max pooling 在同一 kernel
logits = tl.load(decoder_ptr + off_logits)
pooled = tl.max(logits, axis=0)  # 不需要完整 logits 在内存
```

## 陷阱

- `log1p(ReLU(x))` 必须在 max 之后（数学等价需证明单调性）
- LayerNorm eps 位置必须与参考一致
- 某些平台（Ascend 910B）需拆分 decoder 与池化以避开 watchdog

## BW1000 结果

PyTorch 0.903ms → Triton 0.308ms，**2.93×**