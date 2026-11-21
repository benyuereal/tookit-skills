# Task07：mhc_post

## 优化思路

**瓶颈：** Memory-bound，四路 residual 都要重复读 `x[row, hidden]`。

**核心洞察：`(batch, token)` 合并为 row，一次 load，4 次复用**

```python
# 参考：x 被读 4 次（每路一次），4 个独立 kernel
out1 = layer_norm(x + gamma1 * h1)
out2 = layer_norm(x + gamma2 * h2)
...

# 融合：x 一次 load，4 次片上复用，1 个 kernel
x = tl.load(x_ptr + off)          # 一次读
h1 = tl.load(h1_ptr + off)
h2 = tl.load(h2_ptr + off)
...
out1 = layer_norm(x + gamma1 * h1)  # 片上复用
out2 = layer_norm(x + gamma2 * h2)  # ...
...
```

## 关键代码

```python
# grid = (batch * seq,) = (83,)
# 输出 BF16，计算过程 FP32
pid = tl.program_id(0)
row = pid
off = row * hidden + tl.arange(0, hidden)

x = tl.load(x_ptr + off)

# 四路并行计算，x 只加载一次
h1 = tl.load(h1_ptr + off); g1 = tl.load(g1_ptr + off)
h2 = tl.load(h2_ptr + off); g2 = tl.load(g2_ptr + off)
...

def layer_norm(x, w, eps=1e-5):
    mean = tl.sum(x) / N
    var = tl.sum((x - mean) * (x - mean)) / N
    return (x - mean) / tl.sqrt(var + eps) * w

out1 = layer_norm(x + g1 * h1, w1)
...

# store，转换回 BF16
tl.store(out1_ptr + off, out1.to(tl.bfloat16))
```

## 陷阱

- 四路归一化各自独立（每路的 mean/var 不共享）
- C500 上需设置 NCHW 布局

## BW1000 结果

PyTorch 2.569ms → Triton 0.223ms，**11.51×**