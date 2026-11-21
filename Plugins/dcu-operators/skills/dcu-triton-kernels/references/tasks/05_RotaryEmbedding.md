# Task05：MusicFlamingoRotaryEmbedding

## 优化思路

**瓶颈：** Memory-bound，broadcast/concat 后逐元素三角函数链。

**核心洞察：一次 load + 广播融合三角函数 + store**

参考实现经过 `expand → cat → stack → mul_add` 多个小 kernel，每个触发框架调度。融合为单 kernel。

## 关键代码

```python
# program 映射：grid=(seq,)，每 program = 一个 token
pid = tl.program_id(0)
off = pid * dim + tl.arange(0, dim)

# 广播乘法在 Triton 中显式写指针算术
pos = tl.load(pos_ptr + pid * 1)           # [1]
freq = tl.load(inv_freq_ptr + off_half)    # [BLOCK]
angle = pos * freq                         # broadcast to [BLOCK]

# 融合旋转：q * cos + rotate_half(q) * sin
sin = tl.sin(angle)
cos = tl.cos(angle)

x1 = tl.load(x_ptr + off)
x2 = tl.load(x_ptr + off + stride_half)
rot_x1 = x1 * cos - x2 * sin
rot_x2 = x1 * sin + x2 * cos
```

## 陷阱

- 旋转顺序：`rotate_half(q) * sin + q * cos`（不是 `-` 和 `+` 交换）
- `num_warps=1`：逐元素计算，块内并行量小
- 输入通常是 FP16/BF16，计算内部 FP32

## BW1000 结果

PyTorch 0.254ms → Triton 0.098ms，**2.60×**