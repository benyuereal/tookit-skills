# Task10：head_compute_mix_bwd

## 优化思路

**瓶颈：** Launch-bound，多个逐元素 + 多路归约。

**核心洞察：前向+反向一次性完成，双 4096 tile + atomic 汇总**

- 两个主要 program：head_grad 计算（逐元素融合）+ input_grad 汇总（atomic add）
- 双 4096 tile 避免单 tile 过大导致寄存器溢出

## 关键代码

```python
# head_grad 计算：逐元素融合
# grid = (batch * seq, heads * dim_per_head)
pid = tl.program_id(0)
off = pid * BLOCK + tl.arange(0, BLOCK)

grad = tl.load(grad_ptr + off)
head_w = tl.load(head_ptr + off)
mix = grad * head_w  # 融合：不再经过中间张量

# atomic 汇总 input_grad（跨 program 归约）
# input_grad 需要汇总所有 head 的贡献
for head_i in range(num_heads):
    head_offset = head_i * dim_per_head
    mix_h = tl.load(mix_ptr + head_offset + off)  # 每个 head 的 mix
    tl.atomic_add(input_grad_ptr + head_offset + off, mix_h)

# reduce_sum 用于跨 batch/seq 归约
head_grad = tl.sum(mix[:, None] * input[None, :], axis=0)
```

## 陷阱

- `atomic add` 顺序不确定，但加法满足交换律所以正确
- 两个 4096 tile 是实测最优，过大会 VGPR spill
- 反向顺序：先算 `mix` 再算 `head_grad`（依赖关系）

## BW1000 结果

PyTorch 0.176ms → Triton 0.096ms，**1.85×**