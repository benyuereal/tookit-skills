# Task03：FlexAttention

## 优化思路

**瓶颈：** 混合型（score 矩阵 O(n²) 内存 + 大量矩阵计算）。

**核心洞察：在线 Softmax** — 分块处理，状态只保存 `max / sum / acc`，内存 O(n)。

```
传统：完整 [seq, seq] score 矩阵 → O(n²) 内存
在线：每次只加载一个 K/V 块 → O(1) 内存
```

## 关键代码

```python
# 在线 Softmax 公式
m_new = tl.max(tl.concatenate([m_old, m_blk]))
alpha = tl.exp(m_old - m_new)
p_blk = tl.exp(score_blk - m_new)
l_new = alpha * l_old + tl.sum(p_blk, axis=1)
acc_new = alpha * acc_old + p_blk @ v_blk
out = acc_new / l_new[:, None]

# program 映射：grid=(ceil_div(seq, BLOCK_Q), batch*heads)
# query tile 一次加载，K/V 按 BLOCK_KV 流式扫描
pid = tl.program_id(0)
off_q = pid * BLOCK_Q + tl.arange(0, BLOCK_Q)
off_k = tl.arange(0, BLOCK_KV)

for start_k in range(0, seq, BLOCK_KV):
    k = tl.load(k_ptr + off_k[:, None]*stride_k + start_k*stride_kk)  # [BLOCK_KV, head_dim]
    score = tl.dot(q, k) / tl.sqrt(tl.float32(head_dim))               # [BLOCK_Q, BLOCK_KV]
    # ... online softmax 更新 ...
```

## 陷阱

- `num_warps=4`：K/V 循环计算密集，需要更多并行资源
- mask 必须在 exp 前应用，不能在 exp 后
- 每次块计算前重新对齐 max，避免累加溢出

## BW1000 结果

PyTorch 0.144ms → Triton 0.087ms，**1.66×**