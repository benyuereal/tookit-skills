# Task06：MMEncoderAttention

## 优化思路

**瓶颈：** 混合型（Attention 计算 + 内存）。

**核心洞察：** 与 Task03 类似，但 encoder 是双向的（无 causal mask），query tile 共享 K/V 流式扫描。

**跨平台差异：** Ascend 910B 上 query tile 调为 32×32。

## 关键代码

```python
# grid = (ceil_div(seq_q, BLOCK_Q), batch * heads)
# query tile 一次加载完整，K/V 按块流式
pid_q = tl.program_id(0)
off_q = pid_q * BLOCK_Q + tl.arange(0, BLOCK_Q)
off_k = tl.arange(0, BLOCK_KV)

# 双向 attention：不需要 causal mask，直接参与 online softmax
for start_k in range(0, seq_k, BLOCK_KV):
    k = tl.load(k_ptr + off_k[:, None]*stride_k + start_k*stride_kk)
    v = tl.load(v_ptr + off_k[:, None]*stride_v + start_k*stride_vv)

    score = tl.dot(q, k) / tl.sqrt(tl.float32(head_dim))  # [BLOCK_Q, BLOCK_KV]
    # 在线 softmax（无需 mask）
    m_new = tl.max(tl.concatenate([m_old, tl.max(score, axis=1)]))
    alpha = tl.exp(m_old - m_new)
    p = tl.exp(score - m_new)
    l_new = alpha * l_old + tl.sum(p, axis=1)
    acc_new = alpha * acc_old + p @ v
    # ... 状态更新 ...
```

## 陷阱

- encoder 双向，不需要 causal mask
- 处理 `seq_k % BLOCK_KV != 0` 的尾部边界

## BW1000 结果

PyTorch 0.141ms → Triton 0.085ms，**1.66×**