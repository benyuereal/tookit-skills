# Task02：FusedMoE

## 优化思路

**瓶颈：** Compute-bound，大量 GEMM 计算。

**核心洞察（三阶段设计，不是越大越好）：**

```
阶段1：门控与专家选择  — 少量计算，不融合进 GEMM
阶段2：专家激活 GEMM   — 计算密集，tl.dot 分块
阶段3：加权输出归一化  — 小计算，融合
```

强行把两层专家网络+路由+所有 token 放单 kernel → 寄存器状态过大 + 动态布尔分发损害 SIMD 效率。

## 关键代码

```python
# 阶段1：Top-k 路由
topk_weight, topk_idx = torch.topk(gating_output, topk)  # [T, topk]
topk_weight = topk_weight / topk_weight.sum(dim=-1, keepdim=True)

# 阶段2：分块 GEMM（BLOCK_M=32, BLOCK_N=128）
# 注意： BLOCK_K 必须与 hidden 维度对齐
a = tl.load(w1_ptr + expert * intermediate * hidden + pid_m * hidden + off_h)  # [BLOCK_M, hidden]
b = tl.load(w2_ptr + expert * hidden + off_h[:, None] * BLOCK_N)                 # [hidden, BLOCK_N]
acc = tl.dot(a, b)   # FP16 累加 → 内部 FP32

# 阶段3：分阶段加回
result = (acc @ w3.T) * topk_weight[:, None, None]  # [T, topk, intermediate]
result = tl.sum(result, axis=1)                       # [T, intermediate]
```

## 陷阱

- 参数名必须与 reference 的 `state_dict` 对齐（`w1`、`decoder.weight` 等）
- accumulator 用 FP32，最终 store 转 FP16/BF16
- BLOCK_N 检查是否覆盖完整 hidden 维度

## BW1000 结果

PyTorch 2.602ms → Triton 0.145ms，**17.92×**