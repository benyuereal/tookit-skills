# Task01：GroupedTopk

## 优化思路

**瓶颈：** Launch-bound，多个小 kernel 调度开销 > 算术量。

**核心洞察（两步数学证明）：**

1. `exp` 单调 → **在 logits 上直接选 Top-k，不先算 Softmax**
2. 二次归一化时分母抵消 → **只对选中的 8 个专家归一化，跳过 256 路分母**

## 关键代码

```python
# program 映射：grid=(83,)，每 program = 一行 256 专家
row = tl.program_id(0)
expert = tl.arange(0, 256)
logits = tl.load(gating_ptr + row * 256 + expert)  # [256]

# 组分数：reshape 后取 max → 8 个组分数
group_logits = tl.reshape(logits, (8, 32))
group_scores = tl.max(group_logits, axis=1)          # [8]

# 选 Top-4 组：逐轮 argmax → 置 -inf → 再选（编译器展开，无循环开销）
for i in range(4):
    g = tl.argmax(group_scores)
    group_scores = tl.where(tl.arange(8) == g, -float("inf"), group_scores)

# 专家候选：选中的组内专家 mask
selected_mask = (expert_group == g0) | (expert_group == g1) | ...
candidates = tl.where(selected_mask, logits, -float("inf"))

# 8 轮 argmax 选 Top-8 专家（每轮把已选置 -inf）
for i in range(8):
    idx_i = tl.argmax(candidates)
    candidates = tl.where(tl.arange(256) == idx_i, -float("inf"), candidates)

# 稳定归一化：减最大值避免 exp 溢出（v0=exp(0)=1）
v0 = tl.max(selected_logits)
e = tl.exp(selected_logits - v0)
denom = tl.sum(e)
weight = e / denom  # 8 个权重，FP32
```

## 陷阱

- `num_warps=1`：块内只有 256 元素，增加 warps 只增调度开销
- ID 次序用 `torch.equal` 比较，不是 `allclose`
- 配置专用化：`assert (topk, G, topk_group) == (8, 8, 4)`

## BW1000 结果

PyTorch 0.276ms → Triton 0.089ms，**3.10×**