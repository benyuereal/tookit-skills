# Task08：hc_split_sinkhorn

## 优化思路

**瓶颈：** Launch-bound，20 轮 × 4 个小操作 → 40+ 个小 kernel。

**核心洞察：4×4 矩阵片上驻留 20 轮，最后一次性写回**

- `comb` 只有 16 个 FP32 元素，row/col sum 各 4 个
- `sinkhorn_iters = tl.constexpr(20)` 循环在编译期展开
- 循环期间没有 `tl.store`，状态永不写回全局内存

**这是寄存器驻留的理想场景：状态极小、重复使用次数极多。**

## 关键代码

```python
# grid = (B*S,) = (16,)，每 program = 一行（一个 4×4 矩阵）
pid = tl.program_id(0)
hc = tl.arange(0, 4)

# 一次 load 得到 [4,4] 块张量
matrix_offsets = hc[:, None] * 4 + hc[None, :]  # [[0,1,2,3],[4,5,6,7],...]
comb = tl.load(comb_ptr + pid * 16 + matrix_offsets)  # [4, 4]

# 静态循环，片上迭代 20 轮
for _ in range(sinkhorn_iters - 1):
    # 行归一化：[:, None] 广播行统计量到行
    row_sum = tl.sum(comb, axis=1)          # [4]
    comb = comb / row_sum[:, None]

    # 列归一化：[None, :] 广播列统计量到列
    col_sum = tl.sum(comb, axis=0)          # [4]
    comb = comb / (col_sum[None, :] + eps)

# 最后一次性写 pre/post/comb
tl.store(pre_ptr + pid * 4 + hc, pre.to(tl.float16))
tl.store(post_ptr + pid * 4 + hc, post.to(tl.float16))
tl.store(comb_ptr + pid * 16 + matrix_offsets, comb)
```

## 陷阱

- `num_warps=1`：状态极小（16 元素），更多 warps 增加调度开销
- eps 顺序：首轮 row normalize 后先 `+eps` 再 column normalize
- exp 前必须减 row max 保持数值稳定

## BW1000 结果

PyTorch 1.399ms → Triton 0.103ms，**13.57×**