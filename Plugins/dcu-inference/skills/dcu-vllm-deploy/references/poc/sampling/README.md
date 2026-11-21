# 采样管线最小化 POC —— temperature / top-k / top-p / penalties 是怎么实现的

一个 ~200 行的纯 Python 采样器，把 vLLM V1 采样管线的核心步骤浓缩出来，
让你**跑起来、看见**每一步的中间状态（logits 怎么被一步步 mask 成最终分布）。

只依赖 Python 标准库（`math` + `random`），CPU 直接跑，不需要 GPU/DCU：

```bash
python3 mini_sampling.py
```

## 对应真实 vLLM 源码

| POC 里的函数 | 真实 vLLM 位置 | 说明 |
|--------------|----------------|------|
| `MiniSampler.sample` 的管线顺序 | `vllm/v1/sample/sampler.py::Sampler.forward` / `sample` | 顺序：penalties → temperature → top_k/top_p → 采样；`temperature < 1e-5`（`_SAMPLING_EPS`）走 `greedy_sample`（argmax）快路径，不采样 |
| `apply_temperature` | `sampler.py::Sampler.apply_temperature` | `logits / temperature`（in-place div） |
| `apply_top_k` | `vllm/v1/sample/ops/topk_topp_sampler.py::apply_top_k_only` | 非 top-k 的 token 置 `-inf` |
| `apply_top_p` | `topk_topp_sampler.py::apply_top_k_top_p` | logits **升序**排序 → softmax → cumsum → mask 掉 `probs_sum <= 1-p` 的 token，最后一个强制保留（`top_p_mask[:, -1] = False`） |
| `random_sample` | `topk_topp_sampler.py::random_sample` | 不用 `multinomial`（会 CPU-GPU 同步），用 `argmax(probs / q)`，`q ~ Exp(1)`（Gumbel-max 技巧） |
| `apply_penalties` | `vllm/v1/sample/ops/penalties.py::apply_all_penalties` → `vllm/model_executor/layers/utils.py::apply_penalties` | OpenAI 定义：`logits -= freq_pen * 出现次数`；`logits -= pres_pen * (是否出现过)` |

## 它演示了什么

| 演示 | 在 trace 里怎么看 |
|------|------------------|
| **完整管线** | 演示 1：同一份 logits 依次经过 temperature → top-k → top-p，每步打印中间 logits，被 mask 的 token 显示 `-inf` |
| **batch 内参数不同** | 演示 2：3 个请求（greedy / top-k=5 / top-p=0.9）共用同一份 logits，打印 mask 前后的概率分布对比 |
| **top-p 截断点** | 演示 3：cumsum 曲线（升序），标出"最后一个 `probs_sum <= 1-p`"的 token 在哪 |
| **temperature 形状** | 演示 4：t=0.1 / 1.0 / 2.0 三组 softmax，看分布从尖锐（接近 one-hot）变平坦（接近均匀） |
| **penalties** | 演示 5：已生成 `[cat, cat, dog]`，看 cat/dog 的 logits 被压低、未出现的 token 不变 |
| **采样正确性** | 演示 6：固定 seed 采 20 次，实测频率 vs 理论概率 |

## 怎么读 trace

### 演示 1：管线每一步

```
  原始 logits | the: 2.10 cat: 3.40 sat: 1.20 on: 0.50 mat: 2.80 and: 0.10 dog: 1.90 saw: 0.30 a: 1.50 fish: 0.80
    temperature  | the:  2.62 cat:  4.25 ...          ← logits / 0.8，整体放大
    top-k=6      | ... on:  -inf and:  -inf saw:  -inf fish:  -inf   ← 只留最大的 6 个
    top-p=0.9    | ... sat:  -inf a:  -inf            ← 再砍掉尾部，只剩 4 个
    最终分布       | the: 0.108 cat: 0.549 mat: 0.259 dog: 0.084 ...  ← 剩余 token 重新归一化
```

注意 top-k 和 top-p 是**叠加**的：先 top-k 砍到 6 个，top-p 在这 6 个里再砍到 4 个。
被 mask 的 token 概率为 0，剩下的概率重新归一化（所以 cat 从 0.397 涨到 0.549）。

### 演示 3：top-p 的 cumsum 曲线

```
    rank 3 fish  p= 0.0295  cumsum= 0.0840  [mask]  ← 截断点(最后一个 probs_sum<=0.1)
    rank 4 sat   p= 0.0440  cumsum= 0.1280  [保留]
```

top-p=0.9 的含义：按概率**从小到大**累加，把 `cumsum <= 1-0.9=0.1` 的 token 全部 mask。
fish 是最后一个 cumsum 还没超过 0.1 的 token，sat 是第一个超过的——截断点就在两者之间。
保留下来的 token 概率之和 ≥ 0.9（实际是 0.928）。

### 演示 4：temperature

```
    t=0.1  | cat: 0.998 ...        ← 尖锐，几乎 one-hot
    t=1.0  | cat: 0.397 mat: 0.218 ...
    t=2.0  | cat: 0.229 mat: 0.170 ...  ← 平坦，接近均匀
```

temperature 是 logits 的除数：t 小 → logits 差距被放大 → softmax 更尖锐；
t 大 → 差距被压缩 → 分布更平坦。t→0 的极限就是 argmax（greedy）。

### 演示 5：penalties

```
    cat  :  3.40 →  0.90  (出现 2 次: -1.0*2-0.5)
    dog  :  1.90 →  0.40  (出现 1 次: -1.0*1-0.5)
    on   :  0.50 →  0.50  (未出现: 不变)
```

- **frequency penalty**：按出现**次数**线性压低（cat 出现 2 次，压 2 倍）
- **presence penalty**：只要出现过就压一次（不管几次）
- 两者都只作用于**已生成**的 token，未出现的 token 完全不变

### 演示 6：采样正确性

固定 `seed=42` 采 20 次，实测频率和理论概率（softmax 值）对比。
20 次样本太少、噪声大（比如 `the` 理论 0.108 实测 0.200）；
把循环改成 2000 次，实测会明显贴近理论值——这就是对 `random_sample`
（Gumbel-max 技巧）正确性的验证。

## 和真实 vLLM 的差异（简化点）

| 简化 | 真实 vLLM |
|------|----------|
| 每个请求单独循环处理 | 整个 batch 是 `[num_reqs, vocab]` 的 2D tensor，一次算完（向量化） |
| top-k 用 Python `sorted` 找 top-k | `apply_top_k_only` 用 `torch.topk` 找阈值，不排序整个 vocab |
| top-p 用 Python 排序 + 累加 | `apply_top_k_top_p` 用 `torch.sort` + `cumsum` + `masked_fill_`，k 和 p 在同一趟排序里一起处理 |
| 采样用 `random.random()` 逐个生成 | `random_sample` 用 `q.exponential_()` 批量生成 Exp(1) 随机数，`argmax(probs/q)` |
| 没有 repetition penalty | 还有 `repetition_penalty`（乘法，作用于 prompt+output 都出现过的 token，走自定义 CUDA op） |
| 没有 bad words / allowed_token_ids / min_tokens / logit bias | `apply_logits_processors` 里还有一整套 logits processor（bad words、allowed ids、min tokens 等） |
| 没有 per-request seed | 支持每个请求独立的 `torch.Generator`（`generators` dict） |
| 没有 FlashInfer / aiter / lightop 加速路径 | CUDA 上可切 `forward_cuda`（FlashInfer rejection sampling，避免排序）等 |
| greedy 判断用 `temperature < 1e-5` | 同（`_SAMPLING_EPS`），且 greedy 结果在 batch 里用 `torch.where` 和随机采样结果合并 |

想扩展的话，可以加：repetition penalty、bad words、min_tokens、per-request seed、
或者把 20 次采样改成 2000 次看频率收敛。
