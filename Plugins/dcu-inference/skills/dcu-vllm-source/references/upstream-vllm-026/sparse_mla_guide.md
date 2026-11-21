# MQA Logits 与 Sparse MLA —— 稀疏注意力的 Indexer + Attention 两阶段流水线

> 在 DeepSeek-V3、GLM-5 等大模型中，处理超长上下文的核心技术是**稀疏注意力**。
> 它将注意力计算拆成两步：先用轻量级 Indexer 快速筛选出重要的 KV 位置，
> 再对这些位置做精确的注意力计算，在保证精度的前提下大幅缩减计算量。

---

## 1. 为什么需要稀疏注意力？

### 1.1 标准注意力的困境

```
              全量 KV ( 131072 个 token)
              ┌────────────────────────────┐
              │ K₀ K₁ K₂ K₃ ... K₁₃₁₀₇₁ │
              │ ██████████████████████████ │
              │ ██████████████████████████ │
Q @ K^T ────→ │ ████████████████████████  │ ← 每个 Q 都要和 ALL KV 做内积！
              │ ████████████████████████  │
              │ ██████████████████████████ │
              └────────────────────────────┘
              
              计算量: O(S × SKV × H × D)
              131072 × 131072 × 32 × 128 ≈ 7 × 10¹³ 次乘加！
```

对于 128K 上下文，标准注意力的计算量是 **灾难性的**，延迟高到无法接受。

### 1.2 稀疏注意力的核心思想

**真相：** 对于每个 query token，真正重要的 KV token 只占极少数（通常只有几千个）。
绝大多数 KV token 的注意力分数接近 0，计算它们是浪费。

```
   Q₀  关注的 KV:  [ ████████████░░░░░░░░░░░░░░░░░░░░ ]  ~2048/131072 有效
   Q₁  关注的 KV:  [ ░░░░████████████░░░░░░░░░░░░░░░░░░ ]  ~2048/131072 有效
   Q₂  关注的 KV:  [ ░░░░░░░░░░██████████████░░░░░░░░░░ ]  ~2048/131072 有效
                        ↑                        ↑
                   TopK ≈ 2048              Skip ≈ 129024
```

**关键问题：** 怎么在不计算全量注意力的前提下，快速找出这 TopK 个重要位置？

**答案：** 用一个轻量级 Indexer 做快速初筛。

---

## 2. 两步走：Indexer + Sparse Attention

整个稀疏注意力流水线分为两个阶段：

```
╔═══════════════════════════════════════════════════════════════════════════╗
║                        稀疏注意力计算流水线                                  ║
║                                                                           ║
║  ┌─────────────────────────────────────┐                                 ║
║  │  阶段1: Indexer                      │                                 ║
║  │  ┌─────────────────────────────┐    │                                 ║
║  │  │       mqa_logits            │    │  输入: Q [S,H,D], KV全量[SKV,D]  ║
║  │  │  ReLU(K@Q^T) × weights     │    │  输出: logits [S,SKV]            ║
║  │  │         ↓                   │    │                                 ║
║  │  │     TopK 排序               │    │  选出每行最大的 TopK 个位置       ║
║  │  │         ↓                   │    │                                 ║
║  │  │  topk_indices [S,HKV,topk]  │    │  传给阶段2                       ║
║  │  └─────────────────────────────┘    │                                 ║
║  └────────────────┬────────────────────┘                                 ║
║                   │ topk_indices                                         ║
║                   ▼                                                      ║
║  ┌─────────────────────────────────────┐                                 ║
║  │  阶段2: Sparse Attention             │                                 ║
║  │  ┌─────────────────────────────┐    │                                 ║
║  │  │     sparse_mla_fwd          │    │  输入: Q, KV全量, topk_indices   ║
║  │  │  对TopK位置做完整 Softmax    │    │                                 ║
║  │  │  + S @ V 加权求和           │    │  输出: output [S,H,D_v]          ║
║  │  │         ↓                   │    │                                 ║
║  │  │  attention_output [S,H,D_v]  │    │  最终的注意力计算结果             ║
║  │  └─────────────────────────────┘    │                                 ║
║  └─────────────────────────────────────┘                                 ║
║                                                                           ║
║  节省计算量:                                                              ║
║    标准注意力: O(S × SKV × H × D)  ≈ 7 × 10¹³ FLOPs                      ║
║    稀疏注意力: O(S × SKV × H_eff × D_indexer) + O(S × TopK × H × D_attn) ║
║              ≈ O(S × SKV × D)        + O(S × 2048 × H × D)               ║
║              ≈ 计算量减少了 30×+                                            ║
╚═══════════════════════════════════════════════════════════════════════════╝
```

---

## 3. vLLM 中的代码模块映射

在 vLLM 中，阶段1 和阶段2 分别由两个模块实现，它们通过 `topk_indices_buffer` 共享数据：

```
文件映射:
  阶段1 (Indexer):   vllm/model_executor/layers/sparse_attn_indexer.py
                     └─ sparse_attn_indexer()  自定义算子入口
                        ├─ mqa_logits()         Prefill: 连续内存 MQA
                        ├─ paged_mqa_logits()   Decode: 分页 MQA
                        └─ top_k_per_row()      TopK 筛选

  阶段2 (Sparse MLA): vllm/v1/attention/backends/mla/flashmla_sparse.py
                     └─ FlashMLASparseImpl.forward()
                        ├─ _bf16_flash_mla_kernel() → flash_mla_sparse_fwd()   ← BF16 注意力
                        └─ _fp8_flash_mla_kernel()  → flash_mla_with_kvcache() ← FP8 注意力

  串联层 (Wrapper):  vllm/model_executor/layers/mla.py
                     └─ MultiHeadLatentAttentionWrapper.forward()
                        ① self.indexer(...)        → 阶段1, 填 topk_indices_buffer
                        ② self.mla_attn(...)       → 阶段2, 读 topk_indices_buffer
```

### 3.1 调用链（精简版）

```
DecoderLayer.forward(hidden_states)
│
├─ MultiHeadLatentAttentionWrapper.forward()
│
│  ┌── ① Indexer.forward() ──────────────────────────┐
│  │     sparse_attn_indexer()                         │
│  │       ├─ 将当前 K 存入 Indexer KV Cache           │
│  │       ├─ [Prefill] mqa_logits() + TopK           │
│  │       └─ [Decode]  paged_mqa_logits() + TopK     │
│  │     → topk_indices_buffer [num_tokens, 2048]     │
│  └──────────────────────────────────────────────────┘
│
│  ┌── ② MLAAttention.forward() ──────────────────────┐
│  │     FlashMLASparseImpl.forward()                  │
│  │       ├─ Q 低秩吸收 (q_nope · W_UK^T)             │
│  │       ├─ 写入 KV Cache                            │
│  │       └─ flash_mla_sparse_fwd() / flash_mla_with_kvcache()
│  │           (仅对 topk_indices 指定的 2048 个位置)   │
│  │     → attn_out [num_tokens, num_heads, kv_lora]  │
│  └──────────────────────────────────────────────────┘
│
│  ┌── ③ v_up_proj + o_proj ──────────────────────────┐
│  │     attn_out → kv_b_proj → o_proj → output        │
│  └──────────────────────────────────────────────────┘
```

---

## 4. 阶段1 拆解：mqa_logits

### 4.1 职责

Indexer 的任务是快速筛选 TopK 候选位置，而非精确计算——它用极简的相似度函数（ReLU，不做 softmax/exp）对全量 KV 快速扫描，输出 TopK 索引给阶段2。

### 4.2 核心公式

```
对于每个 query 位置 i，每个 head h：

  score(i, j, h) = ReLU( Q[i,h] · K[j]^T ) × weight[i,h]

然后对所有 head 求和：

  logits[i, j] = Σₕ score(i, j, h)
```

用更直观的伪代码：

```python
# Q: [S, H, D]  —— 所有 query，所有 head
# K: [SKV, D]   —— 全量 KV cache
# W: [S, H]     —— 每个 query 位置每个 head 的权重

for i in range(S):           # 每个 query token
    for j in range(SKV):     # 每个 KV token（全量扫描！）
        total = 0
        for h in range(H):   # 每个 head
            raw = dot(Q[i,h], K[j])  # 内积
            activated = max(0, raw)   # ReLU —— 简单粗暴！
            weighted = activated * W[i,h]  # 加权
            total += weighted           # 跨 head 求和
        logits[i, j] = total

# 然后取 TopK：
topk_indices[i] = arg_topk(logits[i, :], k=topk)  # 每行选最大的K个
```

### 4.3 为什么用 ReLU 而不是 Softmax？

| 特性 | Softmax Attention | ReLU Indexer |
|------|-------------------|--------------|
| 复杂度 | 需要 exp + sum 归一化 | 只需 max(0, x)，一条指令 |
| 精度需求 | 高（指数运算敏感） | 低（只关心相对顺序） |
| 计算量 | 高 | 低（约 1/5） |
| 目的 | 精确概率分布 | 快速筛选"可能重要"的位置 |
| 适合场景 | 最终输出 | 快速初筛 |

**一句话总结：ReLU 速度快、精度要求低、只要能把 TopK 选对就行。**

### 4.4 mqa_logits 的两种形态

在代码实现中，`mqa_logits` 有两个版本，对应不同的计算模式：

```
┌──────────────────────────────────────────────────────────────┐
│                    Prefill 阶段                                │
│                                                              │
│   mqa_logits (连续内存版本)                                    │
│   ┌────────────────────────────────────────────┐             │
│   │  Q: [S, H, D]     ─连续摆放               │             │
│   │  K: [SKV, D]      ─也是连续摆放！          │             │
│   │                                            │             │
│   │  适用场景：K 已经在连续显存中             │             │
│   │  Kernel: mqa_attn_return_logits_tl         │             │
│   │  文件: perf/sparse_mla/mqa_logits.py       │             │
│   └────────────────────────────────────────────┘             │
└──────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────┐
│                    Decode 阶段                                 │
│                                                              │
│   paged_mqa_logits (分页内存版本)                              │
│   ┌────────────────────────────────────────────┐             │
│   │  Q: [B, H, D]     ─连续摆放               │             │
│   │  K: [num_blocks, block_size, 1, D]        │             │
│   │     ─分页存放！通过 block_table 索引       │             │
│   │                                            │             │
│   │  适用场景：KV cache 被分页管理            │             │
│   │  Kernel: paged_mqa_logits_tl               │             │
│   │  文件: perf/sparse_mla/paged_mqa_logits.py │             │
│   └────────────────────────────────────────────┘             │
└──────────────────────────────────────────────────────────────┘
```

**为什么需要分页版本？** 

vLLM 使用 **PagedAttention** 来管理 KV cache，K 不是连续存放的，而是按 64 个 token 一页分散在各处。所以需要 `block_table` 来查找物理地址。

---

## 5. 阶段2 拆解：sparse_mla_fwd

### 5.1 职责

`sparse_mla_fwd` 收到 indexer 筛选出的 topk_indices 后，只对这些位置做完整的 Softmax 注意力计算。

### 5.2 核心公式

```
对于每个 query 位置 i，每个 head h：

  scores[i, h, :] = softmax( Q[i,h] · K[topk_indices[i,:]]^T / sqrt(d) + sm_scale )
  
  output[i, h] = Σₖ scores[i, h, k] × V[topk_indices[i, k]]
```

与 indexer 的关键区别在于使用 **softmax 而非 ReLU**，并且输出的是 **V 的加权和**，不是单个分数。

### 5.3 MLA (Multi-head Latent Attention) 的特性

`sparse_mla_fwd` 处理的 Q 和 KV 都比普通注意力更复杂：

```
普通 Attention:
  Q: [S, H, D]         → dim = 128
  K: [SKV, H_kv, D]    → dim = 128

MLA (Multi-head Latent Attention):
  Q: [S, H, D + D_tail]   → dim = 576 (512 nope + 64 rope)
  K: [SKV, H_kv, D + D_tail]
  
  其中 D=512 是"nope"部分，D_tail=64 是"rope"（旋转位置编码）部分
```

MLA 把 KV 压缩到一个低秩隐空间，减少了 KV cache 的显存占用（DeepSeek-V3 通过这个技术实现了极低的 KV cache 开销）。

### 5.4 sparse_mla_fwd 的内部分层架构

`sparse_mla_fwd` 内部实际上做了多层计算：

```
对每个 TopK KV 块:

  1. 加载 Q_spilt0~3 + Q_tail → 从 Q 中切分出 5 个子空间
  2. 根据 topk_indices 从 KV 中 gather 对应的 K/V
  3. GEMM: Q @ K^T   → 对 5 个子空间分别做内积，累加到 acc_s
  4. Online Softmax  → 用流式方法计算 softmax（避免存储完整矩阵）
  5. GEMM: S @ V     → 用 softmax 权重对 V 加权求和
  6. Rescale         → 除以 sumexp 做归一化
  7. 如果用了 split-K：combine 阶段合并各 split 的结果
```

```
sparse_mla_fwd 的 5 路 GEMM 结构:

         Q 被切分为 5 个子空间:
         ┌──────────┬──────────┬──────────┬──────────┬─────────┐
         │ Q_spilt0 │ Q_spilt1 │ Q_spilt2 │ Q_spilt3 │ Q_tail  │
         │  D/4     │  D/4     │  D/4     │  D/4     │  D_tail │
         └────┬─────┴────┬─────┴────┬─────┴────┬─────┴────┬────┘
              │          │          │          │          │
         ┌────▼────┐┌───▼────┐┌───▼────┐┌───▼────┐┌───▼────┐
         │ GEMM 0  ││GEMM 1  ││GEMM 2  ││GEMM 3  ││GEMM 4  │
         │Q₀@K₀^T ││Q₁@K₁^T ││Q₂@K₂^T ││Q₃@K₃^T ││Q_t@K_t^T│
         └────┬────┘└───┬────┘└───┬────┘└───┬────┘└───┬────┘
              │          │          │          │          │
              └──────────┴──────────┴──────────┴──────────┘
                                    │
                              ┌─────▼─────┐
                              │  acc_s    │  [H_per_block, BI] 累加
                              │  Softmax  │
                              │  S @ V    │  对 V 做加权求和
                              └───────────┘
```

---

## 6. 端到端流程：从输入到输出

让我们用一个具体例子走通整个流程：

### 场景：128K 上下文的 Decode 阶段

```
输入:
  hidden_states: [1, 7168]         ← 当前 token 的 hidden state
  KV cache: [2050 blocks, 64, 1, 576] ← 分页存储的 128K KV

═══════════════════════════════════════════════════════════════
Step 1: 生成 Q, K, weights
═══════════════════════════════════════════════════════════════

  Q 从 hidden_states 投影得到:
    q_before_rope: [1, 32, 512]    ← nope 部分
    q_rope:        [1, 32, 64]     ← rope 部分
    拼接后 Q:       [1, 32, 576]    ← 576 = 512 + 64
  
  将 Q 量化为 FP8，得到 q_fp8 [1, 32, 576] FP8 ← 这是传给 indexer 的
  
  K 从前一层的输出得到:
    k: [1, 576]                     ← 当前 token 的 key
    存入 KV cache

═══════════════════════════════════════════════════════════════
Step 2: Indexer — 筛选 TopK 位置
═══════════════════════════════════════════════════════════════

  调用链:
    sparse_attn_indexer()
      └→ paged_mqa_logits(q_fp8, kv_cache, weights, ...)
          计算: ReLU(K @ Q^T) × weights

  计算过程:
    Q: [1, 32, 128]          ← 只用 Q 的前 128 维（indexer head_dim）
    K: 从 KV cache 逐页读出，共 131072 行，每行 128 维
    W: [1, 32]               ← 每个 head 的权重

    输出 logits: [1, 131072]  ← 对每个 KV 位置的"重要性分数"

    TopK: 从 131072 个分数中选出最大的 2048 个
    → topk_indices: [1, 1, 2048]  ← 最重要的 2048 个 KV 位置索引

  计算量: 131072 × 32 × 128 ≈ 5 × 10⁸ FLOPs（比全量注意力小 100×）

═══════════════════════════════════════════════════════════════
Step 3: Sparse MLA — 对 TopK 位置做精确注意力
═══════════════════════════════════════════════════════════════

  调用:
    sparse_mla_fwd(Q_full [1, 32, 576], KV_full, topk_indices [1, 1, 2048])

  计算过程:
    只取 topk_indices 指定的 2048 个 KV 位置:
      gather: K_sparse [2048, 1, 576]
      gather: V 也是同 2048 个位置的 value

    Q @ K^T: [32, 576] @ [576, 2048] → [32, 2048]  (scores)
    Softmax: 对 2048 个 score 做归一化
    S @ V:   [32, 2048] @ [2048, 512] → [32, 512]

    输出: output [1, 32, 512]  ← 最终的注意力输出！

  计算量: 32 × 2048 × 576 ≈ 3.8 × 10⁷ FLOPs

═══════════════════════════════════════════════════════════════
总计:
  标准注意力: ~7 × 10¹³ FLOPs
  稀疏注意力: ~5 × 10⁸ + 3.8 × 10⁷ ≈ 5.4 × 10⁸ FLOPs
  加速比:     ~130,000× !!!
═══════════════════════════════════════════════════════════════
```

---

## 7. 两个算子的对比速查表

| 维度 | mqa_logits | sparse_mla_fwd |
|------|-----------|----------------|
| **文件** | `perf/sparse_mla/mqa_logits.py` | `perf/sparse_mla/sparse_mla_fwd.py` |
| **别名** | `paged_mqa_logits.py` (分页版) | — |
| **阶段** | 稀疏注意力的第1步：TopK 筛选 | 稀疏注意力的第2步：精确计算 |
| **Score 函数** | ReLU | Softmax |
| **是否跨 Head 压缩** | 是（sum over heads → logits） | 否（每个 head 独立输出） |
| **输出** | TopK indices `[S, HKV, topk]` | Attention output `[S, H, D_v]` |
| **Q 维度** | dim=128 (只用 index head dim) | dim=576 (512 nope + 64 rope) |
| **KV 维度** | dim=128 (只读 index dim) | dim=576 (全部维度) |
| **精度需求** | 低（FP8/BF16 都可以，只要排序结果对） | 高（推荐 BF16/FP16，累加用 FP32） |
| **Kernel 写法** | 简单的 GEMM + ReLU + reduce | 复杂的分块 GEMM + online softmax + S@V + combine |
| **Block 策略** | block_N=32/64, block_Q=1~4 | block_I=32, split-K, stream-K |

---

## 8. 图文总结

```
                        🧠 Transformer 层
                              │
                    ┌─────────▼─────────┐
                    │  hidden_states    │
                    │  [B, S, model_dim]│
                    └─────────┬─────────┘
                              │
              ┌───────────────┼───────────────┐
              │               │               │
         ┌────▼────┐    ┌────▼────┐    ┌─────▼─────┐
         │ 投影 Q  │    │ 投影 K  │    │ 投影 V    │
         └────┬────┘    └────┬────┘    └─────┬─────┘
              │               │               │
         q_fp8[N,H,576]  k[N,576]       v[N,512]
              │               │               │
              │         ┌─────▼─────┐         │
              │         │存入KV Cache│         │
              │         └─────┬─────┘         │
              │               │               │
     ╔════════▼═══════════════▼═══════════════╗
     ║    阶段1: Indexer                      ║
     ║                                        ║
     ║  mqa_logits / paged_mqa_logits        ║
     ║  ┌──────────────────────────────────┐  ║
     ║  │ Q[:128] @ K[:128]^T              │  ║
     ║  │ ReLU × weights                    │  ║
     ║  │ Σ over heads → logits [S, SKV]   │  ║
     ║  │ TopK → topk_indices [S,1,2048]   │  ║
     ║  └──────────────────────────────────┘  ║
     ╚════════════════▼═══════════════════════╝
                      │ topk_indices
     ╔════════════════▼═══════════════════════╗
     ║    阶段2: Sparse MLA                   ║
     ║                                        ║
     ║  sparse_mla_fwd                        ║
     ║  ┌──────────────────────────────────┐  ║
     ║  │ Gather: 从 KV 中取 TopK 位置     │  ║
     ║  │ GEMM: Q @ K^T → [H, 2048]       │  ║
     ║  │ Online Softmax                   │  ║
     ║  │ GEMM: S @ V → [H, 512]          │  ║
     ║  │ → attention_output [S, H, 512]  │  ║
     ║  └──────────────────────────────────┘  ║
     ╚════════════════▼═══════════════════════╝
                      │
                 ┌────▼────┐
                 │  output │ → 下一层
                 └─────────┘
```

---

> **核心思想：用低精度、低维度的 ReLU Indexer 快速筛选 TopK 候选位置，再用高精度、全维度的 Softmax Attention 仅对这些位置做精确计算，在保证精度的同时大幅缩减计算量。**

---

*文档基于 tilelang 项目 `perf/sparse_mla/` 下的实际代码编写*  
*相关 vLLM 文件: `sparse_attn_indexer.py`, `flashmla_sparse.py`, `mla.py`*