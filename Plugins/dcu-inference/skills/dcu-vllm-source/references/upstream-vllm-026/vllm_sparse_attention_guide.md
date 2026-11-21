# FlashMLA Sparse Attention 执行流程详解

> 基于 vLLM 0.15.x（V1 架构），梳理从 Indexer 稀疏索引到 FlashMLA Sparse Attention Kernel 的完整执行链路。

---

## 目录

1. [概述](#1-概述)
2. [核心组件总览](#2-核心组件总览)
3. [执行时序：五个阶段](#3-执行时序五个阶段)
4. [阶段一：Indexer 前向计算（Q·K^T → TopK 索引）](#4-阶段一indexer-前向计算)
5. [阶段二：FlashMLASparseImpl.forward（Attention 主流程）](#5-阶段二flashmlasparseimplforwardattention-主流程)
6. [阶段三：Sparse MLA Attention Kernel（实际注意力计算）](#6-阶段三sparse-mla-attention-kernel实际注意力计算)
7. [阶段四：v_up_proj（输出投影）](#7-阶段四v_up_proj输出投影)
8. [阶段五：o_proj（最终输出）](#8-阶段五o_proj最终输出)
9. [三条 Kernel 路径详解](#9-三条-kernel-路径详解)
10. [Paged MQA Logits 与 Sparse MLA FWD 的关系](#10-paged-mqa-logits-与-sparse-mla-fwd-的关系)
11. [数据流全景图](#11-数据流全景图)
12. [关键数据结构](#12-关键数据结构)
13. [配置与路由](#13-配置与路由)

---

## 1. 概述

本文档描述 DeepSeek V3.2（V32）模型中 **稀疏注意力（Sparse Attention / DSA）** 的完整执行流程。整个流程涉及两个核心文件：

| 文件 | 职责 |
|------|------|
| `vllm/model_executor/layers/sparse_attn_indexer.py` | **Indexer**：计算 Q·K^T logits，选出每行 TopK 索引，写入 `topk_indices_buffer` |
| `vllm/v1/attention/backends/mla/flashmla_sparse.py` | **Attention Backend**：使用 Indexer 产生的 TopK 索引，对 KV Cache 进行稀疏注意力计算 |

核心思想是**用一个小型 Indexer 快速筛选出每个 Query Token 最相关的 TopK 个 KV Token**，然后只对这些选中的 KV 进行完整的 MLA 注意力计算，从而将注意力计算复杂度从 O(N²) 降到 O(N·K)，其中 K=2048。

---

## 2. 核心组件总览

```
┌──────────────────────────────────────────────────────────────────┐
│                      DeepseekV2DecoderLayer                      │
│                                                                   │
│  ┌─────────────────────────────────────────────────────────────┐ │
│  │            MultiHeadLatentAttentionWrapper                   │ │
│  │                                                              │ │
│  │  ① Indexer.forward()                                         │ │
│  │     └─ SparseAttnIndexer.forward_native()                   │ │
│  │        └─ sparse_attn_indexer()          ← 计算 QK^T logits │ │
│  │           ├─ fp8_mqa_logits()             ← Prefill 密集 MQA │ │
│  │           ├─ fp8_paged_mqa_logits()       ← Decode 分页 MQA │ │
│  │           └─ top_k_per_row()              ← TopK 筛选       │ │
│  │                                                              │ │
│  │  ② MLAAttention.forward()                                    │ │
│  │     └─ FlashMLASparseImpl.forward()                          │ │
│  │        ├─ _forward_bf16_kv()             ← BF16 路径        │ │
│  │        ├─ _forward_fp8_kv_mixed_batch()  ← FP8 mixed 路径  │ │
│  │        └─ _forward_fp8_kv_separate_prefill_decode()          │ │
│  │           ├─ _fp8_flash_mla_kernel()     ← FP8 decode kernel│ │
│  │           └─ _bf16_flash_mla_kernel()    ← BF16 prefill     │ │
│  │              └─ flash_mla_sparse_fwd()   ← Triton 实现       │ │
│  │                                                              │ │
│  │  ③ o_proj(output)                       ← 输出投影         │ │
│  └─────────────────────────────────────────────────────────────┘ │
└──────────────────────────────────────────────────────────────────┘
```

---

## 3. 执行时序：五个阶段

对于 V32 稀疏注意力，每个 Decoder Layer 的 forward 依次执行以下阶段：

```
阶段一：Indexer.forward()
   ↓  输出：topk_indices_buffer [num_tokens, 2048]
   ↓  含义：每个 query token 最相关的前 2048 个 KV token 的索引
   
阶段二：FlashMLASparseImpl.forward()
   ↓  Q 投影：q_nope · W_UK^T → ql_nope
   ↓  写入 KV Cache：store k_c_normed, k_pe
   ↓  稀疏注意力计算（使用 topk_indices_buffer）
   ↓  输出：attn_out [num_tokens, num_heads, kv_lora_rank]
   
阶段三：v_up_proj
   ↓  attn_out → kv_b_proj → output [num_tokens, num_heads * v_head_dim]
   
阶段四：o_proj
   ↓  output → o_proj → final_hidden_states [num_tokens, hidden_size]
```

---

## 4. 阶段一：Indexer 前向计算

### 4.1 入口

调用链：
```
MultiHeadLatentAttentionWrapper.forward()
  → self.indexer(hidden_states, q_c, positions, self.indexer_rope_emb)
    → Indexer.forward()
      → SparseAttnIndexer.forward_native()
        → torch.ops.vllm.sparse_attn_indexer(...)
          → sparse_attn_indexer()
```

文件位置：`vllm/model_executor/layers/sparse_attn_indexer.py` 第 38-351 行。

### 4.2 Indexer 内部计算

Indexer 是一个**独立的轻量级注意力头**，专门用于生成稀疏索引：

| 属性 | 值 | 说明 |
|------|-----|------|
| n_head | 64 | Indexer 注意力头数 |
| head_dim | 128 | 每个头的维度 |
| rope_dim | 64 | RoPE 维度 |
| q_lora_rank | 1536 | Query 的 LoRA 低秩维度 |

Indexer.forward() 执行步骤（第 732-786 行）：

```python
# 1. 计算 Q（来自 q_lora_rank 的低秩瓶颈）
q = self.wq_b(qr)  # [num_tokens, 1536] → [num_tokens, 64, 128]
q_pe, q_nope = split(q, [64, 64], dim=-1)

# 2. 计算 K（来自原始 hidden_states）
k = self.wk(hidden_states)  # [num_tokens, hidden] → [num_tokens, 128]
k = self.k_norm(k)
k_pe, k_nope = split(k, [64, 64], dim=-1)

# 3. RoPE
q_pe, k_pe = indexer_rope_emb(positions, q_pe, k_pe)

# 4. 计算 weights
weights = self.weights_proj(hidden_states)  # [num_tokens, 64]

# 5. 量化 Q（FP8）
q_fp8, q_scale = per_token_group_quant_fp8(q, quant_block_size=128)
weights = weights * q_scale * softmax_scale * n_head**-0.5

# 6. 调用 sparse_attn_indexer
return self.indexer_op(hidden_states, q_fp8, k, weights)
```

### 4.3 sparse_attn_indexer：Prefill vs Decode 分支

```python
def sparse_attn_indexer(...):
    # === 步骤 1：将当前 token 的 K 写入 Indexer KV Cache ===
    ops.indexer_k_quant_and_cache(k, kv_cache, slot_mapping, ...)
    
    # === 步骤 2：Prefill 路径 ===
    if has_prefill:
        for chunk in prefill_metadata.chunks:
            # 2a. 从 KV Cache 中 gather 历史 K（去量化 + 合并）
            ops.cp_gather_indexer_k_quant_cache(
                kv_cache, k_fp8, k_scale, block_table, cu_seq_lens
            )
            
            # 2b. 计算 QK^T logits（密集 MQA）
            logits = fp8_mqa_logits(
                q_slice,          # [q_len, 64, 128] fp8
                (k_fp8, k_scale), # [kv_len, 128] fp8 + scales
                weights,          # [q_len, 64] fp32
                cu_seqlen_ks,     # 每个 query token 的开始位置
                cu_seqlen_ke,     # 每个 query token 的结束位置
            )
            
            # 2c. TopK 筛选
            top_k_per_row_prefill(logits, topk_indices_slice, topk=2048)
    
    # === 步骤 3：Decode 路径 ===
    if has_decode:
        # 3a. 使用分页 KV Cache 计算 QK^T logits
        logits = fp8_paged_mqa_logits(
            q_fp8,              # [batch, next_n, 64, 128] fp8
            kv_cache_fp8,       # [num_blocks, block_size, 1, D+4] uint8
            weights,            # [batch*next_n, 64] fp32
            context_lens,       # [batch] int32
            block_table,        # [batch, max_blocks] int32
            schedule_metadata,  # SM 调度信息
        )
        
        # 3b. TopK 筛选
        top_k_per_row_decode(logits, next_n, context_lens, topk_indices, topk=2048)
    
    return topk_indices_buffer  # [num_tokens, 2048] int32
```

### 4.4 Prefill 与 Decode 的 logits 计算差异

| 维度 | fp8_mqa_logits (Prefill) | fp8_paged_mqa_logits (Decode) |
|------|--------------------------|-------------------------------|
| **K 来源** | `cp_gather` 从 KV Cache 连续拷贝到临时 buffer | 直接从分页 KV Cache 读取 |
| **内存布局** | K 是连续内存 `[kv_len, 128]` | K 是分页的 `[num_blocks, block_size, 1, D+4]` |
| **适用场景** | Prefill 的 KV 历史较长，一次性 gather 更高效 | Decode 每次只算 1~2 个 token，分页直接访问更快 |
| **计算模式** | 一次 MQA 矩阵乘：Q[M, H, D] × K[N, D]^T → logits[M, N] | 先 gather 再 MQA，按 token 分批次处理 |

两者都是 FP8 精度下的**无 softmax 的 MQA logits**。它们不计算完整的注意力，只计算 Q·K^T 部分，用于 TopK 筛选。

---

## 5. 阶段二：FlashMLASparseImpl.forward（Attention 主流程）

文件位置：`vllm/v1/attention/backends/mla/flashmla_sparse.py` 第 997-1070 行。

### 5.1 入口

```python
def forward(self, layer, q, k_c_normed, k_pe, kv_cache, attn_metadata, output, ...):
```

其中 `q` 是从 MLA Wrapper 传入的完整 Query（已 RoPE），`attn_metadata` 是 `FlashMLASparseMetadata`。

### 5.2 执行步骤

```python
# 1. 切片到实际 token 数（去掉 CUDA Graph padding）
q = q[:num_actual_toks, ...]
k_c_normed = k_c_normed[:num_actual_toks, ...]
k_pe = k_pe[:num_actual_toks, ...]

# 2. 获取 Indexer 已经计算好的 TopK 索引
topk_indices = self.topk_indices_buffer[:num_actual_toks]
# shape: [num_tokens, 2048], dtype: int32

# 3. Q 的低秩吸收（吸收 ROPE 前的那部分）
q_nope, q_pe = q.split([qk_nope_head_dim, qk_rope_head_dim], dim=-1)
q_nope = q_nope.transpose(0, 1)          # (B,N,P) → (N,B,P)
ql_nope = torch.bmm(q_nope, self.W_UK_T) # (N,B,P) × (N,P,L) → (N,B,L)
ql_nope = ql_nope.transpose(0, 1)         # (N,B,L) → (B,N,L)

# 4. 拼接完整的 q：ql_nope + q_pe
q = torch.cat([ql_nope, q_pe], dim=-1)

# 5. 将 k_c_normed 和 k_pe 写入 KV Cache
if kv_cache.numel() > 0:
    ops.concat_and_cache_mla(k_c_normed, k_pe, kv_cache, slot_mapping, ...)

# 6. 根据 cache 类型选择不同的注意力路径
if not use_fp8_cache:
    attn_out = self._forward_bf16_kv(q, kv_cache, topk_indices, attn_metadata)
elif attn_metadata.fp8_use_mixed_batch:
    attn_out = self._forward_fp8_kv_mixed_batch(q, kv_cache, topk_indices, attn_metadata)
else:
    attn_out = self._forward_fp8_kv_separate_prefill_decode(q, kv_cache, topk_indices, attn_metadata)
```

---

## 6. 阶段三：Sparse MLA Attention Kernel

### 6.1 三条路径的对比

| 路径 | KV Cache 格式 | Prefill 处理 | Decode 处理 | 条件 |
|------|--------------|-------------|------------|------|
| **`_forward_bf16_kv`** | BF16 (1152 bytes/token) | BF16 Kernel | BF16 Kernel | `cache_dtype` ≠ fp8_ds_mla |
| **`_forward_fp8_kv_mixed_batch`** | FP8 (656 bytes/token) | FP8 decode kernel（不分 pref/decode） | FP8 decode kernel | `fp8_use_mixed_batch=True` |
| **`_forward_fp8_kv_separate_prefill_decode`** | FP8 (656 bytes/token) | BF16 prefill kernel（经 upconvert） | FP8 decode kernel | 默认 FP8 路径 |

### 6.2 BF16 路径（`_forward_bf16_kv`）

```python
def _forward_bf16_kv(self, q, kv_c_and_k_pe_cache, topk_indices, attn_metadata):
    # 1. 将 per-request 的 token 索引转为全局 KV cache 位置
    topk_indices = triton_convert_req_index_to_global_index(
        attn_metadata.req_id_per_token,
        attn_metadata.block_table,
        topk_indices,
    )
    # 转换公式：global_idx = block_table[req, idx // 64] * 64 + idx % 64
    
    # 2. 调用 BF16 稀疏 MLA kernel
    return self._bf16_flash_mla_kernel(q, kv_cache, topk_indices)

def _bf16_flash_mla_kernel(self, q, kv_cache, topk_indices):
    # KV cache reshape: [total_tokens, 576] → [total_tokens, 1, 576]
    kv_cache = kv_cache.view(-1, 1, kv_cache.shape[-1])
    topk_indices = topk_indices.view(num_tokens, 1, -1)
    
    # 调用 Triton 实现的 sparse MLA forward
    output = flash_mla_sparse_fwd(q, kv_cache, topk_indices, softmax_scale)[0]
    output = output[:, :self.num_heads, :]
    return output
```

### 6.3 FP8 Mixed Batch 路径

```python
def _forward_fp8_kv_mixed_batch(self, q, kv_cache, topk_indices, attn_metadata):
    # 不做 pref/decode 区分，所有 token 作为一个 batch
    topk_indices = triton_convert_req_index_to_global_index(...)
    
    _attn_out, _ = self._fp8_flash_mla_kernel(
        q=q.unsqueeze(0),             # (T,H,D) → (1,T,H,D)
        kv_cache=kv_cache.uint8().unsqueeze(-2),
        topk_indices=topk_indices.unsqueeze(0),
        kernel_metadata=fp8_metadata,
    )
    return _attn_out.squeeze(0)

def _fp8_flash_mla_kernel(self, q, kv_cache, topk_indices, kernel_metadata):
    # 调用 FlashMLA 的 FP8 稀疏 decode kernel
    out, lse = flash_mla_with_kvcache(
        q=q,
        k_cache=kv_cache.view(torch.uint8).unsqueeze(-2),
        indices=topk_indices,
        is_fp8_kvcache=True,
        ...
    )
    return out, lse
```

### 6.4 FP8 Separate Prefill/Decode 路径（最复杂，也是默认路径）

```python
def _forward_fp8_kv_separate_prefill_decode(self, q, kv_cache, topk_indices, attn_metadata):
    # 1. 索引转换（区分 pref/decode）
    topk_indices = triton_convert_req_index_to_global_index(
        ...,
        HAS_PREFILL_WORKSPACE=has_prefill,
        # Prefill token 映射到 workspace offset（而非全局 cache slot）
    )
    
    # 2. Decode 部分：FP8 decode kernel
    if num_decode_tokens > 0:
        attn_out[:num_decode_tokens] = _fp8_decode(
            q[:num_decode_tokens],
            topk_indices[:num_decode_tokens]
        )
    
    # 3. Prefill 部分：分 chunk 处理
    for chunk in fp8_metadata.prefill.chunks:
        # 3a. 从 FP8 KV Cache 中 upconvert 为 BF16 workspace
        ops.cp_gather_and_upconvert_fp8_kv_cache(
            kv_cache,                # FP8 格式
            chunk_workspace,         # BF16 workspace (输出)
            chunk.block_table,
            chunk.seq_lens,
            chunk.workspace_starts,
        )
        # FP8 (656 bytes/token) → BF16 chunk_workspace (576 dims × 2 bytes)
        
        # 3b. 用 BF16 prefill kernel 在 workspace 上计算
        attn_out[chunk.tokens_slice] = self._bf16_flash_mla_kernel(
            chunk_q, chunk_workspace, chunk_topk_indices
        )
```

**为什么 Prefill 要 upconvert 到 BF16 再算？**

- FP8 decode kernel 对小批量 decode（seq_len=1~2）高效
- Prefill 的序列长度可能很长，FP8 矩阵乘积累的精度损失比 decode 大
- BF16 prefill kernel 提供更好的数值精度
- workspace 复用（大小固定）避免额外的 GPU 显存开销

---

## 7. 阶段四：v_up_proj（输出投影）

```python
# flashmla_sparse.py 第 1069 行
self._v_up_proj(attn_out, out=output[:num_actual_toks])
```

将注意力输出 `[num_tokens, num_heads, kv_lora_rank]` 通过 `kv_b_proj` 投影到 `[num_tokens, num_heads * v_head_dim]`。

---

## 8. 阶段五：o_proj（最终输出）

```python
# mla.py 第 238 行
return self.o_proj(attn_out)[0]
```

将 `[num_tokens, num_heads * v_head_dim]` 投影回 `[num_tokens, hidden_size]`。

---

## 9. Paged MQA Logits 与 Sparse MLA FWD 的关系

这是整个流程中最关键的概念区分。

### 9.1 它们是两个独立的东西，服务于不同目的

| | Paged MQA Logits | Sparse MLA FWD |
|---|---|---|
| **做什么** | 计算 Q·K^T（只有 logits，没有 softmax 和 V 加权） | 完整的稀疏注意力：Softmax(Q·K^T/√d)·V |
| **精度** | FP8 | BF16（或 FP8） |
| **调用位置** | `sparse_attn_indexer.py`（Indexer） | `flashmla_sparse.py`（Attention Backend） |
| **输入 Q** | Indexer 的小 Q（64 头 × 128 维） | 主注意力的 Q（num_heads × 576 维） |
| **输入 K** | Indexer 的 K Cache（128 维） | 主注意力的 KV Cache（576 维） |
| **输出** | TopK 索引（int32） | 注意力输出（bf16） |
| **执行时机** | 阶段一（先执行） | 阶段二（后执行，依赖阶段一的输出） |

### 9.2 它们的协作关系

```
                 Paged MQA Logits                  Sparse MLA FWD
                 (Indexer 中)                      (Attention Backend 中)
                      │                                    │
  hidden_states ──────┤                                    │
                      ▼                                    │
            ┌─────────────────┐                            │
            │  Indexer 小 Q    │                            │
            │  (64头×128维)    │                            │
            └────────┬────────┘                            │
                     │                                     │
            ┌────────▼────────┐                            │
            │ Indexer 小 K    │                             │
            │   (128维)       │                             │
            └────────┬────────┘                            │
                     │                                     │
            ┌────────▼────────┐                            │
            │ Q·K^T (FP8 MQA) │  ← fp8_paged_mqa_logits    │
            │  → logits       │    / fp8_mqa_logits         │
            └────────┬────────┘                            │
                     │                                     │
            ┌────────▼────────┐                            │
            │   TopK = 2048   │                             │
            │   筛选索引       │                             │
            └────────┬────────┘                            │
                     │                                     │
            ┌────────▼────────┐                            │
            │ topk_indices    │─────────────────────────────┤
            │ [T, 2048] int32 │                             │
            └─────────────────┘                            ▼
                                                  ┌─────────────────┐
                                                  │  主注意力 Q      │
                                                  │  (N头×576维)    │
                                                  └────────┬────────┘
                                                           │
                                                  ┌────────▼────────┐
                                                  │  主 KV Cache     │
                                                  │  (576维)        │
                                                  └────────┬────────┘
                                                           │
                                                  ┌────────▼────────┐
                                                  │ Sparse MLA FWD  │
                                                  │ Softmax(QK^T)·V │
                                                  │ (只算选中部分)   │
                                                  └────────┬────────┘
                                                           │
                                                           ▼
                                                    attn_out
                                                  [T, N, kv_lora]
```

### 9.3 关键洞察

1. **Paged MQA Logits 是"查表"**：快速找出哪些 KV token 与当前 query 最相关。它本身就是一个小型的注意力计算（小头小维），但**不做 softmax，不做 V 加权**，只关心 Q·K^T 的相对大小来选出 TopK。

2. **Sparse MLA FWD 是"精算"**：用主注意力的大维度（576 维 vs 128 维）对已经筛选出的 TopK 个 KV token 进行**完整精度的注意力计算**，包括 softmax 和 V 的加权求和。

3. **为什么需要两个步骤？** 如果对全部 KV token 都做完整的 MLA 注意力，复杂度是 O(T²·576)，T 可能到 128K 以上。通过 Indexer 先用 128 维的小头快速筛选，只对 2048 个候选做 576 维的精算，复杂度降到 O(T·128 + T·2048·576)。

4. **两者都使用分页（Paged）**：
   - `fp8_paged_mqa_logits` 的 "paged" 指的是从分页 KV Cache 中直接读取 K
   - `flash_mla_with_kvcache` / `flash_mla_sparse_fwd` 同样使用分页 KV Cache，通过 `topk_indices` 指定读取哪些位置

---

## 10. 数据流全景图

```
Layer.forward(hidden_states, positions)
│
├─ fused_qkv_a_proj: hidden → q_lora, kv_lora
│
├─ Indexer.forward()                                    ← 阶段一
│  ├─ wq_b: q_lora → q_indexer [T, 64, 128]
│  ├─ wk:   hidden   → k_indexer [T, 128]
│  ├─ weights_proj: hidden → weights [T, 64]
│  ├─ RoPE on indexer_q, indexer_k
│  ├─ FP8 quantize q_indexer
│  └─ sparse_attn_indexer()
│     ├─ Store k_indexer → Indexer KV Cache
│     ├─ [Prefill] cp_gather + fp8_mqa_logits + topk
│     └─ [Decode]  fp8_paged_mqa_logits + topk
│     → topk_indices_buffer [T, 2048]
│
├─ MultiHeadLatentAttentionWrapper.forward()            ← 阶段二~五
│  ├─ q_a_layernorm → q_b_proj → q [T, N, 576]
│  ├─ kv_a_layernorm → kv_c_normed
│  ├─ RoPE on q_pe, k_pe
│  │
│  ├─ MLAAttention.forward(q, kv_c_normed, k_pe)
│  │  └─ FlashMLASparseImpl.forward()
│  │     ├─ q_nope · W_UK^T → ql_nope
│  │     ├─ q = cat(ql_nope, q_pe)  →  [T, N, kv_lora+rope]
│  │     ├─ concat_and_cache_mla(kv_c_normed, k_pe)
│  │     │
│  │     ├─ [根据 KV Cache 类型选择路径]
│  │     │
│  │     ├─ [BF16 路径]
│  │     │  └─ _bf16_flash_mla_kernel → flash_mla_sparse_fwd
│  │     │
│  │     ├─ [FP8 Mixed 路径]
│  │     │  └─ _fp8_flash_mla_kernel → flash_mla_with_kvcache
│  │     │
│  │     └─ [FP8 Separate 路径] (默认)
│  │        ├─ [Decode]  _fp8_flash_mla_kernel → flash_mla_with_kvcache
│  │        └─ [Prefill] cp_gather_and_upconvert_fp8_kv_cache
│  │                      → _bf16_flash_mla_kernel → flash_mla_sparse_fwd
│  │     → attn_out [T, N, kv_lora_rank]
│  │
│  │  └─ v_up_proj(attn_out) → output [T, N*v_head_dim]
│  │
│  └─ o_proj(output) → final [T, hidden_size]
```

---

## 11. 关键数据结构

### 11.1 KV Cache 格式

```
普通 BF16:
  [num_blocks, block_size, head_size]
  head_size = kv_lora_rank + qk_rope_head_dim = 512 + 64 = 576
  每 token: 576 × 2 = 1152 bytes

FP8 (fp8_ds_mla):
  [num_blocks, block_size, 656]
  每 token 656 bytes:
    [0:512]   512 × float8_e4m3  (NoPE 量化部分)
    [512:528] 4 × float32        (每 128 个 fp8 的 scale)
    [528:656] 64 × bfloat16      (RoPE 部分，不量化)
```

### 11.2 Indexer KV Cache 格式

```
[num_blocks, block_size, head_dim]
head_dim = 128 + 128/128*4 = 132 bytes (FP8 with scale)
```

### 11.3 topk_indices_buffer

```
Shape: [max_num_batched_tokens, 2048]
Dtype: int32
含义: topk_indices[i, j] = 第 i 个 token 的第 j 重要的 KV token 的"逻辑索引"
     正数 = token 在其所属 request 中的位置（需经 block_table 转换为全局 cache slot）
     -1   = 无效（该 request 的历史 token 不足 2048 个）
```

### 11.4 FlashMLASparseMetadata

```
@dataclass
class FlashMLASparseMetadata:
    num_reqs: int
    num_actual_tokens: int
    query_start_loc: Tensor         # [num_reqs+1]
    slot_mapping: Tensor            # [num_tokens]
    block_table: Tensor             # [num_reqs, max_blocks]
    req_id_per_token: Tensor        # [num_tokens]
    block_size: int                 # 64
    topk_tokens: int                # 2048
    
    fp8_extra_metadata: FP8SeparatePrefillDecode | FP8KernelMetadata | None
    fp8_use_mixed_batch: bool
```

---

## 12. 配置与路由

### 12.1 稀疏注意力开关

```python
# vllm/model_executor/models/deepseek_v2.py
self.is_v32 = hasattr(config, "index_topk") and not envs.VLLM_DISABLE_DSA
```

- 有 `index_topk` 配置且未设置 `VLLM_DISABLE_DSA` → 启用 DSA 稀疏注意力
- `index_topk` 通常在 `config.json` 中为 2048

### 12.2 FP8 KV Cache 的三条路径选择

```python
# flashmla_sparse.py builder 中
fp8_use_mixed_batch = (
    self.num_heads < MIN_HEADS_FOR_BF16_PREFILL  # < 32
    and envs.FP8_USE_MIXED_BATCH  # 环境变量
)
```

| 条件 | 路径 |
|------|------|
| `cache_dtype != "fp8_ds_mla"` | `_forward_bf16_kv` |
| `cache_dtype == "fp8_ds_mla"` && `fp8_use_mixed_batch=True` | `_forward_fp8_kv_mixed_batch` |
| `cache_dtype == "fp8_ds_mla"` && `fp8_use_mixed_batch=False` | `_forward_fp8_kv_separate_prefill_decode` |

### 12.3 关键环境变量

| 变量 | 默认值 | 说明 |
|------|--------|------|
| `VLLM_DISABLE_DSA` | False | 禁用稀疏注意力，回退到稠密 MLA |
| `FP8_USE_MIXED_BATCH` | False | FP8 模式下使用 mixed batch（不分 prefill/decode） |
| `USE_LIGHTOP_CONVERT_REQ_INDEX_TO_GLOBAL_INDEX` | - | 使用 lightop 加速索引转换 |
| `USE_LIGHTOP_TOPK` | - | 使用 lightop 加速 TopK 筛选 |
| `USE_FUSED_RMS_QUANT` | - | 使用融合的 RMSNorm + 量化 |
| `VLLM_USE_LIGHTOP_RMS_ROPE_CONCAT` | - | 使用 lightop 的融合 RMS+RoPE+Concat |

---

## 13. 总结

1. **Indexer 与 Attention Backend 是两个独立的子系统**，通过 `topk_indices_buffer` 共享数据。
2. **Indexer 先执行**，用小型注意力头（64 头 × 128 维）快速计算 QK^T logits，选出 TopK=2048 个最相关 KV token 的索引。
3. **Attention Backend 后执行**，拿到 Indexer 的 TopK 索引后，对选中的 KV token 进行完整精度的稀疏注意力计算。
4. **Paged MQA Logits**（在 Indexer 中）只算 logits 不做 attention，**Sparse MLA FWD**（在 Attention Backend 中）做完整的 Softmax + V 加权。
5. FP8 KV Cache 场景下，默认使用 **Separate Prefill/Decode** 路径：decode 用 FP8 kernel，prefill 先 upconvert 再走 BF16 kernel，兼顾效率和精度。
6. 当 `num_heads < 32`（高 TP 场景）且 `FP8_USE_MIXED_BATCH=True` 时，使用 Mixed Batch 模式避免 BF16 prefill kernel 的 head padding 开销。
