# vLLM 投机解码（Speculative Decoding）模块详解

> 基于最新 vLLM 版本，全面覆盖投机解码的设计原理、MTP/EAGLE/DFlash 三大方法、拒绝采样算法、动态投机与调度器集成。

---

## 目录

1. [概述](#1-概述)
2. [核心原理](#2-核心原理)
3. [模块架构](#3-模块架构)
4. [SpeculativeConfig 配置](#4-speculativeconfig-配置)
5. [统一基类 SpecDecodeBaseProposer](#5-统一基类-specdecodebaseproposer)
6. [EAGLE / EAGLE3](#6-eagle--eagle3)
7. [MTP（Multi-Token Prediction）](#7-mtpmulti-token-prediction)
8. [DFlash（并行起草）](#8-dflash并行起草)
9. [其他 Proposer](#9-其他-proposer)
10. [GPU 侧 Speculator 体系](#10-gpu-侧-speculator-体系)
11. [拒绝采样（Rejection Sampling）](#11-拒绝采样rejection-sampling)
12. [动态投机解码](#12-动态投机解码)
13. [调度器与 Runner 集成](#13-调度器与-runner-集成)
14. [指标与监控](#14-指标与监控)
15. [方法对比与选择指南](#15-方法对比与选择指南)
16. [配置参数与环境变量](#16-配置参数与环境变量)
17. [常见问题](#17-常见问题)

---

## 1. 概述

投机解码（Speculative Decoding）是 vLLM 四大推理加速技术之一（PagedAttention / 连续批处理 / 编译优化 / 投机解码）。核心思想：用**廉价的 draft 模型**快速生成 K 个候选 token，再用**昂贵的 target 模型**一次性并行验证，从而将自回归生成的串行延迟转化为并行验证。

### 关键文件

| 文件 | 行数 | 职责 |
|------|------|------|
| `vllm/v1/spec_decode/llm_base_proposer.py` | 1883 | 统一基类 SpecDecodeBaseProposer（EAGLE/MTP/DFlash 共享） |
| `vllm/v1/spec_decode/eagle.py` | 22 | EagleProposer（薄封装） |
| `vllm/v1/spec_decode/dflash.py` | 317 | DFlashProposer（并行起草） |
| `vllm/v1/spec_decode/step3p5.py` | 463 | Step3.5 MTP（逐层 draft-step 选择） |
| `vllm/v1/spec_decode/gemma4.py` | 364 | Gemma4 MTP（constant positions + KV 共享） |
| `vllm/v1/spec_decode/draft_model.py` | 115 | DraftModelProposer（独立 draft 模型） |
| `vllm/v1/spec_decode/ngram_proposer.py` | 293 | NgramProposer（CPU n-gram） |
| `vllm/v1/spec_decode/ngram_proposer_gpu.py` | 671 | NgramProposerGPU（GPU 向量化 n-gram） |
| `vllm/v1/spec_decode/medusa.py` | 81 | MedusaProposer（多头预测） |
| `vllm/v1/spec_decode/suffix_decoding.py` | 103 | SuffixDecodingProposer（后缀树，Arctic Inference） |
| `vllm/v1/spec_decode/extract_hidden_states.py` | 410 | ExtractHiddenStatesProposer（KV/hidden 缓存） |
| `vllm/v1/spec_decode/vocab_mapping.py` | 154 | VocabMapping（异构词表 TLI） |
| `vllm/v1/spec_decode/dynamic/utils.py` | 148 | 动态投机调度 |
| `vllm/v1/spec_decode/metadata.py` | 66 | SpecDecodeMetadata（验证元数据） |
| `vllm/v1/spec_decode/metrics.py` | 281 | SpecDecodingStats / Logging / Prom |
| `vllm/v1/worker/gpu/spec_decode/speculator.py` | — | BaseSpeculator / DraftModelSpeculator |
| `vllm/v1/worker/gpu/spec_decode/autoregressive/speculator.py` | 776 | AutoRegressiveSpeculator（EAGLE/MTP GPU 路径） |
| `vllm/v1/worker/gpu/spec_decode/mtp/speculator.py` | — | MTPSpeculator |
| `vllm/v1/worker/gpu/spec_decode/eagle/speculator.py` | — | EagleSpeculator |
| `vllm/v1/worker/gpu/spec_decode/dflash/speculator.py` | 687 | DFlashSpeculator（单 pass 并行） |
| `vllm/v1/worker/gpu/spec_decode/dspark/speculator.py` | 169 | DSparkSpeculator（块式半自回归） |
| `vllm/v1/worker/gpu/spec_decode/gemma4/speculator.py` | 151 | Gemma4Speculator |
| `vllm/v1/worker/gpu/spec_decode/rejection_sampler.py` | — | GPU RejectionSampler |
| `vllm/v1/worker/gpu/spec_decode/rejection_sampler_utils.py` | — | 5-kernel 拒绝采样流水线 |
| `vllm/config/speculative.py` | 1405 | SpeculativeConfig |

---

## 2. 核心原理

### 2.1 基本流程

```
Target 模型生成 1 个 token (bonus token)
       │
       ▼
Draft 模型快速生成 K 个候选 token (draft tokens)
       │
       ▼
Target 模型一次性并行验证 [bonus, draft_0, ..., draft_{K-1}]
       │  (K+1 个位置同时计算 logits)
       ▼
拒绝采样 (Rejection Sampling):
   逐位置比较 target 与 draft 的分布
   接受匹配的前缀，拒绝第一个不匹配
       │
       ▼
输出: 接受的 token 前缀 + 1 个 bonus token
       │
       ▼
下一轮: 用接受的 token 作为新的 bonus，重复
```

### 2.2 加速原理

自回归生成 K+1 个 token 需要 K+1 次串行 forward。投机解码将其转化为：
- 1 次 draft forward（K 个 token，廉价）
- 1 次 target forward（K+1 个 token，并行验证）

若 draft 接受率为 α，期望每轮接受 `1 + α·K` 个 token，而只用了 2 次 forward（1 draft + 1 target），相比 K+1 次 target forward 显著加速。

### 2.3 无损性

拒绝采样保证输出分布与纯 target 模型**完全一致**（lossless）：
- 接受条件：`p_target(x) > u · p_draft(x)`（u ~ Uniform(0,1)）
- 拒绝后从残差分布 `max(p_target - p_draft, 0)` 重采样

---

## 3. 模块架构

### 3.1 两层架构

```
┌─────────────────────────────────────────────────────────┐
│  CPU 侧 Proposer (vllm/v1/spec_decode/)                 │
│  SpecDecodeBaseProposer (llm_base_proposer.py)          │
│  ├── EagleProposer          (eagle.py)                  │
│  ├── DFlashProposer         (dflash.py)                 │
│  ├── Step3p5MTPProposer     (step3p5.py)                │
│  ├── Gemma4Proposer         (gemma4.py)                 │
│  ├── DraftModelProposer     (draft_model.py)            │
│  ├── NgramProposer / NgramProposerGPU                   │
│  ├── MedusaProposer / SuffixDecodingProposer            │
│  └── ExtractHiddenStatesProposer                        │
│                                                         │
│  职责: 输入准备、draft 模型加载、propose() 主流程         │
└─────────────────────────────────────────────────────────┘
                          │
┌─────────────────────────────────────────────────────────┐
│  GPU 侧 Speculator (vllm/v1/worker/gpu/spec_decode/)    │
│  BaseSpeculator (speculator.py)                         │
│  ├── DraftModelSpeculator                               │
│  │   └── AutoRegressiveSpeculator (autoregressive/)     │
│  │       ├── EagleSpeculator (eagle/)                   │
│  │       ├── MTPSpeculator (mtp/)                       │
│  │       └── Gemma4Speculator (gemma4/)                 │
│  ├── DFlashSpeculator (dflash/)                         │
│  │   └── DSparkSpeculator (dspark/)                     │
│  └── RejectionSampler (rejection_sampler.py)            │
│                                                         │
│  职责: CUDA Graph 捕获、GPU 输入准备、draft 采样          │
└─────────────────────────────────────────────────────────┘
```

### 3.2 工厂函数

```python
# vllm/v1/worker/gpu/spec_decode/__init__.py
def init_speculator(vllm_config, device):
    if method == "dflash":  return DFlashSpeculator(...)
    if method == "dspark":  return DSparkSpeculator(...)
    if use_gemma4_mtp():    return Gemma4Speculator(...)
    if method == "mtp":     return MTPSpeculator(...)
    if use_eagle():         return EagleSpeculator(...)
```

---

## 4. SpeculativeConfig 配置

### 4.1 核心字段

```python
@dataclass
class SpeculativeConfig:
    num_speculative_tokens: int          # 投机 token 数 K
    model: str | None                    # draft 模型路径
    method: SpeculativeMethod            # 方法名
    draft_tensor_parallel_size: int      # draft 模型 TP
    quantization: str | None             # draft 模型量化

    # 高级控制
    disable_padded_drafter_batch: bool   # 禁用 padding（EAGLE）
    use_local_argmax_reduction: bool     # 词表并行局部 argmax（O(2·tp) 通信）
    use_heterogeneous_vocab: bool        # 异构词表 TLI
    parallel_drafting: bool              # 并行起草（DFlash/DSpark 自动 True）

    # 动态投机
    num_speculative_tokens_per_batch_size: list[tuple[int,int,int]] | None

    # 拒绝采样
    rejection_sample_method: Literal["standard","synthetic","block"] = "standard"
    draft_sample_method: Literal["greedy","probabilistic"] = "greedy"
    synthetic_acceptance_rates: list[float] | None

    # Ngram
    prompt_lookup_max: int | None
    prompt_lookup_min: int | None

    # Suffix decoding
    suffix_decoding_max_tree_depth: int = 24
    suffix_decoding_max_cached_requests: int = 10000
```

### 4.2 方法类型

```python
SpeculativeMethod = Literal[
    "ngram", "medusa", "mlp_speculator", "draft_model", "suffix",
    "custom_class", "eagle", "eagle3", "extract_hidden_states",
    "mtp",  # 统一 MTP（deepseek_mtp/qwen3_next_mtp/step3p5_mtp/gemma4_mtp 等已废弃合并）
    "dflash", "dspark", "ngram_gpu",
]
```

### 4.3 方法推断逻辑（`__post_init__`）

```python
# 1. 废弃的 *_mtp 名称统一为 "mtp"
# 2. MTP 无 model 时: 使用 target checkpoint 的 MTP 层
#    self.model = self.target_model_config.model
# 3. 模型名启发式推断:
#    "eagle-" in model → eagle
#    "eagle3" in model → eagle3
#    model_type == "medusa" → medusa
#    model_type in MTPModelTypes → mtp
# 4. n_predict 校验: K 必须能被 n_predict 整除（MTP 层复用）
# 5. dflash/dspark: parallel_drafting 自动 True
```

### 4.4 关键辅助方法

```python
def use_eagle(self): return self.method in ("eagle","eagle3","mtp","dflash","dspark")
def use_dflash(self): return self.method == "dflash"
def use_dspark(self): return self.method == "dspark"
def uses_draft_model(self): return self.method == "draft_model"
def uses_dynamic_speculative_decoding(self): return self.num_speculative_tokens_per_batch_size is not None
```

---

## 5. 统一基类 SpecDecodeBaseProposer

`llm_base_proposer.py` 是 EAGLE/MTP/DFlash 共享的统一基类（1883 行）。

### 5.1 关键初始化字段

```python
class SpecDecodeBaseProposer:
    def __init__(self, vllm_config, device, pass_hidden_states_to_model, runner=None):
        self.method = speculative_config.method
        self.num_speculative_tokens = K

        # 并行起草相关
        self.parallel_drafting = speculative_config.parallel_drafting
        self.extra_slots_per_request = 1 if not parallel_drafting else K
        self.needs_extra_input_slots = ...

        # 位置策略
        self.constant_draft_positions = False  # Gemma4: 所有 draft step 用同一位置

        # 异构词表
        self.use_heterogeneous_vocab = ...
        self.vocab_mapping: VocabMapping | None = None

        # 采样
        self.use_local_argmax_reduction = ...
        self.use_fp64_gumbel = ...

        # 持久化 GPU buffer
        self.input_ids, self.positions, self.hidden_states, self.inputs_embeds
        self.cudagraph_dispatcher = CudagraphDispatcher(vllm_config)
```

### 5.2 propose() 主流程

```python
def propose(self, num_speculative_tokens, target_token_ids, target_positions,
            target_hidden_states, next_token_ids, token_indices_to_sample,
            common_attn_metadata, sampling_metadata, mm_embed_inputs=None,
            num_rejected_tokens_gpu=None, slot_mappings=None) -> torch.Tensor:
    # 1. EAGLE3/DFlash: 合并 aux hidden states
    if method in ("eagle3", "dflash"):
        target_hidden_states = self.model.combine_hidden_states(target_hidden_states)

    # 2. 设置首 pass 输入 (子类可 override)
    num_tokens, token_indices_to_sample, cad = self.set_inputs_first_pass(...)

    # 3. 构建 attention metadata
    per_group_attn, per_layer_attn = self.build_per_group_and_layer_attn_metadata(cad)

    # 4. 确定 batch 执行与 padding (CUDA Graph)
    cudagraph_mode, num_input_tokens, num_tokens_across_dp = \
        self._determine_batch_execution_and_padding(num_tokens)

    # 5. 构建模型输入 (子类可 override, DFlash 在此预计算 context KV)
    model_kwargs, slot_mapping_size = self.build_model_inputs_first_pass(...)

    # 6. 运行 draft 模型 (首 pass)
    with set_forward_context(per_layer_attn, ...):
        ret = self.model(**model_kwargs)
        last_hidden_states, hidden_states = ret  # 或单值

    # 7. 采样 draft token
    sample_hidden_states = last_hidden_states[token_indices_to_sample]
    if K == 0: return empty  # 动态投机 K=0
    if K == 1 or self.parallel_drafting:
        # 早退: 单 token 或并行起草 (DFlash)
        draft_token_ids, draft_probs = self._sample_draft_tokens(...)
        return draft_token_ids

    # 8. 自回归循环: 生成剩余 K-1 个 draft token
    for step in range(1, K):
        # 更新位置/seq_len/slot_mapping
        self._update_positions_dependent_metadata(...)
        # 运行 draft 模型
        with set_forward_context(...):
            last_hidden_states, hidden_states = self.model(**model_kwargs)
        # 采样
        draft_token_ids[step] = self._sample_draft_tokens(...)
        # 更新输入 (token + hidden state)
        self.prepare_inputs_padded(...)

    return draft_token_ids  # [batch_size, K]
```

### 5.3 可 override 的关键方法

| 方法 | 作用 | DFlash override |
|------|------|----------------|
| `set_inputs_first_pass` | 设置首 pass 输入 | 并行 query (1+K/req) |
| `build_model_inputs_first_pass` | 构建模型输入 | 预计算 context KV |
| `build_per_group_and_layer_attn_metadata` | 构建 attn metadata | 非因果断言 |
| `dummy_run` | 显存 profiling | 单 pass |
| `_create_draft_vllm_config` | draft 配置 | 清除 mm_prefix |

### 5.4 模型加载与权重共享

```python
def load_model(self, target_model):
    self.model = self._get_model()  # 加载 draft 模型
    self._maybe_share_embeddings(target_language_model)  # 共享 embed_tokens
    self._maybe_share_lm_head(target_language_model)      # 共享 lm_head
    self.initialize_attn_backend(...)  # 初始化 draft attention

def _maybe_share_embeddings(self, target_lm):
    # 若 draft 无 embed_tokens 或权重相同 → 共享 target 的
    # 节省显存，避免重复加载

def _maybe_share_lm_head(self, target_lm):
    # 若 draft 无 lm_head 或权重相同 → 共享 target 的
    # MTP 的 per-layer shared_head 特殊处理
```

---

## 6. EAGLE / EAGLE3

### 6.1 EAGLE 原理

EAGLE（Extrapolation）用 target 模型的 **hidden states** 作为额外输入，训练一个轻量 draft 头预测下一个 token。

```
Target 模型 forward → hidden_states [T, H]
       │
       ▼
EAGLE draft 头:
  输入: token_ids + hidden_states (concat)
  输出: logits → argmax → draft token
       │
       ▼
自回归 K 步: 每步用新 draft token + 新 hidden state 预测下一个
```

### 6.2 EagleProposer

```python
class EagleProposer(SpecDecodeBaseProposer):
    def __init__(self, vllm_config, device, runner=None):
        super().__init__(vllm_config, device,
                         pass_hidden_states_to_model=True,  # 传递 hidden states
                         runner=runner)
```

仅 22 行，完全复用基类的自回归 propose 循环。

### 6.3 EAGLE3 增强

EAGLE3 额外使用 target 模型的**多层 aux hidden states**（而非仅最后一层）：

```python
# 在 propose() 中:
if method == "eagle3":
    target_hidden_states = self.model.combine_hidden_states(
        torch.cat(aux_hidden_states, dim=-1)  # 拼接多层 hidden states
    )
```

`use_aux_hidden_state_outputs` 标志让 runner 从 target 模型提取指定层的 hidden states。

### 6.4 GPU 侧 EagleSpeculator

```python
class EagleSpeculator(AutoRegressiveSpeculator):
    def load_draft_model(self, target_model):
        return load_eagle_model(target_model, vllm_config)
        # 加载 draft 头, 共享 embed_tokens/lm_head
```

`AutoRegressiveSpeculator` 的 GPU 路径：
- **prefill CUDA Graph**: K+1 tokens/req，FULL 或 breakable-PIECEWISE
- **decode CUDA Graph**: 1 token/req，FULL_DECODE_ONLY
- `propose()`:
  1. `prepare_prefill_inputs` (Triton): 移位 target input ids，拼接 bonus token
  2. prefill forward + 采样 token 0
  3. `_multi_step_decode`: step 1..K-1，每步 forward + 采样 + `update_draft_inputs`

---

## 7. MTP（Multi-Token Prediction）

### 7.1 MTP 原理

MTP 是模型**原生**的多 token 预测头（DeepSeek-V3、Qwen3、GLM-5.1 等）。target 模型本身包含 MTP 层，可直接用于投机，无需额外 draft 模型。

```
Target 模型:
  Backbone (N 层) → hidden_states
       │
       ▼
  MTP 层 (1 层, 与 backbone 同结构) → 预测下一个 token
       │
       ▼
  自回归 K 步: 复用同一 MTP 层 K 次
```

### 7.2 MTP 配置

```python
# MTP 无 model 时, 使用 target checkpoint 的 MTP 层
if method == "mtp" and model is None:
    self.model = self.target_model_config.model  # 同一 checkpoint
    self.quantization = self.target_model_config.quantization

# n_predict 校验: K 必须能被 n_predict 整除 (MTP 层复用)
n_predict = draft_hf_config.n_predict
if K > n_predict and K % n_predict != 0:
    raise ValueError(f"num_speculative_tokens:{K} must be divisible by {n_predict=}")
```

### 7.3 MTPSpeculator

```python
class MTPSpeculator(AutoRegressiveSpeculator):
    def load_draft_model(self, target_model):
        return load_eagle_model(target_model, vllm_config)
        # 加载 MTP 层, 共享 embed_tokens/lm_head
        # 修复 per-MTP-layer shared_head.head
        # 共享 topk_indices_buffer (稀疏注意力 MTP)
```

### 7.4 Step3.5 MTP（逐层 draft-step 选择）

`Step3p5MTPProposer` 的特殊之处：每个 draft step 使用**不同的 MTP 层/头**：

```python
def propose(self, ...):
    for step in range(K):
        # 每个 step 用不同的 MTP 层
        logits = self.model.compute_logits(hidden_states, spec_step_idx=step)
        draft_token_ids[step] = sample(logits)
```

checkpoint 携带 per-MTP-layer 的 `shared_head` 权重。

### 7.5 Gemma4 MTP

`Gemma4Proposer` 的特殊之处：
- `constant_draft_positions = True`：所有 draft step 从同一位置预测
- 所有 draft attention 层**共享 target 的 KV cache**（cross-model KV sharing）
- centroids CUDA graphs（masked embedding 的 top-k 采样）

---

## 8. DFlash（并行起草）

### 8.1 DFlash 原理

DFlash 是**并行** drafter，与 EAGLE 的自回归起草根本不同：

```
EAGLE (自回归):
  step 0: forward(token_0) → draft_0
  step 1: forward(draft_0) → draft_1
  step 2: forward(draft_1) → draft_2
  ... K 次 forward

DFlash (并行):
  一次 forward:
    Q = [bonus_token, mask_0, mask_1, ..., mask_{K-1}]  (1+K 个 query)
    K/V = 从 target hidden states 预计算的 context KV
  → 同时采样 K 个 draft token
  ... 1 次 forward
```

### 8.2 DFlashProposer 关键设计

```python
class DFlashProposer(SpecDecodeBaseProposer):
    def __init__(self, ...):
        super().__init__(..., pass_hidden_states_to_model=True)
        # 并行 query: 每请求 1+K 个 query token
        self.max_query_tokens = max_batch_size * (1 + K)
        # 分离 context buffer (保持 query buffer 地址稳定 for CUDA Graph)
        self._context_slot_mapping_buffer = ...
        self._context_positions_buffer = ...
        self.dflash_causal = not dflash_has_any_non_causal(hf_config)

    def set_inputs_first_pass(self, ...):
        # 融合 Triton kernel: copy_and_expand_dflash_inputs_kernel
        # 写入: query input_ids (bonus at 0, mask at 1..K)
        #       context + query positions
        #       context + query slot mappings
        #       token_indices_to_sample (K 个 mask 行)
        # 新 CommonAttentionMetadata:
        #   query_start_loc = arange * (1+K)
        #   seq_lens = effective_seq_lens + (1+K)
        #   max_query_len = 1+K
        #   causal = self.dflash_causal

    def build_model_inputs_first_pass(self, ...):
        # 预计算 context KV 并直接插入 draft KV cache
        self.model.precompute_and_store_context_kv(
            self._dflash_hidden_states,  # target hidden states
            self._context_positions_buffer[:num_context],
            self._context_slot_mapping_buffer[:num_context],
        )
        # 模型 forward 只看 query input_ids/positions
```

### 8.3 DFlash vs EAGLE 对比

| 维度 | EAGLE | DFlash |
|------|-------|--------|
| 起草模式 | 自回归（K 次 forward） | **并行**（1 次 forward） |
| 输入 | target token ids + hidden states | target hidden states → **预计算 context KV** |
| Query | 每步 1 个 token | 1 bonus + K mask tokens |
| 位置 | 每步 +1 | 固定（bonus at last+1, masks at +2..+K+1） |
| KV 写入 | 每步新 KV | context KV 预插入一次 |
| 额外 slot/req | 0（原地轮换） | K（padded batch, mask slots） |
| 注意力 | 因果 | 可非因果（`dflash_causal`） |
| GPU speculator | AutoRegressiveSpeculator | DFlashSpeculator（单 pass） |

### 8.4 DFlashSpeculator（GPU 侧）

```python
class DFlashSpeculator(DraftModelSpeculator):
    num_query_per_req = 1 + K
    parallel_drafting_token_id = get_parallel_drafting_token_id(...)
    # 检查 dflash_config.mask_token_id / mask_token_id / dspark_noise_token_id

    def propose(self, ...):
        # 1. 合并 aux hidden states
        # 2. prepare_dflash_inputs (Triton, per KV group)
        # 3. model.precompute_and_store_context_kv(...)  # eager, 图外
        # 4. dispatch CG (uniform 1+K tokens/req)
        # 5. _generate_draft: 一次 forward + 采样 K 个 token
        # 总共 1 次 forward — 与 AutoRegressiveSpeculator 的根本区别
```

### 8.5 DSpark（块式半自回归）

`DSparkSpeculator` 继承 DFlashSpeculator，增加**块式**起草：
- 复用 DFlash 的并行 backbone（context-KV 预计算 + query-block forward）
- `_sample_sequential`: 并行 backbone forward 后，**从左到右**采样，带 Markov bias：
  ```
  logits_i = base_logits[:, i] + markov_bias(markov_embed(prev))
  ```
- 要求 `num_speculative_tokens >= dspark_block_size`

---

## 9. 其他 Proposer

### 9.1 DraftModelProposer

独立 draft 模型（非 EAGLE/MTP）：
- `pass_hidden_states_to_model=False`（draft 模型只看 token ids）
- 构建 `VocabMapping`（当 `use_heterogeneous_vocab`）
- 不共享 embed/lm_head
- 要求 draft TP == target TP

### 9.2 NgramProposer / NgramProposerGPU

N-gram 匹配，无需 draft 模型：
- **CPU 版**（`ngram_proposer.py`）：numba 加速，在 prompt/context 中查找最长匹配 n-gram
- **GPU 版**（`ngram_proposer_gpu.py`）：`NgramGPUKernel`（`@support_torch_compile`），向量化 GPU n-gram 查找
  - `unfold` 滑动窗口 vs 尾部后缀
  - 找最早匹配，选最长 n-gram
  - 收集匹配后的 K 个 token

### 9.3 MedusaProposer

Medusa：在 target 模型上加多个预测头，每个头预测不同位置的 token。

### 9.4 SuffixDecodingProposer

后缀树解码（Arctic Inference）：
- 全局后缀树 + prompt 树
- 基于前缀匹配长度动态决定投机长度
- `max_spec_tokens = max_spec_factor * prefix_match_length`
- 需要 `pip install arctic-inference`

### 9.5 ExtractHiddenStatesProposer

独立（非基类派生），K 必须为 1：
- 缓存 target 的 aux hidden states 到 cache-only attention 层
- `propose` 返回 target 的采样 token 作为"总是接受的 draft"
- 用途：KV/hidden-state 缓存（如 KV transfer），非投机

---

## 10. GPU 侧 Speculator 体系

### 10.1 BaseSpeculator

```python
class BaseSpeculator(ABC):
    def init_cudagraph_manager(self, cudagraph_mode): ...
    def capture(self): ...
    def propose(self, input_batch, attn_metadata, slot_mappings,
                last_hidden_states, aux_hidden_states, num_sampled,
                num_rejected, last_sampled, next_prefill_tokens,
                temperature, seeds, ...) -> torch.Tensor:  # [num_reqs, K]
```

### 10.2 DraftModelSpeculator

```python
class DraftModelSpeculator(BaseSpeculator):
    def __init__(self, vllm_config, device):
        self.input_buffers = InputBuffers(...)
        self.draft_tokens = torch.zeros(max_reqs, K)
        self.draft_logits = ...  # 仅 probabilistic 采样时分配
        self.use_fp64_gumbel = ...
        self.use_local_argmax_reduction = ...

    def load_model(self, target_model):
        self.model = self.load_draft_model(...)  # 抽象
        self.draft_attn_layer_names = all - target attention layers

    def set_attn(self, model_state, kv_cache_config, block_tables, ...):
        # 初始化 draft 层 attention backend
        # 保留 target buffers/groups for prefill capture

    def sample_draft(self, hidden_states, positions, idx_mapping,
                     temperature, seeds, draft_step, draft_logits):
        # probabilistic → gumbel_sample(logits, ..., positions+1, ...)
        # greedy → _greedy_sample_draft (local argmax 或 compute_logits().argmax)
```

### 10.3 AutoRegressiveSpeculator（EAGLE/MTP GPU 路径）

```python
class AutoRegressiveSpeculator(DraftModelSpeculator):
    advance_draft_positions = True  # Gemma4: False

    def init_cudagraph_manager(self, cudagraph_mode):
        # prefill manager: K+1 tokens/req, FULL 或 breakable-PIECEWISE
        # decode manager: 1 token/req, FULL_DECODE_ONLY 或 NONE

    def capture(self):
        # 捕获 _prefill (forward + compute_logits + sample)
        # 若 K>1, 捕获 _generate_draft (一个 decode step)

    def propose(self, ...):
        # 1. 合并 aux hidden states (eagle3)
        # 2. prepare_prefill_inputs (Triton)
        # 3. dispatch prefill CG
        # 4. _prefill: forward + 采样 token 0
        # 5. prepare_decode_inputs (Triton)
        # 6. _multi_step_decode: step 1..K-1
        # 7. 返回 self.draft_tokens[:num_reqs]
```

### 10.4 各 Speculator 继承关系

```
BaseSpeculator (ABC)
└── DraftModelSpeculator
    ├── AutoRegressiveSpeculator
    │   ├── EagleSpeculator (eagle/)
    │   ├── MTPSpeculator (mtp/)
    │   └── Gemma4Speculator (gemma4/)
    └── DFlashSpeculator (dflash/)
        └── DSparkSpeculator (dspark/)
```

---

## 11. 拒绝采样（Rejection Sampling）

### 11.1 算法（Leviathan et al. 2023）

```
对每个 draft 位置 i:
  若 greedy:
    接受 iff target_argmax == draft_sampled
  若 stochastic:
    接受 iff p_target(x) > u · p_draft(x),  u ~ Uniform(0,1)
    等价: target_logprob > log(u) + draft_logprob

拒绝后:
  从残差分布 max(p_target - p_draft, 0) 重采样 bonus token
  (log-space: target_log_probs + log1p(-exp(draft - target)))
```

### 11.2 GPU RejectionSampler

```python
class RejectionSampler:
    rejection_sample_method: "standard" | "synthetic" | "block"

    def __call__(self, logits, input_batch, draft_logits=None) -> SamplerOutput:
        # 1. draft_sampled = input_ids[logits_indices]
        # 2. _verify_in_chunks: 分块验证 (fp32 buffer < 1GB)
        #    每块: sampler.apply_sampling_params + Triton rejection_sample
        # 3. get_num_sampled_and_rejected
        # 4. 返回 SamplerOutput(sampled_token_ids, logprobs, num_sampled, num_rejected)
```

### 11.3 5-kernel 流水线（rejection_sampler_utils.py）

```
1. _compute_local_logits_stats_kernel
   每 (logit 行, vocab 块) 计算 target argmax+max (greedy) 或 max+sumexp (stochastic)

2. (block verification) _compute_cumulative_log_p_kernel
   运行联合比率 log p_i = min(log p_{i-1} + log p(x_i) - log q(x_i), 0)

3. _rejection_kernel (每请求一个 program, 顺序处理 draft tokens)
   greedy: 接受 iff target_argmax == draft
   stochastic: 接受 iff target_logprob > log(u) + draft_logprob
   block: 接受 iff u <= h (residual mass 阈值)

4. _resample_kernel
   从残差分布 max(p - q, 0) 采样 bonus token
   gumbel_block_argmax (fp32/fp64 Gumbel)

5. _insert_resampled_kernel
   写入重采样 token, 递增计数
```

### 11.4 三种拒绝采样方法

| 方法 | 说明 |
|------|------|
| `standard` | 标准 Leviathan 算法 |
| `synthetic` | 预计算 `synthetic_conditional_rates`（校准的逐位置接受率） |
| `block` | 块验证（Sun et al. 2024），residual-mass 阈值 |

### 11.5 draft 采样方法

| 方法 | 说明 |
|------|------|
| `greedy` | argmax（默认，最快） |
| `probabilistic` | Gumbel 采样（保留 draft 分布，用于 probabilistic rejection） |

---

## 12. 动态投机解码

### 12.1 原理

根据 batch size 动态调整投机 token 数 K：大 batch 时 K 小（验证开销大），小 batch 时 K 大（加速明显）。

### 12.2 配置

```python
# 3-tuple: (range_start, range_end, num_speculative_tokens)
num_speculative_tokens_per_batch_size = [
    (1, 4, 5),    # batch 1-4: K=5
    (5, 16, 3),   # batch 5-16: K=3
    (17, 64, 1),  # batch 17-64: K=1
]
```

### 12.3 调度器集成

```python
# scheduler.py
self.dynamic_sd_lookup = build_dynamic_sd_schedule_lookup(
    schedule, vllm_max_batch_size, vllm_num_speculative_tokens)
# 每步:
num_spec_tokens_to_schedule = self.dynamic_sd_lookup[len(num_scheduled_tokens)]
# → SchedulerOutput.num_spec_tokens_to_schedule
# → runner 传给 drafter.propose(num_speculative_tokens=K)
# K=0 时: 运行 prefill forward (drafter KV 同步), 返回空 tensor
```

---

## 13. 调度器与 Runner 集成

### 13.1 完整数据流

```
Scheduler.schedule():
  ├── 分配 KV cache (含 num_lookahead_tokens = K)
  ├── scheduled_spec_decode_tokens: 上一步的 draft tokens
  └── SchedulerOutput.num_spec_tokens_to_schedule (动态 K)

Runner.execute_model():
  ├── target 模型 forward → hidden_states
  ├── target 采样 → bonus token
  └── (异步) speculator.propose(...) → draft_tokens [num_reqs, K]

Runner.sample_tokens():
  ├── 构建 SpecDecodeMetadata (logits indices for draft + bonus)
  ├── rejection_sampler(metadata, draft_probs, logits, sampling_metadata)
  │   → 接受/拒绝 draft tokens
  └── 输出: 接受的 token 前缀 + bonus token

Scheduler.update_from_output():
  ├── 更新 num_computed_tokens (减去被拒绝的)
  ├── SpecDecodingStats.observe_draft(num_draft, num_accepted)
  └── 下一步: 用接受的 token 作为新 bonus
```

### 13.2 异步投机解码

```python
use_async_spec_decode = use_async_scheduling and num_spec_tokens > 0
# draft 提议与异步输出拷贝重叠
```

### 13.3 KV Cache 分配

```python
# 调度器为 draft tokens 预留 KV cache
num_lookahead_tokens = self.num_spec_tokens  # = K
allocate_slots(request, num_new_tokens, num_lookahead_tokens=K)
# 分配 K 个额外 slot 给 draft tokens
```

---

## 14. 指标与监控

### 14.1 SpecDecodingStats

```python
@dataclass
class SpecDecodingStats:
    num_spec_tokens: int
    num_drafts: int = 0
    num_draft_tokens: int = 0
    num_accepted_tokens: int = 0
    num_accepted_tokens_per_pos: list[int]  # 逐位置接受直方图

    def observe_draft(self, num_draft_tokens, num_accepted_tokens):
        self.num_drafts += 1
        self.num_draft_tokens += num_draft_tokens
        self.num_accepted_tokens += num_accepted_tokens
        for i in range(num_accepted_tokens):
            self.num_accepted_tokens_per_pos[i] += 1
```

### 14.2 关键指标

```
Mean acceptance length = 1 + (num_accepted / num_drafts)
Draft acceptance rate = num_accepted / num_draft_tokens × 100%
Per-position acceptance rate = 逐位置接受率
```

### 14.3 Prometheus 指标

```
vllm:spec_decode_num_drafts
vllm:spec_decode_num_draft_tokens
vllm:spec_decode_num_accepted_tokens
vllm:spec_decode_num_accepted_tokens_per_pos{position=N}
```

---

## 15. 方法对比与选择指南

### 15.1 三大方法对比

| 维度 | EAGLE/EAGLE3 | MTP | DFlash |
|------|-------------|-----|--------|
| 起草模式 | 自回归（K 次 forward） | 自回归（MTP 层复用 K 次） | **并行**（1 次 forward） |
| 额外模型 | 需要 EAGLE 头 | 模型原生 MTP 层 | 需要 DFlash 模型 |
| 输入 | token ids + hidden states | token ids + hidden states | hidden states → context KV |
| 位置 | 每步 +1 | 每步 +1 | 固定 |
| KV 写入 | 每步新 KV | 每步新 KV | context KV 预插入 |
| 额外 slot/req | 0 | 0 | K |
| 适用模型 | 任意（需训练 EAGLE 头） | DeepSeek/Qwen3/GLM 等 | Qwen3 等（需 DFlash 模型） |
| 接受率 | 高 | 高（模型原生） | 中（并行起草） |
| 延迟 | 中（K 次 forward） | 中 | **低**（1 次 forward） |

### 15.2 选择指南

| 场景 | 推荐方法 | 配置 |
|------|---------|------|
| 模型有原生 MTP 层 | MTP | `method=mtp, num_speculative_tokens=K` |
| 有 EAGLE 头 | EAGLE | `method=eagle, model=eagle_path` |
| 有 EAGLE3 头 | EAGLE3 | `method=eagle3, model=eagle3_path` |
| 有 DFlash 模型 | DFlash | `method=dflash, model=dflash_path` |
| 无额外模型 | Ngram | `method=ngram, prompt_lookup_max=5` |
| 高吞吐大 batch | 动态投机 | `num_speculative_tokens_per_batch_size=[...]` |

### 15.3 K 值选择

| K | 加速比 | 适用 |
|---|--------|------|
| 1-2 | 1.2-1.5x | 大 batch / 低接受率 |
| 3-5 | 1.5-2.0x | 中等 batch（推荐） |
| 5-10 | 1.8-2.5x | 小 batch / 高接受率 |

---

## 16. 配置参数与环境变量

### 16.1 启动参数

```bash
# MTP (模型原生)
vllm serve model --speculative_config '{"method": "mtp", "num_speculative_tokens": 3}'

# EAGLE
vllm serve model --speculative_config '{"method": "eagle", "model": "eagle_path", "num_speculative_tokens": 5}'

# DFlash
vllm serve model --speculative_config '{"method": "dflash", "model": "dflash_path", "num_speculative_tokens": 4}'

# Ngram
vllm serve model --speculative_config '{"method": "ngram", "num_speculative_tokens": 5, "prompt_lookup_max": 5}'
```

### 16.2 环境变量

```bash
VLLM_REJECT_SAMPLE_OPT=0        # 拒绝采样优化
VLLM_USE_V2_MODEL_RUNNER=1      # V2 runner (dspark/dflash 强制)
```

### 16.3 SpeculativeConfig 高级字段

```python
use_local_argmax_reduction: bool = False   # 词表并行局部 argmax
use_heterogeneous_vocab: bool = False      # 异构词表 TLI
rejection_sample_method: "standard"         # standard/synthetic/block
draft_sample_method: "greedy"               # greedy/probabilistic
disable_padded_drafter_batch: bool = False  # 禁用 padding (EAGLE)
```

---

## 17. 常见问题

### 17.1 配置问题

| 问题 | 原因 | 解决 |
|------|------|------|
| `num_speculative_tokens must be divisible by n_predict` | MTP 层复用要求 | K 设为 n_predict 的倍数 |
| `Unsupported speculative method` | 方法名错误 | 检查 `SpeculativeMethod` 列表 |
| EAGLE 接受率低 | draft 头质量差 | 用 EAGLE3 或调低 K |
| DFlash 需要非因果后端 | `dflash_causal=False` | 用 FlashAttention 等支持非因果的后端 |

### 17.2 性能问题

| 问题 | 原因 | 解决 |
|------|------|------|
| 投机解码无加速 | 接受率太低 | 调低 K 或换方法 |
| 大 batch 时变慢 | 验证开销大 | 用动态投机（大 batch 小 K） |
| 显存增加 | draft 模型 + KV cache | 共享 embed/lm_head，调低 K |
| CUDA Graph 捕获失败 | draft 模型动态控制流 | 检查 splitting_ops |

### 17.3 正确性问题

| 问题 | 原因 | 解决 |
|------|------|------|
| 输出分布偏移 | 拒绝采样实现错误 | 用 `standard` 方法验证 |
| 异构词表 OOB | draft/target 词表不同 | 启用 `use_heterogeneous_vocab` |
| 结构化输出 + 投机 | grammar 验证 draft tokens | `DraftTokensHandler` 异步 D2H |

---

> **文档版本**：基于最新 vLLM 版本  
> **生成日期**：2026-08-15  
> **核心文件**：`vllm/v1/spec_decode/llm_base_proposer.py`, `eagle.py`, `dflash.py`, `step3p5.py`, `gemma4.py`, `vllm/v1/worker/gpu/spec_decode/`, `vllm/config/speculative.py`
