---
name: dcu-cudagraph-baked-scalars
description: 排查 CUDA/HIP graph 下"每 batch 变化的 Python 标量被烧死在 capture"导致的静默读错（accept 从第 1 轮起为 0、输出错但不崩）。适用于 sglang/vllm 投机解码 draft、attention backend 等 graph 化路径。
version: 1.0.0
author: DCU Engineer
license: MIT
kind: knowledge
---

# CUDA Graph 烧死 Python 标量：排查与修复

## 问题模式

CUDA/HIP graph capture 时，**每 batch 变化的 Python 标量**（int/float，如 `max_seqlen_k`、`max_seqlen_q`、`topk`、`num_tokens`）若被传进 kernel 启动参数，会在 capture 时被**烧死**在 graph 里，replay 时不更新。后果是**静默读错范围/算错**——不崩、不报错、不 NaN，只是结果错。

典型症状（sglang 投机解码 draft graph）：
- accept rate **从第 1 轮起就是 0.00**（不是跨页/跨长度时的悬崖式下跌）
- 吞吐远低于 graph 关闭时的 eager 路径
- target 模型正常（因为 target 路径传的是常量上界）
- 换 tensor 内容（page_table 格式 per-page/per-token 等）无效——因为错的不是 tensor 而是标量

## 排查步骤

1. **先确认走没走 graph**：日志里 `cuda graph: True` 行数 > 0 说明在 replay；`is_cuda_graph()` 对相应 ForwardMode 为 True。
2. **区分"tensor 过期"和"标量烧死"**：
   - 改 A 的两个取值结果完全相同 → A 不是根因（例：page_table 两种格式都 0.00 → 不是 page_table）。
   - 检查 capture 时的 fill value：sglang `get_cuda_graph_seq_len_fill_value()` 返回 1 → capture 时 `max_seqlen_k = 1 + draft_tokens`（如 5）→ replay 永远只读 5 个 KV token。
3. **grep kernel 调用点的所有 Python 标量参数**：对 graph 化的 forward 路径，逐个检查传给 kernel 的 int/float 是否每 batch 变化。
4. **修复**：改用**常量上界**（如 `self.max_context_len`）。常量在 capture 时烧死无影响，因为 replay 时它本来就不该变。
5. **验证**：accept rate 恢复到与 eager 路径一致（不是"非 0 就行"，要和 graph 关的基线对齐）；吞吐应高于 eager（graph 收益）；`cuda graph: True` 持续出现。

## 通用规则

- graph 化的 kernel 调用里，**每 batch 变化的标量必须走 tensor 侧**（写进预分配 buffer 由 kernel 读）或**证明其有常量上界**。
- 排查 graph 下"从第 1 轮就错"时，**优先查被烧死的标量，而不是 tensor 内容**（page_table、cache_seqlens 等）。
- 负收益的 graph 改动先回退再排查（例：graph 开 32 tok/s < graph 关 75 → 先回退到已知可用态，再定位根因，不要在坏态上叠加假设）。

## 实测案例（<device-B>，DFlash2 draft）

- 症状：draft graph 开 → accept 恒 0.00、32 tok/s；graph 关 → 75.8 tok/s、accept 0.72。
- 证伪过程：先怀疑 replay 的 page_table 格式（per-page vs per-token），打 patch 两种都试 → 结果完全相同 → 证伪。
- 根因：fa2 回退调用 `max_seqlen_k=metadata.max_seq_len_k`（每 batch Python int）被烧死；capture 时 fill=1 → 烧死值 5。
- 修复：flashattention_backend.py 三处改 `max_seqlen_k=self.max_context_len`。
- 结果：98.2/98.4 tok/s（+30%），accept 0.68-0.82 与 eager 基线一致，独立验证 PASS（29 样本 0 个 0.00）。
