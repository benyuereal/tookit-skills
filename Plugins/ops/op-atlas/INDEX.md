# 算子图谱索引（op-atlas）

> **图谱是索引层，不是知识副本。** 卡片只回答四个问题：它算什么 / 谁结构上赢 /
> 上限在哪 / 上次结论。技术描述一律指向别处（`libraries.md` 的 §号、
> `pipeline.md` 的 Q 号、kb 台账），卡片 schema 用 `additionalProperties: false` 锁死。
>
> 校验：`python3 -I Plugins/ops/op-atlas/validate.py --atlas Plugins/ops/op-atlas`

## 怎么用

1. 按家族或名字定位到卡 → 读 `impls` 看谁在这个档位结构上赢（含 `best_for` / `not_for`）
2. 读 `ceiling` 看上限在哪（`kind: missing` 是合法状态，表示天花板未知）
3. 读 `last_verdict` 看上次结论（`ratio` 必须带 `vs`，裸倍数不许写）
4. 要优化它 → 进 `op-optimize` 闭环；要查库的脾气 → 读 `libraries.md` 对应 §

## 算子一览

| 算子 | 家族 | 状态 | 首选实现 | 上限 | 上次结论 |
|---|---|---|---|---|---|
| [`embedding-lookup`](cards/embedding-lookup.yaml) | memory | draft | in-house | bandwidth | 无实测：索引卡，实现层待按证据推进 |
| [`flash-attention-fwd`](cards/flash-attention-fwd.yaml) | attention | draft | flash-attn | mixed | 无实测：索引卡，实现层待按证据推进 |
| [`fused-add-rmsnorm`](cards/fused-add-rmsnorm.yaml) | norm | draft | lightop | bandwidth | 无实测：索引卡，实现层待按证据推进 |
| [`gemm-bf16`](cards/gemm-bf16.yaml) | gemm | draft | hipBLASLt | compute | 无实测：索引卡，实现层待按证据推进 |
| [`gemm-fp8`](cards/gemm-fp8.yaml) | gemm | draft | tilelang-hygon | compute | draft：选型与坑已整理，bench 与 adap… |
| [`kv-cache-paged-write`](cards/kv-cache-paged-write.yaml) | kv | draft | flash-attn | bandwidth | 无实测：索引卡，实现层待按证据推进 |
| [`layernorm`](cards/layernorm.yaml) | norm | draft | lightop | bandwidth | 无实测：索引卡，实现层待按证据推进 |
| [`lm-head-with-sampling`](cards/lm-head-with-sampling.yaml) | gemm | draft | lightop | mixed | 无实测：索引卡，实现层待按证据推进 |
| [`mha-prefill`](cards/mha-prefill.yaml) | attention | draft | flash-attn | compute | 无实测：索引卡，实现层待按证据推进 |
| [`mla-absorb`](cards/mla-absorb.yaml) | attention | draft | FlashMLA | mixed | 无实测：索引卡，实现层待按证据推进 |
| [`mla-full`](cards/mla-full.yaml) | attention | draft | FlashMLA | compute | 无实测：索引卡，实现层待按证据推进 |
| [`mla-kv-cache-layout`](cards/mla-kv-cache-layout.yaml) | kv | draft | FlashMLA | bandwidth | 无实测：索引卡，实现层待按证据推进 |
| [`mla-kv-cache`](cards/mla-kv-cache.yaml) | kv | draft | FlashMLA | bandwidth | 无实测：索引卡，实现层待按证据推进 |
| [`mlp-gelu-geglu`](cards/mlp-gelu-geglu.yaml) | activation | draft | lightop | compute | 无实测：索引卡，实现层待按证据推进 |
| [`mlp-swiglu`](cards/mlp-swiglu.yaml) | activation | draft | lightop | compute | 无实测：索引卡，实现层待按证据推进 |
| [`moe-align-permute`](cards/moe-align-permute.yaml) | moe | draft | lightop | bandwidth | ref-heavy 通过正确性门；容差为零（纯搬运）… |
| [`moe-combine-reduce`](cards/moe-combine-reduce.yaml) | moe | draft | lightop | bandwidth | 无实测：索引卡，实现层待按证据推进 |
| [`moe-dispatch-combine`](cards/moe-dispatch-combine.yaml) | moe | draft | DeepGEMM | mixed | 无实测：索引卡，实现层待按证据推进 |
| [`moe-grouped-gemm-bf16`](cards/moe-grouped-gemm-bf16.yaml) | moe | draft | DeepGEMM | compute | 无实测：索引卡，实现层待按证据推进 |
| [`moe-grouped-gemm`](cards/moe-grouped-gemm.yaml) | moe | draft | deepgemm-das | mixed | 无本算子独立实测；grouped 比稠密循环快一条来… |
| [`moe-shared-expert`](cards/moe-shared-expert.yaml) | moe | draft | lightop | compute | 无实测：索引卡，实现层待按证据推进 |
| [`moe-swiglu-fused`](cards/moe-swiglu-fused.yaml) | moe | draft | lightop | compute | 无实测：索引卡，实现层待按证据推进 |
| [`moe-topk-gating`](cards/moe-topk-gating.yaml) | moe | draft | lightop | bandwidth | ref-heavy 通过正确性门；无性能实测；本段通… |
| [`moe-w4a16`](cards/moe-w4a16.yaml) | moe | draft | moe_c | bandwidth | 无实测：索引卡，实现层待按证据推进 |
| [`moe-w8a8`](cards/moe-w8a8.yaml) | moe | draft | lightop | mixed | Marlin GEMM 分量达峰值带宽的 0.888… |
| [`mqa-logits`](cards/mqa-logits.yaml) | attention | measured | lightop | bandwidth | epilogue 跨 lane 归约换 ds_bpe… |
| [`paged-attention-decode`](cards/paged-attention-decode.yaml) | attention | draft | flash-attn | bandwidth | 无实测：索引卡，实现层待按证据推进 |
| [`qk-norm`](cards/qk-norm.yaml) | norm | draft | lightop | bandwidth | 无实测：索引卡，实现层待按证据推进 |
| [`rmsnorm-fused`](cards/rmsnorm-fused.yaml) | elementwise | draft | lightop | bandwidth | ref-heavy 通过正确性门；无性能实测；用于校… |
| [`rope-mla`](cards/rope-mla.yaml) | rope | draft | FlashMLA | bandwidth | 无实测：索引卡，实现层待按证据推进 |
| [`rope`](cards/rope.yaml) | rope | draft | lightop | bandwidth | 无实测：索引卡，实现层待按证据推进 |
| [`sampling-topk-topp`](cards/sampling-topk-topp.yaml) | sampling | draft | lightop | bandwidth | 无实测：索引卡，实现层待按证据推进 |
| [`sparse-mla`](cards/sparse-mla.yaml) | attention | tuned | tilelang-hygon | mixed | 已收敛到指令数地板而非带宽地板；32k 档约 1.7… |
| [`topk-block-index`](cards/topk-block-index.yaml) | attention | draft | in-house | bandwidth | 无实测：索引卡，实现层待按证据推进 |
| [`weight-prepack`](cards/weight-prepack.yaml) | quant | draft | aiter-das | bandwidth | 无实测：索引卡。核心价值是提醒「漏 prepack … |

## 家族

| 家族 | 判据 | 文件 |
|---|---|---|
| gemm | 纯矩阵乘，复用结构由 M 档决定 | [families/gemm.md](families/gemm.md) |
| attention | Q/K/V 参与，瓶颈通常在 KV 搬运与片上归约 | [families/attention.md](families/attention.md) |
| moe | token 按专家重排后分组计算 | [families/moe.md](families/moe.md) |
| elementwise | 逐元素 / norm / 激活 / 量化融合，memory-bound | [families/elementwise.md](families/elementwise.md) |
| norm | 逐元素归一化，字节数确定，最适合校准带宽判据 | [families/norm.md](families/norm.md) |
| rope | 位置编码旋转，语义坑多（half vs interleaved） | [families/rope.md](families/rope.md) |
| activation | 激活与门控，多与 GEMM epilogue 融合 | [families/activation.md](families/activation.md) |
| kv | KV cache 布局与读写 | [families/kv.md](families/kv.md) |
| quant | 量化与反量化，标度粒度改变 kernel 结构 | [families/quant.md](families/quant.md) |
| comm | 集合通信——**不进实现层**，只有重叠编排 | [families/comm.md](families/comm.md) |
| linear-attn | 线性注意力与状态空间模型 | [families/linear-attn.md](families/linear-attn.md) |
| sampling | 采样，tie-break 决定 determinism | [families/sampling.md](families/sampling.md) |
| memory | 访存原语，结论常是「不要物理搬运」 | [families/memory.md](families/memory.md) |

> 其余家族暂无卡片，不建空壳文件。**空壳比没有更糟**（会让人以为这里有内容）。

## 加一个新算子

1. 在 `cards/` 加一个 yaml（字段白名单见 `validate.py`）
2. 重跑本索引（或手加一行）
3. 跑校验器，E1~E11 全过才提交

**别做的事**：卡片里写技术描述、绝对耗时、无分母百分比。
校验器会拦（E6 字段白名单、E7 结论红线、E10 场景字段、E11 层级）。
