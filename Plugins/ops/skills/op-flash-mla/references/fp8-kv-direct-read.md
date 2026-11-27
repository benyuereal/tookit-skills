# fp8 KV 直读配方（从 SKILL.md §八 拆出）


> **一句话**：KV cache 存 fp8，attention kernel 直接读 fp8、用行内 scale 反量化成 **bf16**，
> MMA 仍走 **bf16**；**Q 必须保持 bf16**。目标是省 HBM 字节，不是换 MMA 精度。
> 2026-11-27 / BW1100 / gfx938：decode 生产已验证（TPOT −6.9%）；prefill 待实现（候选 `prefill_fp8kv_qbf16`）。

### 8.1 机制

| 项 | 取值 |
|---|---|
| KV cache | fp8（GLM5.3 吸收式 MLA 行宽 **528 B = 512 fp8 + 4×fp32 scale**，每 128 维一个 scale） |
| kernel 读 | 直接读 fp8 行；用行内 scale 在 kernel 内反量化成 **bf16** |
| Q | **保持 bf16**（不量化） |
| MMA | **bf16**（`mmac_*_bf16`），不是 fp8×fp8 |
| 省下的字节 | 每 token 每层 KV **1024 B（bf16）→ 528 B（fp8）** |

**decode 生产实测（2026-11-27，BW1100/gfx938）**：

| 指标 | bf16 | fp8 KV 直读 | 变化 |
|---|---|---|---|
| TPOT 均值 | 54.70 ms | 50.91 ms | **−6.9%** |
| TPOT median | 56.27 ms | 52.09 ms | **−7.4%** |
| 吞吐 | — | — | **+6.4%** |
| TTFT | — | — | 不变 |
| 精度 | — | cos 0.999991 / max_abs 3.1e-4 | 过容差 |

机制不是「fp8 算得快」，而是删掉了 `decode_gather_and_up_convert_with_indices`（v3 trace 里
3.63 ms/step，占 8.3%），并让每 token KV 字节从 1024 降到 528。

### 8.2 适用条件（缺一条就别照搬）

1. **KV cache 本身已经是 fp8**，且行内带 scale（per-128 分组）；不是「bf16 cache 临时量化一遍再读」——
   那会多一趟写回，HBM 通过次数反而增加。
2. **Q 精度敏感**：吸收式 MLA 的 Q 直接决定 logits。Q 保持 bf16 是本配方的硬条件，不是可调项。
3. **目标是省 HBM 字节**：收益上界由「KV 读在墙钟里的占比」决定，不是「字节减半 ⇒ 快一倍」。
   先用 PMC/roofline 确认 KV 读是主要项，再动手。
4. **先算读放大**：每个 token 起几个 CTA 决定 KV 被读几遍。同族 prefill 实测 `h_q≤16` 时 1 CTA/token、
   无冗余；`h_q=64` 时 4 CTA/token，每 token 读 528×4 B，字节账可能反超 bf16（详见 §8.6 与
   `op-source-patch` 的 `engine-recipes.yaml:sparse-mla-fp8-direct` traps）。
5. **decode 已验证；prefill 未验证**（§8.5）。

### 8.3 反例：Q 也量化成 fp8（最容易踩的坑）

`/workspace/fp8_mla`（2026-11-21）把 **Q 也量化成 fp8**，走 `mmac_f32_16x16x32_fp8_fp8`：

| 维度 | 结果 | 判读 |
|---|---|---|
| decode 形态收益 | 36.81 → 33.71 ms，**仅 −8.4%** | 远小于「字节减半」的预期 |
| 精度 | cos **0.999506** / max_abs **2.44e-2** | **远超容差 3e-4**，不可用 |
| kernel 自身带宽 | **262 GB/s = 13% 峰值** | fp8 码路本身没跑起来 |
| 同任务 bf16 版 | **1165 GB/s = 59% 峰值** | 慢的不是「fp8」，是那条码路 |

⇒ **不要用「fp8 一定更快」当先验**。Q 量化引入的精度损失不可接受，fp8×fp8 MMA 也没有兑现带宽；
正确的做法是只把 KV 的**读**换成 fp8，计算域留在 bf16。

### 8.4 正确的 decode 实现（可抄）

- 目录：`/workspace/FlashMLA-das/csrc/gfx93/decode/sparse_fp8/`
- `Element = bfloat16_t`（Q 与计算域都是 bf16）
- `dequant_fp8_to_bf16()`（`splitkv_mla.cuh:23`）：读 fp8 行 + 行内 scale，反量化成 bf16 后喂 MMA
- 上层入口与开关：`SGLANG_DSA_FP8_NOPE512_DECODE=1`；完整步骤与坑见
  [`../op-source-patch/references/engine-recipes.yaml`](../op-source-patch/references/engine-recipes.yaml)
  的 `sparse-mla-fp8-direct`

### 8.5 prefill：待验证与实现障碍

**现状**：flash_mla 的 sparse prefill 接口文档要求 `kv: bfloat16`，所以 prefill 仍是
「fp8 cache → 上转 bf16 → bf16 kernel」。若照 decode 改成 **Q bf16 + KV fp8 直读 + kernel 内反量化**：

| 项 | 估计 |
|---|---|
| KV 字节 | **−48%**（1024 → 528 B/token） |
| 理论上限 | **~1.9×** |
| 现实预期 | **1.3–1.6×** |

**实现障碍（已确认，必须先实测再投入）**：prefill kernel 的 KV 是
`buffer_load_dwordx4 ... lds` **直接 global→LDS、不经寄存器**
（`/workspace/FlashMLA-das/csrc/gfx93/prefill/sparse/phase1_hip.cuh`）。加反量化必须走
寄存器中转 + 额外的 LDS store，因此必须实测确认：

```
省的 HBM 字节  >  多出的 LDS/VALU 开销
```

**机器裁决状态**：`bench/prefill_fp8_gate.py` 的 tag 只有 `flash_mla` / `fp8-bh64` /
`fp8-bh64-qonly`，后两者是 **Q 也量化**的反例码路；`_common.get_runner` 里没有
「Q bf16 + KV fp8 直读」的 prefill runner。所以候选 `prefill_fp8kv_qbf16` 现在写的是
`blocked_by`，**没有 done_when**——先实现 runner，再在 gate 里加 tag，才谈完成判定。

### 8.6 相邻实测约束（2026-11-27，详见 ledger §六）

- **prefill 注意力已接近带宽天花板**：生产形态 H=8、topk=2048、S=8192、128K 池，隔离测量
  **1.80 µs/token/层 = 1165 GB/s = 59% HBM 峰值**；排序索引 / 传 `topk_length` / 50% `-1` 槽位
  **全部无收益**（1.79–1.80 µs）。⇒ 省字节是剩余的主杠杆，但不要指望 100% 兑现。
- **每 token 起多个 CTA 会重复读 KV**：`B_H=16`、`h_q=64` ⇒ 4 CTA/token ⇒ **2.88 µs/token**
  （比 H=8 慢 1.6×）；`h_q≤16` 时 1 CTA、无冗余。⇒ 任何 fp8 收益先按 CTA 数折算。
- **生产索引已经排好序**：`SGLANG_KERNEL_KPOOL_TOPK_SORT_WRITEBACK=1` 由 aiter 的
  `module_kpool_topk.so` 在回写时排序（`strings` 确认该 .so 读这个环境变量）。
  「排序能省 37%」生产已经吃到，**不要再立项**。
