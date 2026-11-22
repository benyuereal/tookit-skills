# MoE / MHC 覆盖与 HBird 植入方案

> 依据：`/workspace` 实测资产（megamoe.md、HBIRD_LANDING.md、LIGHTOP_MQA_PERF.md、FP8_REPORT.md、graph_parity_*）
> + 仓库现状（`op-atlas` 5 卡、`surfaces.yaml` 11 条、`loop.py` 1207 行）。
> 结论均可回溯到具体文件，不引用印象。

---

## 0. 四个问题的直接回答

| 问题 | 结论 |
|---|---|
| MoE 覆盖全面了吗 | **没有，而且维度错了**——把 MoE 当成 1 个算子登记，实际是 7 个可独立优化的段 + 2 层并行策略 |
| lightop / megamoe 怎么比 | **不在同一层**，不能同表比。lightop 是算子库层，MegaMoE 是引擎并行层 |
| MHC 要覆盖吗 | **要，且优先级高**——它是 hbird 那条线（`w7/mhc-hip`）的底座 |
| 薄约束做了吗 | **做了，是全仓最强的一块**，但有一个执行断点 |

---

## 1. MoE 拆解：它不是 1 张卡，是 7 张

MoE 前向的真实结构：

```
x → gate/softmax → topk → (可选) align_block_size → permute
   → grouped GEMM（w8a8 / w4a16 / bf16）
   → activation（SwiGLU，含 gate/up 两路）
   → unpermute → 加权 reduce
   ↘ (分布式时) dispatch / combine（all2all）
```

现在 atlas 里 MoE 只有 2 条：`moe-w8a8`(draft)、`moe-grouped-gemm`(draft)。
**覆盖的是第 4 段（一半）和第 5 段（未实现）。** gate/topk、align/permute、unpermute/reduce、dispatch/combine 四段完全没有卡。

而这四段恰恰是最容易踩坑的地方：
- `align_block_size` 的 padding 语义错一位 → 结果静默错（对应仓库已有教训「-1 只能垫在队尾」）
- `combine` 的累加顺序影响 determinism（对应 `contract.yaml` 的 `determinism` invariant）
- `topk` 的 tie-break 影响可复现性

→ 所以 MoE 的"覆盖全面"= **7 张卡，不是补 1 张**。

---

## 2. 五方对比：必须分两层，否则结论是错的

### 2.1 算子库层（同层可比，**该进 `last_verdict`**）

| 库 | MoE 形态 | 流水线特征 | 已知档位约束 | 证据来源 |
|---|---|---|---|---|
| **lightop-4stage** | 4 段式（align→grouped→act→reduce） | 包内融合，无 persistent 全局调度 | FP8 走 default 后端；persistent 在 16-row pooled cache 上 **RuntimeError** | `lightop_bench/FP8_REPORT.md` |
| **deepgemm-das** | grouped GEMM + persistent | persistent CTA，跨 tile 驻留 | 需 64-row packed pages | `deepgemm_moe_bench/`、`moe-grouped-gemm.yaml` |
| **hbird-gemm** | GEMM/DSA persistent dispatch | **persistent dispatch（已落地 patch）** | 与 FP8 路线冲突（见 §2.3） | `hbird-gemm/dsa_mqa_logits/patches/0001-*` |
| **aiter-moe** | fused MoE | 上游 AITER 移植 | 待核 | `dcu-aiter-moe` skill |
| **tilelang-hygon** | 可改 kernel，逐 tile 调度 | 无全局 persistent | 长上下文优先 | `sparse-mla` variants |

### 2.2 引擎并行层（**另一层，结论不能写进算子卡**）

| 方案 | 场景 | 实测结论 | 证据 |
|---|---|---|---|
| **MegaMoE + CP8** | Prefill | 44 个达标点里 39 个更快，中位数 **+13.5%** | `megamoe.md` |
| **MegaMoE + DP8 + DSPARK** | Decode | 并发 32 / 输入 ≤32K：**+38% ~ +64%** | 同上 |
| 旧 P-tp8ep / D-TP8DP8 | 对照 | 基线 | 同上 |
| 显存 | — | 权重 49.29 GB + draft 2.94 GB，水位 **0.80**；0.90 第一笔就 OOM | 同上 |

> **纪律**：`megamoe.md` 里 +13.5% / +38~64% 是**端到端引擎数字**，受并行策略、显存水位、预热策略共同影响。
> 写进 `op-atlas` 卡片的 `last_verdict` 会污染算子层判断——这正是仓库反复强调的「口径打架」。
> 它应该进 `data/` 的引擎基线文档，和算子卡分离。

### 2.3 已经踩到的坑，正好证明「覆盖必须带前置条件」

`lightop_bench/FP8_REPORT.md` 里的实测：

- FP8 在 **default** 后端 192×32K = 0.1465 ms、192×128K = 0.5517 ms，比 BF16 快 **9.4×/10.0×**，有效带宽 1417/1505 GB/s
- 但 **persistent backend 对 FP8 是减速**：64-row packed 形状上慢 **1.28–1.51×**
- 且 persistent 在**强制的 16-row pooled cache 上直接 `RuntimeError`** → `LIGHTOP_PAGED_MQA_LOGITS_BACKEND=1` 对 FP8 **必须保持关闭**
- 而 int8 上 persistent/hbird 派发仍然有效

**这张表说明**：如果只写「lightop persistent 更快」进卡片，下一个人照着跑就会撞 `RuntimeError`，或者拿到 1.3× 的负收益。所以每张卡必须写 `precondition` + `disabled_when`，不然"覆盖全面"只是把坑铺得更广。

---

## 3. 覆盖补齐清单

优先级判据：**① 有没有实测证据可引 ② 是否被别的卡依赖 ③ 能否复用现有 opcheck 框架**。

### P0（先做 8 张，把骨架撑起来）

| 卡名 | 家族 | 为什么 P0 | 证据来源 | 关键前置条件 |
|---|---|---|---|---|
| `moe-topk-gating` | moe | 7 段里的第 2 段，所有 MoE 必经 | 待建 | tie-break 规则需固定，否则 determinism 判不了 |
| `moe-align-permute` | moe | padding 语义错一位就静默算错 | 待建 | `-1` 必须是队尾哨兵 |
| `moe-grouped-gemm` | moe | 已有 atlas 卡，补七件套 | `deepgemm_moe_bench/` | 64-row packed pages |
| `moe-w8a8` | moe | 已有 SKILL+contract，补五件 | `lightop_bench/FP8_REPORT.md` | FP8 禁用 persistent |
| `rmsnorm-fused` | norm | 最典型 bandwidth-bound，最好证伪 | 待建 | 残差是否原地写 |
| `paged-kv-write` | kv | 推理必备，能验证 `mutates_inputs` | 待建 | 16-row vs 64-row page 布局 |
| `rope` | rope | 每层必经，档位简单 | 待建 | d/tail_dim 组合 |
| `silu-gelu` | act | MoE 第 5 段依赖它 | 待建 | 是否与 gate 融合 |

### P1（第二批 8 张）

`moe-w4a16`、`moe-combine-dispatch`（含 all2all）、`mla-standard`（非 sparse）、`mha-gqa-prefill`、`bf16-gemm`、`quant-dequant`、`all-reduce`、`layout-transpose`

### P2（随 MHC 一起）

`causal-conv1d`、`fast-hadamard`、`mhc-fused-kernel`（见 §4）

---

## 4. MHC：要覆盖，但是「一个家族三张卡」

MHC = Megatron 的 MHC 三件套（`dcu-mhc-kernels-source` 的 description 已写明）：
`causal-conv1d` + `fast-hadamard-transform` + `mhc-fused-kernel`（融合版）。

**为什么优先级高**：`HBIRD_LANDING.md` 明写 hbird 那条线的 base 是 **`w7/mhc-hip`**。MHC 不稳，hbird 的 persistent dispatch 就没有可信底座。

建议拆 3 张，理由是三类瓶颈完全不同：

| 卡 | 瓶颈类型 | 第一判据 | 为什么不能合成一张 |
|---|---|---|---|
| `causal-conv1d` | memory / scan 依赖 | 有效带宽 vs 流式峰值 | 是串行 scan，优化手段是 tile 内移位，不是 GEMM 那套 |
| `fast-hadamard` | compute（radix/N） | FLOP 利用率 + LDS 争用 | 是蝶形运算，判据在指令数不在字节 |
| `mhc-fused-kernel` | 融合收益 | 融合前后访存字节差 | 它存在的意义就是验证前两张的融合是否划算 |

---

## 5. HBird 植入：两处，不是一个地方

### 5.1 方法层 → `op-roofline/references/pipeline.md` 补 D5

现有 pipeline.md 是「四问」，覆盖：消费者数、在途单位、LDS/VGPR 预算、tile 参数。**缺第五问：跨 iteration 的驻留与派发。**

HBird 的 persistent dispatch 恰好是这个问题的答案。要补的判据：

```
Q5 这个 kernel 的迭代间代价是否大于驻留收益？
  算：每 iteration 的重入开销（grid launch + 地址重算 + prologue）vs 驻留占用的 LDS/寄存器机会成本
  判：
    - 存在「常驻数据不吃 shared」的方案 → 走 persistent
    - 常驻数据必须进 LDS → 先算 shared 预算，算不下就别做（已有反例：CHANGELOG 记的「常驻数据不能吃掉 shared」）
    - 小形状（单 CTA 时间 < 重入开销 5×）→ persistent 必亏
```

并且 HBird 已经给了**反例的证据**：`RESULTS.md` 里 FP8 + persistent 慢 1.28–1.51×、16-row 直接崩。这正好是「Q5 答错会怎样」的实证，比任何说理都硬。

### 5.2 证据层 → 七件套补 3 件

`hbird-gemm/dsa_mqa_logits/` 的包结构比仓库七件套更完整：

| HBird 有 | 仓库七件套 | 判断 |
|---|---|---|
| `reproduce.sh`（精确命令+环境） | ✗ | **必须补**，否则「可复跑」是口号 |
| `trace_budget.py`（外部 trace → 子算子预算） | ✗ | **必须补**，这是把线上问题映射到卡片的唯一通道 |
| `patches/0001-*.patch` + apply/revert | ✗ | **必须补**，否则"改了什么"只能靠 commit 猜 |
| `RESULTS.md` 带 roofline 算式 | ≈ ledger.md | 已有，但缺算式，按 HBird 的写法补 |

→ 建议把七件套升级成 **九件套**：`+ reproduce.sh + trace_budget.py`，`patches/` 作为可选第十件。

---

## 6. 薄约束：做得好，但断了一个执行环节

### 已经做到的（这部分是全仓最强的设计）

| 机制 | 位置 | 实际作用 |
|---|---|---|
| 攻击面预枚举 | `op-optimize/references/surfaces.yaml`（**11 条**） | 把「凭空发明方向」变成「从表里挑」；每条带 `law / precondition / check / evidence / cost` |
| 主张门 | `loop.py:257 claim_gate()`，4 个调用点（461/768/1088/…） | 机器判定「你觉得快了」不成立 |
| 完成裁决 | `loop.py verify/adopt` | `done_when` 是一条**外部 shell 命令**，返回 0 才 adopted，不由模型口述 |
| 方向证伪 | `direction-refuted` + 同方向证伪 2 次封方向 | 不重复探索已死方向 |

这是「约束轻薄」的正确形态：**约束的是"往哪看"，不是"写什么"**。比写一堆规定强得多。

### 断点（两处，都小但都致命）

**断点 1：`allowed_actions` 没人消费**

`op-constrain` 的契约写得很硬：「按 verdict 硬映射允许动作：memory 天花板→只准减字节；compute 天花板→只准减 FLOP」。
但 `loop.py` 里 `grep allowed_actions` **零命中**——只有「回 op-constrain 重定约束」的**提示语**，没有**读取和拦截**。

后果：`op-constrain` 产出的允许集合只是 result.json 里一段文本，`loop.py run` 不校验 → **越界的改法照样能跑**，薄约束在最后一步漏了。

补法（十几行）：

```python
# loop.py run 之前
allowed = load_result(card).get("allowed_actions")
if allowed is not None and candidate["action_class"] not in allowed:
    die(f"候选 {tag} 的动作类 {candidate['action_class']} 不在 allowed_actions {allowed} 内"
        "（op-constrain 已判天花板；要越界先回 op-constrain 重定约束）")
```

**断点 2：`surfaces.yaml` 的 law 是悬空引用**

11 条攻击面每条都写 `law: L1`…，但 `op-roofline/references/` 下只有 `cards.md / pipeline.md / rules.md`，**没有 `pipeline-laws.md`**——定律本体不存在（可能散落在 pipeline.md 里没编号）。

后果：`law` 字段变成一个装饰性的编号，"判死要能追溯到机制"这条纪律落空。

补法：要么新建 `pipeline-laws.md` 把 L1..Ln 定义清楚，要么把字段改成指向 `pipeline.md#锚点`，并在 `op-atlas/validate.py` 里加一条校验（law 必须能被解析到实体）。后者更省事，前者更长久。

---

## 7. 批次 3 建议（等第二批跑完再发）

1. **先补 3 张 P0 卡**：`moe-topk-gating` / `moe-align-permute` / `rmsnorm-fused`
   理由：前两张补齐 MoE 断链，第三张是最好证伪的 bandwidth-bound，能把 roofline 流程跑通第二遍
2. **修薄约束两个断点**（十几行 + 一个校验），因为它是所有后续卡的护栏，先修收益最大
3. **pipeline.md 补 Q5**（persistent dispatch 判据 + HBird 反例）
4. **七件套升九件套**：先只给新卡打样，不要回头改 3 张老卡

暂不做：MHC 三件套（等 hbird Q5 落地后再动，否则判据不全）。
