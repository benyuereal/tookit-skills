# mHC 语料能否吸纳：评估

> 对象：`/workspace/mhc_kernels`（1012 KB，4 个 .hip 共 1711 行 + bench/probe 脚本）
> 结论先行：**吸纳，但不是搬代码 —— 抽成两张卡 + 一份算子级 benchmark 锚点。**

## 一、它是什么

GLM-5.3-Flash 的 mHC（hyper-connection）算子 `hc_pre` / `hc_post`，
H=4096、hc=4、K=16384、24 个 mixing 列。**独立 HIP 原型**，没接进 sglang。
基线与我们同口径：`sglang` 的 `hc_pre`（prefill TC 路径）+ `mhc_post_tilelang`，
ABBA 交替、每臂 120 样本取中位。

## 二、为什么值得吸纳（三条，都不是"有文件"这种理由）

### 1. 它是**第一个「结构优化到底之后仍未达标、且把理由算清」**的样本

| m | 基线 | 新实现 | 倍数 |
|---|---|---|---|
| 1024 | 416.3 µs | 183.8 µs | 2.26x |
| 4096 | 1067.8 µs | 587.8 µs | 1.82x |
| 16384 | 4114.1 µs | 2175.1 µs | 1.89x |
| 32768 | 8193.9 µs | 4389.9 µs | 1.87x |

目标定的是 2.5x，**没达到，但它没含糊过去**：算出天花板是 2.06x，
且把 proj 的 417→527 GB/s 三代演进与「纯 load 探针 1049 GB/s（256B 段）/
1131 GB/s（512B 段）」的差距逐条对账。这正是我们 `floor.py` 想表达的东西，
但它是在**一个我们没覆盖过的算子上**跑出来的。

### 2. 它验证了我们 judge 的一个隐含假设，并给出反例

Step-3 的融合方案「建了、验了、比不融合慢 1.7 倍」，原因是**流碎裂**：
post 半边要 1-token CTA，proj 半边要 128-token CTA。
这对我们是硬反例：`fusion-boundary` surface 的 `expected_gain: 20~60%`
把融合当成普遍利好。**这个反例本身就值得进库。**

### 3. 它的分母口径和我们一致

1870 GB/s 流式读上限，同窗口 `probe_stream` 实测（256MB，五种 grid：1848–1946）。
和我们 bw-ladder 的 `read_only 1965` 是同一量级的独立复现。

## 三、怎么吸纳（三件，按性价比排序）

### A. 建两张卡：`hc-pre` / `hc-post`（优先）

`Plugins/ops/op-atlas/cards/hc-pre.yaml` / `hc-post.yaml`（已有 `mhc-fused.yaml`，
先看它是什么再决定合并或新建）。字段：

```yaml
math: hc_pre:  P = x @ fn.T; weights = softmax-like mix over 24 columns
             hc_post: y = w0*x + w1*(P @ fm.T)   # 4H in / 4H out
variants:
  - import_path: sglang.kernels.ops.layernorm.hcu_mhc.hc_pre
    signature:   hc_pre(x, fn, ...)   # 需现场 inspect 确认
  - import_path: /workspace/mhc_kernels/out/mhc.so (ctypes)
    note: 自研，417–527 GB/s（proj），1456–1827 GB/s（post）
```

**必须先做的核实**（不许照抄 README）：`inspect.signature` 确认 sglang 入口、
`nm -D out/mhc.so` 确认导出符号、跑一次 `bench.py check` 确认正确性仍成立。

### B. 把「融合反而更慢」写进 surfaces.yaml（成本最低，价值最高）

在 `fusion-boundary` 的 `evidence` 里补一条带 scoped 反例：

```
evidence: ... ；**反例（2026-11-24，mHC Step-3，/workspace/mhc_kernels）**：
两半的 CTA 粒度需求相反（post 要 1-token，proj 要 128-token），融合后
流碎裂，实测慢 1.7 倍。⇒ 融合前必须先问「两半的最优 CTA 粒度是否相容」。
```

这条比 A 便宜，而且立刻能让 `fusion-boundary` 方向不再是无条件利好。

### C. 把 proj 的「算得清的天花板」做成校准记录（最有长期价值）

它在 1049 GB/s 纯 load 上限下跑到 527 GB/s（76%），并把差距归因到
weight 双平面把 x 侧上限压到 ~690 GB/s。这是一条**可复算**的归因链，
是我们 `calibrate.py` 一直缺的「真实 realized vs predicted」样本。

## 四、不吸纳的部分（明确不碰）

| 不吸纳 | 原因 |
|---|---|
| `bench.py`(439 行) 的计时框架 | 我们已有 `ab.py` 契约 + `opbench.bench_fn`；再引入一套是第 8 份重复 |
| `explore/` 下的中间代（ovl*.hip / proj4.hip） | 是探索过程，只有结论（RESULTS.md）有留存价值 |
| `out/mhc.so` 二进制 | 编译产物不入库（`.gitignore` 已封 `*.so`） |
| 它的 `sweep_all.py` / `sweep_proj.py` | 与 `design_space.py` 功能重叠，且没有我们的合法性过滤 |

**语料入库范围**：`RESULTS.md`（证据）+ `kernels/*.hip`（4 个源，1711 行）
+ `host/dispatch.cpp`。约 130 KB 文本，不含二进制。

## 五、风险（必须先说清）

1. **它没有 candidates.yaml / contract.yaml / shapes.yaml**。直接拷进来当"卡"
   会缺我们所有门禁要的字段 —— 必须按我们的格式重建，不能整目录搬。
2. **它是 GLM-5.3 的一个具体算子**，形状硬编码（H=4096, hc=4, K=16384）。
   入库时要写清 `model_limit`，别让下一个人以为它对任意 hc 成立。
3. **`/workspace/mhc_kernels` 不在本仓库内**，路径引用会在别的机器上失效。
   入库时必须把 `RESULTS.md` 的关键数字**内联**进卡，不能只留路径。

## 六、建议执行顺序

1. 先做 B（改 surfaces.yaml 一条 evidence）—— 20 分钟，立刻有价值。
2. 再做 A（两张卡 + 实测核实）—— 半天。
3. C 视 `calibrate.py` 的接线进度决定。

**不建议整体搬迁**：那样会让一个 2.3 MB 的探索目录变成仓库负担，
而我们已经有 39 张卡在等填数据。
