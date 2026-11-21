---
name: op-bench
description: "算子微基准的口径与判据：先答口径三问（测什么/含不含分配/冷还是热），再按六条判据做（生产布局复现、跨卡比 ratio、有效量折算、MAD 显著性、地板判定、冷热两轮）。配套 scripts/opbench.py 提供计时/统计/门控，计时统一走 `opbench.timeit`，别手搓 time.time()。触发场景：给算子卡写 bench/bench.py、A/B 两次改动、判断「快了」是不是噪声、宣布 at_floor 前复核。不适用：契约字段怎么填走 op-contract；PMC 读数与瓶颈归因走 op-profiling；整模型推理链路缺 kernel 级 profile 走 dcu-kernel-layer-microbench（算子卡的 A/B 与噪声判定一律在本 skill）。"
version: 1.0.1
author: DCU Toolkit Contributors
license: MIT
kind: knowledge
---

# 算子微基准：口径与判据

## 一、先答口径三问（答不出就别开始测）

| 问题 | 选项 | 影响 |
|---|---|---|
| **测什么** | kernel 单发 / 算子调用（含 wrapper）/ 端到端 step | 决定结论能不能跨人比较；三者常差 10%~50% |
| **含不含分配** | 预分配 vs 每次 `torch.empty` | 含分配时 `num_split` 越大的实现越吃亏（partial buffer 越大） |
| **冷还是热** | 首轮（含 JIT/编译/预热）vs 稳定态 | 首轮绝不能进结论；JIT 算子（tilelang/triton）首次 20~30s |

结论必须在报告头写明三问的答案，否则视为无效数据。

## 二、六条判据（每条都对应一次真实误判）

### 判据 1：微基准输入布局必须与生产 kernel 输出布局一致
做过 topk「量化版」，微基准 bit-identical 且更快，上生产掉分——生产 warmup 后的输入布局与微基准构造的不同，砍掉的其实是有效区。
> **做法**：从**生产路径 dump 的真实张量**入参，或至少用生产 kernel 的输出形状/步长构造；不要用「看起来一样」的 `randn`。

### 判据 2：跨卡只比 ratio，不比绝对 µs
卡 0–5 曾钉 600MHz，卡 6–9 跑 1150MHz；把两边绝对耗时相减，得到过 1.5× 的假收益。
> **做法**：每个变体都在**同一张卡、同一进程、交替执行**（interleaved），报 ratio；跨卡只做 sanity check。

#### 判据 2 的补充：**对标比值本身也不稳**（2026-11-13 实测）

**现象**：同一个 kernel、同一张卡、同一天，S=128 的「我们 / flash_mla」比值在 **0.97 ~ 1.07** 之间漂；跨会话 ±10%，跨卡另测到 0.97 / 0.98 / 1.073。组内 MAD 可以小到 0.1%。
**机制**：sclk 未锁、频率在 1300MHz 上下漂，而两个实现（或两次会话）对频率的敏感度不同，把比值从 0.97 推到 1.07。**组内 MAD 小只说明同一进程内噪声小，不说明两次会话的条件相同**——所以「MAD 0.1%」不能拿来做跨会话的可信度证明。
**判据**：跨实现站位满足其一——① **锁频**；② **同进程 ABBA**（A/B 交替，比值在同一时间窗内取）；③ **报区间**并注明「未锁频」。单独一次的绝对值不当结论；写「领先/落后 x 倍」时必须带区间与上述条件之一。

### 判据 3：折算有效量，不按名义量算吞吐
稀疏算子带 `-1` padding、MoE 带未激活 expert、attention 带 causal 遮蔽；名义 topk=2048 时实际可能只算 1200 个。
> **做法**：报告同时给「名义 / 有效 TFLOPS·带宽」与有效率（`mean(indices>=0)/topk`）。

### 判据 4：显著性用 MAD，不用「跑两次看着差不多」
DCU 上同一 kernel 的抖动可达 5%。跑 3 次取中位数，用 `|Δ| > k × MAD`（k 默认 3）判定；过不了门的结论标 `INCONCLUSIVE`，写成「在噪声内」，不写成「快了 x%」。

### 判据 5：地板必须按带宽/算力模型算，不按直觉
「只用了 45% 带宽」经常是假象——激活 expert 数变化时带宽随激活线性，gfx938 实测已到 88.8%（其他 gfx 未验证）。
> **做法**：先写 `floor.basis` 算式（见 `op-contract` §五），再算达成率；`≥85%` 才允许标 `at_floor`。

### 判据 6：冷/热两轮，跨轮不可混用
首轮含首次 host→device 拷贝、JIT、warmup 缓存；热轮才是稳态。两轮都跑，但**结论只用热轮**，冷轮只用来暴露「冷启动问题」。

## 三、参数与产出

| 参数 | 默认 | 说明 |
|---|---|---|
| `warmup` | 25 次或 100ms（取先到者） | JIT 算子必须 ≥3 次真实调用后再计时 |
| `rep` | ≥100 次或 500ms | 单次耗时 >50ms 的大 kernel 用 20 次即可 |
| `repeats` | 3 轮 | 每轮独立重新取中位数，最后取三轮中位数 |
| `graph` | 先 eager，再试 graph | 含分配的 wrapper 会失败；失败要记录，不静默跳过 |

GPU 侧计时用 CUDA/HIP event，不用 `time.time()`（异步 queue 会让它只量到 launch）。报告必须同时出人读与机读两份，机读直接喂 `scripts/opverify.py --run`：

```
METRIC op=sparse_mla variant=tilelang shape=decode-m32 ms=0.842 ideff=0.73 tflops=182.4 bw_gbs=498.1 status=ok
{"op":"...","variant":"...","shape":"...","caliber":{"what":"op_call","alloc":true,"state":"hot"},
 "metrics":{"ms":0.842,"mad":0.011,"tflops":182.4,"bw_gbs":498.1,"effective_ratio":0.73},"env":{"gfx":"gfx938"},"date":"..."}
```

## 四、用 `scripts/opbench.py`，不要手搓

```python
import sys; sys.path.insert(0, "<repo>/scripts"); from opbench import bench_fn, significant, floor_status, emit_metric
s = bench_fn(lambda: kernel(...), warmup=25, rep=100, repeats=3)   # -> {median_ms, mad_ms, min,max, n}
emit_metric(op="sparse_mla", variant="tilelang", shape="decode-m32", ms=s["median_ms"])
ok, ratio, why = significant(base_ms, cand_ms, k=3.0)              # 判据：3×MAD
print(floor_status(achieved_gbs=1020, peak_gbs=1200))               # -> ('at_floor', 0.85)
```

只用标准库 + 可选 torch（未装 torch 时 `bench_fn` 退化为 CPU 侧 wall clock 并明确告警），可 `python3 -I scripts/opbench.py --selftest` 自检。

### 算「达成率」之前先取分母

```bash
python3 -I scripts/probe_stream.py --sizes 16 256 1024    # read/copy/write 三种口径分别报
python3 -I scripts/probe_stream.py --sizes 1024 --check   # 与 op-roofline/constants.yaml 对账
```

三种口径不可混用：`copy` 的字节按 R+W 计，只算单边会比实际高 2 倍。**小 buffer 不能当峰值**——
实测同一台机器 128 MiB 只有 1124 GB/s、1 GiB 才 1761，差 57%，因为前者跑进了固定开销主导区。

**行内连续 X 字节的访问模式（strided gather / 列读 / 转置读）另有自己的 ceiling**：
先跑 `python3 -I scripts/probe_pattern.py --runs 128 256 512 linear`，用它给的 `pattern ceiling`
当分母；拿流式峰值算这类 kernel 的达成率，会凭空多出几十个百分点的「优化空间」。

写报告时把「分母来自哪个探针、什么 buffer 尺寸」和百分比写在同一行。

## 五、边界：什么时候**不要**测 / 找谁

- 契约还没写：先写契约（`op-contract`），否则测出来的数没有归属；没有生产布局的真实张量且算子对布局敏感：先解决入参，再测。
- 只改了一行常量却想「顺手测一下」：先看 `ledger.md`，`at_floor` 的直接停。
- 分工：热点定位走 `dcu-perf/dcu-perf-profiling` / `dcu-core/dcu-kernel-layer-microbench`，优化闭环走 `dcu-perf/dcu-op-optimization`，判通过/回退/四档决策走 `op-gate`。
