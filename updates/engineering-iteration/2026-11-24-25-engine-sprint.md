# 工程迭代记录 · 2026-11-24 → 11-25

> 这一轮的主题：**让「优化一个算子」这件事从「跑一次积攒经验」
> 变成「可复现、可中断、能自己写实现」的闭环。**
>
> 20 次提交，从 `5c98bf0`（C4 tile 模型）到 `54e5d08`（冷启动）。
> 下面按「当时以为的问题 → 实际的问题 → 怎么修的」来记，
> 因为**大部分时间花在发现真正的问题上，不是写代码上**。

---

## 一、v6 的六个组件（C1–C6）

| 组件 | 内容 | 落地 |
|---|---|---|
| C1 | 17 条 surface 补 `transform` 字段 | `check_transform.py` 覆盖 17/17 |
| C2 | `variant_forge` 参数化变体生成 | 只做 Triton；HIP 侧标 `forge: manual` |
| C3 | `design_space` 枚举设计空间 | 486 组合，每个拒绝带字节数理由 |
| C4 | tile 模型（attention-decode / reduce-lastdim） | 11 点回测误差 ≤0.005 |
| C5 | `kernel_scaffold` 生成可编译骨架 | 参数必须进编译产物 metadata |
| C6 | 真跑：多轮编排 + 自研实现 | 见下 |

**C5 的关键发现**（3 个真错）：
1. `const int threads = N` 在 host 侧**不会**进 kernel metadata；
   只有 `__launch_bounds__(THREADS)` 会。三组 128/256/512 全编成 vgpr=12/threads=256。
2. **静态 LDS 可读，动态 LDS 不可读**（`extern __shared__` 不进 metadata）。
3. 参数必须走编译期宏。`VEC` 若作为运行期变量，`xv[]/rv[]` 整块溢出到私有内存
   —— `private` 0→528B，0.165→0.538ms，**慢 3.3 倍且不报错**。

---

## 二、C6 的三次迭代

### 第一版：多轮循环跑通，但没赢

```
15:20 fused-3pass-triton   REJECTED_NO_GAIN Δ+24.2% event_isolated falsified=True
15:21 fused-3pass-tilelang REJECTED_NO_GAIN Δ+5.4%  falsified=True
15:24 fused-3pass-tilelang INCONCLUSIVE   Δ-2.6%   falsified=None
```

流程对（口径可 claim、证伪正确、自动停），但**所有自研实现都比厂商慢**。

### 第二版：手写 HIP，第一次超过厂商

**做法**：把 `(x+r)` 留在寄存器（16 float/线程），LDS 只放 `WARPS*4` 字节归约表。

| 实现 | 耗时 | 带宽 |
|---|---|---|
| lightop（厂商） | 202.2 µs | 1327 GB/s |
| `rms-v2-reg`（自研） | **184.6 µs** | **1454 GB/s** |

**−8.7%，3×MAD 显著。** 中途踩的坑：`VEC` 运行期传入 → 溢出 → 慢 3.3 倍。
这个坑后来写进了 `kernel_scaffold` 的注释和 `forge_step` 的体检判据。

### 第三版：让 forge 自己造，赢家由 judge 判

一开始 `advance` 是「从候选队列挑一个已有的」。
改成「队列空 → 自动生成 → 编译 → 体检 → 过六道门 → 测」。

**sweep 4 组同批实测**（共用一次 base 0.2019ms）：

| 配置 | 耗时 |
|---|---|
| t64-VEC8 | 0.1805 ms |
| t256-VEC2 | 0.1810 ms |
| t128-VEC4 | 0.1827 ms |
| t512-VEC1 | 0.1835 ms |

**极差 1.24%，全在噪声内。** 这推翻了我上一轮写进代码的结论
（「256/VEC2 比 512/VEC1 快 24%」——那 24% 来自一次孤立单跑）。

**教训**：单点对比永远分不清两个同时变化的变量。sweep 的价值不是找最优，
是**分清哪个变量真在起作用**。

---

## 三、精度门：从「恒 False」到能出结论

### 症状

全仓 **0 条** `acc_ok=true` 记录。

### 排查（三处断链，每处都是「接了但没接上」）

1. `bench/acc_probe.py` **不存在** → `loop.py` 的 `if (card/acc_adapter).exists()`
   静默跳过整块精度检查
2. 补上之后仍不生效：`decide()` 拿到 `maxdiff=None` 就提前返回，
   **根本没走到 acc 判定**
3. 再补之后 acc 字典仍为空：`acc_probe` 收到的是**候选 id**（`verify-acc`），
   而记录查的是**变体名**（`rms-v2-reg`）

### 容差判据也错了

bf16 只有 7 位尾数，ULP 随量级变：

| 值域 | ULP |
|---|---|
| [0.5,1) | 2^-8 = 0.00390625 |
| [1,2) | 2^-7 = 0.0078125 |
| [2,4) | 2^-6 = 0.015625 |

实测 `max_abs=1.5625e-2 = 2^-6`，在 [2,4) 区间**正好 1 ULP**（完美舍入）。
而旧的 `max_abs=2.0e-3` 只够 0.13 ULP。

**铁证：四个变体全 FAIL，含厂商 lightop。** 判据错，不是实现错。

改成 `mode: ulp_bound, max_ulp: 2` 后：四个变体全 PASS，
lightop 最差 2.00 ULP，其余 1.00。

### 结果

`acc_ok=true` 从 0 条 → 全项目第一批可 claim 记录。

---

## 四、语言选型：工具存在，但没人调用

### 发现

`scripts/lang_select.py` 里写着：

```
rmsnorm prefill-8k：lightop(HIP) 197.0µs / tilelang 221.0 / triton 242.0
triton 比厂商慢 23%
```

**判据和数据都齐 —— 但 `grep lang_select` 全仓只有文档引用，零调用方。**

外部证据（opbench 三臂对照）：裸 CC / 本 skill / SkillHub
**三臂都没写过一行 HIP**，而本仓已有实测更快的现成实现。

### 第二层问题：判据不可执行

```yaml
hip:    use_when: ['瓶颈已确认在 LDS bank / 寄存器压力 / MFMA 排布']
triton: avoid_when: ['纯 elementwise，Triton 已足够']
```

「纯 elementwise」怎么判？没人能执行 → 等于没有判据。

### 改法：从 roofline 数字推导

6 条规则，输入是四个数（bound / utilization / remaining / launch_fraction）：

| 规则 | 判据 | 语言 |
|---|---|---|
| launch-bound | `launch/t_actual > 0.30` | 不换语言，做融合 |
| compute-near-ceil | `bound=compute & util>0.85` | 库实现够用 |
| memory-large-gap | `bound=memory & util<0.90 & remaining>0.20` | **HIP 优先** |
| memory-near-ceil | `bound=memory & util>=0.90` | triton |
| reduce-dense | `is_reduction & util<0.85` | HIP |

在 `advance.py` 的 `build_plan` **之前**执行。
实测两个场景给出相反建议（0.831→HIP / 1.002→triton）。

---

## 五、第二张卡：fused-add-rmsnorm

**结果：140.0 µs vs lightop 255.2 µs = 1.82x，精度也更好（0.50 vs 1.50 ULP）。**

### 三个真 bug

1. **适配器参数顺序错位**：模板写死 `run(x, w, r)`，本卡输入顺序是
   `(x, residual, weight)` → `w` 收到 residual，grid 算成 2097152 → **VMFault**
2. **输出份数多写 25% 流量**：生成器默认写两份输出（含 `residual_out`），
   本卡只有一份 —— 拿它比少写一份的实现就是「比两件不同的活」
3. **`ab.py` 没有 `--measure-mode`**：本卡是 96 行旧版，只测 `back_to_back`
   （不可 claim）—— D1 七份分叉的直接后果

### DS 从 32 到 92

成绩是真的、精度是真的、口径是 claim 级的 —— **只是记分器的输入格式没对上**：

| 项 | 问题 | 修后 |
|---|---|---|
| U | `ratio.kind` 写的是 `speedup`，DS 只认 `utilization` | 0.98 |
| S | 只有单形状 | 1.0（三形状 geomean 0.4792） |
| R | ledger 有内容但无 `### 轮 N` 标题 | 1.0（补 5 轮） |

**这是「两处记账」的第 5 次复现。**

---

## 六、第三张卡：norm-act-swiglu（用户出题）

### 题目

```python
x[M,4096] + residual[M,4096] → h              # add
h = w * h / sqrt(mean(h²) + eps)              # RMSNorm
g, u = h.chunk(2, dim=-1)                     # gate/up
out[M,2048] = silu(g) * u                     # silu-and-mul
```

基线两步法 0.29904 ms。

### 关键洞察：砍掉中间张量

| | 流量 | 明细 |
|---|---|---|
| 两步法 | 369.1 MB | ①读x,r 写x,r (4H) ②读h 写out (1.5H) |
| 三合一 | **167.8 MB** | 读x+读r+读w+写out (2.5H) |

**降到 45.5%。** 融合成立的根据：`gate` 与 `up` 来自同一个 `h`，
归一化因子对两者相同 —— 一个 kernel 内先归约、再同时算两半。

### 结果

| BLOCK_ROWS | 耗时 | 地板占比 | vgpr |
|---|---|---|---|
| 1 | 148.0 µs | 66% | 39 |
| **2** | **140.5 µs** | **69%** | 57 |
| 4 | 153.3 µs | 63% | 84 |

**2 行最优。4 行因寄存器压力（vgpr 84）退化** —— 多行摊薄 barrier 的收益，
在某个点被寄存器成本吃掉。

最终：**2.03x，1.00 ULP，DS 89**。holdout（M=13312）达 73% 地板，比 public 更高。

### 顺带做了冷启动

用户的题暴露了一个真缺口：**新算子没有入口**。
`derive.py` 只认 sparse-mla；`grep "建卡" scripts/*.py` = 空；
卡片 schema 20+ 字段且 `additionalProperties: false` → 手写必撞 E6。

写了 `op_bootstrap.py`。它拒了这个题（含 `@` 但 pattern=elementwise），
提示「复合算子要拆开」—— 这是对的，GEMM 由 FLOPs 约束、elementwise 由字节约束。

---

## 七、外部语料吸纳（mHC / mqa_opt）

### mHC（GLM-5.3 的 hc_pre/hc_post）

实跑验证 8 条声明，7 条复验：

| 声明 | 实测 |
|---|---|
| 整体 m=16384 = 1.89x | **1.89x**（精确吻合） |
| proj4 的 +17% | 507.3 → 593.1 GB/s，且 `max|dP|=0.00e+00` |
| weight 复用证伪 | MB=2 342 vs MB=1 584 GB/s |
| NOB 反证 | 去掉 weight 流只到 662 GB/s（+13%） |

### mqa_opt：一条漂亮的证伪

PMC 实测 `SQ_LDS_BANK_CONFLICT / SQ_INSTS_LDS = 9.6`，分析性根因清楚
（256B 行跨距是 0 mod 32 banks）。XOR swizzle 也确实解决了它。

**但实测只快 0.36%。**

**教训：PMC 计数器非零 ≠ 它是瓶颈。** 修掉一个真实存在的问题，
可能一点收益都没有。

### 消融归因的第一次真实使用

`ablate.py` 收到 `abl_noepi`/`abl_noshfl`/`abl_neither` 时算出全负 share，
只报 `unexplained` 没拦住。查源码 584 行：

```cpp
// ablations (not numerically valid; timing-only diagnostics)
```

**这些是计时探针，故意算错以测某段耗时**，根本不是消融项。
加了 `derived_from` / `numerically_valid` 强制声明，三类分流。

修复后：只剩 `no_bp` 归因（share 0.039），status 如实报
「**96% 的收益说不清**」。比负数表诚实。

---

## 八、三条待裁决的债（已解决其一）

| 债 | 内容 | 状态 |
|---|---|---|
| D3 | 6 张卡缺可补证据 → 降 draft | ✅ 已执行（measured 27→22，C1 1.0→0.815） |
| D5 | `mlp-gelu-geglu` 语义矛盾 | ⏳ 待拆 |
| D6 | 前置闸门强制会 100% 停跑 | ⏳ 待分级 |

D3 的代价是**覆盖分下降**（6.2→6.0），但裁决理由是「measured 的本义是有数且可追，
追不了就不是」。**诚实优先于分数。**

---

## 九、如果重来，我会先做什么

1. **先建精度门**，再谈性能。没有它，所有「快了多少」都缺一半证据。
2. **先接调用链**，再写工具。`lang_select` 存在了三周没人调用，
   写它的时间等于白花。
3. **先做多形状**，再抠单点。单形状的卡 DS 上限很低，
   而且很容易在单个形状上过拟合。
4. **先跑 sweep**，再写结论。「256 比 512 快 24%」那条错误结论，
   如果当时做了 sweep，五分钟就能推翻。
