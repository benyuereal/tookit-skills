# Agent Skill 路线图 v5.1：先让数字可信，再谈能力

**日期**：2026-11-24
**业主定性**：交付物 = 一个 agent skill；算子优化只是反哺手段
**唯一验收**：拿 agent 没见过的新算子，不给提示，看它能否自主走完六步

> **v5.1 是「停止评审、去执行」版。** v6 只能是执行中发现的修复记录，不许是重新设计。
> **最大风险不是架构，是范围蔓延**：S0 七项做完就该去重测，别顺手把 S1 也做了（§9.1）。

---

## 第零章 · 已经做了什么（本机实测）

### 0.1 状态（截至 `6901d51`）

| 项 | 状态 | 实测 |
|---|---|---|
| YAML 缓存 | ✅ | `validate.py` 15.4s → **0.999s** |
| bound 词表统一 + fail-closed | ✅ | 39/53 张卡的词表对齐，未知值报错 |
| R0 端到端跑通 | 🟡 | 链路通，但**数字方向是反的** |
| `roofline.json` + allowed_actions | ✅ | 三项拦截实测生效 |
| `judge()` 唯一判定 + 4 字段 | ✅ | 四态 + `falsified` 三态映射 |
| S0 可信度四项（P0-2/3/4/6） | ✅ | 见 §0.5，全部实测生效 |
| 21 个 commit | ✅ | 已 push（`origin/main..HEAD` = 0） |

**当前分数**：深度 2.9 / 覆盖 6.5 / 质量 3.8（Q5 从 196 涨到 241，已超 200）

### 0.2 R0 的第一个结论，方向是反的

`096d1dc` 落盘的第一条记录：
```
tag=fused-3pass-triton  base(lightop)=194.68µs  cand=160.81µs  Δ=-17.4%
```
同一形状、同一对实现，`ledger.md`（2026-11-23，event 隔离，40 次中位）记的是：

| 实现 | R0（背靠背） | ledger（event 隔离） |
|---|---|---|
| fused-3pass-triton | 160.81 µs | 242.2 µs |
| lightop-fused | 194.68 µs | 227.0 µs |
| **比值** | **0.826（triton 快）** | **1.067（triton 慢）** |

摆动 29%，跨过 1.0。**符号相反。**

### 0.3 独立成立的只有两条（v5 的折算推导是循环论证）

v5 曾写「k_triton=1.506、k_lightop=1.166，折算后 242.2/227.0=1.067，完全一致」——
**这是同义反复**：乘数就是从这两个端点反解出来的，再用它验证端点必然成立。
两个未知、两个方程，解不唯一。

**真正独立成立的两条：**

**① 符号相反（稳健）**
```
背靠背     160.81/194.68 = 0.826  → triton 快
event 隔离 242.2 /227.0  = 1.067  → triton 慢
```

**② 不存在全局修正因子（这条才是真正的发现）**
若存在均匀乐观 k，则 `c·k=a` 且 `d·k=b` ⇒ `c/d` 必须 = `a/b` = 1.067。
实测 `c/d` = 0.826 ⇒ **不存在这样的 k**。等价地 `k₁/k₂ = 1.506/1.166 = 1.292`（差 29%）。

这顺手**否掉了 ab.py 自己的「偏乐观约 23%」**：均匀 1.23 会预测背靠背应是
196.9 / 184.6 µs，实测 160.8 / 194.7——两个方向都错。

**机制自洽**：triton 更短（160 vs 195µs）→ 固定开销占比更高 → 背靠背时被隐藏得更多 → 更乐观。

⇒ `claimable=False` 的理由比原来更强：不是「偏乐观要打标」，而是
**偏差逐实现、逐形状变化，不存在任何可移植的修正因子**。

### 0.4 MAD 小 ≠ 准确：`judge()` 只能判噪声，判不了偏差

**本机实测（这是 v5.1 自查发现的额外缺口）**：

| 口径 | Δ | 3×MAD | Δ/(3×MAD) |
|---|---|---|---|
| R0 背靠背 | 33.87 µs | 0.43 µs | **79 倍** |
| ledger event 隔离 | 15.2 µs | 15.0 µs | **1.01 倍** |

**荒谬点：符号错的口径，"显著性"高 78 倍；符号对的口径只勉强过线。**

原因：`judge()` 检测离散度（MAD），检测不了偏差（bias）。`back_to_back` 的 MAD
极小（它测的是理想流水），偏差却是系统性的、被极小的 MAD 完全掩盖。

⇒ **`measure_mode` 必须是独立前置门，不能折进显著性检验。**
这条要写进 `judge()` 的 docstring。

### 0.5 五个可信度缺口 → 四个已修

| # | 缺口 | 实测证据 | 状态 |
|---|---|---|---|
| K1 | 测量模式不进记录 | `grep -c measure_mode loop.py` = **0** | 🔄 agent 在做 |
| K2 | 基线角色不校验 | `grep -c role loop.py` = **0**；`ref-heavy` role=`correctness_only` | ✅ 已修，实测拒绝 |
| K3 | `remaining<0` 进决策 | `-0.0404` 放行；`floor.py` 三处写着「负值=判据错」 | ✅ 已修，实测拒绝 |
| K4 | 跨口径相减 | `t_floor` 端到端 vs `t_actual` 背靠背 | ✅ 已修，`remaining` −4.0%→**+10.9%** |
| K5 | dead_end 不参与过滤 | 2 条 `verdict: dead_end` 仍进推荐 | ✅ 已修，标 `DEAD_END` |

**另有两条永久失败的 `precondition_check`**（`slc` / `vmcnt` 在 PMC 列里各 0 命中）
——agent 在做 S1 时一并修。

---

## 第一章 · 核心判断

> 传动轴在，油也在加；**但仪表盘是坏的**。
> 现在每转一圈都产出一个数字，而这个数字的符号取决于用哪把尺子。

**先修仪表盘，再踩油门。** 四句话：

1. 所有结论必须带测量模式，且模式与结论强度**绑定**
2. 所有判据必须能跑通，跑不通的标 `unrunnable`，不许「默认永久失败」
3. ISA 层不是「没有」，是**被文档写死了**——反汇编实测是通的
4. **`judge()` 只判噪声、判不了偏差** ⇒ `measure_mode` 是独立前置门

---

## 第二章 · 目标架构

一句话：不是「一条严谨的流水线」，而是「**便宜可靠的评测 × 广覆盖的候选 × 严密的账本**」。

现在的设计严谨有余、吞吐不足——一轮只试一个候选。而公开工作（AVO 连续 7 天、
KernelEvolve 维护种群、Kevin 16 轨迹 × 8 轮）真正出成绩靠的是**大量评测**。

```
L6 反作弊    形状分离 · 隐藏压力分布 · 禁止行为 AST · 审计追踪(sha256)
L0 契约      semantics / shapes / dtype / 容差 / 压力分布
L5 记忆      recall(bound,signal) · calibrate(predicted vs realized) · 死路库
L1 度量      三层分母 + 双地板 + ABBA + 3×MAD + 噪声门 + **measure_mode**
L2 诊断      判据B(launch) → 判据A(roofline) → 归因(唯一主导) + PMC三态 + ISA
L3 决策      EV = headroom × recoverable_share × 折扣 × 前置通过
L4 执行      advance.py（唯一入口，只编排，继承 loop.py 六道门）
```

### 2.1 诊断信号阶梯

```
D1 哪条腿绑      AI vs ridge · 双地板              🟡 模型少（只 2 个特例）
D2 硬件在等什么  PMC: stall/带宽/发射率            ✅ 但 3 个计数器坏
D3 资源为什么不够 VGPR/LDS/spill                   ✅
D4 为什么 stall  指令流: waitcnt/向量宽度/展开       ✅ 实测可用，但被文档误判
D6 别人怎么做    同算子厂商 kernel 的 ISA 对照      ✅ 102 kernel 可对照
```

**现在断在 D2 → D4**：PMC 说「有 stall」，只有 ISA 说「为什么 stall」。
**静态先于动态**：ISA 编译一次就有，PMC 要专门采集且开销大。

---

## 第三章 · P0 可信度层（S0，1 天）

### P0-1 · `measure_mode` 进 record，且决定结论强度

```python
MODES = {
    "event_isolated": {"claimable": True,  "note": "event 隔离 + 预分配 + 40 次中位"},
    "end_to_end":     {"claimable": True,  "note": "每次 empty_like，含分配"},
    "back_to_back":   {"claimable": False, "note": "无 per-iter sync，kernel 间可重叠"},
}
```

三条硬规则：
1. `record.measure_mode` **必填**；ab.py 的值原样搬运，不许丢
2. `back_to_back` **永远 claimable=False**：可筛方向，不得进入「已验证」
3. base 与 cand **必须同模式**；不一致直接 `INCONCLUSIVE`

**依据（v5.1 改写）**：跨模式比较不仅翻转符号，而且乐观度**逐实现变化**
（`k₁/k₂ = 1.292`），所以**不存在可移植的修正因子**——只有「同模式 + 不可 claim」一条路。

**附加要求**：`judge()` 的 docstring 必须写明能力边界——只能判噪声，判不了偏差。

### P0-2 · 基线角色校验 ✅

`role=correctness_only` 的变体不得作性能对比对象。

### P0-3 · remaining 符号门 + 同口径校验 ✅

```python
if r is not None and r < 0:
    probs.append("remaining<0 —— floor.py 已定义为「判据本身错了」，先查分母")
if roof.get("t_actual_mode") != roof.get("t_floor_mode"):
    probs.append("t_actual 与 t_floor 口径不同，相减无意义")
```

### P0-4 · dead_end 过滤 ✅

`verdict: dead_end` 的两条不进推荐，只进 recall。

### P0-5 · 修掉永久失败的 precondition_check

| surface | 现在 | 应改为 |
|---|---|---|
| `slc-policy` | `grep -c slc` 查 PMC（0 命中） | `dccobjdump ... && grep -c " slc" X.ISA` |
| `prefetch-depth-isa` | `grep vmcnt` 查 PMC（0 命中） | `grep -oE "s_waitcnt (v\|lgkm)cnt\([0-9]+\)" X.ISA` |

**纪律**：跑不通的标 `unrunnable` 并说明缺什么输入。
「永久 False」与「默认通过」是同一类错误，只是方向相反。

### P0-6 · 轮次与重跑分离 ✅

```python
round_key = (card, shape, direction, tag, impl_hash)   # ← 必须加 impl_hash
round = 该 key 首次出现；rerun = 同 key 且同 impl_hash
```

**已实测**：R0 的 5 条记录是 **1 轮实验 + 4 次重跑**。

**为什么必须加 `impl_hash`**：`ledger.md` 里 triton 有 172.5µs（无 `residual_out`）
和 242.2µs（补上后）两个版本——同 tag、同 shape、同称「event 隔离」，**相差 40%**。
不加 hash，修好实现后的重测会被误判成 rerun 而不计入轮次。

### P0-7 · 两个测试失败 ✅

- `test_adapter`：reference 改返回 2 份后测试没跟着改
- `test_allowed_actions`：**按 fail-closed 哲学答案是「拦」**，不留成开放问题

---

## 第四章 · P1 ISA 层复活（S1，1 天）

### 4.1 先修文档（0.5h，最划算的一笔）

| 文档 | 写的 | 实际 |
|---|---|---|
| `isa-census.md` | `library_gfx938/` | **不存在**，实际 `library_bw_nmz/` |
| `isa-census.md` | 「厂商已带好反汇编 461MB *.ISA」 | **全机 `find *.ISA` 零结果** |
| `isa-census.md` | `--inputs=<.hsaco>` | `.hsaco` 报 not recognized，必须 `.co` |
| `tool-status.md` | `dccobjdump: degraded`（`--architecture` 不支持） | **用错参数，`--target=gfx938` 完全可用** |

**为什么最划算**：`tool-status.md` 规定「判据里只能写 available 的工具」。
agent 读完判定反汇编不可用 → 永远只走 PMC，而 PMC 的三个关键计数器采不到。

### 4.2 汇编里的 6 个杠杆，13 条 surface 一条都表达不了

**⚠️ v5.1 必读：静态计数 ≠ 动态权重。**
GEMM 主循环的 load 每个 K-step 都发生，epilogue 的 store 每 tile 才一次——
静态比例 12:32 完全可能对应动态占比 <5%。

**正确顺序：ISA 生成假设 → PMC 定级（动态占比）→ A/B 裁决。**
四条 surface 的 `evidence` **一律只是上界**；`expected_gain` 在被 PMC 定级前
仅作排序用，不得当量级引用。

### 4.3 补 4 条活的 ISA surface

| id | statement | class |
|---|---|---|
| `vmem-vector-width` | 读/写向量宽度是否用满（`buffer_load/store dwordxN`） | isa-policy |
| `mls-path` | LDS 读走矩阵通路还是标量 `ds_read_b32` | isa-policy |
| `lds-read-form` | `ds_read_b128` 与 `ds_read_b32` 比例 | isa-policy |
| `epilogue-expansion` | 收尾段是否完全展开（热循环内无 `s_cbranch`） | pipeline |

每条 evidence 必须带「静态证据，动态占比未知」这句。

### 4.4 attribution 允许 ISA 证据

现在 `attribution.md` 规定 evidence「必须是 PMC 数值」。改成**允许二选一**
（`kind: pmc` 或 `kind: isa`），否则这四条 surface 永远无法产出「主导瓶颈」结论。

---

## 第五章 · P2 诊断 → 决策的断链（S2，0.5 天）

`attribution.md` 的五个 dominant（`hbm` / `l2-miss` / `lds-conflict` / `vgpr-spill` / `issue-stall`），
在 `op-optimize/` 和 `scripts/` 全目录 **grep 零命中**。

```
PMC → attribution.dominant = "l2-miss"    ← 诊断链终点
        ↓（无映射，这里断了）
surfaces.yaml 13 条 id                    ← 决策链起点
```

agent 得出「主导是 L2 miss」之后，**没有任何机检路径**把它翻译成「该试哪条 surface」。
它只能自由发挥——而方向预算、refuted 拒跑、动作类约束**全挂在 surface id 上**，
这一步一自由发挥，后面所有约束都绕过。

补 `references/dominant-to-surface.yaml`：
```yaml
hbm:          [layout-pack, fusion-boundary, gather-path]
l2-miss:      [layout-pack, gather-path]
lds-conflict: [layout-pack, lds-read-form]
vgpr-spill:   [occupancy, epilogue-expansion]
issue-stall:  [cross-lane-fold, barrier-width, prefetch-depth, vmem-vector-width]
```

**配套**：`unknown` 拆三态（`csv_missing` / `counter_incomplete` / `counter_suspect`）。
同卡连续 2 轮 `unknown` → `blocked`，要求**换采集方式**，不许再采一次。

---

## 第六章 · P3 发动机（S3，1 天）

### 6.1 澄清：不建 `runner.py`，但要有唯一发动机

`loop.py run` 已有**六道门**：candidates 登记 → bottleneck 归因 → done_when 外部裁决
→ 方向预算 → refuted 拒跑 → allowed_actions 天花板约束。
**新模块自己跑实验，这六道全绕过。那不是重构，是安全性倒退。**

### 6.2 `advance.py` 的唯一职责

```
每轮：plan → gate(precondition) → run(调 loop.py) → judge → record
      → replan → stop → calibrate
```

**唯一验收**：advance.py 跑完一轮，中间没人敲任何命令，而 ledger 多一条记录、stage 前进。

### 6.3 replan 规则（防 EV 塌缩）

v5 原写法「top3 变了→继续；top1 未变→stalled」有两个洞：
**洞 1** EV 塌缩（top1 从 0.9 掉到 0.3 仍排第一 → 判「继续」）；
**洞 2** 分不清健康轮换与 tie-break 抖动。

```python
def replan(prev, new, ledger):
    if not new:                                return STOP("无可挖方向")
    if new[0].ev < 0.30:                       return STOP("EV 塌缩到 0.3 以下")
    if new[0].ev < 0.5 * prev[0].ev:           return STOP("EV 掉了一半以上")
    if prev[0].id == new[0].id:                return CONTINUE_SAME
    if new[0].ev < 1.5 * new[1].ev:            return STALLED("没拉开差距，可能是抖动")
    cited = ledger_has_citation(new.rejected, prev[0])
    return CONTINUE_SWITCH(cited)
```

### 6.4 `calibrate`：排进 S3 判据，不许悬空

它是最早发现 EV **系统性高估**的机制（每条 surface 预测 15%、实际兑现 3% ⇒ EV 排序整体失真）。

```
surface                n   predicted_med   realized_med   ratio
fusion-boundary        3   15%             4.2%           0.28  ← EV 该打折
```

**S3 完成判据加一条**：累计 ≥5 轮后必须能输出这张表。

---

## 第七章 · P4 roofline 求值器（S4，1.5 天）

### 7.1 粒度就错了：`bound` 是 shape 的函数

`gemm-bf16` 卡片自己写着：「拐点随 N 移动：K=N=4096 时 M≈256，N=14336 时 M≈512」。
`rmsnorm-fused` 有四个形状档，`decode-m1` 和 `prefill-8k` 的 launch 占比完全不同。

⇒ `load_allowed_actions(card)` 的签名是错的，必须 `load_allowed_actions(card, shape_id)`。

### 7.2 模板驱动，不是每算子一个模型

10 个访问模式模板覆盖 53 张卡：

| 模板 | 覆盖 | FLOPs | Bytes |
|---|---|---|---|
| `elementwise-unary` | rope · quant · gelu | ~N | 2·N·d |
| `elementwise-binary` | silu×mul · add · combine | ~N | 3·N·d |
| `reduce-lastdim` | rmsnorm · layernorm · qk-norm | ~3N | (2~3)·N·d |
| `matmul` | gemm-* · qkv · o_proj | 2MNK | (MK+KN+MN)·d |
| `attention-prefill` | flash-attn · mha | 4S²hd | KV 全读 |
| `attention-decode` | paged · mla-* | 4Shd | KV 全读 |
| `gather-scatter` | moe-permute · embedding | ~0 | moved·d |
| `topk-reduce` | topk-gating · sampling · mqa | ~N | N·d + k·d |
| `matvec` | lm-head | 2NK | (N+1)·K·d |

`reduce-lastdim` 写一次，四个算子自动有 roofline。

### 7.3 special-case 必须声明理由

```yaml
model_override: models/sparse_mla.py
why_not_generic: "topk 稀疏导致字节依赖数据分布，模板只能给上界"
```
`validate.py` 加硬判据：`model_override` 缺 `why_not_generic` 直接 error。

### 7.4 存放：按 shape 索引

```json
{"schema_version":"1.0","card":"rmsnorm-fused",
 "model":{"kind":"template","name":"reduce-lastdim"},
 "by_shape":{"decode-m1":{...},"prefill-8k":{"bound":"memory","remaining":0.31,
   "t_actual_mode":"event_isolated","t_floor_mode":"event_isolated"}}}
```

### 7.5 S4 完成判据：**必须加 golden 校验**（v5.1 新增）

v5 原判据「同一 gemm 卡两个 shape 的 bound 翻转」**只测方向敏感性，不测绝对正确性**。

**洞**：一个 `bytes` 公式系统性差 2× 的模板，bound 翻转照样正确，
但会毒化所有 `remaining` 和 EV，而且错误安静地住在每张卡的 `roofline.json` 里。

**补两条 golden 判据**：

```
G1 绝对正确性（每个模板 ≥1 个算子交叉验证）
   公式算出的 bytes ÷ 实测时间 推出的带宽利用率 必须 ∈ [0.6, 1.0]
   依据：lightop 实测 1182 GB/s ÷ 1680 = 0.70 ∈ 区间 ✅
        若模板 bytes 差 2× → 利用率 1.40 → 超界 → 报错

G2 方向敏感性（v5 原有，保留）
   gemm-bf16：M=128 应判 memory / M=8192 应判 compute
```

G1 落不进区间有两个可能，**必须区分**：模板错（公式）或 分母错（`bw-ladder` 口径）。
不许二选一硬判。

---

## 第八章 · P5 反作弊（S5，0.5 天，**必须在盲测之前**）

公开审计的真实案例：KDA 分布感知、MinGRU 覆盖攻击。
共同点：**agent 没变强，它学会了骗评测**，而且是「高级工作 + 一处作弊」。

| # | 检查 | 实现 | 分工 |
|---|---|---|---|
| 1 | 形状分离 | `correctness_shapes ∩ bench_shapes = ∅` | 防「按 shape 分派」 |
| 2 | 扰动分布 | ×3 / ×0.01 / 取负 | 防**统计量**分派（AST 抓不到） |
| 3 | 禁止行为 AST | 候选 kernel 里出现「对字面常数/具体 shape 的分支」→ 判 dead | 只抓字面量，**别对 AST 产生假信心** |
| 4 | **输入隔离双测** | 两个不同输入各跑一次，分别与 reference 对比 | 见下 |

**v5.1 修正：第 4 条 v5 写反了。**
v5 写「连调两次输出逐位相同（防缓存）」——**这是错的**：
缓存作弊恰恰会让两次输出相同（第二次直接返回第一次结果）。

```python
out_A = run(kernel, input_A);  assert allclose(out_A, ref(input_A))
out_B = run(kernel, input_B);  assert allclose(out_B, ref(input_B))
# ← 关键：缓存 kernel 第二次返回 A 的结果，对不上 ref(B)
if not kernel_uses_atomics(source):        # 排除 atomics 的合法非确定性
    assert bitexact(run(kernel, input_A), out_A)
```
另需：kernel 文件计时前后 **sha256 不变**（防中途换实现）。

---

## 第九章 · 执行顺序与时间盒

| 阶段 | 内容 | 时间 | 完成判据（可机检） |
|---|---|---|---|
| **S0** | P0 可信度 7 项 | 1 天 | **★ 可证伪重测**：event 隔离重测两实现各 40 次，比值 ∈ `1.067 ± 3×MAD_base`，且 pin `impl_hash` |
| S1 | ISA 文档 + 4 条 surface + dead_end + 助记符 | 1 天 | 反汇编成功一次并贴 exit code；4 条前置全 rc=0 |
| S2 | dominant→surface 映射 + unknown 三态 | 0.5 天 | 五个 dominant 全映射到 ≥1 条 surface，有测试 |
| S3 | `advance.py` | 1 天 | ①无人干预跑完一轮 ②≥5 轮能输出 predicted vs realized |
| S4 | roofline 求值器 | 1.5 天 | ①**golden**：利用率 ∈[0.6,1.0] ②方向敏感性 |
| S5 | 反作弊四件套 | 0.5 天 | ①作弊 kernel（`if n<=2048`）被抓 ②缓存 kernel 被抓 |
| S6 | R1/R2 盲测 | 2 天 | 六步产物齐全 + 第二轮 rejected 引用第一轮 verdict |

### 9.1 范围蔓延是本阶段最大风险

> **S0 七项做完就该去重测，别顺手把 S1 也做了。**

理由：S0 的正确性**只有重测能验**；如果先做 S1，重测时已经混入 ISA 方向的变量，
说不清是谁的影响。

---

## 第十章 · 机器验收（每条实跑过）

```bash
python3 -I Plugins/ops/op-atlas/validate.py     # 0 错误 / 17 警告，0.999s
python3 -m pytest Plugins/ops -q -p no:cacheprovider
cd /tmp && dccobjdump --inputs=extop_gfx938.co --target=gfx938 --recompile
/opt/dtk/llvm/bin/llvm-readelf --notes /opt/dtk/lib/hipblaslt/library_bw_nmz/Kernels.so-000-gfx938.hsaco
hipcc --save-temps --offload-arch=gfx938 --genco k.hip -o k.hsaco
python3 -I Plugins/ops/skills/op-optimize/loop.py -C Plugins/ops/skills/op-rmsnorm-fused status
python3 -I Plugins/ops/skills/op-roofline/derive.py --list   # 只有 2 个模型
```

---

## 第十一章 · 不做什么

1. 不新建 `ledger.py` / `runner.py`，但要建 `advance.py`（唯一编排入口）
2. 不加新门禁（E39–E43 已够）
3. 不手工优化单个算子
4. 不在 schema 里写没有实测支撑的因子
5. **不在没有 `measure_mode` 的情况下宣布任何加速**
6. **不在没有形状分离的情况下跑盲测**
7. 不留「永久失败」的 `precondition_check`

---

## 第十二章 · v5.1 修正清单

### 12.1 五条实质缺口（第五轮评审，全部接受）

| # | 指正 | 修正位置 |
|---|---|---|
| 1 | **S0 判据是循环论证**（乘数由端点反解再验证端点） | §0.3 重写为两条独立事实；§9 S0 判据改为可证伪重测 |
| 2 | S4 只测方向不测绝对正确性 | 新增 §7.5 golden 校验 |
| 3 | P5「双调用一致性」语义写反 | §八 改为输入隔离双测 |
| 4 | 静态计数 ≠ 动态权重 | §4.2 加警告，四条 evidence 降级为上界 |
| 5 | `calibrate` 无位置 | 新增 §6.4，S3 判据加一条 |

### 12.2 四条小问题

| # | 指正 | 修正 |
|---|---|---|
| 6 | replan 有 EV 塌缩盲区 | §6.3 重写 |
| 7 | round key 吞合法重试 | 加 `impl_hash`（实证：同 tag 两版本差 40%） |
| 8 | decision 词表在漂移 | schema 枚举 + D4 映射写死 |
| 9 | P0-7 别留成开放问题 | 按 fail-closed 答「拦」 |

### 12.3 自查发现的额外缺口（原评审未提）

**`judge()` 的能力边界 = 只能判噪声，不能判偏差**（§0.4）。
本机实测：符号错的口径 Δ/(3×MAD) = **79 倍**，符号对的只有 **1.01 倍**。

### 12.4 版本弧线

| 轮 | 状态 |
|---|---|
| v1 | 图纸有虚线 |
| v2 | 虚线画实，曲轴量错 |
| v3 | 装了不该装的零件（基于未核实论断） |
| v4 | 拆误装件，验证顺序倒置 |
| v5 | 跑通了，发现仪表是坏的 |
| **v5.1** | 修正 v5 自己的循环论证；**停止评审、去执行** |

### 12.5 下一版唯一值得看的指标

> **重测后的 1.067 落不落进 3×MAD。**
> 落进去 → 五轮弧线闭环。
> 落不进去 → `ledger.md` 也要进复核名单，那会是第六个、也是更深的可信度缺口。

---

## 第十三章 · 一句话

> 现在每转一圈都产出一个数字，而这个数字的符号取决于用哪把尺子。
> 同一对实现：event 隔离下慢 6.7%，背靠背下快 17.4%。
> 而且 ab.py 自己写的「偏乐观 23%」也被数据否掉了——乐观度逐实现变化 1.29×，
> **不存在可移植的修正因子**。
>
> **R0 的价值恰恰是它「打脸」了测量而不是优化**：
> 一个流程的第一次实机产出是「我的测量不可信」，这恰恰说明流程在自我生效。
