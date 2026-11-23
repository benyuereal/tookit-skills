# Case 001：SwiGLU 融合 vs 分拆 —— 场景驱动的完整优化链

> 「场景 → 优先攻击面 → 实测 → 结论」整条链的一次完整走通，供照抄格式。
> 每一步都可在本机复跑：`python3 bench.py`

## 0. 输入：算子 + 场景

| 项 | 值 |
|---|---|
| 算子 | `mlp_swiglu`（MLP 的 gate/up + 激活 + 乘） |
| 形状 | M=32768, H=4096（输入 `[M, 2H]` = 8192 宽），bf16 |
| 场景归属 | `bandwidth-bound` |

判定依据（按 `scenarios.yaml` 的 `detect`）：字节数确定、无复用结构可调、
AI 远低于机器平衡点 ⇒ 访存受限。

## 1. 按场景取优先攻击面

```bash
python3 loop.py -C <卡> enumerate --scenario bandwidth-bound
```

```
序   surface          为什么在这个场景优先
1   fusion-boundary  这一档唯一的真收益是省 HBM 往返；融合是主要手段
2   layout-pack      每多一次布局转换就多一趟读写
3   gather-path      非合并访问到不了流式峰值
已知死路：
  ✗ 优化计算部分：这段没在等算力
  ✗ 用标称带宽做分母：会把已有空间算成已到顶
```

→ 第 1 优先 **fusion-boundary**：融合是否真的省了往返。

## 2. 定位到库.函数（L1）

| 字段 | 值 |
|---|---|
| `import_path` | `lightop.FuseBiasSwiGLU` |
| `signature` | `FuseBiasSwiGLU()(input, bias) -> Tensor`；input `[M, 2H]`，输出 `[M, H]` |
| `smoke_test` | `python3 -c "import lightop; print(hasattr(lightop,'FuseBiasSwiGLU'))"` |

**同族备选（L2 选型）**：

| 实现 | 为什么选/不选 |
|---|---|
| `lightop.FuseBiasSwiGLU` | **选**：bias + silu + 乘三者融进一个 kernel |
| `aiter.fuse_silu_and_mul_ep` | 不含 bias；带 bias 的 MLP 要先 `x+b`，多一趟往返 |
| `aiter.fuse_silu_mul_fp8_quant_ep` | 融合量化；本例不需要量化 |
| `lightop.GNSilu` | 同族但语义不同（不含 bias 加法） |

## 3. 实测

```bash
python3 Plugins/ops/skills/op-optimize/cases/001-swiglu-fused-vs-split/bench.py
```

本机（BW1100 / gfx938）实测输出：

```
== 1. 本机带宽峰值标定（buffer 1024 MB）==
  流式只读 :   1728 GB/s
  copy R+W :   1125 GB/s

== 2. 融合 vs 分拆（M=32768 H=4096 bf16）==
  语义 cosine = 0.9999947
  融合 : 1.5696 ms  MAD 0.0294
  分拆 : 3.5242 ms  MAD 0.0466
  ratio: 0.445x (融合/分拆)        ⇒ 融合快 2.25x
  3×MAD = 0.1397 ⇒ 显著

== 3. 达成率 ==
  融合字节数 : 805.3 MB（读 x + 读 b + 写 out）
  实测带宽   : 513 GB/s
  vs 只读峰值 1728 : 30%
  vs copy 峰值 1125 : 46%
```

**结论：融合有明确收益（2.25x），但离峰值还很远（39-46%，取决于分母）。**

### caveat 达成率的分母会浮动，必须同时报区间

同一份 kernel 时间（1.5163-1.5184 ms，波动 0.14%，极稳），
达成率却在两次测量中给出 **46%** 与 **39%**。差异全部来自分母：

| 测量 | copy 峰值 | 达成率 |
|---|---|---|
| 首次（19:0x） | 1125 GB/s | 46% |
| 复测（19:5x） | 1373 GB/s | 39% |

两次的卡状态都是 HCU% 0.0（空闲）。**kernel 时间稳定、分母不稳定**——
说明带宽标定对运行环境敏感（可能的因素：其他进程残留的 L2 状态、
显存碎片、驱动态）。

**纪律**：报达成率时必须同时给出「所用分母的实测值」与「本次测量的时间戳」，
否则同一个 kernel 的达成率可以在 39%-46% 之间随便挑，这本身就是一种分母漂移。
更稳的做法是报「时间」为主、「达成率」为辅——时间是直接测的，达成率是算的。

## 4. 分母怎么选（这步最容易错）

SwiGLU 的访存比是**读 2 : 写 1**（读 x 的 `[M,2H]`、读小 bias、写 `[M,H]`）。
所以：

| 口径 | 本机实测 | 适用 |
|---|---|---|
| 流式只读 | 1728 GB/s | 纯读算子 |
| copy（读+写） | 1125 GB/s | **读写混合（本例最接近）** |
| 标称 | 1100 GB/s | 型号名，**不是能力** |

用 copy 口径 ⇒ **46%**。用标称 1100 会得到 47%——看起来差不多，但那是巧合；
真正危险的是纯读算子用标称会把 60% 算成 94%，直接误判到顶。

### caveat 踩过的坑：小 buffer 测带宽会虚高

最初用 25MB buffer 测出 **4571 GB/s**，比只读峰值还高一倍多。
原因是小 buffer 反复读会部分命中 L2。**所有带宽数字必须用 ≥1GB buffer 实测**，
`bench.py` 里已固定为 1024MB。

## 5. 下一个攻击面（不是「已优化完」）

46% 说明访存模式离流式峰值还很远。按场景表第 2/3 优先：

| 候选 | 方向 | 预期机制 |
|---|---|---|
| `vectorize-width` | layout-pack | bf16 下向量化宽度是否被限制在 8B 而非 16B |
| `coalesce-check` | gather-path | 读 `[M,2H]` 是否连续；非连续会掉到 46% 以下 |

**为什么不下「已到顶」的结论**：46% 的达成率在 memory-bound 算子里属于偏低。
如果是合并访问 + 满向量化，这类逐元素 kernel 通常能到 70-85%。

## 6. 这个 case 示范了什么

1. **场景先行**：先判 bandwidth-bound，再按表取 fusion-boundary，不是凭直觉试
2. **分母实测**：不引用历史数字，每次现测；且按访存比选口径
3. **错误可查**：小 buffer 虚高这件事记录在案，避免后人重踩
4. **结论带缺口**：给了 2.25x 的收益，也明确 46% 的达成率说明没到顶

## 复跑

```bash
python3 Plugins/ops/skills/op-optimize/cases/001-swiglu-fused-vs-split/bench.py
# 或跳过带宽标定（快，但没有分母）
python3 .../bench.py --quick
```
