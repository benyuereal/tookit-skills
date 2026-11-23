# Case 007：MLA 非吸收 vs 吸收式 —— 同一库，两个形态差 4.5 倍

> 起点：`mla-full` 卡 status=draft，`mla-absorb` 已 measured。
> 两张卡都指向 `flash_mla`，但形态完全不同：
> 吸收式 q_len=1（decode，memory-bound）；非吸收 q_len>1（prefill，compute-bound）。

## 0. 判据：必须等 FLOPs 比

D=576 的 attention 没有现成的 flash_attn 对照（flash_attn 上限 D=256）。
直接拿「D=576 的 TFLOPS」比「D=128 的 TFLOPS」不公平——D 变大会增加
每元素的 MMA 与 LDS 压力。

所以用**等 FLOPs** 构造对照：FLOPs 线性于 SQ·D，
把 flash_attn 的 SQ 从 512 放大到 2304（=512×576/128），两边都是约 309 G FLOPs。

## 1. 轮 1：MLA 非吸收，扫 q_len（KV=8192）

| q_len | median | MAD | TFLOPS | of peak |
|---|---|---|---|---|
| 1 | 0.0872 ms | 0.0003 | 6.9 | 0.02 |
| 64 | 1.0133 ms | 0.0010 | 38.1 | 0.12 |
| 256 | 3.9003 ms | 0.0014 | 39.6 | 0.13 |
| 512 | 7.7498 ms | 0.0016 | 39.9 | 0.13 |
| 1024 | 15.4625 ms | 0.0053 | 40.0 | 0.13 |

**q_len ≥ 64 之后稳定在 0.13 of peak，完全不随规模改善。**
⇒ 不是固定开销，是**结构性的算力利用率低**。

## 2. 轮 2：等 FLOPs 对照（关键证据）

| 实现 | 形状 | median | TFLOPS | of peak |
|---|---|---|---|---|
| flash_attn | SQ=2304, D=128, KV=8192 | 1.7058 ms | **181.3** | 0.58 |
| flash_mla 非吸收 | SQ=512, D=576, KV=8192 | 7.7498 ms | **39.9** | 0.13 |

两边 FLOPs 都是 309 G。**等工作量下 flash_attn 快 4.54×。**

这是干净的同口径对照，不是「D 不同所以不能比」的模糊结论。

## 3. 轮 3：吸收式能不能代替非吸收做 prefill？

直觉上「吸收式更省 KV 读」，但 prefill 有 512 个 query：

| 路径 | 总耗时 |
|---|---|
| 非吸收，一次处理 512 个 query | 7.75 ms |
| 吸收式，调 512 次（每次 q_len=1） | 0.0862 × 512 = **44.15 ms** |

**吸收式分步慢 5.70×。** 因为它把每步都做成 memory-bound 的短 kernel，
固定开销乘了 512 倍。**prefill 场景下吸收式不可用**——这条否证了
「吸收式总能省」的直觉。

## 4. 结论

```
① MLA 非吸收在 q_len>=64 后稳定在 0.13 of peak，不随规模改善
② 等 FLOPs 下 flash_attn 快 4.54×（181.3 vs 39.9 TF，都是 309 G FLOPs）
③ 吸收式无法代替非吸收做 prefill：512 个 query 分步比一次做完慢 5.70×
④ 所以「需要完整注意力权重」这个理由的代价是 4.5× 的算力效率
```

### 对选型的含义

| 场景 | 该用什么 | 依据 |
|---|---|---|
| decode（q_len=1，KV 长） | 吸收式 | case 006：达峰值 0.35，且只读 KV |
| prefill（q_len 大） | **非吸收，但别指望它快** | 本 case：0.13 of peak，比 flash_attn 慢 4.5× |
| prefill 且不需 MLA 语义 | flash_attn | 0.58 of peak |

**`mla-full` 的 `best_for` 应改成「需要完整注意力权重的场景」，
并明确写出代价**——而不是笼统写「长上下文 prefill」，
那会让人以为它在 prefill 上是好选择。

## 5. 未测定项（不插值）

| 项 | 状态 |
|---|---|
| 0.13 的根因（是 D=576 的 LDS 压力还是调度） | **未测定**（需 PMC，本机无 rocprof） |
| 换 tilelang 实现能否改善 | 未测定（未跑通等价形状） |
| sparse MLA 的 prefill 形态 | 未测定（属 `sparse-mla` 卡） |
| 吸收式的 prefill 批量实现（一次做多 query） | 未测定（flash_mla 无此接口） |

## 6. 记账守卫

本 case 也带 `check_ceiling()`（同 case 006）。
第一次做「按 D 外推」时算出 581 TF > 峰值 310——那是线性外推的产物，
不是测量。**外推值不许进任何结论**，所以改为实测等 FLOPs 对照。

## 复跑

```bash
python3 -I Plugins/ops/skills/op-optimize/cases/007-mla-full-vs-absorb/bench.py
# 落盘：results/mla-full-vs-absorb.json
```
