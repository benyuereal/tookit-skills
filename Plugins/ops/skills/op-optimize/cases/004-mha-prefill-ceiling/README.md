# Case 004：MHA prefill 的上限在哪 —— 先量到顶，再谈优化

> 起点：`mha-prefill` 卡 status=draft，没有实测。
> 场景归属 `compute-bound`。按场景表第一批攻击面是 tile / config 类。
> **本 case 的产物是「上限」而不是「改法」**——理由在最后。

## 0. 分母先于结论

达成率的分母不能用抄来的数。这个 case 每一轮都**同进程重测 bf16 峰值**：

```
轮0  bf16 峰值 = 310.8 TF（N=6144，1.492 ms）
```

为什么坚持同进程测：2026-11-23 查出 constants.yaml 里的 bf16 峰值 185 TF 是错的
（由 19.7 条 MFMA/cyc 反推，标定用的是 N=2048 这种固定开销占大头的小形状）。
同机实测 310 TF —— **实测高于声称的峰值，说明常数错**。
用错分母，达成率会系统性高估 1.7×。详见 `op-roofline/tools/calibrate_bf16_peak.py`。

## 1. 轮 1：causal prefill 扫 S

| S | median | MAD | TFLOPS | of peak |
|---|---|---|---|---|
| 1024 | 0.3578 ms | 0.0050 | 96.0 | 0.31 |
| 2048 | 0.9470 ms | 0.0019 | 145.1 | 0.47 |
| 4096 | 3.1216 ms | 0.0030 | 176.1 | 0.57 |
| 8192 | 11.4541 ms | 0.0053 | 192.0 | **0.62** |

S 放大 8 倍，达成率 0.31 → 0.62，**单调上升但收敛**。
S=8192 已接近本机 flash_attn 的常态上限。

## 2. 轮 2：非 causal 对照 —— 判 causal 的块跳过做没做干净

| | TFLOPS |
|---|---|
| causal（工作量减半） | 176.1 |
| 非 causal（全量） | 198.0 |

非 causal 的 TF 只高 12%，**说明 causal 的三角块跳过是有效的**——
如果跳过没做干净，非 causal 会明显更快（因为同样时间做了两倍活）。

## 3. 轮 3：GQA 对照 —— 判瓶颈是不是 KV 读

| | median | TFLOPS | of peak |
|---|---|---|---|
| MHA（H_kv=32） | 3.1216 ms | 176.1 | 0.57 |
| GQA（H_kv=8） | 3.0086 ms | 182.7 | 0.59 |

**只快 3.6%。** 这条否证了一个常见假设：「prefill 慢是因为 KV 读得多」。
KV 读取量砍到 1/4，耗时几乎不动 ⇒ **瓶颈在算力/指令侧，不在访存侧**。

## 4. 轮 4：换库对照 —— 为什么用 flash_attn 不用 SDPA

| 路径 | median | TFLOPS | of peak |
|---|---|---|---|
| flash_attn（BSHD） | 3.1216 ms | 176.1 | 0.57 |
| `F.scaled_dot_product_attention`（BHSD） | 4.9226 ms | 111.7 | 0.36 |

**SDPA 慢 1.58×。** 这是 L2 选型的直接证据：
本机 SDPA 走的是 flash-attn 的 adapter 但多一层分派，且对 BHSD 布局要先 transpose。
→ **tier=primary 给 flash_attn，SDPA 只能当对照**，理由是可复跑的耗时比，不是文档说法。

## 5. 轮 5：varlen —— prefill 的真实形态

| 形态 | median | TFLOPS | of peak |
|---|---|---|---|
| 2 条 × 2048 | 0.5002 ms | 137.4 | 0.44 |
| 8 条 × 2048 | 1.7946 ms | 153.2 | 0.49 |
| 固定 B=4 × 2048（对照） | 0.9470 ms | 145.1 | 0.47 |

变长与固定 batch 同量级 ⇒ **变长本身不额外掉点**，不需要为变长单独设计。

### 这里踩过一个坑（记账错）

第一次测 varlen 算出 **546~611 TF，1.8~2.0× 峰值**。
实测值不可能超过峰值——所以是 FLOPs 记错了：varlen 张量是
`[total_tokens, H, D]`，**没有 batch 维**，我把 B=4 又乘了一遍。
改回 `4·n·S²·H·D·0.5` 后落到 135~153 TF，与固定 batch 对照吻合。

**教训**：任何「超过峰值」的结果先怀疑记账，不要先怀疑机器。
这与 185 那次是同一类错误的两面——那次是分母错，这次是分子错。

## 6. 结论：为什么本 case 不产出「改法」

```
flash_attn causal prefill 在本机的达成率：
  S=1024 → 0.31，S=8192 → 0.62（趋势收敛）
瓶颈定位：
  ③ GQA 把 KV 读砍 4×，耗时只降 3.6% ⇒ 不是访存
  ② 非 causal 工作量 2×，TF 只高 12% ⇒ causal 跳过有效，无残余浪费
⇒ 剩下的 38% 差距在 flash_attn 的 tile 调度 / MMA 发射，属库内部
```

**要再往上走必须改 flash_attn 源码的 tile 逻辑**，那是 source-patch，
不在本轮的「选型 + 改法」范围内。诚实标注：

| 项 | 状态 |
|---|---|
| 手工调 tile 能到多少 | **未测定**（需改源码，属 op-source-patch） |
| S=8192 之后是否继续上升 | 未测定（显存限制，H=32 D=128 B=4 时 S=16384 需约 50 GB） |
| 与 aiter 的 prefill 实现对比 | **未测定**（本轮未找到 aiter 的等价入口） |

## 7. 对本卡（`mha-prefill`）的落点

- `ratio` = 0.62，vs 写「本机实测 bf16 峰值 310 TF（同进程标定）」，
  scope 写清 B/H/D/S/是否 causal。
- `best_for`：S≥2048 的 causal prefill；S=1024 只有 0.31，**小 S 别指望它**。
- `not_for`：decode（M 小，是 memory-bound，结论不可互换）；需要改算法时用 tilelang。
- 结论只写「达峰值的 0.62，瓶颈在算力侧不在访存侧」，不写「还能快 1.6×」——
  后者需要改源码才能验证，没验证就不能写。

## 复跑

```bash
python3 -I Plugins/ops/skills/op-optimize/cases/004-mha-prefill-ceiling/bench.py
# 落盘：results/mha-prefill.json
```
