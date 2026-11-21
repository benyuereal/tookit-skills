---
name: op-moe-w8a8
description: "FP8 MoE 前向（routed experts w8a8）：GEMM1(gate/up) → silu-mul → FP8 量化 → GEMM2 → moe_sum 加共享专家。覆盖 lightop 4-stage / ASM 变体 / aiter / triton 多条实现的适用面与已知正确性缺陷（ASM mode>=1000 batch 串读）、融合收益判据（epilogue 融 moe_sum 慢 2.5-3×）。触发场景：MoE kernel 选型、M 分档调参、融合方案评估、MoE 精度/自洽性排查、ASM mode 1000 能不能用、routed 与 shared 两条路径怎么核。不适用：单发 FP8 GEMM 的 dtype/warp_size/k_pack 与三角选型走 op-gemm-fp8；稀疏注意力的 topk/-1 语义走 op-sparse-mla；四闸四档判定走 op-gate。"
version: 0.1.0
author: DCU Toolkit Contributors
license: MIT
---

# FP8 MoE 前向（op-moe-w8a8）· draft

> 状态：`draft`。结构已建，契约与判据已定；**bench/ 与 adapter（`_common.py`）尚未落地**（缺哪些符号见 `contract.yaml` 各门的 `skip_reason`，待办登记在 `data/ops/registry.json` 的 `open_questions`）。

## 一、是什么

```
x[B, H] --gate/up GEMM--> [B, 2*I] --silu(gate)*up--> [B, I] --fp8 quant-->
        --down GEMM--> [B, H]  (按 expert 分组)  ──┐
共享专家路径 ─────────────────────────────────────┴--> moe_sum --> out[B, H]
```

| 张量 | 形状 | 说明 |
|---|---|---|
| `x` | `[M, H]` | M = token 数（decode 时很小，prefill 时很大） |
| `w1`（gate/up） | `[E, 2I, H]` | FP8，按 expert 打包 |
| `w2`（down） | `[E, H, I]` | FP8 |
| `topk_ids` / `topk_w` | `[M, K]` | 路由结果 |
| `out` | `[M, H]` | = routed + shared |

**M 是决定性变量**：decode（M=1~64）与 prefill（M≥1024）是两种完全不同的算子形态，结论不可互换。

## 二、变体矩阵

| 变体 | 来源 | 适用 | 状态 |
|---|---|---|---|
| `lightop-4stage` | lightop `moe_w8a8_fp8`（mode 500–525） | decode + prefill 通用 | **安全区** |
| `lightop-asm` | 同库 mode ≥1000（预编译 `.co`，无源码） | 快，但**有正确性 bug** | **不可用**（见下） |
| `aiter-moe` | aiter fused moe | 视形状 | 待评 |
| `triton-shared` | 共享专家路径（triton + Cijk） | 生产并行的第二条 MoE 路径 | 现役 |
| `tilelang` | 自研 | 仅研究 | 未评 |

> 生产有**两条** MoE 路径：routed 走 lightop，shared 走 triton+Cijk。分析时不要漏掉 shared 路径的耗时。

## 三、正确性判据：ASM mode ≥ 1000

**已知 bug**：`lightop` 的 MoE Marlin ASM kernel（`mode >= 1000`，预编译 `.co` 无源码）在**混合专家 batch** 下会串读其他 token / 专家的数据。

- 判据不是「对比 torch 参考」，而是**自洽性**：同一个 token 在 batch 里的输出不依赖它的 batch 同伴。
- 实测：`(16, 0)` 配置 bit-exact 自洽；`(32, 1000)` 不自洽（max_diff 量级 1e5）。
- 源码注释里「(32,1000) verified lossless 1.6e-8」**不可信**。
- 安全区：`mode 500–525`（非 ASM 4-stage 家族）。

推论：**MoE 的无损性判定用自洽性测试**（batch 独立性），不要依赖自己写的 torch 参考——参考写错会同时放过 bug。

## 四、融合判据（已证伪，别再重开）

| 方案 | 结果 | 根因 |
|---|---|---|
| GEMM2 epilogue 融 `moe_sum`（atomicAdd） | 慢 **2.5–3×** | 全局 RMW + 写冲突 |
| GEMM1 epilogue 融 silu-mul + fp8 量化（g1f） | 结构不可行 | mode 0 decode 的 `BLOCK_N=16`，gate 列 `[0,512)` 与 up 列 `[512,1024)` 跨 block，纯 epilogue 融不到一起；要融必须拉宽 `BLOCK_N`，等于动 GEMM 主体 |
| `moe_sum` 用 triton 替代 | 慢 **8×** | bit-exact 但无收益，多一次读写 |
| Marlin GEMM 再优化 | 已贴地板 | gfx938 隔离实测 88.8% 峰值带宽（其他 gfx 未验证），且随激活 expert 数线性 |

**默认假设：融合净负。** 要推翻必须同时证明「收益不被搬移吃回去」和「端到端真收益」（见 `op-gate` §三）。

## 五、判据

- 正确性：vs 参考（按 expert 分组的 dequant 参考）容差见 `contract.yaml`
- 不变量：**`batch_independence`（自洽性，本算子最重要的一项）** + `determinism`
- 性能：按 M 分档，同卡同进程交替 A/B；`M<64` 与 `M≥64` 分开报

## 六、已知坑

| 坑 | 现象 | 规避 |
|---|---|---|
| ASM `mode ≥ 1000` | 混合专家 batch 串读 | 禁用；用 `mode 500–525` |
| 把 `moe_sum` 当成可省的一步 | 84 次调用看似冗余 | 42+42 分属 routed / shared 两条路径，**不是冗余** |
| 用小 batch 微基准推断生产 | M=1/MTP 数据有水分 | 以 M=32 并发 decode 为准 |
| 漏统计 shared 路径 | 只优化 routed | 两条路径都要入 profile |
| 假设「激活 expert 数少所以还有空间」 | 误判 45% 带宽 | 带宽随激活 expert 线性，gfx938 实测按 88.8% 峰值算 |

## 七、待补

`contract.yaml` 已就位；`bench/` 与 adapter `_common.py` 待写（复用 `op-sparse-mla` 的骨架）。补完后把状态推进到 `measured`。
