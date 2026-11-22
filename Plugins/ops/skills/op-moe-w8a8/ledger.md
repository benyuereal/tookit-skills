# op-moe-w8a8 台账

> 绝对数字进 `~/.config/dcu-toolkit/kb/op-opt/`；本文件只留**相对倍数、机制与判定**。
> 机器权威在 `candidates.yaml:refuted`（进优化闭环后才有）。

## 一、当前状态

**draft**。契约、形状档位、变体矩阵、adapter、A/B 适配器均已就绪；
缺的是**本机安装 lightop / aiter 的 MoE 后端**——门会因此 FAIL 并说明原因，
这是正确行为，不是缺陷。

## 二、已确立的事实（来自 contract.yaml 与 SKILL.md，已有实测支撑）

| 事实 | 机制 | 影响 |
|---|---|---|
| ASM `mode >= 1000` 禁用 | 混合专家 batch 下串读其他 token/专家数据 | 违反 `batch_independence` 不变量；安全区 `mode 500–525` |
| M 分档是两个算子形态 | decode（M=1~64）权重主导；prefill（M≥1024）算力主导 | 两档结论不可互换 |
| routed 与 shared 是两条路径 | routed 走 lightop；shared 走 triton+Cijk | 只测 routed 会漏掉一半耗时 |
| epilogue 融 moe_sum 慢 2.5–3× | 融合把可并行部分串成依赖链并多占 smem | 先量分拆版的和再决定 |
| Marlin GEMM 分量达峰值带宽的 0.888 | 该分量已隔离实测 | 只说明分量，**不代表整体** |
| 权重必须 shuffle | `aiter_moe_shfl_weight` 未调用是**静默错误** | AITER 路径的前置条件 |

## 三、待验证（open_questions，来自 registry.json）

1. shared 路径（triton+Cijk）与 routed 路径的耗时占比拆分
2. aiter-moe 尚未评测
3. 整体达成率未判（只有 Marlin 分量的 0.888）

## 四、复现方式

```bash
# 正确性 / 不变量门（tiny 档，adapter 自检用；真实档留给真机 bench）
python3 scripts/opcheck.py correctness \
    --contract Plugins/ops/skills/op-moe-w8a8/contract.yaml \
    --shape tiny --variants lightop-4stage

# A/B（需先装后端）
python3 Plugins/ops/skills/op-moe-w8a8/bench/ab.py \
    --variants lightop-4stage --shape decode-m1 --rounds 5
```

## 五、纪律

- 本卡任何「快了 x%」必须过 `claim_gate` 三条硬门（口径 / 3×MAD / 正确性）。
- 分母固定用实测峰值 1724 GB/s（不是标称 1100——用标称会把已有空间误判成到顶）。
- `tiny` 档只用于验证门是否通，**绝不用于性能结论**。
