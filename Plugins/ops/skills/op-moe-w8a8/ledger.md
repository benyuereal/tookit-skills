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

---

## 六、S3 迭代记录（3 轮）

### 轮 1：moe_align_block_size 的 pad_sorted_ids 开关 —— **抓到库 bug**

**假设**：`pad_sorted_ids=True` 能让 padding 显式可见，便于核对哨兵位置。

**测法**：M=256 E=32 K=4 BS=128，同一个 topk_ids 跑两种取值。
（签名实测：`moe_align_block_size(topk_ids, block_size, num_experts, expert_map, expert_mask, pad_sorted_ids, num_token, ep_size, is_EP, local_num_tokens)`）

| 取值 | 结果 |
|---|---|
| `pad_sorted_ids=False` | 84.00 µs，sids (5088,)，ntp=4096 |
| `pad_sorted_ids=True` | **NameError: name 'round_up' is not defined** |

**判定**：**这个开关是坏的**。lightop 在 `pad_sorted_ids=True` 分支引用了未定义的 `round_up`。
不是用法错——参数名与位置按 `inspect.signature` 拿的。

```
mechanism: 库实现缺陷。pad_sorted_ids=True 走未定义符号 round_up 的分支。
           影响：想用「显式 padding」核对哨兵位置的尝试都走不通，
           只能靠 pad_sorted_ids=False 的输出反推。
           判据：NameError（不是静默错，容易确认）。
next:      要么改 lightop 源码补 round_up（source-patch），
           要么在卡上写明「本开关不可用」避免后人踩 → 已写入 SKILL.md 已知坑。
```

### 轮 2：fuse-moe-sum-epilogue —— **证伪**

**假设**：把 `moe_sum` 融进 grouped GEMM 的 epilogue 能省一次写回。

**判定**：**证伪**，慢 2.5-3×。

```
mechanism: 融合把本可并行的 moe_sum 串进 grouped GEMM 的依赖链，并多占 smem。
           对已有 4 段的流水线，多一段融合反而减少 CTA 驻留。
           判据：libraries.md §3 记录的反例 + SKILL.md 的融合判据。
```

### 轮 3：asm-mode-ge-1000 —— **方向证伪**

**假设**：ASM 路径（mode≥1000）可能更快。

**判定**：**方向证伪**，且是正确性问题不是性能问题。

```
mechanism: 该 kernel 在混合专家 batch 下串读其他 token/专家的数据，
           违反 contract.yaml 的 batch_independence 不变量。
           判据：SKILL.md 的已知 bug 记录；安全区是 mode 500-525。
```

### 三轮小结

| 轮 | 方向 | 判定 | 类型 |
|---|---|---|---|
| 1 | pad_sorted_ids 开关 | **库 bug（NameError）** | 新发现 |
| 2 | fuse moe_sum epilogue | 证伪，慢 2.5-3× | 既有，补机制 |
| 3 | asm mode>=1000 | 方向证伪（正确性） | 既有，补机制 |

**轮 1 的价值**：这是第一次在「改法」层（L3 的 env-flag 类）发现库本身的缺陷。
说明「参数开关」不是无风险的——开关存在不代表能跑。
