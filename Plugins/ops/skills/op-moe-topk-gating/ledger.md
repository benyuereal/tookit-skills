# op-moe-topk-gating 台账

> 绝对数字进 `~/.config/dcu-toolkit/kb/op-opt/`；本文件只留相对倍数、机制与判定。

## 一、当前状态

**draft**。九件套已齐（contract / shapes / variants / _common / bench/ab /
tests / ledger / reproduce.sh / trace_budget.py），`ref-heavy` 变体
**已通过 correctness 门**（PASS，max_abs=0.0）。缺的是本机的 lightop /
aiter gate+topk 后端。

## 二、天花板算式（带宽地板）

```
读 W_gate：E * H * 2B
decode-m1（M=1, H=4096, E=128）：128 * 4096 * 2 = 1.05 MB
内存地板 = 1.05 MB / 1724 GB/s = 0.61 µs
```

M=1 时这一段的地板在微秒级——**它不该是瓶颈**。如果 profile 显示它占比高，
说明问题在 launch 开销或融合边界，不在带宽。

## 三、已确立的事实

| 事实 | 机制 | 影响 |
|---|---|---|
| `topk_idx` 是离散量 | 容差对索引无意义 | 必须逐位比较，不能用 max_abs 判 |
| tie-break 未在 torch.topk 文档中保证 | 跨版本/后端可能变 | 参考实现自己实现稳定排序（取小索引） |
| softmax 在全分布上做 | 先 topk 再 softmax 会在子集上归一 | `sum(w)` 会不等于 1 |
| E 不一定是 2 的幂 | 某些 kernel 的 padding 假设 E 是 2 的幂 | 用 E=100 测 |

## 四、待验证

1. 融合 gate+topk 的 tie-break 是否与参考一致（这是最容易静默错的地方）
2. 本段在真实 trace 里的占比（跑 `trace_budget.py`）

## 五、复现

```bash
./reproduce.sh check       # 正确性门（无需后端）
./reproduce.sh bench       # 需要后端与空闲卡
./reproduce.sh trace       # 需要 TRACE_DECODE
```

## 六、纪律

- 索引比较一律逐位；权重比较用契约容差。
- 任何「快了 x%」必须过 `claim_gate` 三条硬门。
- 本段通常不是瓶颈，先跑 `trace_budget.py` 确认占比再投入。
