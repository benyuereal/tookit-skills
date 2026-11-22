# op-moe-align-permute 台账

> 绝对数字进 kb；本文件只留相对倍数、机制与判定。

## 一、当前状态

**draft**。九件套已齐，`ref-heavy` 变体**已通过 correctness 门**
（PASS，max_abs=0.0，限 0.0——纯搬运，容差为零）。缺本机实现后端。

## 二、天花板算式（纯读写搬运）

```
字节 = 读 M*H*2B + 写 M*K*H*2B
prefill-8k（M=8192, K=8, H=4096）：读 67 MB + 写 537 MB = 604 MB
内存地板 = 604 MB / 1724 GB/s = 350 µs
```

**这一段是纯 bandwidth-bound**：没有计算，没有复用，字节数完全确定。
达成率低于地板说明访存模式有问题（非合并访问、向量化不足）。

## 三、已确立的事实

| 事实 | 机制 | 影响 |
|---|---|---|
| `-1` 只能垫队尾 | `valid_NI` 按块首计数截断循环 | **前置 `-1` 会让整个输出恒为 0**（实测） |
| 对齐倍数是 kernel 契约 | 与下游 grouped GEMM 的 tile 高度绑定 | 对齐错了会读越界 |
| counts/offsets 用 int32 可能溢出 | M*K 很大时 | 需显式检查 |
| 置换必须可逆 | `unpermute(permute(x))` 应逐位还原 | 这是最直接的正确性判据 |

## 四、待验证

1. lightop-align 的 padding 实现是否真的只垫队尾
2. align=64 与 align=128 在同一个 kernel 上的行为差异

## 五、复现

```bash
./reproduce.sh check
./reproduce.sh bench
```

## 六、纪律

- 本卡容差为 0：任何位差都是搬运错误，不是数值误差。
- 多变体对比必须各喂独立副本（原地写的实体会互相污染）。
- 报告达成率必须用实测 1724 GB/s 做分母。
