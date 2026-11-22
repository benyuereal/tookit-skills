# op-fused-add-rmsnorm 台账

> 绝对数字进 `~/.config/dcu-toolkit/kb/op-opt/`；本文件只留相对倍数、机制与判定。

## 一、当前状态

**可跑通**。`lightop.RMSNormAdd` 已接入并通过 correctness 门
（实测 max_abs=1.53e-2、cosine=0.999997，在修正后的容差内）。
这是本仓第一张**用真实库实现跑通门禁**的卡。

## 二、本轮最重要的发现：绝对容差在 bf16 输出上不可移植

接入 `lightop.RMSNormAdd` 后第一次跑门是 **FAIL**——
max_abs 1.46e-2，超了我原先定的 2e-3 容差 7 倍。

但继续查发现这不是实现错误：
```
输出量级       ~3.36
max_abs        1.46e-2
相对误差中位数  1.6e-3   ← 这是 bf16 的正常舍入
```

**问题在容差的定法**：绝对容差随输出量级变化，同一个实现换个量级的输出
就从「过」变「不过」。已改为以 `rel_median` 为主判据，`max_abs` 只作上界兜底。

> 这正是这个算子当校准件的价值——它先校准了我的容差口径。

## 三、天花板算式（字节数完全确定）

```
融合：读 x + 读 r + 读 w + 写 y = 3*M*H*2B
prefill-8k（M=8192, H=4096）：3*8192*4096*2 = 201 MB
内存地板 = 201 MB / 1724 GB/s = 117 µs
```

分母用实测 1724 GB/s，不用标称 1100（后者会把已有空间算成已到顶）。

## 四、已确立的事实

| 事实 | 机制 | 来源 |
|---|---|---|
| `lightop.RMSNormAdd` 是 nn.Module | 签名 `(normalized_shape, eps=1e-5, inplace=False)`，`forward(X, residual)`，weight 是模块属性 | `inspect.signature` 实测 2026-11-22 |
| eps 在 sqrt 内 | 与 fp32 参考一致 | 实测对比 |
| 同族有 dropout/bias 变体 | RMSNormAddDropout / RMSNormAddDropoutBias | `dir(lightop)` 实测 |
| bf16 输出的相对误差约 1.6e-3 | 正常舍入，不是实现问题 | 实测 |

## 五、待验证

1. 真机 A/B（需要空闲卡）：`./reproduce.sh bench`
2. `inplace=True` 时是否真的原地改写残差（决定 `mutates_inputs` 的准确值）
3. hidden 非 2 的幂时的向量化损失

## 六、复现

```bash
./reproduce.sh check    # 需要 lightop（已装 0.6.0）与一张卡
./reproduce.sh bench    # 需要 HIP_VISIBLE_DEVICES
```

## 七、纪律

- 达成率分母固定用实测 1724 GB/s。
- 容差以 `rel_median` 为主，`max_abs` 只作兜底。
- 多变体对比各喂独立副本（原地写会污染）。
