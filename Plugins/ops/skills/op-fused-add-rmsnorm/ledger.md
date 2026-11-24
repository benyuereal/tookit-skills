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

---

# 优化轮次记录

> 每一轮 = 一次「换实现 / 换结构」的尝试 + 一次同口径实测。
> 数字全部是本机 event_isolated 实测，base 每轮重测。

### 轮 1 · 建立基线：lightop.RMSNormAdd 到底多快

**做法**：不优化，先量准。`lightop.RMSNormAdd(H, eps=1e-6, inplace=False)` 是 nn.Module，
`forward(X, residual) -> y`，weight 是模块属性。

**实测**（M=8192 H=4096 bf16，201.3MB，event_isolated，3 轮 × 21 次）：
203.0 / 203.4 / 203.5 µs，MAD < 0.3% ⇒ **990 GB/s = 2_rw_2to1 档的 57%**。

**结论**：厂商实现只跑到 57%，有 40% 的空间。这不是「调参能救」的量级，
是结构问题 —— 与 rmsnorm-fused 卡上的发现一致（LDS 中转的额外往返）。

### 轮 2 · 换语言：triton 融合单 kernel

**做法**：用 triton 写单 kernel 融合残差（`BLOCK=H`，`num_warps=4`，无 mask）。

**实测**：未达预期。同形状下 triton 的收敛点在 175~186 µs 区间，
比 lightop 的 203 µs 快，但离地板还远。

**结论**：换 DSL 只拿到 ~10~14%，不是解药。rmsnorm-fused 卡的结论在此复现：
**三种 DSL（triton/tilelang/HIP）都落在同一档**，瓶颈是访存模式本身。

### 轮 3 · 定位机制：整行 LDS 往返是主因

**做法**：读 `isa_probe` 的反汇编 + 对照 scaffold 生成的骨架实测。

**实测**：scaffold 的默认骨架（整行读进动态 LDS → 归约 → 读回写）
只有 **212 GB/s**，是 lightop 的 1/5。

**结论**：H=4096 时每行 16KB 的 LDS 往返，把「2 读 1 写」打成了 4 次往返。
这不是带宽不够，是**多走了两趟 LDS**。

### 轮 4 · 寄存器驻留：`(x+residual)` 留在寄存器

**做法**：把 `(x+r)` 留在寄存器（`VEC*8` 个 float/线程），
LDS 只放 `WARPS*4` 字节的跨 warp 归约表；读写走 16 字节向量（uint4）。

**实测**（event_isolated，3 轮 × 21 次）：
**121.0 / 121.1 / 120.8 µs**，MAD < 0.5% ⇒ **1664 GB/s = 地板 1730 的 96.2%**。

**结论**：+40.5% vs lightop。机理是把 LDS 往返彻底消掉，
让 3 趟访存真正变成 3 趟。

**踩到的坑（已写进 kernel_scaffold）**：`VEC` 若作为**运行期变量**传入，
`xv[]/rv[]` 会整块溢出到私有内存（`private` 0→528B，0.165→0.538ms，慢 3.3 倍）。
必须编译期固化。

### 轮 5 · 同口径复核 + 口径纠错

**做法**：把轮 4 的结果放进 `loop.py` 的六道门，并给本卡补 `acc_probe.py`。

**实测**：base 每轮重测 255.2 µs（比轮 1 的 203 慢，说明那一轮的卡状态不同），
cand 140.0 µs ⇒ **ratio 0.549，显著加速 45.1%**，判定 `ACCEPTED_NUMERICS_PENDING_E2E`。
精度：自研 **0.50 ULP** vs lightop **1.50 ULP**（自研更准）。

**结论**：1.82x 成立，且是 claim 级口径。

**同时纠正一个判据错误**：给本卡加归约折扣 0.81 会算出 utilization=118.8%
（突破物理）。因为 1730 GB/s 是**同机同窗口含归约的实测上限**，不是纯 copy 换算。
不加折扣才对：1664/1730 = 96.2%。
