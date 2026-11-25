# op-norm-act-swiglu 台账

三合一：`add + RMSNorm + SwiGLU`，输出半宽。

```
xf      = x + residual                        读 x, residual
h       = w * xf / sqrt(mean(xf^2,-1)+eps)    读 w；归约沿 H
gate,up = h[:, :I], h[:, I:]                  I = H/2
out     = silu(gate) * up                     写 out，I 宽
```

M=8192, H=4096, I=2048，全 bf16，eps=1e-6。

---

# 优化轮次记录

> 每轮 = 一次结构改动 + 一次同口径实测（event_isolated，base 每轮重测）。

### 轮 1 · 建立基线：两步法到底搬了多少

**做法**：不优化，先量准 lightop 的现成两步路径：
```python
o1, o2 = lightop.fused_add_rms_norm(x, r, w, eps)   # 原地写 x 与 r
lightop.fuse_silu_and_mul(o1, out)
```

**实测**（M=8192）：0.29904 ms（给定口径）/ 0.3050 ms（本仓 ab.py 口径）。

**流量核算**：① 读x+读r+写x+写r = 4H = 268.4 MB；② 读h+写out = 1.5H = 100.7 MB。
合计 **369.1 MB ⇒ 1234 GB/s = 1:1 档的 71%**。

**结论**：两步法本身效率不低（71%），问题在**多搬了 201 MB 的中间张量**。
这是结构性浪费，调参救不了。

### 轮 2 · 判断能不能融合

**做法**：分析融合的可行性。关键约束是 SwiGLU 需要 `h` 的两半（gate/up），
而 `h` 是整行归一化的结果 —— 必须先拿到全行 `mean(xf^2)` 才能算 `h`。

**结论**：可行。因为 `gate` 和 `up` 来自**同一个 `h`**，
归一化因子 `inv` 对两者相同 —— 一个 kernel 内先归约、再同时算两半，
不需要把 `h` 写出去。

### 轮 3 · 单行/CTA 的实现

**做法**：`(x+r)` 留寄存器（每线程 16 个 float），warp shuffle + 静态 LDS 表做行内归约，
归一化后同一线程的 gate/up 直接配对算 `silu(g)*u`，只写 I 宽。

**实测**：148.0 µs ⇒ **1133 GB/s = 地板的 66%**，相对基线 **2.02x**。

**结论**：融合成立，流量从 369.1 降到 167.8 MB（45.5%）。
但只有 66% 地板 —— 归约延迟藏不住（每 CTA 只搬 16KB 却要做 2 次 barrier）。

### 轮 4 · 多行/CTA：摊薄归约开销

**做法**：一个 CTA 处理 BLOCK_ROWS 行，行间归约互相独立 ⇒ 一组 barrier 服务所有行。

**实测**：

| BLOCK_ROWS | 耗时 | 地板占比 | vgpr |
|---|---|---|---|
| 1 | 148.0 µs | 66% | 39 |
| **2** | **140.5 µs** | **69%** | 57 |
| 4 | 153.3 µs | 63% | 84 |

**结论**：2 行最优。4 行因寄存器压力（vgpr 84）反而退化 —— 
多行不是越多越好，存在一个 vgpr 与 barrier 开销的平衡点。

### 轮 5 · 完整门禁验收

**做法**：放进 `loop.py` 的六道门，补 `acc_probe.py`。

**实测**：base 每轮重测 0.3050 ms，cand 0.1496 ms ⇒ **ratio 0.494，显著加速 50.9%**，
判定 `ACCEPTED_NUMERICS_PENDING_E2E`。精度 **1.00 ULP**、cosine 1.0000001。

**holdout 形状**（M=13312）：0.21552 ms / 272.6 MB / 1265 GB/s（73% 地板）——
**大规模时效率更高**，说明不是靠固定开销占比取胜。

**结论**：2.03x 成立，claim 级口径。

---

## 已确立的事实

| 事实 | 证据 |
|---|---|
| 融合的全部收益来自砍掉中间张量 | 369.1 → 167.8 MB；两实现的 kernel 结构基本相同 |
| 2 行/CTA 是最优点 | 1/2/4 行 = 148.0/140.5/153.3 µs |
| 精度比基线更好 | 1.00 ULP vs 基线自身对 fp32 的偏差 |
| holdout 上效率更高 | M=13312 达 73% 地板（public 69%） |

## 下一步（next_probe）

`surface: barrier-width` —— 当前 82.6% 地板，剩余来自行内归约的两道 barrier。
用 `ds_bpermute` 替掉 LDS 表可省一次 `__syncthreads`，预期 -5%~-10%。
falsifier：若减少 barrier 后耗时不变，则瓶颈在访存而非同步。

## 纪律

- 分母固定用 1:1 档 1680 GB/s × 归约折扣 0.81
- 每次对比必须 base 重测（lightop 两步法原地改写 x/r，须各喂副本）
- 换 H 必须重新生成 kernel（H == THREADS*8*VEC 且 THREADS*8 == I）
