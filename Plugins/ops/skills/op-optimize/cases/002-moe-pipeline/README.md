# Case 002：MoE pipeline 段级拆解（MegaMoE 式融合的可行性排查）

> 前面 001 是单算子级（一个 kernel、一个场景、一个 ratio）。
> 这一例是 **pipeline 级**：MoE 的收益不是把某个 GEMM 写快，
> 而是把五段收进一个持久化算子、消掉中间落盘。
>
> **所以先要的不是优化，是段占比。** 没有它，后面全是猜。

## 阶段 0：五段分段计时

```bash
python3 stage_timing.py --M 4096    # P 档（大 M，compute-bound）
python3 stage_timing.py --M 32      # D 档（小 M，latency-bound）
```

口径：**CUDA event，不是 time.time**（异步 kernel 用 time.time 会被量成 0）；
每段之间 `synchronize`，不含前后段；每段单独 warmup；报 median + MAD。

### P 档（M=4096, H=2048, I=1408, E=32, K=4）

| 段 | median_ms | MAD | 占比 |
|---|---|---|---|
| gate_up_gemm | 7.0837 | 0.0045 | **47.8%** |
| down_gemm | 6.1352 | 0.0022 | **41.4%** |
| combine | 1.0074 | 0.0003 | 6.8% |
| swiglu | 0.3891 | 0.0010 | 2.6% |
| dispatch | 0.2190 | 0.0120 | 1.5% |
| **合计** | **14.8344** | | 100% |

### D 档（M=32）

| 段 | median_ms | MAD | 占比 |
|---|---|---|---|
| gate_up_gemm | 5.4931 | 0.0118 | **48.0%** |
| down_gemm | 5.4867 | 0.0069 | **47.9%** |
| combine | 0.2187 | 0.0040 | 1.9% |
| dispatch | 0.1717 | 0.0010 | 1.5% |
| swiglu | 0.0784 | 0.0026 | 0.7% |
| **合计** | **11.4486** | | 100% |

## 阶段 0 的结论（会影响后面所有阶段的假设）

**两个 GEMM 占了绝对大头，而且小 M 时占比更高：**

| 档 | 两个 GEMM 合计 | 三段非 GEMM（dispatch+swiglu+combine） |
|---|---|---|
| P（M=4096） | **89.2%** | 10.9% |
| D（M=32） | **95.9%** | 4.1% |

**这推翻了「先做融合消落盘」的默认假设。**

「把五次下发收进一个算子、消掉四趟中间落盘」这个思路，收益上限是
**非 GEMM 的那 4-11%**——因为它只能省 dispatch / swiglu / combine 之间的
中间态，动不了两个 GEMM 本身。

也就是说：
- 如果两个 GEMM 的实现已经接近地板，**融合最多省 4-11%，且要付融合的复杂度**
- 如果两个 GEMM 离地板很远，**先优化 GEMM 比先融合划算得多**

**下一步必须先回答**：这两个 GEMM 各自离各自的地板有多远？
在回答之前，做融合是在优化那 4-11%，而不是那 89-96%。

## 一个必须说明的口径限制

这里的五段是**用 torch 逐专家循环搭的未融合基线**，不是任何生产实现。
它的用途是**拿占比**，不是拿倍数——逐专家循环本身比真正的 grouped GEMM 慢，
所以「两个 GEMM 占 89-96%」这个结论在真实实现里会被稀释。

**但它稀释的方向是确定的**：真实实现里 GEMM 更快，
所以非 GEMM 段的占比会**上升**。这意味着融合的上限比 4-11% 高一些，
但不会高到「融合优先于 GEMM 优化」的程度。

后续阶段要换成真实 grouped GEMM 重测，才能给出准数。
**在那之前，任何「融合能省 X%」的说法都是未测定的。**

## 未测定项（不插值、不假装）

| 项 | 状态 |
|---|---|
| 真实 grouped GEMM 下的段占比 | **未测定**（当前是逐专家循环基线） |
| 两个 GEMM 各自的地板与达成率 | **未测定**（阶段 1 做） |
| 融合 vs 拆分的交叉点 M* | **未测定**（阶段 2 做） |
| 重叠是否真发生 | **未测定**（阶段 3 做） |

## 复跑

```bash
python3 stage_timing.py --M 4096   # 约 40s
python3 stage_timing.py --M 32
```

结果落盘到 `results/stage0_<M>.json`，含完整口径字段（what/alloc/state/timer/rounds）。

---

## 阶段 1：段级选型（两个 GEMM 的地板与达成率）

```bash
python3 stage1_gemm.py    # 见 results/stage1_gemm.json
```

### grouped vs 逐专家循环

形状 E=32, M/exp=128, N=1536, K=2048, bf16：

| 实现 | median | MAD | TFLOPS |
|---|---|---|---|
| `deepgemm.m_grouped_bf16_gemm_nt_masked`（mode 1000） | 0.3205 ms | 0.0149 | **80.4** |
| 逐专家 `torch.mm` 循环 | 1.0171 ms | 0.0149 | 25.3 |
| | | | **3.17x，|Δ|=0.697 > 3×MAD=0.045，显著** |

### 达成率：43%（远未到顶）

```
bytes = 230.7 MB，grouped 带宽 720 GB/s
算力地板（bf16 185 TF）= 0.1393 ms
达成率 = 0.1393 / 0.3205 = 43%
```

**这更正了阶段 0 的推论。** 阶段 0 用的是逐专家循环（25.3 TFLOPS），
它让两个 GEMM 看起来占了 89-96%。真实 grouped 实现快 3.17x，
所以**非 GEMM 段的真实占比会比阶段 0 高**——但 GEMM 本身 43% 的达成率
说明它也远没到顶。

**净结论**：现在既不能断言「融合优先」，也不能断言「GEMM 优先」。
两个方向都还有空间，需要阶段 2 的拐点来定。

### 实测到的硬约束（L2 选型必须记）

| 约束 | 现象 |
|---|---|
| masked 形态要求 **N 被 256 整除** | 真实 MoE 的 I=1408 被直接拒绝 |
| `a` 必须 **3D `[E, M, K]`**，`b` 必须 **3D `[E, N, K]`** | 传 2D 会 IndexError |
| `mode=1000` 是 asm layout，`pack_bf16_masked_weight` 只做 contiguous | — |
| `mode=2000` 需真 pack | — |

### ll 路径拐点：**未测定**（诚实标注）

`m_grouped_fp8_gemm_nt_masked_ll` 要求 `b` 是 **6D packed layout**
（报错：`expect tensor matrix_b is 6 which is 3`），而 deepgemm
**没有公开的 6D pack 函数**。

试过的路径：
- 2D/3D + `pack_bf16_masked_weight` → matrix_b 维度不够
- fp8 tuple `(tensor, scale)` 输入 → 同样维度不够

**这正是 MegaMoE 的 pack5 权重布局缺口在 DCU 上的对应物。**
按纪律标为「未测定」，**不插值、不假装测过**。

> 「ll 路径在多少 token 以下才用」这个问题，在当前可见的 API 下**无法回答**。
> 要回答它得先有 6D pack 实现——那是阶段 5 失败模式库里该记的一条。
