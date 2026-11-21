# MQA logits 攻坚 · 交接（2026-11-13）

> 目标算子：DSA indexer 打分 `logits[m,n] = Σ_h relu(q[m,h,:]·k[n,:])·w[m,h]`，窗口 `[ks,ke)`。
> 对手：**lightop** 的 `mqa_logits`（本机已装，`from lightop.attention import mqa_logits`）。
> 目标：**超越该基线**。当前状态：**未达成**。

## 一、当前最好成绩

| 实现 | 耗时（M=8/N=128k/H=32/D=128/fp8） | vs lightop |
|---|---:|---:|
| lightop（基线） | 0.152~0.163 ms | 1.00× |
| **tilelang v3（BM8/BN96/HC4/t256）** | **0.245 ms** | **1.51× 慢** |

正确性：对 fp32 参考 cos **1.000000**（反量化后的 q/k）。

## 二、模型（先算再写）

`op-roofline/models/mqa_logits.py` 已给出：`AI = 2·q_per_ktile·H/D`。
H=32/D=128/fp8 ⇒ `AI = q_per_ktile/2`；机器平衡点 **313 FLOP/B**
⇒ 现成 decode 内核（qpk=1）AI=64，**纯访存受限**；杠杆是「把跨 query 的复用并进 GEMM 的 N 维」。

## 三、已实测排除的九条（别重走）

| # | 杠杆 | 结果 |
|---|---|---|
| 1 | 每条 query 一个 gemm（v1） | 6.7× 慢（gemm 只有 32 行） |
| 2 | (query×head) 摊平进 GEMM 的 M 维（v2） | 2.7× |
| 3 | 按 head 分块 + 去冗余 fragment 拷贝（v3） | **1.51×** ← 最好 |
| 4 | head 块 HC=8/16/32 | 1.75× / 2.69× / 3.06× |
| 5 | 线程数 128/384/512 | 更差或编译失败 |
| 6 | BN ∈ {64,80,96,112,128,160,256} 全扫 | 96~128 最好 |
| 7 | MLS `prefer_instruction="matrix_load"` | 单边挂**静默算错**（cos 0.90）；双边挂被动态子切片拒绝 |
| 8 | persistent CTA + Q 常驻 shared(32KB) | **2.87×（更差）**：occupancy 2→1 |
| 9 | persistent CTA + Q 常驻寄存器 | 2.09× |
| 10 | 把 w 折进 q（`relu(x)·w = relu(x·w)`） | 1.60×，**无收益** |

## 四、三个静默错误（编译通过、不报错、结果错）

1. **3D→2D 拷贝**：`T.copy(Q[m,0,0], q_sh)` 从 `[M,H,D]` 拷进 `[BM*H,D]` ⇒ cos 0.48。
   必须**逐条 2D 拷贝**。
2. **MLS 是双边的**：只给一边挂 `matrix_load` ⇒ cos 0.90（见 `op-hip-compile/references/isa-census.md`）。
3. **串行循环不能访问线程分布维**：必须 `T.reduce_sum` 或经 shared 中转。

## 五、为什么 tilelang 到顶了

lightop 的 kernel（`csrc/mqa_logits/mqa_logits_asm.cu`）参数结构暴露了它的做法：
除了 `cu_seq_len_k_start/end` 和 `num_heads`，其余全是 **rocBLAS/Tensile 的标准 GEMM ABI**
（`problemNumGroupTiles0/1`、`staggerUIter`、`wgmRemainder1`、`numFullBlocks`）。
⇒ 它是**成熟 Tensile GEMM + 轻量 epilogue**，不是自研 tile 循环。

tilelang 的 `T.gemm` 在 tile 选择与工作组调度上够不到那一层。

## 六、下一步：手写 HIP/MMA（M1）

**已验的底座**：`/workspace/_ext/hbird-gemm/kernels/prefill256_fp8.hip` —— 本机 `aicc` 可编译
（实测产出 36 KB .so）。它的 fp8 MMA 用法：

```c
// A 片段：LDS → 寄存器，32 行一个 sub-quarter（2 m-frag = 8 dwords）
RD_A = tl::mls::ds_read_format_tensor_a<SQ_MK, SQ_Q, SQ_TILE, 1, 8, 1,
                                        tl::fp8_t, 1, true, hcu_target_enum::gfx938>(...)
// MMA：A/B 各 int32x2（8 fp8），accum float32x4
C_l[..] = __builtin_hcu_mmac_f32_16x16x32_fp8_fp8_lit_lts(A, B, C_l[..], 1, 0);
```

hbird 的几何可**直接映射**到 MQA：tile 256(KV 行) × 256(=M·H 列) × 64(D)，512 线程，1×8 wave 切分
⇒ **每个 wave 正好拥有 32 列 = 一条 query 的 32 个 head**，h 归约落在 wave 内。

**A/B 的 LDS 布局已从 hbird 的 packkernel 里解出来了**（这是原来卡住的那一步）：

```c
// pack A（A 操作数 = 我们的 K 块；X 是 M×K row-major，P 是打包后）
idx = blockIdx.x*512 + threadIdx.x;              // 一个线程 16 B
s = idx & 7;  t = idx >> 3;  rp = M >> 1;        // rp = 行对数
pr = t % rp;  j = t / rp;                        // j = k-block 号（64 宽）
rr = s >> 2;  c = (((s & 3) - (pr & 7)) + 4) & 3;
P[idx*16] = X[(pr*2 + rr)*K + j*64 + c*16];      // 16 B

// pack B（B 操作数 = 我们的 Q；N 行 × K 列）—— 按「目标地址线性」写
idx = blockIdx.x*256 + threadIdx.x;  ng = (N+255) >> 8;
j = idx >> 10; within = idx & 1023;  wv = within >> 7;
l = (within >> 1) & 63;  ki = within & 1;
b = j / ng;  bx = j % ng;
n0 = bx*256 + wv*32 + (l & 15);
k0 = b*64 + 32*ki + (l >> 4)*8;
P[idx*16 .. +8]  = X[n0*K      + k0 .. +8];
P[idx*16+8 .. +8] = X[(n0+16)*K + k0 .. +8];
```

B 的「256 列一组 + 一个 k-block 交织」不是随便定的：hbird 实测同一份字节，
交织粒度 16/32/64/128/320/640/1280 KB 对应 **1545/1525/1474/1390/1354/1277/1171 GB/s**，
越细越好 ⇒ 一个 k-block 粒度最优。

**M1 待办**：
1. K 侧：hbird 的 A 是 pack 过的，我们的 K 是普通 `[N,D]` ⇒ 要么写 pack，要么直接构造 LDS 布局
2. epilogue：relu → 按 h 归约（wave 内）→ ×SQ×KSV → 写 OUT
3. 验收：编译 → cos ≥0.999 → ABBA 对 lightop → **>1.3× 就回流水线四问重新设计，别调参**

骨架在 `/tmp/mqa/mqa_logits_hip.hip`（含 TODO 与判据）。

## 七、可以复用的现成件

| 能力 | 现成代码 |
|---|---|
| gather + 硬件 OOB 钳位 | `tl::hcu_buffer_load<...,8,true>`（本卡 `dev/sparse_mla_hip_glm.hip`） |
| LDS 矩阵读（MLS） | `ds_read_format_tensor_a/b`（hbird） |
| fp8 MMA | `__builtin_hcu_mmac_f32_16x16x32_fp8_fp8`（hbird） |
| 块内归约 | `tl::AllReduce<SumOp,...>::run_batch` |
| 编译配方 | `aicc` + 11 条 `-mllvm`（见 `dev/COMPILE.md`） |
| 基线/评测脚本 | `/tmp/mqa/sweep5.py`（同进程 ABBA + cos 校验） |
