# 指令面普查（gfx938，2026-11-13）

> 目的：**别再靠翻源码偶遇**。写 GEMM/主循环之前，先知道这台机器上「有哪些 MMA 宽度、哪些 LDS 读原语、
> 它们的编译期前置条件是什么」。
>
> 每条都要有 `验证方式`。没验过的标 `unverified`，不许当判据用。

## 一、怎么刷新（三条路，按可用性排序）

**① 工具链头文件枚举（最快，本机可用）**

```bash
grep -rhoE "__builtin_hcu_mmac_[a-z0-9_]+|ds_read_[a-z0-9_]+|buffer_load_[a-z0-9_]+" \
     $TILELANG_HYGON_ROOT/src/tl_templates/hcu/ | sort -u
```

**② 厂商库的 kernel 元数据（配置基准，本机已验证可读）**

```bash
# gfx938 的 Tensile 库
H=/opt/dtk/lib/hipblaslt/library_bw_nmz/Kernels.so-000-gfx938.hsaco
/opt/dtk/llvm/bin/llvm-readelf --notes "$H" | grep -E "\.name:|vgpr_count|group_segment"
```
给出每个厂商 kernel 的**名字（编码了 MacroTile / 向量宽度）+ VGPR/LDS 用量**，
是判断「我们的资源用量是不是偏胖」的外部基准。

**③ 反汇编厂商 kernel（拿真实指令，已打通）**

两条路，都不用自己装工具：

```bash
# (a) 厂商已经带好了反汇编（461 MB，matmul 全家）
D=/opt/dtk/lib/hipblaslt/library_gfx938
grep -ohE "v_mmac[a-z0-9_]*|v_mfma[a-z0-9_]*" $D/FusedMatmulKernel_gfx938.co.ISA | sort | uniq -c | sort -rn

# (b) 自己反汇编任意 .hsaco / .co —— **输出写到当前目录的 <file>.ISA**，不在 stdout
cd /tmp && /opt/dtk/llvm/bin/dccobjdump --inputs=<file.hsaco> --recompile --target=gfx938
```

`--recompile` 不打印到终端，只在 cwd 落 `<file>.ISA` + `<file>-kernelMetadata.metadata`。
（`--disassembleAll` 只做 host，别用它找设备指令。）

## 二、普查表

| dtype | MMA 指令 | 宽度 (M×N×K) | LDS 读原语 | `k_pack` | 可用 MLS | 验证方式 | 状态 |
|---|---|---|---|---|---|---|---|
| bf16 | `__builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts` | 16×16×16 | `ds_read_format_tensor_a/b` | **2** | ❌ | 本项目 kernel 在用；micro GEMM 逐位对 fp32 参考 | verified |
| fp8 (e4m3fn) | `__builtin_hcu_mmac_f32_16x16x32_fp8_fp8` | 16×16×**32** | `ds_read_m32x32_b8` | **1** | ✅ | 8192³ 达 432 TF（`gemm_fp8_channelwise`）；本项目 fp8 kernel 精度 PASS | verified |

### 厂商实测（来自 `/opt/dtk/lib/hipblaslt/library_gfx938` 的反汇编，2026-11-13）

| 指令 | 出现次数 | 说明 |
|---|---:|---|
| `v_mmac_f32_16x16x32_fp8_fp8` | 170,490 | fp8 主力，与上表一致 |
| `v_mmac_i32_16x16x32_i8` | 128,820 | int8 同宽度 32 |
| `v_mmac_f32_16x16x32_fp8_bf8` / `bf8_fp8` | 各 44,040 | 混合 fp8/bf8 |
| **`ds_read_m32x32_b8`** | **22,786** | **MLS 矩阵读，量产在用** |

**没有出现任何 bf16 MMA**（`16x16x16_bf16` 为 0）——该库是 fp8/int8 家族，bf16 走别的库。
⇒ 「fp8 的 MMA 宽度是 bf16 的两倍」不只是头文件说的，厂商 kernel 里就是这么用的。

### 编译期前置条件（最容易踩）

| 事实 | 后果 | 来源 |
|---|---|---|
| **MLS 要求 `kPack==1`** | bf16 的 `kpack=2` 用不了 MLS；想用只能降到 1，那会把 gather 宽度砍半 | 编译期报错原文：`MatrixLoad dst Gemm consumer must have kPack=1, got 2` |
| **MLS 是双边的**：A 和 B 都要 `matrix_load` | 只给一边挂，**编译通过但静默算错**（实测 cos 0.90/1.00，无报错、无 inf）；且目标区域必须是**静态**形状（`q_sh[i*H:(i+1)*H]` 这种动态子切片会报 `matrix_load dst last-2 region must be static`）| 2026-11-13 MQA logits 实测 |
| `k_pack=2` 要求 `K % 64 == 0` | fp8 想用 k_pack=2 得把 block_I 提到 64 | 本项目 fp8 移植实测 |

### 全局访存原语

| 原语 | 走哪 | 用途 |
|---|---|---|
| `tl::hcu_buffer_load<..., 8, true>` | global → **寄存器** | 向量化 + **硬件 OOB 钳位**（`-1` 哨兵自动 0-fill，免分支） |
| `buffer_load_dwordx4 ... offen lds` | global → **LDS** | 直接落 LDS，省一次寄存器中转 |
| `prefer_instruction="matrix_load"` | 走 MLS 路径 | 需 `kPack==1` |

## 三、怎么加一行（micro 验证流程）

不要在大 kernel 里试。写 **20 行**最小 kernel：

```
A[M,K] @ B[N,K]^T -> C[M,N]，M=N=64、K=512
对 C 与 (A.float() @ B.float().T) 比 max_abs
```

**判据：`max_abs` 必须为 0（逐位一致）。** 不为 0 就换 dtype，别去调 scale ——
本项目实测 `float8_e4m3fn` 逐位精确，而 `float8_e4m3fnuz` 有整体比例偏差（cosine 仍是 1.0）。

## 四、已知缺口 / 推翻它的观测

- **反汇编这条路没通**（缺 `llvm-objdump`）⇒ 表里「厂商实际用什么指令」一列是空的。
- **别的 gfx 未普查**：本表全部是 gfx938。VGRP 预算 196608 只对 768-VGPR 模式成立，
  非 768 模式是 65536，occupancy 会变——换设备**必须重跑 §一 的 ①**。
- `e4m3fnuz` 的偏差是**本机组合**的观测；换 DTK/驱动版本可能不同，用它之前先跑 §三。
- 表里没有 tf32/bf8/int8 的行 = 没测，不是不存在。
