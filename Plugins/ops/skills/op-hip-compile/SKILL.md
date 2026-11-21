---
name: op-hip-compile
description: "手写 HIP kernel 的编译调优：为什么用 aicc 而不是 hipcc、怎样从对标实现（tilelang JIT）逐字镜像 -mllvm 集合（而不是抄静态列表）、FAST_MATH/WDRA 条件项、从 .so 里读 VGPR/SGPR 算占用率、codegen 报错（PrintVecElemLoad Check failed）怎么二分到具体构造。触发场景：手写 HIP 正确但比同体 tilelang 慢 10%+、同体不同速、aicc/hipcc 报 Unknown command line argument、要判断 occupancy 是不是 1 CTA/CU、要给新 kernel 定编译参数、codegen 在 Check failed 处 abort 而大 kernel 里定位不到源头。不适用：换算法/调参/候选队列与续跑走 op-optimize；计时口径与噪声判定走 op-bench；fp8 dtype 与 tilelang 写法走 op-gemm-fp8。"
version: 1.0.0
author: DCU Toolkit Contributors
license: MIT
---

# 手写 HIP 的编译调优（op-hip-compile）

## 一、解决什么

手写 kernel「数值对、逻辑对，但比对标实现慢 10~30%」，十次里有九次不是算法，是**编译选项没镜像全**。
本模块给两条判据 + 两个工具，把这件事从「凭记忆抄 flag」变成「可复现、可验证」。

| 做 | 不做 |
|---|---|
| 从对标实现推导出应给的编译选项 | 不调度优化（那是 `op-optimize`） |
| 从产物读 VGPR/SGPR/LDS，算占用率 | 不做计时（那是 `op-bench`） |
| 给出「慢在 codegen 还是慢在算法」的判据 | 不改 kernel 本体 |

## 二、两条铁律

1. **镜像对标实现的实际选项，不是抄一份静态 flag 列表。**
   对标实现（这里是 tilelang JIT）的 `-mllvm` 集合是**运行时**按 arch + `pass_configs` 拼出来的，
   还有条件项（`TL_ENABLE_FAST_MATH` → 追加 `-enable-hcu-approx-func-fp-math`）。
   实测代价（gfx938）：漏一条条件项，同体 kernel 慢 **1.16×/CTA**（0.810 → 0.952 ms），而**输出位级一致**。
   ⇒ 这类差异**永远不会被正确性测试抓到**，只能逐 flag A/B。
2. **怀疑 codegen 就读二进制，不要读源码猜。**
   `aicc -shared` 的 .so 里，host ELF 之后内嵌一份设备 ELF；设备 ELF 的 `.note` 带 AMDGPU metadata，
   里面有每 kernel 的 `.vgpr_count` / `.sgpr_count` / `.group_segment_fixed_size`。用 VGPR 反推
   「每 CU 能驻几个 CTA」。

## 三、为什么是 aicc，不是 hipcc

| 项 | aicc | hipcc |
|---|---|---|
| 接受这批 `-mllvm` | 是（本机 tilelang JIT 就是调它） | **否**，后端 dcc/lld 报 `Unknown command line argument` |
| 产物 | `-shared -fPIC` 出 .so，内嵌设备 ELF | 同 |

这些 flag 是后端 LLVM 的**指令调度 / 延迟建模 / 寄存器预算**，**不改变数值**
（带 flag 与不带 flag 路径输出位级一致）。所以：**数值对不上不是 flag 的问题**，别往这查。

## 四、怎么用

```bash
# ① 打印该 arch 的完整 flag 集（直接调 tilelang 的同一个函数，保证逐字镜像）
python3 derive_flags.py --arch gfx938 --fast-math

# ② 打印并顺手编译
python3 derive_flags.py --arch gfx938 --fast-math --hip dev/x.hip --out dev/libx.so

# ③ 不从命令行猜：从对标 kernel 源码嗅 pass_configs，再推导
python3 derive_flags.py --arch gfx938 --sniff /path/to/tilelang/ref_kernel.py

# ④ 读产物：每个 kernel 的 VGPR/SGPR/spill 与占用率
python3 inspect_so.py dev/libx.so --threads 256 --lds 28672
```

生成代码（HIP）读法四条 —— 数着读比猜快：

- `__builtin_hcu_mmac_f32_16x16x16_bf16_lit_lts` 出现几次 = 几条 MFMA；FLOPs/8192 可交叉验。
- 操作数是 `*(bfloat16x4_vec*)(X + k)` ⇒ X 在寄存器；出现 `i_16` / `local_id` 循环 ⇒ 从 shared 读。
- **内层下标带 `(threadIdx.x & 7)` 但循环步长是 128 的，就是标量 2 字节读**（无法向量化）。
- `__syncthreads()` 条数 = 每块的排空次数；occupancy=1 CTA/CU 时每一条都是硬损失。

`derive_flags.py` 的目标是**不重复实现** tilelang 的拼装逻辑：它 `import
tilelang.contrib.hcu.get_hcu_compile_flags` 并显式传 `pass_configs`。若 aicc 不在 PATH，
该函数会按 DTK 判定直接返回 `[]`——工具的退出码 3 就是拦这个「静默空集」。
`--print-static` 的静态副本**只用于对照**，判据永远是那个函数。

## 五、gfx938 的 11 条（`--fast-math` 后的完整集）

| # | flag | 类别 | 说明 |
|---|---|---|---|
| 1 | `-enable-latency-hack=true` | 调度 | 打开后端延迟 hack |
| 2 | `-mmac-latency=5` | 延迟建模 | 矩阵核（MMAC）延迟按 5 建模，影响发射间隔与重排 |
| 3 | `-ds-load-store-latency=6` | 延迟建模 | LDS load/store 延迟按 6 建模 |
| 4 | `-disable-machine-sink=True` | 调度 | 关掉 machine sink（避免指令下沉/重物化破坏流水形状） |
| 5 | `-check-valu-data-forward-hazards=0` | 调度 | 不做 VALU 写后读前递冒险检查 |
| 6 | `-disable-cluster-lds-memops=true` | LDS | 不生成 cluster LDS memops |
| 7 | `-amdgpu-disable-backoff-barrier=false` | barrier | 保留 backoff barrier |
| 8 | `-support-768-vgprs=true` | 寄存器 | 放开 768 VGPR 预算（>256 的分配才可能；本机主 kernel 252，未触发但不要省） |
| 9 | `-enable-hcu-approx-func-fp-math=true` | 数学 | **FAST_MATH 条件项**；漏了 1.16×/CTA（见 §二） |
| 10 | `-hcu-update-wait-by-reverse-search=true` | 调度 | 用反向搜索决定 waitcnt 插入位置（gfx938/92a/946 专属） |
| 11 | `-hcu-pre-emit-load-store-opt=false` | 调度 | 关闭 pre-emit load/store 优化（同上，仅这三个 arch） |

> 1~8 是 arch 无条件项（gfx92a/946 把第 8 条换成 `-support-512-vgprs=true`）；
> 9 由 `TL_ENABLE_FAST_MATH` 触发；10~11 只在 gfx938/92a/946 追加。
> 表的「说明」按名与后端语义分类；**真正要信的是 `derive_flags.py` 的实际输出**。

## 六、已知坑

| 坑 | 现象 | 规避 |
|---|---|---|
| 只抄 10 条无条件项 | 慢 1.16×/CTA，输出位级一致，正确性测试全绿 | 用 `derive_flags.py --fast-math` / `--sniff`，别手抄 |
| 顺手加上 WDRA 项 | `-vgpr-greedy-alloc-mode=local-wave` 实测 **+1.2% 更慢** | 只在对标 kernel 真的开了 `TL_ENABLE_HCU_WDRA` 时才镜像；镜像 ≠ 越多越好 |
| 以为 `__launch_bounds__(256,2)` 能换 2 CTA/CU | aicc **忽略 minBlocks 请求**：VGPR 252 纹丝不动，性能 1.000× | 占用率墙只能靠 kernel 重分片（压 VGPR）破，不能靠编译期声明 |
| 把 `-mllvm` 集合当成「性能开关」盲试 | 单条加/减在统计噪声内，看不出 | 只在「与对标实现差 10%+ 且假设同体」时用；且同进程 A/B、多次取中位 |
| 拿 host `.text` 当 kernel 体积 | host `.text` 只有几 KB，看不出差别 | 用 `inspect_so.py` 读设备 ELF 的 `.text`（本机 GLM 档 34996 B） |

## 七、codegen 报错怎么二分（`PrintVecElemLoad` / `Check failed`）

**现象**：`codegen_hcu.cc: PrintVecElemLoad: Check failed: (!type_name.empty())`——编译期直接 abort，没有行列号，大 kernel 里完全看不出是哪一处构造触发。
**机制**：这类 check 是 codegen 对「元素类型还没落地」的断言，触发点通常落在新引入的数据类型上：fp32 fragment 窄化写进 fp8 shared、`if_then_else` 补零到 fp8、fp8 gather。同一个 dtype 在大 kernel 里会经过十几处 lowering，**报错点不等于源头**。
**判据（做法）**：把新数据类型相关的构造拆成 **15–25 行**的最小 kernel 逐个试，一次只留一个构造：

| 最小用例 | 单独验证的构造 |
|---|---|
| fp32 fragment → fp8 shared | 窄化赋值 / `T.copy` 的 dtype 转换 |
| `if_then_else` 补零到 fp8 | 条件表达式里的常量 0 与结果 dtype |
| fp8 gather | 带索引的向量读 |

能复现的那条就是元凶；单条不复现就两两组合，仍然只在最小 kernel 里做。**不要在大 kernel 里猜**：每猜一次都要重编译整个 kernel，而 abort 前已经执行了几十处 lowering，猜错的成本远高于拆最小用例。

## 七·五、指令面普查（写 kernel 之前先查）

这台机器上「有哪些 MMA 宽度、哪些 LDS 读原语、它们的编译期前置条件」见
[`references/isa-census.md`](references/isa-census.md)（含刷新命令、micro 验证流程、以及
**MLS 要求 `kPack==1`** 这类会直接否掉方案的约束）。

## 八、与其它 skill 的关系

| skill | 关系 |
|---|---|
| `op-bench` | 编译差异必须用同进程 A/B 才能看出（跨进程绝对 µs 不可比） |
| `op-profiling` | VGPR→占用率→「等内存时无可切换」的推理链在这里收口 |
| `op-optimize` | 「所有调参都撞墙」时，先按 §二 排除 codegen，再判「需重写 kernel」 |
| `op-sparse-mla` | 本模块的来源案例：`../op-sparse-mla/dev/sparse_mla_hip_glm.hip` 的 11-flag 构建与 1.16×/CTA 代价 |

## 九、落点

- 绝对 ms / 体积字节只进 `~/.config/dcu-toolkit/kb/op-opt/`；本 skill 只放相对倍数与口径。
- 「镜像」的判据是**跑 `derive_flags.py`**，不是这份文档里的表——表会漂，函数不会。
