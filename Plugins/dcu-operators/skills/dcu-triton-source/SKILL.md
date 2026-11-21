---
name: dcu-triton-source
description: Triton DCU/HCU 源码结构——独立 hcu backend（DTK 特化：libgalaxyhip/aillvm/gfx936-938 768-vgprs）+ 调优 CLI
version: 1.1.1
author: DCU Engineer
license: MIT
---

# Triton DCU/HCU 源码适配索引（dcu-triton-source）

> 定位：**Triton 在 DCU 上有独立的 `hcu` backend**（与 `amd`/`nvidia` 并列的第三 backend），是 amd backend 的 fork + DTK 特化。本 skill 讲 hcu backend 的结构、DTK 适配点、调优 CLI。
> 与 `dcu-triton-kernels` 的关系：那个讲**怎么写/优化 Triton kernel**（DSL 层）；本 skill 讲 **Triton 编译器在 DCU 上的 backend 结构**（编译器层）。写 kernel 用前者，排查编译/工具链/调优用本 skill。
> 实测来源：`triton-3.6.0+das185.dtk2604.torch2100.2608131719.gb796b1-cp310` wheel（release/dtk2604 源），解包 384 文件。

## §0 定位：hcu backend 是什么

Triton 的 `triton/backends/` 下有 `amd` / `nvidia` / **`hcu`** 三个 GPU backend。`hcu` 是海光从 `amd` fork 出来的，**DCU 设备走 hcu backend**（不是 amd）。

**backend 选择机制**：`triton/compiler/compiler.py:367` 定义 `_HCU_ARCHS = frozenset({'gfx926','gfx928','gfx936','gfx938','gfx92a','gfx946'})`——设备 arch 命中即走 hcu backend。`triton/backends/amd/driver.py:844` 注释明示：*"Hygon deployments use triton.backends.hcu for HIP; AMD driver must not"*（amd driver 对 HCU arch 让位）。

**hcu vs amd 差异**（DTK 特化）：
| 维度 | amd backend | hcu backend（DCU） |
|---|---|---|
| HIP runtime 库 | `libamdhip64.so` | **`libgalaxyhip.so`**（DTK 主库） |
| 设备属性 API | `hipGetDevicePropertiesR0600` | **`hipGetDeviceProperties_v2`** |
| LLVM 工具链 | `<rocm>/llvm` | **`<dtk>/aillvm`**（AI LLVM） |
| 安装路径候选 | `/opt/rocm` | `/opt/rocm` + **`/opt/dtk`** |
| 编译失败归档 | 无 | **`~/.triton/cache/failed_kernels/`**（海光新增） |
| 调优 CLI | 无 | **`hcutune_cli` / `opt_config_cli`**（海光新增） |

## §1 源码地图（wheel 实测目录）

```
triton/
├── backends/
│   ├── hcu/                    # ★ 海光独立 backend（本 skill 主体）
│   │   ├── __init__.py         # 空（backend 注册在 compiler.py）
│   │   ├── compiler.py         # ★ HIPBackend（hcu 版）：DTK 工具链 + gfx arch 参数 + 失败归档
│   │   ├── driver.py           # ★ HIP driver：libgalaxyhip 定位 + hipGetDeviceProperties_v2
│   │   ├── driver.c            # C 侧 driver（DTK/ROCm 双路径设备属性）
│   │   ├── _utils.py / autotuner.py / jit.py / testing.py
│   │   ├── hcutune_cli.py      # ★ 分布式 auto-tuning CLI（海光新增）
│   │   ├── opt_config_cli.py   # ★ 最优 config 查询 CLI（海光新增）
│   │   └── lib/librocshmem_device.bc
│   ├── amd/                    # AMD 上游 backend（hcu 的 fork 源，含 include/ hip 头文件）
│   ├── nvidia/
│   └── compiler.py / driver.py # backend 基类
├── compiler/compiler.py        # _HCU_ARCHS 定义（L367）+ backend 分发
├── language/target_info.py     # HIP_CDNA3_ARCHS（含 gfx936/938）
├── _internal_testing.py        # HIP_HCU_ARCHS（L28）
├── _C/libtriton/               # 编译产物（ir/passes/llvm/amd/hcu/distributed 绑定）
└── libs/                       # 依赖库
```

## §2 DTK 工具链特化（compiler.py，精确落点）

| 函数 | 行号 | 作用 |
|---|---|---|
| `HIPBackend.is_hcu` 判定 | `compiler.py:180` | `self.is_hcu = target.arch in ('gfx926','gfx928','gfx936','gfx938','gfx92a','gfx946')` |
| `path_to_rocm()` | `compiler.py:337` | 取 ROCm/DTK 根：`ROCM_PATH` env → `/opt/rocm` → **`/opt/dtk`** |
| `path_to_rocm_llvm()` | `compiler.py:350` | ★ DTK 用 **`<dtk>/aillvm`**（AI LLVM），缺失时打印 `HCU/DTK TOOLCHAIN ERROR: AI LLVM (aillvm) not found` 并回退 `<dtk>/llvm` |
| `_archive_failed_compilation()` | `compiler.py:372` | ★ 编译失败归档到 `~/.triton/cache/failed_kernels/<时间>_<kernel>_<stage>/`（context.txt + 中间产物），海光新增调试设施 |
| `path_to_rocm_lld()` | `compiler.py:401` | 找 `ld.lld`（env `TRITON_HIP_LLD_PATH` → backend llvm/bin → `<llvm>/bin` → `/usr/bin`） |
| `path_to_rocm_clang()` | `compiler.py:425` | 找 clang（env `TRITON_HIP_CLANG_PATH` → 默认 `clang-18` → `clang`） |
| `_get_clang_args()` | `compiler.py:447` | ★ gfx arch → clang 参数映射（见 §3） |

**DTK 工具链排查**：编译报 `HCU/DTK TOOLCHAIN ERROR: AI LLVM (aillvm) not found` → 检查 `<ROCM_PATH 或 /opt/dtk>/aillvm` 是否存在（DTK 安装不完整，见 `dcu-dtk`）。

## §3 gfx arch → clang 参数矩阵（compiler.py:447 `_get_clang_args`）

| arch | clang 参数 | 说明 |
|---|---|---|
| gfx928 | `-mllvm=-support-512-vgprs=true` | 512 VGPR |
| **gfx936** | **`-mllvm=-support-768-vgprs=true`** | ★ 768 VGPR（DCU 主力） |
| **gfx938** | **`-mllvm=-support-768-vgprs=true`** | ★ 768 VGPR |
| gfx946 | `-mllvm=-support-512-vgprs=true` | 512 VGPR |
| gfx92a | `-mllvm=-support-512-vgprs=true` | 512 VGPR |

> 未知 arch 直接 `raise ValueError(f"Unknown arch: {options.arch}")`。gfx936/938 的 768-vgprs 是 DCU 相对 AMD CDNA 的关键差异（更多向量寄存器）。

## §4 DTK driver 特化（driver.py，精确落点）

| 函数 | 行号 | 作用 |
|---|---|---|
| `_is_dtk()` | `driver.py:69` | ★ 判定 DTK vs ROCm：`ROCM_PATH` 含 "dtk" 或 `/opt/dtk` 存在 |
| `_get_path_to_hip_runtime_dylib()` | `driver.py:78` | ★ DTK 用 **`libgalaxyhip.so`**，ROCm 用 `libamdhip64.so`；env `TRITON_LIBHIP_PATH` 可覆盖 |
| DTK 安装路径 | `driver.py:166-167` | DTK 时 common_install_path = `/opt/dtk/lib/libgalaxyhip.so` |
| 设备属性符号 | `driver.py:191-192` | ★ DTK 用 **`hipGetDeviceProperties_v2`**，ROCm 用 `hipGetDevicePropertiesR0600` |
| noload 主/次库 | `driver.py:368-369` | DTK：主 `libgalaxyhip.so` 次 `libamdhip64.so`（互斥加载） |

**driver.c 侧**（`driver.c:121/152-157/200`）：C 代码同样 DTK/ROCm 双路径——DTK 用 `hipDeviceProp_t_v2`/`hipGetDeviceProperties_v2`，ROCm 用 `hipDeviceProp_t`/`R0600` 变体。

## §5 调优 CLI（海光新增，amd backend 没有）

| CLI | 文件 | 用法 |
|---|---|---|
| **hcutune_cli** | `hcu/hcutune_cli.py` | 分布式 auto-tuning：`hcutune_cli --nproc=8 YOUR_SCRIPT.py [args]`，起 N 个进程分摊 benchmark case（须配 `triton.utils.dist`） |
| **opt_config_cli** | `hcu/opt_config_cli.py` | 查最优 config：`python -m triton.tools.opt_config_cli show [--all] [--kernel ...] [--device ...]`，从 config cache 取最优配置 |

> 这两个是海光在 hcu backend 加的**调优基础设施**（amd backend 无对应物），做 Triton kernel 调优时优先用（见 `dcu-triton-kernels` 的 Benchmark 流程）。

## §6 与兄弟 skill 关系

- `dcu-triton-kernels`：Triton DSL 写/优化 kernel（十题实战 + 五大优化模式 + DCU 陷阱）→ 本 skill 是它的**编译器 backend 层**支撑。
- **Inductor 代码生成层**（FX图→IR→Triton kernel 的中间层）：见 `references/inductor-dcu-verified.md`（2026-09-17 在 <device-1>BW150 实测验证：融合/CSE/persistent_reduction + 文章与 DCU 实际差异，如 `libdevice.floor` 非 `tl.math.floor`）。
- `dcu-kernel-dev`：HIP/TileLang 两条 kernel 路径 → Triton 是第三条路径，backend 结构见本 skill。
- `dcu-dtk` / `dcu-dtk-layout`：DTK 安装/布局 → 本 skill 的 `aillvm`/`libgalaxyhip` 路径依赖 DTK 布局。
- `dcu-op-optimization` / `dcu-perf-profiling`：优化闭环 → 本 skill 的 hcutune_cli/opt_config_cli 是调优工具。

## §7 验证入口

```bash
# 确认走 hcu backend（设备 arch 命中 _HCU_ARCHS）
python3 -c "import triton; print(triton.runtime.driver.active.get_current_target())"

# 确认 DTK 工具链
python3 -c "from triton.backends.hcu.compiler import HIPBackend; print(HIPBackend.path_to_rocm_llvm())"
# 期望: /opt/dtk/aillvm（不是 /opt/dtk/llvm）

# 编译失败归档位置
ls ~/.triton/cache/failed_kernels/   # 有失败 kernel 时这里有 context.txt + 中间产物

# 调优 CLI
python -m triton.tools.opt_config_cli show --all
```

## 何时选它 / 何时别选

- **选它**：算子的形状随数据变化（topk、ragged、gather 类），写不出固定的 tile 切分，需要 DSL 的表达力换正确性。
- **选它**：要快速得到一版可用的正确实现做对照（写 Triton 的周期通常短于写 HIP 同等功能）。
- **选它**：问题出在编译器层——`AI LLVM (aillvm) not found`、kernel 编译失败要归档、要找最优 config（`opt_config_cli`）。
- **别选（结构性吃亏）**：GEMM 主体。`tl.dot` 在 DCU 上 M=64–256 只有参考实现的 0.29–0.41×，且这是 lowering 路径的结构位置，调 BLOCK/num_warps 救不回来。
- **别选**：需要直接控制寄存器/LDS 或手排 barrier 的 kernel——hcu backend 暴露不到那一层，硬需求走手写 HIP。
- **别选**：M=1 的极瘦 GEMV——非 dot 的 broadcast 写法能打满流式峰值，但这种形状手写 ladder 更省编译与缓存的不确定性。
- **反例（易混）**：编译问题先看 `~/.triton/cache/failed_kernels/` 的归档；工具链缺失报的是安装不完整，不是本 skill 能修的代码问题。

- **DCC 编译器官方文档索引**（8 章 56 叶：编译选项/-mllvm/GCVM/Triton 接入/LingShu 后端/dcc-autotune/dccprof）：见 `references/dcc-compiler-docs.md`（2026-10-14 从文档中心实测建档）。

- DCU 设备**走 hcu backend**（gfx926/928/936/938/92a/946 命中），不是 amd backend。排查 Triton 编译问题先看 hcu backend 代码。
- **DTK 工具链依赖**：hcu backend 用 `<dtk>/aillvm`（AI LLVM）+ `libgalaxyhip.so`，DTK 安装不完整会报 `AI LLVM (aillvm) not found`。
- **gfx936/938 是 768-vgprs**（AMD CDNA 是 512），写 kernel 时寄存器压力预算不同。
- 编译失败有**自动归档**（`~/.triton/cache/failed_kernels/`），排查编译错误先看归档的 context.txt。
- 版本演进：triton 3.1~3.6 都有 hcu backend（源里 9 个版本），本 skill 基于 3.6.0（最新）。
- 不存性能数据/基线；数字用 `dcu-op-optimization` 模板留痕。
- 高危操作谨慎，优先测试机验证。
