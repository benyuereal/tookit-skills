---
name: dcu-tilelang-source
description: TileLang DCU 适配源码结构——TVM 系 DSL 编译器（contrib/hcu.py 海光 HIP 后处理 + TILELANG_OVERRIDE_DEVICE_SOURCE + libtilelang 链接 DTK）
version: 1.0.1
author: DCU Engineer
license: MIT
kind: knowledge
---

# TileLang DCU 适配源码索引（dcu-tilelang-source）

> 定位：**TileLang 是 TVM 系的 tile 级 DSL 编译器**（Python 写 tile 算子 → TVM IR → HIP/CUDA kernel），海光版支持 DCU。本 skill 讲 TileLang 在 DCU 上的包结构、海光 hcu 适配点、与 kernel 开发 skill 的衔接。
> 与 `dcu-kernel-dev` 的关系：`dcu-kernel-dev` 讲 **HIP/TileLang 两条 kernel 开发路径**（方法学）；本 skill 讲 **TileLang 编译器在 DCU 上的源码结构**（编译器层）。SGLang 的 `dsv4/tilelang_kernel.py` 是 TileLang 在 DCU 生产路径的实战样板（见 `dcu-sglang-source`）。
> 实测来源：`tilelang-0.1.9+das185.dtk2604.torch2100.2608121523.gf631c9-cp38-abi3` wheel（release/dtk2604 源），解包 3638 文件（含完整 TVM 源码）+ ldd 实测。

## §0 定位：TileLang 是什么

TileLang = **tile 级编程 DSL**（比 Triton 更底层，比手写 HIP 更高层）：
- Python 写 tile 算子（`@tilelang.jit` 装饰器）→ TVM TIR → 调度 → HIP/CUDA codegen → hipcc/clang 编译。
- 内置 GEMM/attention 等 tile 模板（`tileop/`）+ autotuner（`autotuner/`）+ carver（自动调度搜索 `carver/`）。
- **DCU 生产使用**：SGLang 海光 fork 的 `dsv4/tilelang_kernel.py`（DeepSeek-V4 注意力）已用 TileLang（见 `dcu-sglang-source`）。

**DCU 适配**：海光在 `contrib/hcu.py` 加 HIP 后处理回调 + `TILELANG_OVERRIDE_DEVICE_SOURCE` 环境变量（替换生成的 HIP 源码），`libtilelang.so`/`libtvm.so` 链接 DTK 的 `libgalaxyhip.so.5`。

## §1 源码地图（wheel 实测目录，含完整 TVM 源码）

```
tilelang/
├── __init__.py / _ffi_api.py / ir.py / dtypes.py / env.py / libinfo.py
├── language/                    # ★ DSL 语言层（45 文件：gemm_op/allocate/builtin/parser/ast）
├── tileop/                      # ★ tile 算子模板（16 文件：GEMM/attention 等）
├── engine/                      # 编译引擎（lower.py/callback.py，5 文件）
├── transform/                   # TVM pass（9 文件）
├── analysis/ / layout/ / intrinsics/ / math/ / quantize/
├── autotuner/                   # ★ 自动调优（tuner.py/param.py，4 文件）
├── carver/                      # ★ 自动调度搜索（34 文件，arch/cdna.py 等）
├── jit/                         # JIT 编译（31 文件：adapter/cython/libgen/execution_backend）
├── contrib/                     # ★ 后端工具
│   ├── hcu.py                   # ★★ 海光 HCU 后端（HIP 后处理回调 + override env）
│   ├── rocm.py                  # ROCm 后端（find_lld 等）
│   ├── hipcc.py                 # hipcc 编译器封装
│   └── cc.py / dlpack.py
├── src/                         # ★ C++ 源码（294 文件，TVM 系 codegen/schedule）
├── 3rdparty/                    # TVM 第三方（3109 文件，含 tvm 本体）
├── lib/                         # 编译产物
│   ├── libtilelang.so           # ★ TileLang 核心（链接 libgalaxyhip.so.5 + libhsa-runtime64）
│   ├── libtvm.so                # ★ TVM 核心（链接 libgalaxyhip.so.5 + libz3）
│   └── tilelang_cython_wrapper.abi3.so
└── testing/ / tools/ / cache/ / profiler/
```

## §2 海光 hcu 适配（contrib/hcu.py，精确落点）

`contrib/hcu.py` 是海光在 TileLang 加的 **HCU 后端**（ROCm 后端的 DCU 特化）：

| 函数 | 作用 |
|---|---|
| `hcu_recompute_from_source` | ★ `@register_hip_postproc_callback` 注册的 HIP 后处理回调——codegen 出 HIP 源码后、hipcc 编译前，可替换设备源码 |
| `TILELANG_OVERRIDE_DEVICE_SOURCE` | 环境变量：指定一个 `.cu` 文件**整体替换**生成的 HIP translation unit |
| `TILELANG_OVERRIDE_DEVICE_SOURCE_DIR` | 环境变量：目录里放 `{kernel_symbol}.cu`（仅当 codegen 暴露单个 `__global__` kernel 时） |
| `_extract_hip_kernel_symbols` | 正则提取 `extern "C" __global__` kernel 符号名 |
| `_override_declares_symbols` | 校验 override 文件是否声明了全部 kernel 符号 |

**关键行为**：
- `tilelang.engine.lower` import 本模块时**自动注册回调**（无需手动 import hcu）。
- **kernel cache 命中会跳过 lowering**（override 不生效）——强制重跑设 `TILELANG_DISABLE_CACHE=1` 或清 cache。
- 两个 override env 都设时，**单文件优先**。
- `target="auto"` 时 ROCm PyTorch 构建选 HIP（即使 PATH 有 CUDA toolkit）；override 只影响 HIP 设备 codegen，不影响 `tilelang_cuda`。

> **用途**：TileLang 生成的 HIP kernel 在 DCU 上有 bug 时，可用 `TILELANG_OVERRIDE_DEVICE_SOURCE` 替换成手修版本（不改 TileLang 本体，可回滚）——同 `dcu-vllm-runtime-patch` 的 monkey-patch 思路，但作用在 codegen 层。

## §2b MLS / GEMM_MLS 架构白名单（gfx936 限制，源码实证 + 2026-09-10 BW1000 真机复现）

TileLang 的 HCU GEMM 有两条数据搬运路径：
- **普通路径**：`ldmatrix_a/ldmatrix_b + mmac`（`tileop/gemm/gemm_hcu_mmac.py:302-340`）——**支持 gfx936**。
- **MLS 路径**（Matrix Load Store，LDS 矩阵搬运优化）：`matrix_load` / `ds_read_format` tile op + `gemm_mls` 模板——**仅白名单 gfx938 / gfx92a / gfx946**。

**硬限制落点**：`src/target/utils.cc:150-159` 的 `GetHcuArchString()`：
```cpp
static const std::set<std::string> supported = {"gfx938", "gfx92a", "gfx946"};
ICHECK(supported.count(mcpu))
    << "HCU arch " << mcpu
    << " not supported for MLS/GEMM_MLS; supported: gfx938, gfx92a, gfx946";
```
该函数**只被 MLS/GEMM_MLS 路径调用**（`op/mls.cc:42/327/501`、`op/mls_gemm_dep.cc:64`、`op/ds_read_format.cc:385/394/403`），普通 GEMM 不碰它。

**触发链**：kernel 含 `matrix_load`/`ds_read_format` tile op（`transform/annotate_mls_gemm_dep.cc:89` 的 `ContainsMlsTileOps` 把关）→ GEMM lowering `use_gemm_mls = a_from_mls || b_from_mls`（`gemm_hcu_mmac.py:256`）→ 走 MLS 模板 → `GetHcuArchString` ICHECK Fatal。

**关键坑**：MLS 路径选型**只看 kernel 里有没有 MLS op，不看 arch**——没有 gfx936 的 fallback 门。所以只要 kernel 用了 MLS，在 gfx936（BW100/BW1000/BW150/K100-AI 这一档老架构）上必崩。对照 `contrib/hcu.py:133`：`get_hcu_compile_flags` 里 gfx936 是**合法编译目标**（`["gfx928","gfx936","gfx938","gfx92a","gfx946"]`）——即 gfx936 整体支持，只是 MLS 子特性没在 gfx936 上开放。

**真机复现（2026-09-10，BW1000 容器，gfx936/80CU/64GB，tilelang 0.1.9+das185.dtk2604.torch2100.2608121523.gf631c9 + torch 2.10.0+das.opt1）**：
- 对照组（`T.copy` shared load + `T.gemm`，128×128×64 tile，256 线程，fp16 4096³）：编译+运行 PASS，max_abs_err=0.1251（rel 3.4e-4，fp16 正常），粗测 ~74 TFLOPS。
- 实验组（`T.matrix_load` 搬 A/B + `T.gemm`）：编译期即 `InternalError: Check failed: (supported.count(mcpu)) is false: HCU arch gfx936 not supported for MLS/GEMM_MLS; supported: gfx938, gfx92a, gfx946`——**ICHECK 报错文案与行号（utils.cc:155）与源码实证完全一致**。
- 复现脚本要点：`tilelang.compile(pf, out_idx=-1)`（`@T.prim_func` 装饰后的对象**不能再包 `tilelang.jit()`**——eager builder 的 `inspect.signature` 会对 PrimFunc 对象抛 `TypeError: <源码片段> is not a callable object`）；eager 模式 `inspect.getsourcelines` 要求脚本落盘，stdin/heredoc 跑会 `OSError: could not get source code`。
- `triton.language` 有 `matrix_load`（无 `ds_read_format`）——triton 侧 MLS 触发面待查，triton 基础 kernel（vadd/tl.dot matmul 64³ tile）在 gfx936 均 PASS。

**规避/修复**：
- 自己写 TileLang kernel：gfx936 上**不要用** `T.matrix_load`/`T.ds_read_format`，也别把 MLS 格式 shared 喂给 `T.gemm`，改用普通 shared load + `T.gemm`（走 ldmatrix+mmac 路径）。
- aiter/SGLang 自动生成的 kernel 触发：属**框架侧缺 arch 门**——MLS 选型处应加 `mcpu in {gfx938,gfx92a,gfx946}` 判断、gfx936 自动回退非 MLS；需改框架 fork 或找海光要带 fallback 的构建。
- 换/对齐 tilelang 构建版本：若 MLS 是近期才默认开启，换在 gfx936 上能正确回退的构建可绕开。
- 先 `rocminfo | grep gcnArchName` 确认卡确实是 gfx936（排除 target 被错设成别的 arch）。

## §3 DTK 链接（lib 产物，ldd 实测）

| .so | DTK 库链接 |
|---|---|
| `libtilelang.so` | **`libgalaxyhip.so.5`** + `libhsa-runtime64.so.1` + `libtvm_ffi.so` + `libz3.so` |
| `libtvm.so` | **`libgalaxyhip.so.5`** + `libhsa-runtime64.so.1` + `libz3.so` |

> 同 triton/ORT/DeepEP 的 DTK 库替换规律：`libgalaxyhip.so.5` 是 DTK 主 HIP 库。

## §4 与 kernel 开发 skill 的衔接

- **写 TileLang kernel**：`dcu-kernel-dev` 的 TileLang 路径（方法学）+ 本 skill（编译器结构）。
- **DCU 实战样板**：SGLang `dsv4/tilelang_kernel.py`（DTK26.04 镜像内 `sglang/attention/dsv4/tilelang_kernel.py`，源码未随本包分发，见 `dcu-sglang-source` 离线参考代码节）——海光已在生产路径用 TileLang。
- **Triton vs TileLang**：Triton（`dcu-triton-source`）是独立编译器 + hcu backend；TileLang 是 TVM 系 + contrib/hcu。两者都支持 DCU，Triton 更主流，TileLang 在 SGLang DSV4 有生产使用。
- **autotuner/carver**：TileLang 自带自动调优（`autotuner/tuner.py`）+ 调度搜索（`carver/`），做 tile 算子调优时优先用。

## §5 与兄弟 skill 关系

- `dcu-kernel-dev`：HIP/TileLang 两条 kernel 路径（方法学）→ 本 skill 是 TileLang 编译器层。
- `dcu-triton-source` / `dcu-triton-kernels`：Triton 编译器/kernel → 与 TileLang 并列的 DSL 路径。
- `dcu-sglang-source`：SGLang DSV4 的 TileLang 实战样板。
- `dcu-op-optimization`：优化闭环 → TileLang autotuner 是调优工具。
- `dcu-dtk`：DTK 安装 → `libgalaxyhip.so.5` 来自 DTK。

## §6 验证入口

```bash
# 安装（torch 标记对齐，本版 torch2100）
pip install tilelang -i https://pypi.sourcefind.cn/release/dtk2604/

# 确认 DTK 链接（DCU 容器内）
ldd $(python3 -c "import tilelang,os;print(os.path.dirname(tilelang.__file__))")/lib/libtilelang.so | grep -E 'galaxyhip|hsa'

# 确认 hcu 后端可用
python3 -c "from tilelang.contrib import hcu; print('hcu backend OK')"

# 基本 TileLang kernel（参考 dcu-kernel-dev TileLang 路径）
python3 -c "
import tilelang, tilelang.language as T
@tilelang.jit
def matmul(A, B, C):
    ...
print('tilelang OK')
"
```

## 使用须知

- TileLang 是 **TVM 系 DSL 编译器**，wheel 内含完整 TVM 源码（3638 文件）——可看 codegen/schedule 实现。
- **海光 hcu 后端**：`contrib/hcu.py` 的 `hcu_recompute_from_source` 回调 + `TILELANG_OVERRIDE_DEVICE_SOURCE` env（替换生成的 HIP 源码，不改 TileLang 本体）。
- **kernel cache 陷阱**：cache 命中跳过 lowering，override 不生效——强制重跑设 `TILELANG_DISABLE_CACHE=1`。
- **MLS/GEMM_MLS 架构白名单**：MLS 路径仅 gfx938/gfx92a/gfx946，gfx936（BW100/BW1000/BW150/K100-AI）上走 MLS 必崩（`GetHcuArchString` ICHECK，`utils.cc:155`）——gfx936 上 GEMM 走普通 ldmatrix+mmac 路径，详见 §2b。
- **DTK 链接**：`libtilelang.so`/`libtvm.so` 链 `libgalaxyhip.so.5` + `libhsa-runtime64`。
- **DCU 生产使用**：SGLang DSV4 的 `tilelang_kernel.py` 是实战样板（见 `dcu-sglang-source`）。
- 不存性能数据/基线；数字用 `dcu-op-optimization` 模板留痕。
- 高危操作谨慎，优先测试机验证。
