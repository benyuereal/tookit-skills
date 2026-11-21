---
name: dcu-ort-source
description: ONNX Runtime DCU 适配源码结构——ROCm EP 链接 DTK libgalaxyhip + MIGraphX EP + 二进制 wheel 适配点
version: 1.0.0
author: DCU Engineer
license: MIT
---

# ONNX Runtime DCU 适配源码索引（dcu-ort-source）

> 定位：**ONNX Runtime 在 DCU 上是二进制 wheel 交付**（无 C++ 源码在包内），DCU 适配体现在**编译产物的库链接**上。本 skill 讲 ORT wheel 的结构、DCU 适配点（ROCm EP / MIGraphX EP 链接 DTK 库）、与调优 skill 的衔接。
> 与 `dcu-ort-perf` / `dcu-ort-profile` 的关系：那两个讲**怎么调优/剖析** ORT；本 skill 讲 **ORT 在 DCU 上的包结构与适配落点**。
> 实测来源：`onnxruntime-1.22.2+das.opt1.dtk2604.2607061906.gc3f2dd-py3-none-manylinux_2_35_x86_64.whl`（release/dtk2604 源），解包 313 文件 + ldd 实测。

## §0 定位：ORT 在 DCU 上的交付形态

ORT DCU 版是 **py3-none wheel**（纯 Python 包 + 预编译 .so），**C++ 源码不在包内**（与 vLLM 0.21+ wheel 同形态）。DCU 适配在**编译期**完成，体现在：
1. **ROCm EP**（`libonnxruntime_providers_rocm.so`，99.4MB）：链接 DTK 的 `libgalaxyhip.so.5`（非 ROCm 的 `libamdhip64`）+ `libhipblas`/`libMIOpen`/`libhipfft`/`librocm_smi64`。
2. **MIGraphX EP**（`libonnxruntime_providers_migraphx.so`）：链接 `libmigraphx_c.so.3` + `libgalaxyhip.so.5`。
3. **核心 runtime**（`libonnxruntime.so.1.22.2` + `onnxruntime_pybind11_state.so`）：平台无关 + pybind 绑定。

> 关键判别：ORT 在 DCU 上**不是**用 ROCm 原生 `libamdhip64`，而是 DTK 的 `libgalaxyhip.so.5`——这是 DTK 相对 ROCm 的核心库替换（同 triton hcu backend 的 `libgalaxyhip` 特化，见 `dcu-triton-source` §4）。

## §1 源码地图（wheel 实测目录）

```
onnxruntime/                          # 经 .data/platlib 安装
├── __init__.py
├── capi/
│   ├── libonnxruntime.so.1.22.2              # 核心 C runtime
│   ├── onnxruntime_pybind11_state.so         # pybind 绑定（22.4MB）
│   ├── libonnxruntime_providers_rocm.so      # ★ ROCm EP（99.4MB，DCU 适配主体）
│   ├── libonnxruntime_providers_migraphx.so  # ★ MIGraphX EP（0.3MB，DCU 适配）
│   ├── libonnxruntime_providers_shared.so    # EP 共享
│   ├── _pybind_state.py / _ld_preload.py
│   └── onnxruntime_collect_build_info.py     # 构建信息收集
├── quantization/                       # 量化工具（QDQ/静态量化，平台无关）
├── transformers/                       # transformers 优化工具（fusion_*.py 系列）
├── tools/                              # onnx 工具（ort_format_model / qdq_helpers）
├── backend/                            # ONNX backend API
└── datasets/                           # 测试模型（logreg_iris/mul_1/sigmoid.onnx）
```

## §2 DCU 适配点（ldd 实测，.so 库链接）

| .so | 大小 | DTK/DCU 库链接（ldd 实测） | 说明 |
|---|---|---|---|
| `libonnxruntime_providers_rocm.so` | 99.4MB | **`libgalaxyhip.so.5`** + `libhipblas.so.2` + `libMIOpen.so.1` + `libhipfft.so.0` + `librocm_smi64.so.2` | ★ ROCm EP，DCU 适配主体（GEMM/Conv/Attention 走 hipblas/MIOpen） |
| `libonnxruntime_providers_migraphx.so` | 0.3MB | **`libmigraphx_c.so.3`** + `libgalaxyhip.so.5` | ★ MIGraphX EP（图级融合，DCU 上常比 ROCm EP 快） |
| `onnxruntime_pybind11_state.so` | 22.4MB | （平台无关 + pybind） | Python 绑定 |
| `libonnxruntime.so.1.22.2` | — | （核心 runtime） | 执行引擎 |

> **EP 选择**：DCU 上 `providers=['MIGraphXExecutionProvider', 'ROCMExecutionProvider', 'CPUExecutionProvider']` 或 `['ROCMExecutionProvider', ...]`。MIGraphX EP 做图级融合（整图编译），ROCm EP 做算子级（hipblas/MIOpen）。选型与调优见 `dcu-ort-perf`（ROCm/MIGraphX EP 选择节）。

## §3 版本与 DTK 绑定

- 源里 ORT 版本：`1.19.2+das.opt1.dtk2604.torch251/torch271`（cp312）+ `1.22.2+das.opt1.dtk2604.2607061906.gc3f2dd`（py3-none，最新）。
- **1.22.2 是 py3-none**（不绑 torch/cp 版本），1.19.2 绑 torch251/271（cp312）。装 1.22.2 更省事（无 torch ABI 约束）。
- DTK 绑定：`dtk2604`（DTK 26.04）。

## §4 与兄弟 skill 关系

- `dcu-ort-perf`：ORT 推理调优（perf_test 测量/批处理/FP16/EP 选择/图优化/多线程/io_binding/多实例/Triton Server 集成）→ 本 skill 是它的**包结构/适配落点**支撑。
- `dcu-ort-profile`：ORT Profile 机制 + MOE 显存修复 → 剖析时用。
- `dcu-migraphx-perf` / `dcu-migraphx-custom-op`：MIGraphX 调优/自定义算子 → ORT 的 MIGraphX EP 底层就是 MIGraphX。
- `dcu-dtk`：DTK 安装 → `libgalaxyhip.so.5` 等 DTK 库来自 DTK。

## §5 验证入口

```bash
# 安装（1.22.2 py3-none，无 torch 约束）
pip install onnxruntime -i https://pypi.sourcefind.cn/release/dtk2604/

# 确认 DCU EP 可用
python3 -c "import onnxruntime as ort; print(ort.get_available_providers())"
# 期望含 MIGraphXExecutionProvider / ROCMExecutionProvider

# 确认 ROCm EP 链接 DTK 库（DCU 容器内）
ldd $(python3 -c "import onnxruntime,os;print(os.path.dirname(onnxruntime.__file__))")/capi/libonnxruntime_providers_rocm.so | grep -E 'galaxyhip|hipblas|MIOpen'
# 期望: libgalaxyhip.so.5（不是 libamdhip64）

# 构建信息
python3 -c "import onnxruntime as ort; print(ort.get_build_info())"
```

## 使用须知

- ORT DCU 版是**二进制 wheel**，C++ 源码不在包内——要看 EP 源码需拉 ORT 上游 + DTK 构建配置（本 skill 只到 .so 链接级）。
- **DCU 用 `libgalaxyhip.so.5`**（DTK），不是 ROCm 的 `libamdhip64`——排查 EP 加载失败先确认 DTK 库路径（`dcu-dtk`）。
- **EP 选择**：MIGraphX EP（图级融合）vs ROCm EP（算子级），DCU 上 MIGraphX 常更快，选型见 `dcu-ort-perf`。
- 1.22.2 是 py3-none（无 torch 约束），1.19.2 绑 torch 版本——优先装 1.22.2。
- 不存性能数据/基线；数字用 `dcu-ort-perf` 模板留痕。
- 高危操作谨慎，优先测试机验证。
