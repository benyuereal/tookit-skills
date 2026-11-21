---
name: dcu-fastsafetensors-source
description: fastsafetensors DCU 适配源码结构——safetensors 并行加载（gpu_compat.h 运行时 dlopen CUDA/HIP 双路径 + GDS/ThreeFS copier）
version: 1.0.0
author: DCU Engineer
license: MIT
kind: knowledge
---

# fastsafetensors DCU 适配源码索引（dcu-fastsafetensors-source）

> 定位：**fastsafetensors 是 safetensors 权重的并行/异步加载库**（多进程 + GPU 直读，加速大模型权重加载），海光版支持 DCU（HIP）。本 skill 讲 fastsafetensors 在 DCU 上的包结构、GPU 兼容层、copier 后端。
> 与 `dcu-vllm-source` / `dcu-sglang-source` 的关系：vLLM/SGLang 加载权重时可走 fastsafetensors（`--load-format` 或 safetensors 并行加载路径），本 skill 是**权重加载层**的 DCU 适配。
> 实测来源：`fastsafetensors-0.3.2+das185.dtk2604.torch2100.2608121445.g499e61-cp310` wheel（release/dtk2604 源），解包 34 文件 + strings 实测。

## §0 定位：fastsafetensors 是什么

fastsafetensors = safetensors 的**高性能加载器**：
- **并行加载**：多进程分片读 + GPU 直读（GDS/ThreeFS），绕过 CPU 中转。
- **异步**：`parallel_loader.py` 多进程 + `file_buffer.py` 文件缓冲。
- **框架适配**：`frameworks/_torch.py`（PyTorch）/ `_paddle.py`（Paddle）。

**DCU 适配关键点**：`cpp/gpu_compat.h` 是 **CUDA/HIP 双路径兼容层**——所有 GPU 函数**运行时 dlopen/dlsym**（不编译期链接 CUDA/HIP 头），单一 universal 二进制运行时检测平台。DCU 上走 HIP 路径（`libamdhip64.so` 符号）。

## §1 源码地图（wheel 实测目录）

```
fastsafetensors/
├── __init__.py
├── auto_loader.py               # 自动加载器（入口）
├── loader.py                    # 基础 loader
├── parallel_loader.py           # ★ 并行加载器（多进程）
├── file_buffer.py               # 文件缓冲
├── common.py / config.py / st_types.py / dlpack.py / tensor_factory.py
├── threefs_loader.py            # ThreeFS 加载器
├── cpp.cpython-310-x86_64-linux-gnu.so   # ★ C++ 扩展（运行时 dlopen GPU 库）
├── cpp/                         # ★ C++ 源码（wheel 内带，可看）
│   ├── gpu_compat.h             # ★★ CUDA/HIP 双路径兼容层（dlopen 符号表）
│   ├── dstorage.h / dstorageerr.h
│   ├── ext.cpp / ext.hpp        # pybind 扩展
├── copier/                      # ★ 拷贝后端（多策略）
│   ├── base.py / registry.py / unified.py
│   ├── gds.py                   # GDS（GPU Direct Storage）
│   ├── nogds.py                 # 无 GDS 回退
│   ├── threefs.py               # ThreeFS
│   ├── dstorage.py              # DStorage
│   └── example_copier.py
└── frameworks/
    ├── _torch.py                # PyTorch 框架适配
    └── _paddle.py               # Paddle 框架适配
```

## §2 GPU 兼容层（cpp/gpu_compat.h，DCU 适配核心）

`gpu_compat.h` 定义 CUDA/HIP 两套**库名 + 符号名**，运行时 dlopen 选择：

| 平台 | runtime 库 | 关键符号（dlsym） |
|---|---|---|
| CUDA | `libcudart.so` | `cudaGetDeviceCount` / `cudaMemcpy` / `cudaMemcpyAsync` / `cudaDeviceSynchronize` / `cudaHostAlloc` / `cudaFreeHost` / `cudaDeviceGetPCIBusId` / `cudaMalloc` / `cudaFree` / `cudaDriverGetVersion` / `cudaDeviceGetAttribute` / `cudaSetDevice` |
| **HIP（DCU）** | **`libamdhip64.so`** | `hipGetDeviceCount` / `hipMemcpy` / `hipMemcpyAsync` / `hipDeviceSynchronize` / `hipHostMalloc` / `hipHostFree` / `hipDeviceGetPCIBusId` / `hipMalloc` / `hipFree` / `hipDriverGetVersion` / `hipDeviceGetAttribute` / `hipSetDevice` |

**设计**：不 include CUDA/HIP 头、不编译期链接 GPU runtime——`cpp.so` 里 `dlopen("libamdhip64.so")` + `dlsym("hipMemcpy")` 等（strings 实测 `cpp.so` 含 `dlopen`/`hipGetDeviceCount`/`libamdhip64.so` 字符串）。

> ⚠ **DCU 注意**：gpu_compat.h 的 HIP 路径 dlopen 的是 **`libamdhip64.so`**（ROCm 标准名），**不是** DTK 的 `libgalaxyhip.so`。DCU 上 `libamdhip64.so` 通常由 DTK 提供兼容层（或软链到 libgalaxyhip）——若 dlopen 失败，检查 DTK 是否提供 `libamdhip64.so` 符号链接（见 `dcu-dtk-layout`）。这是 fastsafetensors 与 triton/ORT/DeepEP（直接链 libgalaxyhip）的**差异点**。

## §3 copier 后端（copier/）

| copier | 文件 | 说明 |
|---|---|---|
| GDS | `copier/gds.py` | GPU Direct Storage（NVMe→GPU 直读，需 cuFile/对应 HIP 支持） |
| 无 GDS | `copier/nogds.py` | 回退路径（CPU 中转） |
| ThreeFS | `copier/threefs.py` | ThreeFS 分布式文件系统 |
| DStorage | `copier/dstorage.py` | DStorage |
| 统一 | `copier/unified.py` | 统一接口（按环境选后端） |
| 注册 | `copier/registry.py` | copier 注册表 |

> DCU 上 GDS 路径依赖 DTK 的 GPU Direct Storage 支持；不支持时自动回退 `nogds`（CPU 中转，慢但可用）。

## §4 与 vLLM/SGLang 的衔接

- vLLM/SGLang 加载 safetensors 权重时可走 fastsafetensors 并行加载（多进程 + GPU 直读），加速大模型启动。
- 权重加载慢（启动时间长）时，确认是否启用 fastsafetensors 并行加载 + GDS 路径。

## §5 与兄弟 skill 关系

- `dcu-vllm-source` / `dcu-vllm-hcu-source` / `dcu-sglang-source`：推理框架 → fastsafetensors 是权重加载层。
- `dcu-model-deploy-verify`：模型部署验证 → 权重加载是部署第一步。
- `dcu-dtk` / `dcu-dtk-layout`：DTK 布局 → `libamdhip64.so` 兼容层来自 DTK。
- `dcu-inference-infra`：推理 infra 总入口。

## §6 验证入口

```bash
# 安装（torch 标记对齐，本版 torch2100）
pip install fastsafetensors -i https://pypi.sourcefind.cn/release/dtk2604/

# 确认 GPU 兼容层（DCU 容器内，看 dlopen 的库）
python3 -c "
import fastsafetensors
from fastsafetensors import parallel_loader
print('fastsafetensors OK')
"

# 确认 cpp.so 的 GPU 库依赖（strings 看 dlopen 目标）
strings $(python3 -c "import fastsafetensors,os;print(os.path.dirname(fastsafetensors.__file__))")/cpp*.so | grep -E 'libamdhip64|libcudart|hipMemcpy'
```

## 使用须知

- fastsafetensors 是**权重加载库**，不是推理引擎——加速模型启动，不影响推理性能。
- **GPU 兼容层走 dlopen**：DCU 上 dlopen `libamdhip64.so`（非 libgalaxyhip）——若加载失败检查 DTK 的 libamdhip64 兼容层。
- **GDS 路径**：DCU 上 GDS 依赖 DTK 支持，不支持时回退 nogds（CPU 中转）。
- **torch 标记对齐**：本版绑 torch2100。
- 不存性能数据/基线；数字用 `dcu-op-optimization` 模板留痕。
- 高危操作谨慎，优先测试机验证。
