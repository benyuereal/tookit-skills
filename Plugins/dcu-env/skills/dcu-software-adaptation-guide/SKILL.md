---
name: dcu-software-adaptation-guide
description: DCU 软件适配通用参考——拿到任意新软件包，按 5 大适配形态决策 + 7 步 SOP + DTK 核心差异 + 适配点定位技巧 + 验证清单完成适配（19 包实测归纳）
version: 1.1.1
author: DCU Engineer
license: MIT
kind: knowledge
---

# DCU 软件适配通用参考（dcu-software-adaptation-guide）

> 定位：**把任意软件包适配到 DCU 的通用方法论**。从 19 个实测包（2026-09-01）归纳，让其他 agent 拿到新软件时能按本 SOP 快速完成适配，不必从零摸索。
> 与 `dcu-cuda-hip-port` 的关系：那个讲 **CUDA 代码怎么移植到 HIP**（构建/hipify/API 差异/kernel 对拍），是**子步骤**；本 skill 讲**拿到一个软件包后整体怎么适配**（侦察→分类→定位→实现→构建→验证→打包→入库），`dcu-cuda-hip-port` 是其中"实现"步的一个分支。
> 与 10 个 source skill 的关系：那些是**具体包的源码地图**（vllm-hcu/triton/deep-ep 等）；本 skill 是**跨包的方法论**，source skill 是它的案例库。
> 适用：拿到一个上游软件包（pip 包/源码仓库），要让它跑在 DCU 上。

## 工作原则：容器优先（适配在容器里做，不在物理机上做）

> **推荐：软件适配直接用官方（光源）镜像作为底座起容器，在容器环境里完成适配；尽量不要在物理机上装软件栈做适配。**

- **底座镜像选择**：按目标"DTK × 框架 × Python 版本"选官方（光源）镜像——先查 `dcu-adaptation-overview` §1 定版本组合，再用 `dcu-image-hub` 查具体镜像/tag（如 `pytorch:2.9.0-ubuntu22.04-dtk26.04-py3.10` 这类 base 镜像）；官方无合适镜像时，退而在容器内用 DTK base 镜像 + pip install 自建环境。
- **物理机职责最小化**：只装驱动 + hyhal（`dcu-driver`），按 `dcu-docker`「DCU 镜像启动」模板起容器（透传 kfd/mkfd/dri + 挂 hyhal + `--shm-size=16G`）；DTK 容器自带，不在物理机上装（`dcu-dtk`）。
- **为什么不在物理机上适配**：
  1. **DTK 版本坑**：宿主机 `/opt/dtk` 常是旧版（如 <device-alias> 宿主 = DTK-25.04.4，不满足 BW 系列 DTK-26.04+ 要求）；把宿主 `/opt/dtk` 挂载覆盖容器自带 DTK 会直接编译/运行失败（gfx938 librccl 缺失、invalid device function，实测）。
  2. **适配环境即交付环境**：最终交付是容器化环境，容器内直接适配，产物 `docker commit` 固化成镜像即可交付（见 `dcu-docker`「DCU 镜像打包/导出与离线部署」）；物理机适配完还要搬进容器复验，做两遍。
  3. **不污染共享物理机**：宿主不装一堆包、不影响他人环境；做坏了重建容器即可，回滚成本低。
- **例外**：确需在物理机上适配时（驱动层问题、宿主侧工具链需求），先确认 DTK 版本与卡型系列匹配，并把结果 1:1 搬进容器环境复验后再交付。

## 0. 快速决策：这个包属于哪种适配形态？

拿到包先做 3 个判断，定位到 5 大形态之一（§1）：

```
Q1: 包里有 C++/CUDA 源码要编译吗？
├─ 否（纯 Python）→ 看 Q2
└─ 是 → 看 Q3

Q2: 纯 Python 里有平台分支（is_cuda/is_rocm/cuda 调用）吗？
├─ 有少量（<20 处）→ 形态 D（纯 Python 内容级适配）
└─ 无/极少 → 形态 D（几乎不改，构建对齐即可）

Q3: 是"框架本体"还是"框架的配套/插件"？
├─ 框架本体（vLLM/SGLang/Megatron 这种要跑服务的）→ 看 Q4
└─ 配套库（算子/通信/工具）→ 形态 A（二进制 wheel 重编译）

Q4: 上游版本迭代快、海光不想 fork 维护吗？
├─ 是（想保持上游干净、可回滚）→ 形态 B（插件 patch 层）
└─ 否（可以 fork 改源码）→ 形态 A（fork 重编译）
```

**编译器类**（triton/tilelang 这种自己带 codegen 的）特殊：形态 E（独立 backend），在编译器里加一个 DCU backend 目录。

## 1. 五大适配形态（19 包归纳）

### 形态 A：二进制 wheel 重编译（CUDA→HIP 移植）
**适用**：算子库/通信库/工具库，有 C++/CUDA 源码，编译成 .so。
**代表**：onnxruntime、apex、transformer-engine、triton、tilelang、deep-ep、mooncake、causal-conv1d、fast-hadamard、mhc-fused-kernel、hip-dsa、deepspeed（inference/v2 kernels）。
**核心步骤**：
1. 拿源码（sdist 或上游 git，**wheel 里通常无 csrc**——vLLM 0.21+/apex/TE 的 wheel 只有 .so，要看源码得拉上游）。
2. CUDA→HIP 移植（hipify + 手改，见 `dcu-cuda-hip-port`）。
3. 构建（hipcc 或 setup.py，见 §5）。
4. 链接 DTK 库（`libgalaxyhip.so` 等，见 §3）。
5. 验证（import + ldd + 数值对拍，见 §6）。
**关键点**：
- **wheel 里有没有源码决定工作量**：deepspeed/hip-dsa 的 wheel 带 .hip 源码（改后重编即可）；apex/TE/vLLM 的 wheel 只有 .so（要拉上游源码）。
- **DTK 库替换**是核心：`libamdhip64.so` → `libgalaxyhip.so`（见 §3.1）。
- **规避手段：FastPT 不转码编译**（`HYGON-AI/FastPT`，工单 <internal-ticket> 验证）：包内含需 nvcc 编译的 CUDA 代码、又不想改码时，`source /opt/dtk/cuda/env.sh` 走 CUDA 兼容编译；或用 `source /usr/local/bin/fastpt -C` 让 PyTorch 应用组件的 CUDA 源码**不转码**直接编译。优先于 hipify 手改（零代码变更、可回滚）。
- **torch 绑定**：编译型包要绑 torch 版本（`torch290`/`torch2100` 标记），torch ABI 不匹配会 `undefined symbol: c10::...`（见 §7.1）。

### 形态 B：插件 patch 层（不改上游，运行时 patch）
**适用**：框架本体（vLLM/Megatron），上游迭代快、海光不想 fork 维护、要可回滚。
**代表**：vllm-hcu（vLLM 0.21+）、dcu-megatron（Megatron）。
**核心步骤**：
1. 建独立 pip 包（`vllm_hcu`/`dcu_megatron`），装到框架旁。
2. 经框架的 **entry point 插件机制**钩入（vLLM 的 `vllm.platform_plugins`/`vllm.general_plugins`）。
3. 三种 patch 手段（按需组合）：
   - **字符串 patch**：运行时读框架模块源码，做 `(old, new)` 子串替换（vllm-hcu 105 个 `*.patch.py`）。
   - **模块交换**：`sys.modules` 里把 `vllm.xxx` 替换成 `vllm_hcu.xxx` 实现（vllm-hcu `MODULE_EXCHANGE_MAP`）。
   - **函数/类 patch**：替换函数引用（dcu-megatron `Patch` 类）。
4. 平台插件：注册 `HCUPlatform`（继承框架 `Platform`），接管设备/算子分发。
5. 唯一改框架安装目录处：post_install 软链个别文件（vllm-hcu 软链 `modular_kernel.py`）。
**关键点**：
- **可回滚**：卸载插件包 + 删软链即恢复上游。
- **字符串 patch 会静默失败**：框架升级后 `old` 子串匹配不到，不报错——升级后必须重跑 patch 看日志（见 §7.5）。
- **模块交换要回滚**：任一交换失败回滚已注册项，不半套生效（vllm-hcu `module_exchanges_all_ok`）。
- **平台插件是核心**：`HCUPlatform` 接管 `get_attn_backend_cls`/`device_count`/`import_kernels` 等，框架据此走 DCU 路径。

### 形态 C：运行时 dlopen 双路径（universal 二进制）
**适用**：工具库，想一个二进制同时支持 CUDA/DCU，不分别编译。
**代表**：fastsafetensors（`cpp/gpu_compat.h`）。
**核心步骤**：
1. 写 GPU 兼容层头文件（`gpu_compat.h`）：定义 CUDA/HIP 两套**库名 + 符号名**。
2. 不 include CUDA/HIP 头、不编译期链接 GPU runtime——运行时 `dlopen("libamdhip64.so")` + `dlsym("hipMemcpy")`。
3. 单一 universal 二进制，运行时检测平台选路径。
**关键点**：
- **DCU 上 dlopen 的是 `libamdhip64.so`**（ROCm 标准名），**不是** `libgalaxyhip.so`——依赖 DTK 提供 `libamdhip64.so` 兼容层/软链（见 §7.6）。这是 fastsafetensors 与 triton/ORT/DeepEP（直接链 libgalaxyhip）的**差异点**。
- 适合**轻量 GPU 调用**（memcpy/malloc 等），不适合重 kernel（kernel 还是要编译）。

### 形态 D：纯 Python 内容级适配（几乎不改）
**适用**：纯 Python 包，平台分支少或无。
**代表**：megatron-core（hip 提及仅 1 处且是注释误命中）、sglang-router（纯 CPU 路由，.so 无 DTK 依赖）。
**核心步骤**：
1. 确认无 C++/CUDA 源码（或 .so 是纯 CPU）。
2. grep 平台分支（`is_cuda`/`is_rocm`/`cuda`），少量则改，无则不改。
3. 构建对齐（torch 标记/Python 版本），打 wheel。
**关键点**：
- **适配量最小**，主要是"确认能跑 + 版本对齐"。
- **特化在配套包**：megatron-core 本体几乎无 DCU 代码，DCU 特化在配套的 dcu-megatron（形态 B）——**框架本体 + 配套 patch 层**是常见组合。

### 形态 E：独立 backend（编译器类）
**适用**：编译器/DSL（自己带 codegen），要加一个 DCU 后端。
**代表**：triton（`backends/hcu/` 独立 backend）、tilelang（`contrib/hcu.py` 海光后处理）。
**核心步骤**：
1. 从现有 backend（amd/rocm）fork 一个 DCU backend 目录（triton `backends/hcu/`）。
2. 加 DTK 特化（工具链/库/设备属性 API，见 §3）。
3. 注册 backend 选择机制（arch 命中即走 DCU backend，triton `_HCU_ARCHS`）。
4. 原 backend 对 DCU arch 让位（triton amd driver 注释 "Hygon deployments use triton.backends.hcu"）。
**关键点**：
- **DTK 工具链特化**是核心：`aillvm`（AI LLVM）+ `libgalaxyhip.so` + `hipGetDeviceProperties_v2`（见 §3）。
- **arch 判定**：DCU 的 gfx（gfx936/938 等）命中即走 DCU backend。
- **可加调优设施**：triton hcu backend 加了 `hcutune_cli`/`opt_config_cli`（amd backend 没有）。
- **科学计算语言 GPU 后端同理**：Julia（JuliaGPU）这类带自己 GPU 后端/代码生成的语言，适配 DCU 走"独立 backend"路线——为 DTK 写/改 GPU 后端，在 BW 系列验证。已有 DTK 26.04 的 JuliaGPU 初步适配方案可参考（见内部《DCU JuliaGPU DTK+DCU 适配方案》）。

## 2. 适配 SOP（7 步）

### Step 0：侦察（拿包 + 定形态）
```bash
# 1. 从源下 wheel（见 dcu-das-hub-api）
# 2. 解包看结构
unzip -l <pkg>.whl | head -50          # 看文件清单
unzip <pkg>.whl -d /tmp/<pkg>_x
# 3. 判断形态（§0 决策树）
find /tmp/<pkg>_x -name '*.cu' -o -name '*.cpp' -o -name '*.hip' | head   # 有源码？
find /tmp/<pkg>_x -name '*.so' | head                                    # 有编译产物？
cat /tmp/<pkg>_x/<pkg>-*.dist-info/entry_points.txt 2>/dev/null          # 有插件注册？
cat /tmp/<pkg>_x/<pkg>-*.dist-info/METADATA | head -30                   # 版本/依赖
```
**产出**：包形态（纯 Python/编译/二进制 wheel）+ 适配形态（A/B/C/D/E）+ 源码在不在 wheel 里。

### Step 1：分类（定适配策略）
按 §0 决策树定形态，按 §1 取对应策略。
**产出**：适配形态 + 核心步骤清单。

### Step 2：适配点定位（找哪里要改）
用 §4 的技巧（grep 平台分支 / ldd 依赖 / strings / entry_points）定位所有 DCU 适配点。
**产出**：适配点清单（文件:行号 / .so 依赖 / 插件注册点）。

### Step 3：适配实现（按形态）
- 形态 A：CUDA→HIP 移植（`dcu-cuda-hip-port`）+ DTK 库链接。
- 形态 B：建插件包 + entry point + patch（字符串/模块交换/函数 patch）+ 平台插件。
- 形态 C：写 gpu_compat.h 双路径 + dlopen。
- 形态 D：改少量平台分支（或不改）。
- 形态 E：fork backend 目录 + DTK 特化 + arch 注册。

### Step 4：构建（见 §5）
hipcc 或 setup.py，显式 `--offload-arch`，绑 torch 版本。

### Step 5：验证（见 §6）
import + ldd + 数值对拍 + e2e + torch ABI + 版本号。

### Step 6：打包与版本（见 §3.5）
按 das 版本号约定打 wheel：`<base>+das.<optN>.<dtk>.<torch>.<时间戳>.<commit>`。

### Step 7：入库（源 + skill）
- 上传 sourcefind 源（release/dtkXXXX 或 customer/dtkXXXX-rcX 仓库，见 `dcu-das-hub-api`）。
- 写 source skill（源码地图，六要素落点标准，见 `dcu-inference-infra-skill-builder/references/source-skill-landing-standard.md`——该内部 skill 未随本包分发，标准见内部版）。
- 注册到 `dcu-skill-router` 路由表 + `dcu-env-packages` 包清单。

## 3. 核心适配内容：DTK vs ROCm 差异（必背）

### 3.1 库替换（最核心）
| 用途 | ROCm | DTK（DCU） |
|---|---|---|
| HIP runtime 主库 | `libamdhip64.so` | **`libgalaxyhip.so`**（.so.5） |
| 设备属性 API | `hipGetDevicePropertiesR0600` | **`hipGetDeviceProperties_v2`** |
| 设备属性结构 | `hipDeviceProp_t` | **`hipDeviceProp_t_v2`** |
| LLVM 工具链 | `<rocm>/llvm` | **`<dtk>/aillvm`**（AI LLVM） |
| 安装路径 | `/opt/rocm` | `/opt/dtk`（候选含 /opt/rocm） |

> **几乎所有编译型包的 DCU 适配核心就是 `libamdhip64` → `libgalaxyhip` 替换**（triton/ORT/DeepEP/Mooncake/TileLang/apex/TE/causal-conv1d/fast-hadamard/mhc/hip-dsa 全链 libgalaxyhip.so.5）。
> **例外**：fastsafetensors 走 dlopen `libamdhip64.so`（依赖 DTK 兼容层，见 §7.6）。

### 3.2 gfx 架构差异
| gfx | VGPR | 备注 |
|---|---|---|
| gfx928 | 512 | K100AI 类 |
| **gfx936** | **768** | ★ BW100/BW150，DCU 主力，有 FP8 matrix core |
| **gfx938** | **768** | ★ 有 FP8 |
| gfx946 | 512 | |
| gfx92a | 512 | |

> gfx936/938 的 **768-vgprs** 是 DCU 相对 AMD CDNA 的关键差异（triton `_get_clang_args` 加 `-mllvm=-support-768-vgprs=true`）。写 kernel 时寄存器压力预算不同。
> **FP8**：仅 gfx90a/gfx936/938 有 matrix core；gfx906/908 无 FP8（见 `dcu-cuda-hip-port` §0）。

### 3.3 平台判定函数
不同框架命名不同，**别混用**：
| 框架 | DCU 判定 | 备注 |
|---|---|---|
| SGLang 0.5.12 | `is_dcu()` | 旧术语 |
| SGLang 0.5.15+ | `is_hcu()` | 新术语（DCU→HCU 演进） |
| vLLM | `HCUPlatform`（vllm-hcu 插件） | 平台插件 |
| triton | `_HCU_ARCHS`（arch 命中） | gfx926/928/936/938/92a/946 |
| tilelang | `contrib/hcu.py` | 海光后处理回调 |

> **版本演进**：海光把 DCU 术语演进为 **HCU**（sglang 0.5.15+、vllm-hcu、triton hcu backend）。写代码先确认目标版本术语，别把旧版 `is_dcu` 硬套到新版 `is_hcu`。

### 3.4 torch 绑定标记
编译型包绑 torch 版本，标记格式 `torch<版本>`：
| 标记 | torch 版本 |
|---|---|
| torch251 | 2.5.1 |
| torch271 | 2.7.1 |
| torch280 | 2.8.0 |
| torch290 | 2.9.0 |
| torch2100 | 2.10.0 |

> **torch 是基座，不绑自己**；其他包（triton/flash_attn/vllm 等）绑 torch 标记。**换 torch 大版本要整体重装一整套**，不能只升 torch（见 `dcu-env-packages` §5.6）。

### 3.5 das 版本号约定
海光发布轮版本字符串格式固定：
```
<包基础版本>+das.<optN>.<dtk标记>.<torch绑定?>.<时间戳>.<commit>
```
例：`vllm-0.15.1+das.opt1.alpha.dtk2604.torch290.2604081832.gbcb2ba`
- `das` = 海光 DAS 适配标识（区别于上游原生轮）
- `dtk<数字>` = 目标 DTK 版本（dtk25041/dtk2604）
- `torch<版本>` = 绑定的 torch（部分包不带，如 torch 自身）
- `opt<N>` = 优化/构建轮次
- 时间戳 + commit = 可追溯

> **装包时版本里必须有 `+das`**，没有就是上游原生轮（DCU 上跑不起来）。

## 4. 适配点定位技巧（新包怎么快速找哪里改）

| 技巧 | 命令 | 找什么 |
|---|---|---|
| **grep 平台分支** | `grep -rnE 'is_cuda\|is_rocm\|is_hip\|cudaMalloc\|rocm' <src> --include=*.py --include=*.cu --include=*.cpp` | 平台判定/分支（要加 DCU 分支的地方） |
| **ldd 看 .so 依赖** | `ldd <pkg>.so \| grep -E 'hip\|galaxy\|dtk\|not found'` | 链的 GPU 库（libgalaxyhip vs libamdhip64） |
| **strings 看 dlopen/错误串** | `strings <pkg>.so \| grep -E 'dlopen\|libamdhip64\|libgalaxyhip\|block size'` | 运行时 dlopen 目标 + 约束/报错字符串（Python 源码 grep 不到的） |
| **entry_points.txt** | `cat <pkg>-*.dist-info/entry_points.txt` | 插件注册点（形态 B 的钩子） |
| **RECORD** | `cat <pkg>-*.dist-info/RECORD` | 完整文件清单（wheel 里有什么） |
| **METADATA** | `cat <pkg>-*.dist-info/METADATA` | 版本/依赖/Python 约束 |
| **wheel vs sdist** | `unzip -l <pkg>.whl \| grep -E '\.cu\|\.cpp\|\.hip'` | 源码在不在 wheel 里（决定要不要拉上游） |
| **grep DTK 特化** | `grep -rnE 'galaxyhip\|aillvm\|dtk\|hygon\|hcu' <src>` | 已有的 DTK/海光特化点 |

> **关键**：约束/报错字符串常在 .so 里（如 flash-attn 的 "Paged KV cache block size must be 64"），Python 源码 grep 不到，必须 `strings <so>`（见 `dcu-sglang-source` 三·补）。

## 5. 构建方法

### 5.1 hipcc（CMake HIP 语言是坏的）
⚠️ DTK 26.04 的 CMake HIP 语言检测是坏的（`CMake can not determine linker language`）。**直接用 hipcc**：
```bash
/opt/dtk/bin/hipcc -O2 -std=c++20 --offload-arch=gfx936 main.cpp kernel.cu -o app
```
- **必须显式 `--offload-arch=gfxXXX`**——默认目标是 gfx928/906/926，不是你的卡。
- 大项目用构建脚本（bash/Makefile）调 hipcc，别硬修 CMake。
- 详见 `dcu-cuda-hip-port` §1。

### 5.2 setup.py bdist_wheel
```bash
PYTORCH_ROCM_ARCH=gfx936 python3 setup.py bdist_wheel
```
- 绑 torch 版本（编译时链接当前 torch 的 HIP 库）。
- 打 wheel 后按 §3.5 版本号约定命名。

### 5.3 构建产物检查
```bash
# 确认链了 DTK 库
ldd <built>.so | grep galaxyhip
# 确认无 CUDA 残留
nm -u <built>.so | grep -i cuda   # 应为空
```

## 6. 验证清单

| # | 验证项 | 命令/方法 | 通过标准 |
|---|---|---|---|
| 1 | **import 成功** | `python3 -c "import <pkg>"` | 无 ImportError |
| 2 | **DTK 库链接** | `ldd <pkg>.so \| grep galaxyhip` | 链 `libgalaxyhip.so.5`（非 libamdhip64，或兼容层） |
| 3 | **torch ABI 对齐** | import 编译型包 | 无 `undefined symbol: c10::...`（torch 标记匹配） |
| 4 | **数值对拍**（kernel 包） | 抽 kernel + CPU 参考 + 随机数据比 max err | max err < 容差（fp32 ~5e-3） |
| 5 | **e2e**（框架包） | 起服务跑真实模型 | 输出正确（与参考对齐） |
| 6 | **版本号合规** | `pip show <pkg>` | Version 含 `+das` + dtk 标记 + torch 标记 |
| 7 | **Python 版本** | wheel 文件名 cpXXX | 匹配目标 Python（cp310/311/312） |
| 8 | **patch 应用**（形态 B） | 启动框架看日志 | "patches applied successfully"，无 "Failed to apply" |

> **别只信编译通过**——kernel 必须数值对拍（`dcu-cuda-hip-port` §4）；框架必须 e2e（起服务跑真实模型）。

## 7. 踩坑（19 包实测）

### 7.1 torch ABI 不匹配（编译型包最常见）
装了"为 torch X 编译"的轮到 torch Y 环境，import 报 `undefined symbol: _ZN3c10...`（如 `c10::MessageLogger::stream` 是 torch 2.10 才有）。**根因是 torch 大版本不匹配，不是包损坏**。诊断：`nm -u <kernel>.so \| grep c10`（kernel 需要哪些符号）+ `nm -D <torch>/lib/libc10.so \| grep <符号>`（环境 torch 有没有）。规避：torch 标记必须和环境 torch 大版本一致（见 `dcu-das-hub-api` §5）。

### 7.2 CMake HIP 语言是坏的
DTK 26.04 的 CMake `project(... LANGUAGES HIP)` 报 `can not determine linker language`，`CMAKE_HIP_COMPILER`/`CXX` 都无效。**直接 hipcc 构建**（§5.1）。

### 7.3 hipcc 默认 offload 不是你的卡
hipcc 默认 offload 目标是 gfx928/906/926，**必须显式 `--offload-arch=gfxXXX`**，否则编出来的 .so 在你的卡上跑不了。

### 7.4 warp shuffle 无 mask
DTK 提供 `__shfl`/`__shfl_xor` 等（无 mask 版），**没有** `__shfl_*_sync`。用 shim 头映射，**别加 `__HIP_DEVICE_COMPILE__` 守卫**（host pass 会报未声明）。见 `dcu-cuda-hip-port` §3。

### 7.5 字符串 patch 静默失败（形态 B）
vllm-hcu 的字符串 patch 在框架升级后 `old` 子串匹配不到时**不报错**（静默跳过）。升级框架后必须重跑 `vllm-hcu-apply-patches` 看日志有无 "Failed to apply"。模块交换则相反——**任一失败回滚**（不半套生效）。

### 7.6 dlopen libamdhip64 兼容层（形态 C）
fastsafetensors 的 gpu_compat.h dlopen 的是 `libamdhip64.so`（ROCm 标准名），**不是** `libgalaxyhip.so`。DCU 上依赖 DTK 提供 `libamdhip64.so` 兼容层/软链——若 dlopen 失败，检查 DTK 是否提供该符号链接（`dcu-dtk-layout`）。这是形态 C 与形态 A（直接链 libgalaxyhip）的差异点。

### 7.7 kernel cache 跳过 lowering（tilelang）
tilelang 的 kernel cache 命中会跳过 lowering，`TILELANG_OVERRIDE_DEVICE_SOURCE` 不生效。强制重跑设 `TILELANG_DISABLE_CACHE=1` 或清 cache。

### 7.8 rc4 vs release 仓库
部分包只在 `customer/dtk2604-rc4` 仓库（causal-conv1d/fast-hadamard/mhc/hip-dsa/cupy/mori 等 14 包），release/dtk2604 主索引没有。装包用 rc4 仓库（`dcu-das-hub-api` 跨仓库枚举）。

### 7.9 py3-none vs cpXXX
部分包是 py3-none wheel（不绑 Python/torch，如 onnxruntime 1.22.2），部分是 cpXXX（绑 Python 版本，如 apex cp312/hipdnn cp311）。装前确认目标 Python 版本匹配。

### 7.10 框架本体 + 配套 patch 层组合
megatron-core 本体几乎无 DCU 代码（形态 D），DCU 特化在配套的 dcu-megatron（形态 B）。**框架本体 + 配套 patch 层**是常见组合——适配框架时别只改本体，要看有没有配套 patch 层。

### 7.11 形态 B 变体：运行期查表注入（env-gated best_config）
**不改算子数学**，只在调用点按 `(M,K,N)` 查表传最佳启动配置（num_stages/num_warps/BK/BN/SPLIT_K）；env 门控 + 未列回退 stock（fail-closed）。这是 DCU 上**零精度风险**的性能适配范式（改动是启动配置而非数值路径/布局），适于 GEMM / attention kernel 的启动参数调优（K100-AI SGLang w8a8_int8 + GDN chunked 实测：4K c1 TTFT −16%、64K c1 −14%、吞吐 +7.9%，TPOT ~−8%）。

### 7.12 字符串 patch `|| true` 会吞失败
`patch ... || true` 在补丁上下文漂移/匹配不到时**不报错**（静默跳过），比 §7.5 更隐蔽。补丁必须用**运行时日志（如 ACTIVE 打印）**证明生效，否则静默 no-op。

## 8. 上游 GPU 库部署验证速查（2026-10 实测：opencv/faiss/spconv/TE/LMCache/FastPT 六包）

> 场景：**已适配好的 das 构建装进容器跑通**（不是移植，是部署验证）。<device-A>（BW150×8）+ sglang0.5.18-dtk2604 容器基线 torch 2.11 实测。

### 8.1 前置条件：LD_LIBRARY_PATH 挂 DTK 兼容层
上游 GPU 库（opencv/faiss/spconv/TE 等）的 .so 链 `libcudart.so.12`、`librocm_smi64.so.2` 等，DCU 上由 DTK 兼容层提供。**不挂会 import 即挂**。关键路径（DTK26.04 容器实测）：
- `libcudart.so.12` → `/opt/dtk/cuda/cuda-12/targets/x86_64-linux/lib`（opencv 必挂）
- `libgalaxyhip.so.5` → `/opt/dtk/hip/lib`；`librocm_smi64.so.2` → `/opt/hyhal/lib`（挂载点）
- **最稳做法：`source /opt/dtk/env.sh` 后不要手动覆盖 `LD_LIBRARY_PATH`**——手动 export 会把 env.sh 自带路径顶掉，反而弄坏 torch（实测：手动 export 后 torch import 报 `librocm_smi64.so.2 not found`）。
- 验证脚本放 `/root/` 等干净目录跑，**文件名别和包同名**（`spconv.py` 会 shadow 掉 spconv 包 → circular import）。

### 8.2 DTK 真实版本核实（镜像 tag 会骗人）
镜像 tag 写 `dtk2604` 不等于容器内 DTK 是 26.04——实测 sglang0.5.18-`dtk2604` 容器内 `/opt/dtk/lib/libhipblaslt.so.0.10`（26.10 系）。**排查符号缺失先核实真实版本**：`ls /opt/dtk/lib/libhipblaslt.so.*` + `nm -D <lib> | grep <符号>`，别信 tag。

### 8.3 torch 版本矩阵（nightly/dtk 源，2026-10-14 实测）
- 源上有 2.5.1/2.6.0/2.7.1/2.8.0/2.9.0/2.10.0/2.11.0/**2.12.0** 全版本 DCU wheel（全 dtk2604 标记，cp310）；**没有 dtk26.10 的 torch**，光源也没有 dtk26.10 base 镜像（最新 base = torch 2.11 dtk26.04）。
- 容器 torch 升降级：`pip install --no-deps -i https://pypi.sourcefind.cn/nightly/dtk/ torch==<ver>+dtk2604...`（wheel 大，~10 分钟；`--no-deps` 防连带）。
- **torch 升降级后，所有 torch 绑定包（TE/flash-attn/fastpt 等）必须同步换对应 `torchXXX` 构建**，否则 `undefined symbol: c10::...`（§7.1 的延伸：不只是环境错配，主动换 torch 同样触发）。

### 8.4 TransformerEngine-das 铁三角（实测 `TE_GPU_PASS`）
**TE 2.13.0 + torch 2.10.0 + flash-attn 2.8.4(torch2100)** 组合跑通（Linear forward）。
- TE **2.14.1** 不行：要 `hipblaslt_ext::GemmEpilogue` **V1 构造器**符号，只在比容器 DTK 更新的新 DTK-26.10 里（单库前置宿主 26.10 hipblaslt 会因 `liborigami.so.1` 等运行时依赖缺而失败——DTK 库是整套耦合，别单库混用）。
- TE **2.13.0** 要 `GemmEpilogueV2` 系列符号，容器 DTK（26.10 系）8/8 全满足。
- 残留小瑕疵：TE 2.13.0 声明支持 flash-attn 2.1.1~2.8.3，装 2.8.4 打一条 warning，不影响跑通。

### 8.5 FastPT：入口按 torch 精确版本门控
`source /usr/local/bin/fastpt -C` 会检查 torch **精确大版本**（`2.12.0+dtk2604.torch2120` 构建只认 2.12），不匹配直接 `ERROR: Torch version is not supported`。解法 = 按 wheel 标记升/降容器 torch（§8.3），不是换 fastpt 版本。

### 8.6 各包 API 形态（老文档会误导）
- **faiss-das**：GPU 索引是新形态 `GpuIndex*`/`GpuIndexFlatConfig`（经典 `IndexFlatGpu` 直接构造会 TypeError）；稳路径 `faiss.index_cpu_to_gpu(res, 0, cpu_idx)`。`StandardGpuResources` 可用，self-search dist=0 验证。
- **spconv-das**：API 挪到 **`spconv.pytorch`** 子包（顶层 import `SparseConvTensor` 不到）；coordinates 要 **4 列** `[batch, x, y, z]`。
- **opencv-das**：产物名是 `opencv-python`（4.9.0+das.opt1.dtk2604，abi3），GPU 扩展在（`cv2.cuda.getCudaEnabledDeviceCount()`=卡数）；**别和容器自带 `opencv-python-headless` 混装**（同装会导致 `cv2` import 出残缺模块）。
- **LMCache-das**：pypi 源最新构建是 **0.3.13**（torch2100），import 通过；注意与 GitHub `HYGON-AI/LMCache-das` 锚点（0.4.2）**不同步**，部署以源上实际构建为准。

## 9. 交叉引用

| 场景 | 用哪个 skill |
|---|---|
| CUDA 代码怎么移植到 HIP（构建/hipify/API 差异/对拍） | `dcu-cuda-hip-port` |
| 从零写 HIP/TileLang kernel | `dcu-kernel-dev` |
| 查源里某包版本/下 wheel/跨仓库枚举 | `dcu-das-hub-api` |
| 查哪些包已适配/装哪个版本 | `dcu-env-packages` |
| 适配用哪个官方底座镜像 / 镜像与 tag 查询 | `dcu-image-hub` |
| DTK 安装/布局/工具链路径 | `dcu-dtk` / `dcu-dtk-layout` |
| 具体包的源码地图（案例库） | 10 个 source skill：`dcu-vllm-hcu-source` / `dcu-triton-source` / `dcu-ort-source` / `dcu-deepep-source` / `dcu-mooncake-source` / `dcu-fastsafetensors-source` / `dcu-tilelang-source` / `dcu-mhc-kernels-source` / `dcu-hipdsa-source` / `dcu-training-frameworks-source` |
| 写 source skill 的落点标准 | `dcu-inference-infra-skill-builder/references/source-skill-landing-standard.md`（内部 skill，未随本包分发） |
| 算子优化闭环 | `dcu-op-optimization` |

## 使用须知

- 本 skill 是**方法论**，不含具体包的源码地图（那是 10 个 source skill 的事）。拿到新包：先 §0 定形态 → §2 走 SOP → §3 背核心差异 → §4 定位适配点 → §5 构建 → §6 验证 → §7 避坑。
- **DTK 库替换（libamdhip64→libgalaxyhip）是绝大多数编译型包的核心**，先确认这个。
- **torch 标记必须对齐**，否则 ABI 不匹配（§7.1）。
- **别只信编译通过**——kernel 数值对拍、框架 e2e（§6）。
- **上游 GPU 库部署验证**（已适配 das 构建装容器跑通）：§8 速查——兼容层 LD_LIBRARY_PATH、DTK 真实版本核实、torch 矩阵、TE 铁三角、FastPT 版本门、faiss/spconv 新 API。
- **形态 B（插件 patch）的字符串 patch 会静默失败**，升级框架后重跑看日志（§7.5）。
- 适配完要**写 source skill + 注册路由表 + 更新包清单**（Step 7），沉淀给后续 agent。
- 不存性能数据/基线；数字用 `dcu-op-optimization` 模板留痕。
- 高危操作谨慎，优先测试机验证。
