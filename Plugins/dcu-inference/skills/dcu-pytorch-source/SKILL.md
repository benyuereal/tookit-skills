---
name: dcu-pytorch-source
description: DCU源码编译低版本PyTorch与LibTorch提取
version: 1.0.1
author: DCU Engineer
license: MIT
kind: tool
---

# DCU PyTorch 源码编译适配与算子验证（dcu-pytorch-source）

> 来源：内部 DCU 实践，已脱敏。本 skill 将其提炼为可操作的标准动作。
> 验证环境：Kylin V10 SP2 / GCC 7.3.0 / CMake 3.16.3 / DTK 26.04 / Python 3.10 / LibTorch 2.1.0 / C++ ABI=1 / 加速卡 BW100(gfx936) / 基础镜像 `<internal_registry>/dcu/admin/base/pytorch:2.5.1-kylin10sp3-dtk26.04-py3.10`。
> 注：BW100 代号对应 gfx936；其他卡先用 `rocminfo` 确认 `gcn_arch` 再改 `PYTORCH_ROCM_ARCH`（见 `dcu-env`）。

## 核心目标
在高版本 DTK（26.04，ROCm/HIP 6.x）上，**源码编译**低版本 PyTorch（2.1.0），**提取**独立可部署的 LibTorch C++ 运行时（`.so`+`include`），并做 **17 类 196 项算子 CPU vs DCU 交叉精度验证**，为模型迁移提供可靠性依据。

## 三阶段总览
1. **源码编译 → Python wheel**：在 DTK 26.04 上编译出 `torch-2.1.0+...-cp310-cp310-linux_x86_64.whl`。
2. **提取 C++ LibTorch 运行时**：从编译产物收集 `.so`+`include`，patchelf 修正 RPATH 为 `$ORIGIN`。
3. **算子兼容性验证**：C++ demo + 196 项全量测试（`op_test.cpp`），CPU 参考 vs DCU 结果 max_diff 容差判定。

---

## 阶段一：源码编译 Python wheel

### 1.1 镜像与依赖
```bash
docker pull <internal_registry>/dcu/admin/base/pytorch:2.5.1-kylin10sp3-dtk26.04-py3.10
# 进容器后：
pip install -U pip -i https://pypi.tuna.tsinghua.edu.cn/simple
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
python -c "import astunparse, yaml, numpy; print(numpy.__version__)"
cmake --version && ninja --version
```

### 1.2 环境变量（DTK/HIP 工具链 + 构建开关）
直接用脚本：`bash scripts/env_setup.sh`（源码见下）。关键点：
- `USE_ROCM=1`、`USE_CUDA=0`、`USE_NINJA=1`、`BUILD_NVFUSER=0`（**必须关，否则编译失败**）、`USE_CUSPARSELT=0`
- `PYTORCH_ROCM_ARCH=gfx936`（BW100）、`_GLIBCXX_USE_CXX11_ABI=1`
- `PYTORCH_BUILD_VERSION=2.1.0`、`PYTORCH_BUILD_NUMBER=11`

### 1.3 关键步骤：修正 version-dev（不修 CMake 阶段必败）
DTK 26.04 的 `/opt/dtk/.info/version-dev` 默认仅 `26.04`，不满足 `LoadHIP.cmake` 的 `x.y.z` 正则。
```bash
grep ROCM_VERSION_ /opt/dtk/include/rocm_version.h   # 例: MAJOR=6 MINOR=3 PATCH=3
cp -a /opt/dtk/.info/version-dev /opt/dtk/.info/version-dev.bak
echo "6.3.3-26.04" > /opt/dtk/.info/version-dev        # 前三位对齐 rocm_version.h
```
→ 脚本化：`bash scripts/00_fix_version_dev.sh`

### 1.4 源码获取与 HIP 化
```bash
cd /path/to/pytorch
git checkout <your-2.1.0-das-branch>      # 海光 DAS 分支/Tag，以仓库实际为准
git submodule sync --recursive && git submodule update --init --recursive
python tools/amd_build/build_amd.py       # HIP 化；'skipped, already hipified' 正常
```

### 1.5 编译执行（增量，严禁 clean）
```bash
mkdir -p /path/to/pytorch-wheels
python setup.py bdist_wheel -d /path/to/pytorch-wheels 2>&1 | tee /path/to/pytorch-build.log
# 成功产物: torch-2.1.0+das.opt1.dtk2604-cp310-cp310-linux_x86_64.whl
```
⚠ **增量编译铁律**：遇到编译错误，只改相关源文件后**重跑 setup.py bdist_wheel**，ninja 自动增量重编；**绝对不要**执行 `setup.py clean`（否则 2-4h 全量重编）。

### 1.6 四个典型编译坑及修复

| # | 现象 | 根因 | 修复 | 涉及文件 |
|---|------|------|------|----------|
| 一 | CMake 报 ROCm 版本异常 / HIP 组件检测失败 | version-dev 内容 `26.04` 无法匹配 `x.y.z` 正则 | `echo "6.3.3-26.04" > /opt/dtk/.info/version-dev` | `cmake/public/LoadHIP.cmake` |
| 二 | `error: cannot initialize 'int*' with 'uint32_t[3]'` | DTK26 设备属性 `maxThreadsDim/maxGridSize` 类型变 `uint32_t[3]`，源码仍按 `int*` | 把这几处声明改 `auto` | `aten/.../hip/UpSampleBilinear2d.hip`、`UpSampleNearest2d.hip`、`AdaptiveAveragePooling.hip`、`DilatedMaxPool2d.hip` |
| 三 | `could not convert 'hipsparseDestroyDnMat' from const to non-const` | DTK26/ROCm6.x hipsparse 已改 const 签名，旧宏在 `ROCM>=50200` 打开非 const 分支 | `ROCM_VERSION>=60000` 启用 `AT_USE_CUSPARSE_CONST_DESCRIPTORS`；`AT_USE_HIPSPARSE_GENERIC_52_API` 仅限 `50200≤v<60000` | `aten/src/ATen/hip/HIPSparse.h` 等 |
| 四 | `nvfuser/executor_utils.cpp: 'args' was not declared` | `USE_ROCM` 分支变量声明顺序错误 | 直接关 nvfuser（LibTorch 不依赖） | `BUILD_NVFUSER=0` 或 CMakeCache `BUILD_NVFUSER:BOOL=OFF` |

修复脚本（幂等）：
- 坑二：`bash scripts/patches/patch_maxThreadsDim.sh /path/to/pytorch`
- 坑三：`bash scripts/patches/patch_hipsparse_h.sh /path/to/pytorch`（向 `HIPSparse.h` 追加正确宏分支）
- 坑四：`bash scripts/patches/disable_nvfuser.sh [pytorch目录]`

> 所有修复后**不要** `setup.py clean`，直接重跑 `02_build_wheel.sh` 增量续编。

---

## 阶段二：提取独立 LibTorch C++ 运行时
编译成功只是第一步；对 LibTorch C++ 推理场景，需从产物提取 `.so`+`include` 并修正 RPATH 使可移植。

### 2.1 收集 so 库
```bash
PREFIX=/path/to/torch-2.1.0
rm -rf "$PREFIX"; mkdir -p "$PREFIX/lib"
cp -a /path/to/pytorch/torch/lib/*.so* "$PREFIX/lib/"
ls -lh "$PREFIX/lib/" | grep -E "libtorch|libc10|libtorch_hip"
```
关键库：

| 库文件 | 功能 |
|--------|------|
| `libtorch.so` | LibTorch 主库（聚合所有 Torch C++ API） |
| `libtorch_cpu.so` | CPU 后端算子 |
| `libtorch_hip.so` | DCU 后端算子（核心） |
| `libc10.so` | C10 核心基础库（Tensor 结构） |
| `libc10_hip.so` | C10 的 HIP 后端 |
| `libtorch_global_deps.so` | 全局依赖聚合 |

### 2.2 收集 include 头文件（必须完整目录结构）
```bash
cp -a /path/to/pytorch/torch/include/. "$PREFIX/include/"
# 校验: at/ c10/ torch/ torch/csrc/api/include/ 均存在
```
⚠ 缺任何路径都会让下游 C++ `#include` 失败，用 `cp -a` 保留符号链接，别手挑文件。

### 2.3 RPATH 修正（决定可移植性）
编译产物 `.so` 默认 RPATH 指向编译时绝对路径，部署到其他环境会 `libc10.so not found`。修正为 `$ORIGIN`：
```bash
for so in "$PREFIX"/lib/*.so; do
  [ -f "$so" ] || continue
  patchelf --set-rpath '$ORIGIN' "$so"   # 原文此处被截断，已修复
done
patchelf --print-rpath "$PREFIX/lib/libtorch.so"   # 应输出 $ORIGIN
```
> 未装 patchelf：`pip install patchelf` 或 `yum install patchelf`。

### 2.4 预期产物目录
```
/path/to/torch-2.1.0/
├── BUILD_INFO.txt
├── include/  (ATen/ c10/ torch/ torch/csrc/api/include/ ...)
└── lib/      (libtorch.so libtorch_cpu.so libtorch_hip.so libc10.so c10_hip.so ...)
```
→ 一键执行：`bash scripts/collect_libtorch.sh /path/to/pytorch /path/to/torch-2.1.0`

---

## 阶段三：算子兼容性验证（17 类 196 项）

### 3.1 验证原理（同输入 → 双后端 → 交叉对比）
每个用例用相同随机种子（CPU 直接生成，`to(kCUDA)` 拷贝保证数值一致）：
1. 生成标准输入 Tensor（同种子）
2. CPU 执行（`torch::kCPU`） / DCU 执行（`torch::kCUDA`）
3. DCU 结果 `.to(torch::kCPU)` 拷回
4. 计算 `max_diff = max|DCU - CPU|`；整型/Bool 用 `torch::equal()` 精确匹配，浮点用分层容差
5. `max_diff ≤ atol` → PASS，否则 FAIL

### 3.2 精度判定标准（分层容差）

| 数据类型 | 绝对容差 atol | 适用范围 |
|----------|---------------|----------|
| float64 | 1e-8 | 高精度线性代数（矩阵求逆、伪逆） |
| float32（常规） | 1e-5 | 算术、激活、逐元素数学 |
| float32（规约） | 1e-3 ~ 1e-2 | sum/mean/var 累加、大矩阵乘法 |
| float16 | 1e-2 | 半精度 |
| int32/int64/bool | 0（精确） | 相等比较 |

### 3.3 算子覆盖范围（17 类 196 项）

| 类别 | 数量 | 部分列举 |
|------|------|----------|
| Tensor 创建与属性 | 9 | zeros, ones, arange, linspace, eye, rand, randn, full, empty+fill_ |
| Dtype 类型转换 | 5 | float32→float64/float16/int32/int64/bool |
| 基本算术运算 | 16 | add, sub, mul, div, neg, abs, floor_divide, fmod, clamp, sign, round |
| 矩阵运算 | 13 | matmul(2D/3D/4D), bmm, addmm, addmv, dot, outer, transpose, inverse, pinverse |
| 激活函数 | 21 | relu, leaky_relu, elu, selu, celu, gelu, silu, sigmoid, tanh, softmax, prelu |
| 逐元素数学 | 28 | exp, log, sqrt, pow, sin, cos, tan, erf, erfc, erfinv, frac, ceil, floor |
| 规约操作 | 22 | sum, mean, std, var, prod, max, min, argmax, argmin, logsumexp, norm, cumsum |
| 比较操作 | 12 | eq, ne, gt, lt, ge, le, isnan, isinf, isfinite, maximum, minimum, where |
| 形状与视图 | 15 | reshape, view, unsqueeze, squeeze, flatten, cat, stack, split, chunk, repeat |
| 索引与切片 | 10 | index_select, slice, narrow, gather, scatter_, masked_select, nonzero, take |
| 卷积与池化 | 11 | conv2d(4种), max_pool2d, avg_pool2d, adaptive_avg/max_pool2d |
| 归一化 | 7 | batch_norm, layer_norm(3种), group_norm, instance_norm, normalize |
| 损失函数 | 7 | mse_loss, l1_loss, cross_entropy, nll_loss, bce, smooth_l1_loss, kl_div |
| 线性代数 | 8 | cross, trace, diag, diagonal, tril, triu, matrix_rank, matrix_power |
| 插值与上采样 | 2 | interpolate(nearest), interpolate(bilinear) |
| 其他 | 6 | topk, sort, kthvalue, unique, one_hot, multinomial |
| 自动微分 | 3 | backward(simple), backward(matmul), backward(relu) |
| **合计** | **196** | — |

### 3.4 工程构建（CMakeLists.txt）
见 `scripts/CMakeLists.txt`（CMake 3.16.3 + C++17 + `_GLIBCXX_USE_CXX11_ABI=1`）。
- `TORCH_ROOT` 指向提取出的 `torch-2.1.0`；`HIP_ROOT` 指向 `/opt/dtk-26.04`（部署时按实际改）
- 两个 target：`demo`（冒烟测试）、`op_test`（196 项全量测试）
- 链接库：`torch torch_cpu torch_hip c10 c10_hip pthread dl rt`
- 原文 `${HIP_LIBDIR}` 被 Markdown 吞成 `{HIP_LIBDIR}`，脚本里已还原为 `${HIP_LIBDIR}` + `BUILD_RPATH` 设置

### 3.5 部署与运行
```bash
# 部署 LibTorch 运行时到目标节点项目目录下；确认 CMakeLists 的 TORCH_ROOT / HIP_ROOT
cd torch_optest && mkdir -p build && cd build && cmake .. && make -j$(nproc)
./op_test          # 196 个算子依次 CPU vs DCU 执行并交叉对比
# 一键（含 PASS/FAIL 统计）: bash scripts/run_op_test.sh
```

### 3.6 验证结论（经验，非基线数据）
原文在 DTK 26.04 + Torch 2.1.0 环境跑通 **196/196 全部 PASS**（CPU vs DCU 交叉对比）。
> 注：原文 4.6 有两张 `![image]()` 空占位图（原始图未随文档导出），此处不引用图片、不编造日志；结论性经验可直接复用，但实测请以你本地 `op_test` 输出为准（遵守"汇报测试结果不裁剪/不自行下结论"的纪律）。
> 完整 196 项用例实现见 `scripts/op_test.cpp`；冒烟测试见 `scripts/demo.cpp`。

---

## 高版本 PyTorch（预编译 wheel，如 2.9.0）适配要点

> 实测来源：<device-B> 设备（<<internal-ip>>，BW100X8）`<container_vllm015>` 容器，DTK 26.04，已装 `torch 2.9.0+das.opt1.dtk2604.20260331.g4e3c1e7`（预编译 wheel，非源码构建）。

本 skill 的"阶段一源码编译"针对**低版本**（如 2.1.0，社区无对应 whl、必须自编译）。高版本（≥2.9.0）海光已提供**预编译 wheel**，走 `pip install` 即可，**不需要**重走源码编译，且阶段一的四个源码编译坑大多**已不适用**：

| 维度 | 低版本 2.1.0（源码编译） | 高版本 2.9.0（预编译 wheel，实测） | 高版本是否需要本文档的修复 |
|------|------|------|------|
| 获取方式 | 源码 `bdist_wheel` 自编译 | 直接装 `torch 2.9.0+das.opt1.dtk2604...whl` | 否，免编译 |
| 坑一 version-dev | 编译期必踩，需改 `x.y.z-26.04` | 容器内 `version-dev` 仍 `26.04`，但预编译 wheel **不触发** `LoadHIP.cmake`，运行时无碍 | 仅当你**从源码重编** 2.9.0 时才需 `00_fix_version_dev.sh` |
| 坑二 maxThreadsDim | `.hip` 文件需 sed 改 `auto` | 2.9.0 头已用 `->maxGridSize[...]` 正确类型，无 `int*` 误用 | 否 |
| 坑三 hipsparse const | 需补 const descriptor 宏 | 2.9.0 `HIPSparse.h` 已用 `HIPSPARSE_VERSION>=200400` 正确判定（不再依赖 `ROCM_VERSION>=50200` 旧分支） | 否（`patches/patch_hipsparse_h.sh` 对 2.9.0 不适用，脚本已幂等跳过，但**无需执行**） |
| 坑四 nvfuser | 需 `BUILD_NVFUSER=0` | 2.9.0 wheel **不含** nvfuser lib | 否 |

### 高版本仍需保留的动作
- **LibTorch C++ 提取（阶段二）仍然适用**：2.9.0 wheel 自带完整 `include/`（含 `torch/csrc/api/include/`）与 `lib/`。直接对**已安装的 torch 包目录**跑 `collect_libtorch.sh` 即可：
  ```bash
  # 高版本：PT 指向已安装的 torch 包目录（而非源码树）
  bash scripts/collect_libtorch.sh \
    /usr/local/lib/python3.10/dist-packages/torch \
    /path/to/torch-2.9.0
  ```
  之后阶段三的 `CMakeLists.txt` / `op_test.cpp` 验证框架可直接复用（链接 `torch torch_cpu torch_hip c10 c10_hip`）。
- **算子兼容性验证（阶段三）完全复用**：196 项 CPU vs DCU 交叉对比框架与高版本无关，可直接验证 2.9.0 的 DCU 后端数值精度。
- **`version-dev` 仍建议保持 `x.y.z` 格式**：若你后续基于 2.9.0 做二次源码编译（如自定义算子），`00_fix_version_dev.sh` 仍然需要。

> 结论：低版本走"源码编译 + 4 修复"全量流程；高版本（预编译 wheel）只需"装 wheel → 提取 LibTorch → 跑 196 项验证"，源码编译坑基本被上游修复吸收。版本差异本质是"海光是否已发布该版本的预编译 wheel"。

## 关键经验总结（落地 checklist）
1. **version-dev 是关键**：`/opt/dtk/.info/version-dev` 必须与 `rocm_version.h` 一致，采用 `x.y.z-26.04` 格式，否则 CMake 阶段即败。
2. **增量编译省时**：绝对避免 `setup.py clean`；用 ninja 增量把"修复-编译"循环压到分钟级。
3. **HIP API 需手动适配**：DTK26 的 const 描述符、设备属性类型与旧版不同，按坑二/三逐一修。
4. **RPATH 决定可移植性**：部署 LibTorch 必须 `patchelf --set-rpath '$ORIGIN'`，否则跨环境动态链接失败。
5. **容差分层**：不同 dtype/算子类别用差异化 atol，避免误报/漏报（见 3.2）。
6. **全量测试是底线**：适配后必须过 196 项全量交叉验证，不应仅凭 Smoke Test 判可用。

---

## 目录与脚本清单（scripts/）
| 文件 | 作用 |
|------|------|
| `env_setup.sh` | source 注入 DTK/PyTorch 编译环境变量 |
| `00_fix_version_dev.sh` | 修正 `/opt/dtk/.info/version-dev` 为 `x.y.z-26.04` |
| `01_prepare_source.sh` | checkout 2.1.0 DAS 分支、子模块、HIP 化 |
| `02_build_wheel.sh` | 增量编译 wheel（不 clean） |
| `collect_libtorch.sh` | 收集 so+include 并 patchelf 修正 RPATH |
| `patches/patch_maxThreadsDim.sh` | 坑二修复（auto 推断） |
| `patches/patch_hipsparse_h.sh` | 坑三修复（const descriptor 宏） |
| `patches/disable_nvfuser.sh` | 坑四修复（关 nvfuser） |
| `demo.cpp` | LibTorch DCU 冒烟测试（matmul + CPU 交叉对比） |
| `op_test.cpp` | 17 类 196 项算子全量验证框架 |
| `CMakeLists.txt` | demo + op_test 构建配置 |
| `run_op_test.sh` | 一键编译+运行+PASS/FAIL 统计 |

## 与兄弟 skill 关系
- `dcu-dtk`：DTK 安装/版本/`/opt/dtk` 布局基础。
- `dcu-env`：容器拓扑、`rocminfo` 查 gfx arch、CPU 绑核等环境确认。
- `dcu-vllm-op-replace`：torch 层适配之上再接 vLLM 算子替换；本 skill 是更底层的 PyTorch C++ 运行时适配。
- 入口与组合剧本见 `dcu-inference-infra`。

## 验证步骤（本 skill 自检）
- [x] 所有 `*.sh` 通过 `bash -n` 语法检查
- [x] `demo.cpp` / `op_test.cpp` 括号配平、含 `int main`、正常收尾
- [x] 修复原文三处渲染瑕疵（镜像 `<br>`、patchelf 截断、`${HIP_LIBDIR}` 被吞）
- [x] 来源已脱敏、性能/日志已最小化（遵守 skill 内容纪律）
