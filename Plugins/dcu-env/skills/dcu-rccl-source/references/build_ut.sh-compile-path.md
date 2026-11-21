# build_ut.sh 编译/UT 流程（DTK 26.04.2 RCCL 参考）

> 本文件是对 DTK 26.04.2 RCCL 源码中 `build_ut.sh` 的提炼（源码树未随本包分发，原文以本地 DTK RCCL 源码为准），与源码树地图互补。用于在 DCU 上编译 RCCL 动态库 + UT 测试用例。

## 0. 一句话

`build_ut.sh` = 用 hipcc 编译 RCCL 动态库（libnccl.so）+ Googletest UT 测试用例，输出到 `rccl/build/test/`。UT 依赖 Googletest 1.10+。

## 1. 编译路径（build_ut.sh 提炼）

```bash
#!/bin/bash
# 用法：./build_ut.sh <HIPCC_PATH> <ROCM_PATH> <LIB_TARGET> <UT_TARGET>
# 例：./build_ut.sh /opt/rocm/bin/hipcc /opt/rocm /opt/rocm/lib/librccl.so /opt/rocm/lib/librccl_ut.so

HIPCC=${1:-/opt/rocm/bin/hipcc}
ROCM_PATH=${2:-/opt/rocm}
LIB_TARGET=${3:-/opt/rocm/lib/librccl.so}
UT_TARGET=${4:-/opt/rocm/lib/librccl_ut.so}

# 用 CXX=hipcc 编译动态库（CMakeLists 底层走 hipcc）
CXX=$HIPCC cmake .. -DBUILD_TESTS=ON
# 编译并输出
make -j 16
# 把 UT 库拷贝到目标位置
cp librccl_ut.so $UT_TARGET
```

关键点：
- **CXX=hipcc** 是 RCCL 编译核心（用 HIP 编译器，非 gcc）
- `-DBUILD_TESTS=ON` 才会连带编译 UT 用例
- 输出在 `rccl/build/test/` 目录（ut 用例编译产物）

## 2. UT 用例编译产物

- RCCL UT 名称格式：`CollectiveCall.<Type of test>`（如 `AllReduce*`、`AllGather*`、`ReduceScatter*`、`AllToAll*`）
- 全量测试命令：`cd rccl/build/test; ./rccl-UnitTests`
- 过滤测试：`./rccl-UnitTests --gtest_filter="AllReduce*"`
- UT 框架：Googletest 1.10 或以上，依赖跟随 RCCL 动态库一起构建

## 3. 与 CMakeLists/README 的关系

- `CMakeLists.txt` 是 RCCL 的 CMake 主配置（含 `ROCM_VERSION` 检测、`EXPLICIT_ROCM_VERSION` cache、`BUILD_TESTS=ON` 选项）
- `src/Makefile` 定义动态库 libnccl.so 的构建规则（`lib : staticlib binary`、`HIPCUFLAGS`、`GPU_TARGETS`）
- `README.md` 给出动态库编译法：`CXX=hipcc cmake ..; make -j 32`

## 4. 适合场景

- 在 DCU 上编译 RCCL 动态库/UT 时，参考本文件核对 hipcc 路径 / ROCM 路径 / BUILD_TESTS=ON 是否真实命中
- 调 PCIe 混合链路 env 后，用 `dcu-rccl-test` 实测调优（本文件只给编译路径，不留性能数字）

## 5. 注意事项

- 本文件不存性能基线/逐行日志；调优效果用 `dcu-rccl-test` 实测留痕。
- 行号随版本漂移；编译路径以本地 DTK RCCL 源码的 `build_ut.sh` 为准实时核验。
- 高危操作谨慎，优先测试机验证。