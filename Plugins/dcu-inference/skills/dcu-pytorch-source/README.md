# dcu-pytorch-source — DCU PyTorch 源码编译适配与算子验证

在 DCU（海光）DTK 26.04 高版本栈上源码编译适配低版本 PyTorch（如 2.1.0），提取独立 LibTorch C++ 运行时，并通过 17 类 196 项算子 CPU vs DCU 交叉对比做数值精度验证。

内容整理自内部 DCU 实践，已脱敏。

## 安装 / 启用
本 skill 已随 `dcu-*` 技能包就位，无需额外安装：
```
<agent-home>/skills/dcu/inference/dcu-pytorch-source/
├── SKILL.md          # 完整流程、坑、表格、脚本索引
├── README.md         # 本文件
└── scripts/
    ├── env_setup.sh
    ├── 00_fix_version_dev.sh
    ├── 01_prepare_source.sh
    ├── 02_build_wheel.sh
    ├── collect_libtorch.sh
    ├── run_op_test.sh
    ├── CMakeLists.txt
    ├── demo.cpp
    ├── op_test.cpp
    └── patches/
        ├── patch_maxThreadsDim.sh
        ├── patch_hipsparse_h.sh
        └── disable_nvfuser.sh
```

## 快速用法
1. 在 DTK 26.04 容器内：`bash scripts/env_setup.sh`
2. 修 version-dev：`bash scripts/00_fix_version_dev.sh`
3. 准备源码并 HIP 化：`bash scripts/01_prepare_source.sh /path/to/pytorch`
4. 遇编译坑按需打 patch（`scripts/patches/*`），再 `bash scripts/02_build_wheel.sh /path/to/pytorch`
5. 提取 LibTorch：`bash scripts/collect_libtorch.sh /path/to/pytorch /path/to/torch-2.1.0`
6. 算子验证：`cd torch_optest && cmake .. && make -j`，`./op_test`（或 `bash scripts/run_op_test.sh`）

> 详细步骤、表、坑见 `SKILL.md`。所有 `*.sh` 已通过 `bash -n` 语法检查；`demo.cpp`/`op_test.cpp` 括号配平、含 `int main`。
