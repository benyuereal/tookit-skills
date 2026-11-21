# dcu-hytlass-source — hytlass（CUTLASS 3.5.0 的 HCU 移植）源码与用法

hytlass（HYGON HCU Templates for Linear Algebra Subroutines）是 CUTLASS 3.5.0 的 HCU 移植，C++ 模板库，
BSD-3-Clause。它补的是「用 C++ 模板搭一类 GEMM」这条中间路径——既不是用现成库，也不是从零写 HIP。

## 目录

```
dcu-hytlass-source/
└── SKILL.md
```

## 快速用法 / 要点

- 两套编程模型选一套：CUTLASS 2.x 风格（threadblock/warp + mmac + `ds_read_matrix`）/ 3.x CuTe（仓内称 HuTe）
- 11 个代表性示例按「是否 batch/group → 网格够不够 → epilogue 融不融」挑最近的一个改
- 门槛：DTK 25.10+ / CMake 3.19+ / C++17；上游是固定 commit，升级要整块 re-port
- 示例是 best practice 示范、**不保证最优**；要单 shape 极限仍走手写 ladder

> 本 skill 详细内容（示例清单、构建、与手写 HIP 的分界、已知边界）见 `SKILL.md`。
