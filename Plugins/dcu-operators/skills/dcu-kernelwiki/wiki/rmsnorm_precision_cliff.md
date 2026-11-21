---
title: RMSNorm bf16 舍入悬崖（DCU 精度陷阱）
scope: [DCU, all gfx]
source: "内部文档 主文档 §4–§5 (file_id=<internal-file-id>) / dcu-kernel-dev §八"
verified_on: "DCU 工程师实测 (K100-AI, TileLang, bf16)"
status: verified
---

# RMSNorm bf16 舍入悬崖（DCU 精度陷阱）

## 现象

RMSNorm 用 block 级规约求整行 x² 和的标量 RMS。当 block_size 增大，规约的
**加法顺序改变** → IEEE754 舍入顺序变化 → bf16 写回撞**舍入悬崖（Rounding Cliff）**。

机理：`(A+B)+C != A+(B+C)`；block=1024 时 `local_sum` 在规约树最后阶段才相加远距元素，
顺序与 `torch.mean()` 不同；强转 bf16 时一方向上舍入、另一方下舍入，差达 1 ULP。

## 量级（示例，非基线，以本地实测为准）

| block_size | max diff |
|---|---|
| 256 | ~0 |
| 1024 | 0.03125（bf16 在 [4,8) 区间 1 ULP） |

## 判据（放行标准）

- **max diff < 0.03（≈1 ULP）视为安全范围内，可放行**；
- 否则回退较小 block 或对齐 torch 的规约顺序。
- 用独立 f32 reference 对比（见 `dcu-kernel-dev/scripts/tl_rmsnorm_precision_test.py`），
  输出 max/mean diff 与 NaN 检查。

## 优先级结论（方法论）

RMSNorm 在推理耗时中占比极小（量级 ~0.01%），对端到端吞吐几乎无影响。
**DCU 上应优先把精力投入 fa / gemm / elementwise 等热点算子**，
RMSNorm 做到「正确 + 不拖后腿」即可。

## 关联

- 调试工具：`dcu-kernel-dev/scripts/tl_rmsnorm_precision_test.py`（正确性）
- 端到端 profiling：`dcu-kernel-dev/scripts/tl_rmsnorm_benchmark.sh`（hipprof 包裹）
