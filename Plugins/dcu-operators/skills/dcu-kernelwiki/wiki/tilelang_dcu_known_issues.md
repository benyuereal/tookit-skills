---
title: TileLang 在 DCU 的已知问题（实测汇总）
scope: [DCU, all gfx, TileLang]
source: "内部文档 三篇 tilelang 文档 (file_id=<internal-file-id> / <internal-file-id> / <internal-file-id>) / dcu-kernel-dev §四"
verified_on: "DCU 工程师实测 (K100-AI, DTK 2604, tilelang 0.1.7.post3 / 0.1.9 / 0.1.10)"
status: verified
---

# TileLang 在 DCU 的已知问题（实测汇总）

> 完整 12 项见 `dcu-kernel-dev` §四。此处列最高频命中的几条（agent 写 TileLang kernel 必查）。

| # | 问题 | 现象 / 根因 | 对策 |
|---|---|---|---|
| 1 | 后端选错 | 未显式 `target="hip"` → 编译出 CUDA 代码在 DCU 跑错 | `backend_target = "hip" if torch.version.hip else "cuda"` |
| 2 | warp_size | DCU=64，shfl 默认 32 → 越界返回自身值 → 规约结果错 | 显式 `warp_size=64` |
| 3 | alloc_shared 竞争写 | 误当寄存器用 → 结果随机、每跑一次不一样 | 精度敏感量用 `alloc_local` |
| 4 | 动态参数 M | 编译器不识别 Python 变量 | `T.dynamic('M','int32')` |
| 6 | hipprof libLLVM | `libLLVM-17git.so: cannot open` | `export LD_LIBRARY_PATH=/opt/dtk-.../dcc/lib:$LD_LIBRARY_PATH` |
| 9 | 不支持 CUDA arch 引 | `from tilelang.carver.arch import CUDA` 报错 | 改用 `from tilelang.carver.arch import CDNA` |
| 10 | 不支持 93 arch | Analyzer 的 ARCH_CONFIGS 无 93 | 等框架支持 |
| 12 | invalid argument | 一般资源越界 | 调低块尺寸；`assert total_shared <= max_shared` 提前发现 |

## 通用修复「资源越界 / invalid argument」

```python
assert total_shared <= max_shared, "Shared memory too high"
# 调低：block_M=32, block_N=32, num_stages=1, threads=128
```

## 通过性矩阵（结论性，非性能基线）

- `tilelang-0.1.10/examples/amd`：`autotune` 阶段 acc_s / acc_s_cast 布局推断冲突
  → 两者 dtype 不同，换 `acc_s_cast` 为 shared memory 做 cast 可解。
- `examples/attention_sink/example_mha_sink_fwd_bhsd.py`：OK，tilelang vs torch 加速比约 **3.85×**。
- `testing/python/dcu`：0.1.9 无法跑通；前期 0.1.6 可跑通。
