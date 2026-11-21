---
title: Profile 驱动优化循环协议（DCU 版）
scope: [DCU, kernel optimization]
source: "github.com/Dominic789654/auto-kernel-research (SKILL.md) / dcu-op-optimization"
verified_on: "协议源自社区 auto-kernel-research；DCU 化映射：profiler→hipprof/rocprof, device→DCU"
status: verified
---

# Profile 驱动优化循环协议（DCU 版）

> 改编自 `Dominic789654/auto-kernel-research`（MIT）的纯协议。核心：**优化由 profiler 证据驱动，而非直觉**。
> 与 `dcu-op-optimization`（profile→定位→改→验证）同构，此处给可执行的循环骨架。

## 铁律：数字必须"赚来"

- 一个 metric 仅当来自**真实命令的真实输出**才算数。测量命令必须打印 `METRIC name=value` 行，只解析这些行。
- 一个"瓶颈声明"（memory-bound / occupancy 受限 / hotspot）**只能来自本次会话落到磁盘的 profiler 报告**。
- 正确性门禁（vs reference，在容差内）不过 → 再快也是废纸，结果不计。
- git 为真相：每个实验一次 commit；败者 `git reset --hard HEAD~1`。

## 循环（每轮）

1. **HEARTBEAT**：更新 `state/heartbeat`（迭代 + 时间戳）第一件事。
2. **THINK**：提一个假设，**必须引用 `profile_summary.md` 的某一行**（如 "38% stall 是 Long Scoreboard → tile load 进 LDS"）。无 profiler 依据 → 下一步是 profile 而非 edit。
3. **EDIT**：聚焦改动，≤5 文件，diff 小易测易回退。
4. **COMMIT**：先提交再运行，崩了也能 reset。
5. **CHECK**：正确性门禁退出 0。
6. **MEASURE**：跑 metric 命令，只解析 `METRIC` 行。无行 → `crash`，不是 0。
7. **DECIDE**：超 best-kept 噪声带 → keep；持平/变差 → loss 回退；crash/check 失败 → 回退。
8. **RE-PROFILE**（重大结构性胜出后）：重跑 profiler，刷新瓶颈图（瓶颈会移动）。
9. **LOG**：追加 `findings.jsonl`；更新 `progress.json`、`directions_tried.json`。

## 置信度：别为噪声庆祝

- GPU jitter 大，metric 跑 **5+ 次**；平台允许则锁时钟（`nvidia-smi --lock-gpu-clocks` 对应 DCU 用 `rocm-smi --setperflevel` 或固定频率）。
- 用 **MAD**（中位数绝对偏差）估噪声；`|delta| > 2×MAD` 才算真改善。
- <5 次测量或 MAD≈0 → 不能声称置信，只保留明确可重复的增益。

## DCU 适配映射

| 原协议 | DCU 对应 |
|---|---|
| profiler: ncu / nsys | `hipprof --hip-trace` / `rocprof` / `rocTracer` |
| device: H100 / MI300X | 物理机 `rocminfo` 查 gfx（容器不可靠）；编译用 `dcu_build_kernel.sh` 现查 |
| 时钟锁定 | `rocm-smi --setclock` / DTK 提供的频率固定手段 |
| 编译 | `hipcc --offload-arch=<探测gfx>`（绝不写死） |

## 优化方向菜单（让 profile 选，别盲扫）

- **memory-bound** → coalesce global 访问、经 LDS 分块、tiling 提 cache 复用、向量化 128-bit load、砍冗余流量、`__ldg` 只读路径。
- **compute-bound** → 用 Matrix Core / tensor core、提算术强度、融合减 launch、选更快 math（fp8 等 checks 容忍时）。
- **latency / 低占用** → 调 block size 与每 block 寄存器/LDS、提 ILP、async copy / double buffer 重叠、减 `__syncthreads`。
- **warp stall / 发散** → 重构分支、再平衡负载、重排访存藏延迟、解 bank conflict。
- **launch/overhead-bound** → 融合 kernel、CUDA/ROCm graph、persistent kernel、批小 launch。

## 反停滞

- `stale_count ≥ 2`：停调当前 frame 的旋钮（block/unroll），**重 profile 换瓶颈**。
- `stale_count ≥ 4`：标 `status:"stuck"`，换结构性方向（tiling 方案 / 内存布局 / 算法 / 并行策略）。
