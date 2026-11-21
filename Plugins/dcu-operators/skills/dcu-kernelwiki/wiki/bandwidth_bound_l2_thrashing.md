---
title: bandwidth-bound kernel 的 L2 thrashing 悬崖与 NVIDIA cache-hint 优化的 DCU 适用性
scope: [gfx936, BW150, BW100]
source: "知乎专栏《QKNorm+RoPE Kernel 优化：TMA Prefetch、L2 Cache 分层与 Occupancy 调优》(zhuanlan.zhihu.com/p/2018036507324216990, H100 NVL 实测) + <device-A>/<device-B> 真机对照实验"
verified_on: "2026-09-28 <device-A>(BW150) + <device-B>(BW100), gfx936, DTK 26.04 容器, hipcc --offload-arch=gfx936"
status: verified
---

# bandwidth-bound kernel 的 L2 thrashing 悬崖与 NVIDIA cache-hint 优化的 DCU 适用性

## 背景

NVIDIA 侧的 bandwidth-bound kernel 优化文章（如 QKNorm+RoPE：TMA prefetch 预热 L2 +
streaming store 防写污染 + evict-first load 防 read-once 数据 thrash L2）常报告
"数据量越大带宽越差"的悬崖（H100 naive 从 ~3400 掉到 ~2500 GB/s，-26%），
并靠 cache 分层策略救回。评估这类优化能否迁移 DCU 时，**先查目标卡 L2 大小**。

## 已验证事实（2026-09-28，<device-A> BW150 + <device-B> BW100）

1. **BW150/BW100 的 L2 = 8MB**（`hipDeviceProp_t.l2CacheSize` 实测，8388608 bytes）。
   H100 是 50MB，DCU 小 6 倍。
2. **8MB L2 上不出现 L2 thrashing 悬崖**：QKNorm+RoPE 型 kernel（read-once 大数据
   Q/K + 高复用小数据 cos_sin/weight）用 naive 直接 global load/store 实现，
   带宽随 token 数单调上升后平台化（~1360 GB/s），512→61440 tokens 全程不衰减。
   正确性 max_diff=0.0000。
3. **根因**：悬崖的本质是"小数据 fit L2（快）→ 大数据 overflow L2（thrash）"的相变。
   H100 50MB L2 存在这个相变（4k token 的 Q/K ~24MB fit，10k+ ~60MB overflow）；
   DCU 8MB L2 下数据从一开始就不 fit（1024 token 的 Q/K 已 8.4MB > 8MB），
   硬件直接流式走 HBM，无相变、无 thrashing。
4. **结论**：此类依赖 L2 容量相变的 NVIDIA cache-hint 优化（TMA prefetch /
   streaming store / evict-first load）在 BW 系列（小 L2）上**无 headroom**——
   naive 实现即 HBM 最优（实测已打满 copy 峰值 ~1360 GB/s）。
5. TMA 类指令（`cp.async.bulk.prefetch.tensor`）DCU 无对应硬件，直接排除；
   PTX cache hint（`ld.global.cs`/`ld.global.lu`）无 DCU 等价物，AMD 侧对应机制是
   buffer 指令 + buffer descriptor cache-control 位（本 DTK 无 nontemporal_store builtin）。

## 可复用方法（评估 NVIDIA kernel 优化文章迁移 DCU 的判据）

1. 先查目标卡 L2：`hipDeviceProp_t.l2CacheSize`（容器内 hipcc 编译 props 程序即可）。
2. 判断优化是否依赖"数据量跨 L2 容量"的相变：
   - 依赖（prefetch 预热 / evict-first / streaming store 防 thrash）→ 只在
     "工作集会从小于 L2 长到大于 L2"的卡上有收益；L2 小、数据天然不 fit 的卡上无效。
   - 不依赖（occupancy/寄存器压力、算法级 layout 如 dual-half 消 shuffle、
     向量化访存）→ 架构无关，可迁移，但收益上限受 HBM 带宽约束。
3. 任何"打满带宽"的结论先测纯 copy 的 HBM 实测峰值（多 size × 多 block 扫描），
   再对比 kernel 实测带宽，确认 headroom 是否存在。

## unverified

- **BW1100（gfx938）L2 大小未测**（2026-09-28 <device-alias> 跳板隧道不稳未连上）。
  若其 L2 显著大于 8MB，本结论对 gfx938 需修正。补测方法同上（props 程序 + naive kernel 带宽曲线）。
  > ✅ **2026-10-20 更新：BW1100 L2 = 8MB 已实测**（<device-D> 机，HIP l2CacheSize，P0 采集）——“若 L2 显著大于 8MB 需修正”对 BW1100 不成立，本结论对 gfx938 成立（其余指标见 dcu-perf-cases references/hwprobe-baseline-20261020）。

## 关联

- 性能数字（HBM 峰值、带宽曲线）→ `dcu-perf-cases` 案例 38
- bandwidth-bound kernel 的通用优化轴 → `dcu-kernel-dev` §15.2/§15.3
- profile 定位（stall/内存指标）→ `dcu-perf-profiling`
